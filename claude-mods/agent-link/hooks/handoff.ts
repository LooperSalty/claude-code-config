import type { EngineInterface, On, SessionCompactInput, SessionCompactResult } from 'claude-code'

import type { AgentLinkHandoff } from '../types'
import { filesTouched, renderArchive, renderBrief, requestsOf, splitParts, tailOf } from './handoff-text'
import { MAIN_LABEL, readHandoffs, recordHandoff } from './state'

/**
 * The relay: where the engine would compact a conversation into a summary,
 * the mod archives it word for word and hands the loop a fresh context that
 * opens on the person's requests, the files touched and the last exchanges.
 */
export const HANDOFF_COMMAND = 'relais'
const MODE_KEY = 'handoffMode'
const MODES = ['on', 'auto', 'off'] as const
type HandoffMode = (typeof MODES)[number]
const DEFAULT_MODE: HandoffMode = 'on'
const ARCHIVE_DIR = '.claude/agent-link/handoffs'
const SECTION_ID = 'agent-link:handoff'
const OLDER_ARCHIVES = 3
const PRECOMPUTE_SKIP = 'agent-link : relais actif, pas de résumé pré-calculé'

const MODE_TEXT: Readonly<Record<HandoffMode, string>> = {
  on: 'la compaction automatique et /compact passent par le relais (contexte neuf + archive mot pour mot)',
  auto: 'seule la compaction automatique passe par le relais ; /compact garde le résumé classique',
  off: 'compaction classique (résumé) partout',
}

function isMode(value: unknown): value is HandoffMode {
  return MODES.some(mode => mode === value)
}

export async function readMode($: EngineInterface): Promise<HandoffMode> {
  const stored = await $.store.get(MODE_KEY)
  return isMode(stored) ? stored : DEFAULT_MODE
}

function relays(mode: HandoffMode, trigger: SessionCompactInput['trigger']): boolean {
  return mode === 'on' || (mode === 'auto' && trigger === 'auto')
}

async function homeDir($: EngineInterface): Promise<string> {
  const home = (await $.env.get('USERPROFILE')) ?? (await $.env.get('HOME')) ?? (await $.session.root())
  return home.replace(/\\/g, '/').replace(/\/$/, '')
}

function stampOf(at: number): string {
  return new Date(at).toISOString().replace(/[:.]/g, '-')
}

function partPath(base: string, index: number): string {
  return index === 0 ? `${base}.md` : `${base}-p${index + 1}.md`
}

/** Writes the archive, records the relay, and answers the compaction with the fresh context. */
async function relay($: EngineInterface, e: SessionCompactInput): Promise<SessionCompactResult> {
  const at = await $.clock.now()
  const sessionId = await $.session.id()
  const loop = e.agentId ?? MAIN_LABEL
  const earlier = await readHandoffs($)
  const base = `${await homeDir($)}/${ARCHIVE_DIR}/${sessionId}/${stampOf(at)}-${loop}-r${earlier.length + 1}`
  const meta = { sessionId, loop, trigger: e.trigger, cwd: await $.session.cwd(), at }
  const parts = splitParts(renderArchive(e.messages, meta))
  const paths = parts.map((_, index) => partPath(base, index))
  await Promise.all(parts.map((text, index) => $.fs.write(partPath(base, index), text)))

  const previous = earlier.filter(one => one.loop === loop).map(one => one.archive)
  const brief = renderBrief({
    trigger: e.trigger,
    archive: paths,
    previous: previous.slice(-OLDER_ARCHIVES),
    messages: e.messages.length,
    requests: requestsOf(e.messages),
    files: filesTouched(e.messages),
    tail: tailOf(e.messages),
  })
  const handoff: AgentLinkHandoff = { at, trigger: e.trigger, loop, archive: paths[0] ?? base, parts: paths.length, messages: e.messages.length }
  await recordHandoff($, handoff)
  $.ui.toast(`Relais : contexte remis à neuf, ${e.messages.length} messages archivés`)
  return { messages: [{ role: 'user', text: brief, toolUses: [] }] }
}

function describeHandoff(handoff: AgentLinkHandoff): string {
  const time = new Date(handoff.at).toISOString().slice(11, 16)
  const parts = handoff.parts > 1 ? ` (+${handoff.parts - 1} parties)` : ''
  return `- ${time} ${handoff.loop} · ${handoff.trigger} · ${handoff.messages} messages → ${handoff.archive}${parts}`
}

async function statusText($: EngineInterface): Promise<string> {
  const mode = await readMode($)
  const handoffs = await readHandoffs($)
  const lines = handoffs.length === 0 ? ['Aucun relais dans cette session.'] : handoffs.map(describeHandoff)
  return [`Relais : ${mode} — ${MODE_TEXT[mode]}.`, 'Changer : /relais on | auto | off', ...lines].join('\n')
}

async function runCommand($: EngineInterface, args: string): Promise<string> {
  if (args === '' || args === 'status') return statusText($)
  if (!isMode(args)) return `Option inconnue « ${args} ». Utilise : /relais on | auto | off | status`
  await $.store.set(MODE_KEY, args)
  return `Relais : ${args} — ${MODE_TEXT[args]}.`
}

export async function registerHandoffCommand($: EngineInterface): Promise<void> {
  await $.command.register({
    name: HANDOFF_COMMAND,
    description: 'Relais de session à la place du résumé de compaction : contexte neuf + archive mot pour mot',
    argumentHint: '[on|auto|off|status]',
  })
}

function sectionText(handoff: AgentLinkHandoff): string {
  const parts = handoff.parts > 1 ? ` (et ${handoff.parts - 1} parties suivantes, -p2.md…)` : ''
  return `# Relais de session (mod agent-link)
Ton contexte a été relayé, pas résumé : l'historique complet d'avant est archivé mot pour mot dans ${handoff.archive}${parts}.
Pour tout détail ancien (décision, chemin, erreur, demande), cherche-le là avec Grep ou un Read ciblé au lieu de le deviner.`
}

export function hookHandoff(on: On): void {
  on('session.compact', async ($, e, next) => {
    const mode = await readMode($)
    if (e.trigger === 'precompute') return mode === 'off' ? next(e) : { skip: PRECOMPUTE_SKIP }
    if (!relays(mode, e.trigger) || e.messages.length === 0) return next(e)
    return relay($, e)
  }).catch(($, e, next) => {
    $.ui.log('agent-link : relais impossible, compaction classique à la place')
    return next(e)
  })

  on('command.run', { command: HANDOFF_COMMAND }, async ($, e) => ({
    text: await runCommand($, e.args.trim().toLowerCase()),
  })).catch(() => ({ text: 'agent-link : commande /relais indisponible pour le moment.' }))

}

/** The main prompt's pointer to the last relay's archive, once the main loop has been relayed. */
export async function handoffSection(
  $: EngineInterface,
): Promise<{ id: string; text: string; scope: 'session' } | undefined> {
  const last = (await readHandoffs($)).filter(one => one.loop === MAIN_LABEL).at(-1)
  return last === undefined ? undefined : { id: SECTION_ID, text: sectionText(last), scope: 'session' }
}
