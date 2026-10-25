import type { SessionMessage, ToolResultSummary, ToolUseSummary } from 'claude-code'

/**
 * The relay's text, with no `$`: the word-for-word archive of a transcript and
 * the brief a fresh context opens on. Nothing here summarizes.
 */
export const HANDOFF_MARKER = '[agent-link · relais de session]'
export const MAX_REQUESTS = 25
export const REQUEST_CHARS = 1500
export const TAIL_CHARS = 20_000
export const TAIL_RESULT_CHARS = 1500
/** Characters per archive file: at most 4 bytes each in UTF-8, under `$.fs.write`'s 4 MiB. */
export const PART_CHARS = 1_000_000

const FILE_TOOLS: readonly string[] = ['Edit', 'Write', 'MultiEdit', 'NotebookEdit']
const REMINDER = /<system-reminder>[\s\S]*?<\/system-reminder>/g
const REQUESTS_BLOCK = 'agent-link:requests'
const FILES_BLOCK = 'agent-link:files'
const REQUEST_HEADING = /^### Demande \d+\n/m
const FENCE = '````'

export type ArchiveMeta = {
  sessionId: string
  loop: string
  trigger: string
  cwd: string
  at: number
}

export type Brief = {
  trigger: string
  archive: readonly string[]
  previous: readonly string[]
  messages: number
  requests: readonly string[]
  files: readonly string[]
  tail: string
}

function clip(text: string, max: number): string {
  if (text.length <= max) return text
  return `${text.slice(0, max)}\n… [${text.length - max} caractères coupés : texte entier dans l'archive]`
}

function fence(text: string): string {
  return [FENCE, text, FENCE].join('\n')
}

function isHandoff(message: SessionMessage): boolean {
  return message.role === 'user' && message.text.startsWith(HANDOFF_MARKER)
}

function withoutReminders(text: string): string {
  return text.replace(REMINDER, '').trim()
}

/** The text between `<!-- name -->` and `<!-- /name -->`, or '' when the block is absent. */
function blockOf(text: string, name: string): string {
  const open = `<!-- ${name} -->`
  const start = text.indexOf(open)
  const end = text.indexOf(`<!-- /${name} -->`)
  return start < 0 || end < start ? '' : text.slice(start + open.length, end).trim()
}

function block(name: string, body: string): string {
  return [`<!-- ${name} -->`, body, `<!-- /${name} -->`].join('\n')
}

/** Tool calls whose answer the transcript holds on the call itself, so a result row need not repeat it. */
function answeredIds(messages: readonly SessionMessage[]): ReadonlySet<string> {
  return new Set(
    messages.flatMap(message => message.toolUses.filter(use => use.text !== undefined).map(use => use.tool_use_id)),
  )
}

function renderToolUse(use: ToolUseSummary, resultChars: number): string {
  const head = `### → ${use.tool} (${use.tool_use_id})${use.isError === true ? ' [erreur]' : ''}`
  const input = fence(JSON.stringify(use.input, null, 2))
  return use.text === undefined ? `${head}\n${input}` : `${head}\n${input}\n${fence(clip(use.text, resultChars))}`
}

function renderResult(result: ToolResultSummary, resultChars: number): string {
  const head = `### ← résultat (${result.tool_use_id})${result.isError ? ' [erreur]' : ''}`
  return `${head}\n${fence(clip(result.text, resultChars))}`
}

/** One message as markdown, or '' when it holds nothing a reader needs (results shown on their call). */
function renderMessage(
  message: SessionMessage,
  index: number,
  answered: ReadonlySet<string>,
  resultChars: number,
): string {
  const text = message.text.trim()
  const uses = message.toolUses.map(use => renderToolUse(use, resultChars))
  const results = (message.toolResults ?? [])
    .filter(result => !answered.has(result.tool_use_id))
    .map(result => renderResult(result, resultChars))
  const body = [...(text === '' ? [] : [text]), ...uses, ...results]
  return body.length === 0 ? '' : [`## #${index + 1} · ${message.role}`, ...body].join('\n\n')
}

export function renderArchive(messages: readonly SessionMessage[], meta: ArchiveMeta): string {
  const answered = answeredIds(messages)
  const header = [
    `# Archive agent-link · session ${meta.sessionId}`,
    `Créée : ${new Date(meta.at).toISOString()} · déclencheur : ${meta.trigger} · boucle : ${meta.loop} · ${messages.length} messages`,
    `Dossier : ${meta.cwd}`,
    "Historique complet, mot pour mot, sans résumé. Cherche avec Grep (`## #12 · user`, `### → Edit`, un nom de fichier) plutôt que de tout lire.",
  ].join('\n')
  const body = messages.map((message, index) => renderMessage(message, index, answered, Infinity))
  return `${[header, ...body.filter(part => part !== '')].join('\n\n')}\n`
}

/** Cuts a long archive at line ends into parts of at most `max` characters. */
export function splitParts(text: string, max: number = PART_CHARS): string[] {
  if (text.length <= max) return [text]
  const cut = text.lastIndexOf('\n', max - 1)
  const at = cut > 0 ? cut + 1 : max
  return [text.slice(0, at), ...splitParts(text.slice(at), max)]
}

/** What the person asked, word for word, oldest first; a previous relay's list carried over. */
export function requestsOf(messages: readonly SessionMessage[]): string[] {
  const all = messages.flatMap(message => {
    if (message.role !== 'user') return []
    if (isHandoff(message)) {
      return blockOf(message.text, REQUESTS_BLOCK)
        .split(REQUEST_HEADING)
        .map(part => part.trim())
        .filter(part => part !== '')
    }
    const text = withoutReminders(message.text)
    return text === '' ? [] : [clip(text, REQUEST_CHARS)]
  })
  return all.slice(-MAX_REQUESTS)
}

function pathOf(use: ToolUseSummary): string | undefined {
  const path = use.input.file_path ?? use.input.notebook_path
  return typeof path === 'string' ? path : undefined
}

/** Files written or edited, each once, in the order first touched; a previous relay's list carried over. */
export function filesTouched(messages: readonly SessionMessage[]): string[] {
  const carried = messages
    .filter(isHandoff)
    .flatMap(message => blockOf(message.text, FILES_BLOCK).split('\n'))
    .map(line => line.replace(/^- /, '').trim())
  const touched = messages
    .flatMap(message => message.toolUses)
    .filter(use => FILE_TOOLS.includes(use.tool) && use.isError !== true)
    .map(pathOf)
  return [...new Set([...carried, ...touched])].filter((path): path is string => path !== undefined && path !== '')
}

type Tail = { picked: string[]; used: number; isFull: boolean }

/** The newest messages, word for word, within `budget` characters; tool results cut short. */
export function tailOf(messages: readonly SessionMessage[], budget: number = TAIL_CHARS): string {
  const answered = answeredIds(messages)
  const rendered = messages.map((message, index) =>
    isHandoff(message)
      ? `## #${index + 1} · user\n(relais précédent : ses demandes et fichiers sont repris plus haut)`
      : renderMessage({ ...message, text: withoutReminders(message.text) }, index, answered, TAIL_RESULT_CHARS),
  )
  const tail = rendered.reduceRight<Tail>(
    (acc, part) => {
      if (acc.isFull || part === '') return acc
      const used = acc.used + part.length
      if (used > budget && acc.picked.length > 0) return { ...acc, isFull: true }
      return { picked: [clip(part, budget), ...acc.picked], used, isFull: false }
    },
    { picked: [], used: 0, isFull: false },
  )
  return tail.picked.join('\n\n')
}

function renderRequests(requests: readonly string[]): string {
  return requests.map((request, index) => `### Demande ${index + 1}\n${request}`).join('\n\n')
}

export function renderBrief(brief: Brief): string {
  const requests = brief.requests.length === 0 ? '(aucune)' : block(REQUESTS_BLOCK, renderRequests(brief.requests))
  const files = brief.files.length === 0 ? '(aucun)' : block(FILES_BLOCK, brief.files.map(path => `- ${path}`).join('\n'))
  const older = brief.previous.length === 0 ? [] : [`Archives plus anciennes : ${brief.previous.join(' ; ')}`]
  return [
    HANDOFF_MARKER,
    `Ton contexte vient d'être remis à neuf (déclencheur : ${brief.trigger}). Rien n'a été résumé : les ${brief.messages} messages précédents sont archivés mot pour mot.`,
    `Archive : ${brief.archive.join(' ; ')}`,
    ...older,
    '',
    'Règles de reprise :',
    "- Ne suppose rien de l'historique qui n'est pas écrit ici : vérifie dans l'archive avec Grep ou un Read ciblé (offset/limit), jamais en entier.",
    "- Les fichiers font foi : relis un fichier avant de le modifier plutôt que de te fier à un souvenir.",
    "- Continue la tâche là où le dernier échange s'arrête, sans refaire ce qui est fait.",
    '',
    "## Demandes de l'utilisateur (mot pour mot, de la plus ancienne à la plus récente)",
    requests,
    '',
    '## Fichiers modifiés',
    files,
    '',
    `## Derniers échanges (mot pour mot ; résultats d'outils coupés à ${TAIL_RESULT_CHARS} caractères)`,
    brief.tail === '' ? '(aucun)' : brief.tail,
  ].join('\n')
}
