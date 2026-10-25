import type { AgentSpawnInput, EngineInterface, On } from 'claude-code'

import { handoffSection } from './handoff'
import { refreshAgents, refreshStatus } from './state'
import { BOARD_TOOL, ROSTER_TOOL } from './tools'

const SECTION_ID = 'agent-link:collab'
const MARKER = '[agent-link]'

/** Appended to a new subagent's or teammate's task: kept short, it rides every spawn. */
export const SPAWN_NOTE = `

${MARKER} Coordination : d'autres agents peuvent travailler en parallèle avec toi.
- \`${ROSTER_TOOL}\` : qui existe, son statut, et le nom à donner à SendMessage.
- \`${BOARD_TOOL}\` : action=read avant de commencer (évite les doublons) ; action=post pour une découverte utile aux autres (1 à 3 lignes factuelles, un topic court).
- SendMessage : seulement pour une question ou une info qui débloque un agent précis.
Messages courts et factuels, sans politesse ni répétition. Si ces outils ne sont pas disponibles, ignore cette note.`

export const MAIN_SECTION = `# Coordination des agents (mod agent-link)
Tu as \`${ROSTER_TOOL}\` (liste des agents : id, nom SendMessage, statut) et \`${BOARD_TOOL}\` (tableau partagé : post / read par topic).
Quand tu lances plusieurs agents en parallèle, ils reçoivent ces consignes : ils lisent le board, y postent leurs découvertes et s'écrivent avec SendMessage. Lis le board (action=read, since=...) pour suivre leur avancement au lieu de leur redemander.`

/** Resumes keep their own prompt, forks inherit the parent's context, workflow scripts own theirs. */
export function shouldInject(e: AgentSpawnInput): boolean {
  return e.resume === undefined && !e.fork && e.workflow === undefined && !e.prompt.includes(MARKER)
}

function offersAgentTools(tools: readonly string[]): boolean {
  return tools.includes(ROSTER_TOOL) || tools.includes(BOARD_TOOL)
}

async function refreshQuietly($: EngineInterface): Promise<void> {
  try {
    await refreshAgents($)
    await refreshStatus($)
  } catch (error) {
    $.ui.log(`refresh failed: ${String(error)}`, { to: 'debug' })
  }
}

export function hookCollab(on: On): void {
  on('agent.spawn', async ($, e, next) => {
    const spawned = await next(shouldInject(e) ? { ...e, prompt: `${e.prompt}${SPAWN_NOTE}` } : e)
    await refreshQuietly($)
    return spawned
  }).catch(($, e, next) => next(e))

  on('prompt.compose', async ($, e, next) => {
    const composed = await next(e)
    const handoff = await handoffSection($)
    const own = [
      ...(offersAgentTools(e.tools) ? [{ id: SECTION_ID, text: MAIN_SECTION, scope: 'session' as const }] : []),
      ...(handoff === undefined ? [] : [handoff]),
    ]
    if (own.length === 0) return composed
    const ids = own.map(section => section.id)
    const others = composed.sections.filter(section => !ids.includes(section.id))
    return { sections: [...others, ...own] }
  })

  on('turn.complete', async ($, e, next) => {
    const completed = await next(e)
    if (e.agentId !== undefined) await refreshQuietly($)
    return completed
  })
}
