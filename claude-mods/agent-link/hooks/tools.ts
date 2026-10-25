import type { EngineInterface, On } from 'claude-code'

import type { AgentLinkAgent } from '../types'
import { handleBoard } from './board'
import { MAIN_LABEL, isActive, labelOf, refreshAgents, refreshStatus } from './state'

export const ROSTER_TOOL = 'mcp__agent-link__roster'
export const BOARD_TOOL = 'mcp__agent-link__board'

const ROSTER_SPEC = {
  name: 'roster',
  description:
    "Liste les agents de cette session (subagents, teammates) : id, nom à donner à SendMessage, type, statut, parent et tâche. Sert à savoir qui existe et à qui écrire. Indique aussi qui tu es.",
  inputSchema: {
    type: 'object',
    properties: {
      includeEnded: {
        type: 'boolean',
        description: 'Inclure aussi les agents terminés (completed, failed, killed). Défaut : false.',
      },
    },
  },
  isDeferred: false,
} as const

const BOARD_SPEC = {
  name: 'board',
  description:
    "Tableau partagé entre tous les agents de la session. action=post publie une découverte courte et factuelle sous un topic ; action=read lit les posts (tout, un topic, ou depuis un horodatage). Évite de dupliquer le travail des autres agents et de tout faire remonter au parent.",
  inputSchema: {
    type: 'object',
    properties: {
      action: { type: 'string', enum: ['post', 'read'] },
      topic: { type: 'string', description: 'Sujet court (ex. "api", "bugs"). Défaut : general.' },
      text: { type: 'string', description: 'Pour post : le contenu, court et factuel (2000 caractères max).' },
      since: {
        type: ['number', 'string'],
        description: 'Pour read : ne lire que les posts après cet horodatage (ms, ou date ISO).',
      },
      limit: { type: 'number', description: 'Pour read : nombre maximum de posts (défaut 30, max 100).' },
    },
    required: ['action'],
  },
  isDeferred: false,
} as const

function describeAgent(agents: readonly AgentLinkAgent[], agent: AgentLinkAgent): string {
  const address = agent.name ?? agent.id
  const parent = agent.parentId === undefined ? MAIN_LABEL : labelOf(agents, agent.parentId)
  return `- ${agent.name ?? agent.type} [${agent.type}] ${agent.status} · SendMessage to="${address}" · id=${agent.id} · parent=${parent} · « ${agent.description} »`
}

export async function rosterText(
  $: EngineInterface,
  callerId: string | undefined,
  includeEnded: boolean,
): Promise<string> {
  const agents = await refreshAgents($)
  await refreshStatus($)
  const shown = includeEnded ? agents : agents.filter(isActive)
  const self = callerId === undefined ? 'le fil principal (main)' : `${labelOf(agents, callerId)} (id=${callerId})`
  const lines = shown.map(agent => describeAgent(agents, agent))
  const body = lines.length === 0 ? ['Aucun agent actif.'] : lines
  return [`Tu es : ${self}.`, `Agents (${shown.length}/${agents.length}) :`, ...body].join('\n')
}

export async function registerTools($: EngineInterface): Promise<void> {
  await $.tool.register(ROSTER_SPEC)
  await $.tool.register(BOARD_SPEC)
}

export function hookTools(on: On): void {
  on('tool.call', { tool: ROSTER_TOOL }, async ($, e) => ({
    result: await rosterText($, e.agentId, e.includeEnded === true),
  })).catch(() => ({ deny: 'agent-link : roster indisponible pour le moment.' }))

  on('tool.call', { tool: BOARD_TOOL }, async ($, e) => ({
    result: await handleBoard($, e.agentId, e),
  })).catch(() => ({ deny: 'agent-link : board indisponible pour le moment.' }))
}
