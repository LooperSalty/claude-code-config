import type { EngineInterface, On } from 'claude-code'

import type { AgentLinkAgent, AgentLinkEntry } from '../types'
import { clearBoardAndJournal, isActive, readAgents, readJournal, refreshAgents, refreshStatus } from './state'

export const PANE = 'agents-chat'
export const COMMAND = 'agents-chat'
const TITLE = 'Agents'
const MIN_JOURNAL_ROWS = 5
const CHROME_ROWS = 4

const STATUS_ICONS: Readonly<Record<string, string>> = {
  pending: '…',
  running: '●',
  waiting: '◐',
  idle: '○',
  completed: '✓',
  failed: '✗',
  killed: '■',
}

function iconOf(status: string): string {
  return STATUS_ICONS[status] ?? '?'
}

function timeOf(at: number): string {
  return new Date(at).toISOString().slice(11, 16)
}

function lineOf(entry: AgentLinkEntry): string {
  if (entry.kind === 'post') {
    return `${timeOf(entry.at)} [${entry.topic ?? 'general'}] ${entry.from} : ${entry.excerpt}`
  }
  const failed = entry.isDelivered === false ? ' (non délivré)' : ''
  return `${timeOf(entry.at)} ${entry.from} → ${entry.to}${failed} : ${entry.excerpt}`
}

function agentLabel(agent: AgentLinkAgent, viewed: string | undefined): string {
  return `${agent.id === viewed ? '▶ ' : ''}${iconOf(agent.status)} ${agent.name ?? agent.type} `
}

async function switchView($: EngineInterface, agentId: string | undefined): Promise<void> {
  const moved = await $.ui.view(agentId === undefined ? {} : { agentId })
  if (moved.deny !== undefined) $.ui.toast(`Vue inchangée : ${moved.deny}`)
}

async function clearAll($: EngineInterface): Promise<string> {
  await clearBoardAndJournal($)
  await refreshStatus($)
  return 'agent-link : board et journal vidés.'
}

export async function registerCommand($: EngineInterface): Promise<void> {
  await $.command.register({
    name: COMMAND,
    description: 'Ouvre le pane Agents (agents actifs et leurs échanges) ; "clear" vide le board et le journal',
    argumentHint: '[clear]',
  })
}

export function hookPane(on: On): void {
  on('command.run', { command: COMMAND }, async ($, e) => {
    if (e.args.trim() === 'clear') return { text: await clearAll($) }
    await refreshAgents($)
    const opened = await $.ui.open({ id: PANE, title: TITLE })
    return { text: opened.isPlaced ? 'Pane Agents ouvert.' : `Pane Agents en attente : ${opened.reason}` }
  }).catch(() => ({ text: 'agent-link : impossible d’ouvrir le pane Agents.' }))

  on('ui.render', { component: 'Pane', requestId: PANE }, async ($, e) => {
    const { Box, Text, Button } = $.ui.resolve(e)
    const agents = await readAgents($)
    const journal = await readJournal($)
    const viewed = e.props.view.agentId
    const room = Math.max(MIN_JOURNAL_ROWS, e.props.scroll.bodyRows - agents.length - CHROME_ROWS)
    const active = agents.filter(isActive).length
    return (
      <Box flexDirection="column">
        <Text weight="bold">{`Agents · ${active} actifs / ${agents.length}`}</Text>
        <Button key="main" plain onPress={() => switchView($, undefined)}>
          {`${viewed === undefined ? '▶ ' : ''}◆ Fil principal`}
        </Button>
        {agents.length === 0 && <Text dimColor>Aucun agent pour l’instant.</Text>}
        {agents.map(agent => (
          <Button key={`agent-${agent.id}`} plain onPress={() => switchView($, agent.id)}>
            {agentLabel(agent, viewed)}
            <Text dimColor>{`${agent.type} · ${agent.description}`}</Text>
          </Button>
        ))}
        <Text weight="bold">{`Échanges · ${journal.length}`}</Text>
        {journal.length === 0 && <Text dimColor>Pas encore de message ni de post.</Text>}
        {journal.slice(-room).map(entry => (
          <Text wrap="truncate-end" dimColor={entry.kind === 'post'}>
            {lineOf(entry)}
          </Text>
        ))}
      </Box>
    )
  })
}
