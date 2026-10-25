import type { EngineInterface, On, SessionReceiveOrigin } from 'claude-code'

import { MAIN_LABEL, appendEntry, excerpt, hasRecentMessage, labelOf, readAgents, refreshStatus } from './state'

/** A receive that repeats a send logged this recently (same text) is the same message. */
const DEDUPE_WINDOW_MS = 10_000
const AGENT_ORIGINS: readonly string[] = ['peer', 'coordinator', 'peer-send-message']

function senderOf(origin: SessionReceiveOrigin): string {
  if ('teammate' in origin) return origin.teammate
  if (origin.kind === 'coordinator') return MAIN_LABEL
  return 'pair'
}

function isAgentDelivery(origin: SessionReceiveOrigin): boolean {
  return AGENT_ORIGINS.includes(origin.kind)
}

async function logSend(
  $: EngineInterface,
  from: string | undefined,
  to: string,
  text: string,
  isDelivered: boolean,
): Promise<void> {
  const agents = await readAgents($)
  const at = await $.clock.now()
  await appendEntry($, { at, kind: 'message', from: labelOf(agents, from), to, excerpt: excerpt(text), isDelivered })
}

async function logReceive(
  $: EngineInterface,
  origin: SessionReceiveOrigin,
  to: string | undefined,
  text: string,
): Promise<void> {
  const at = await $.clock.now()
  const line = excerpt(text)
  if (await hasRecentMessage($, line, at, DEDUPE_WINDOW_MS)) return
  const agents = await readAgents($)
  await appendEntry($, { at, kind: 'message', from: senderOf(origin), to: labelOf(agents, to), excerpt: line, isDelivered: true })
}

/** Logging never blocks a message: a failure goes to the debug log. */
async function quietly($: EngineInterface, work: () => Promise<void>): Promise<void> {
  try {
    await work()
    await refreshStatus($)
  } catch (error) {
    $.ui.log(`journal failed: ${String(error)}`, { to: 'debug' })
  }
}

export function hookJournal(on: On): void {
  on('session.send', async ($, e, next) => {
    const sent = await next(e)
    await quietly($, () => logSend($, e.agentId, e.to, e.text, sent.isDelivered))
    return sent
  }).catch(($, e, next) => next(e))

  on('session.receive', async ($, e, next) => {
    const received = await next(e)
    if (received.consumed !== undefined || !isAgentDelivery(e.origin)) return received
    await quietly($, () => logReceive($, e.origin, e.agentId, e.text))
    return received
  }).catch(($, e, next) => next(e))
}
