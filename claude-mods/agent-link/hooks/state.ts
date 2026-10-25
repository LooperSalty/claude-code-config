import { atom, read, update } from 'claude-code'
import type { AgentInfo, EngineInterface } from 'claude-code'

import type { AgentLinkAgent, AgentLinkEntry, AgentLinkHandoff, AgentLinkPost } from '../types'

/**
 * Every `$.state` read and write of the mod lives here: the scan that checks
 * the module wants each atom used in the file that declares it.
 */
export const MAX_ENTRIES = 200
export const EXCERPT_CHARS = 160
export const MAIN_LABEL = 'main'
export const MAX_HANDOFFS = 20

const ENDED_STATUSES: readonly string[] = ['completed', 'failed', 'killed']

const agentsAtom = atom({ plugin: 'agent-link', key: 'agents' } as const, [])
const boardAtom = atom({ plugin: 'agent-link', key: 'board' } as const, [])
const journalAtom = atom({ plugin: 'agent-link', key: 'journal' } as const, [])
const seqAtom = atom({ plugin: 'agent-link', key: 'seq' } as const, 0)
const handoffsAtom = atom({ plugin: 'agent-link', key: 'handoffs' } as const, [])

/** Keeps the newest `max` items of a list, as a new list. */
export function keepLast<T>(list: readonly T[], max: number): T[] {
  return list.length <= max ? [...list] : list.slice(list.length - max)
}

/** One line, at most `max` characters, for logs and the pane. */
export function excerpt(text: string, max = EXCERPT_CHARS): string {
  const flat = text.replace(/\s+/g, ' ').trim()
  return flat.length <= max ? flat : `${flat.slice(0, max - 1)}…`
}

export function isActive(agent: AgentLinkAgent): boolean {
  return !ENDED_STATUSES.includes(agent.status)
}

/** A JSON-safe snapshot of one agent: no field is ever `undefined`. */
export function toSnapshot(info: AgentInfo): AgentLinkAgent {
  return {
    id: info.id,
    type: info.type,
    description: info.description,
    status: info.status,
    ...(info.name === undefined ? {} : { name: info.name }),
    ...(info.parentId === undefined ? {} : { parentId: info.parentId }),
  }
}

export function labelOf(agents: readonly AgentLinkAgent[], agentId: string | undefined): string {
  if (agentId === undefined) return MAIN_LABEL
  const agent = agents.find(one => one.id === agentId)
  return agent?.name ?? agent?.type ?? agentId.slice(0, 8)
}

function randomId(): string {
  return Math.random().toString(36).slice(2, 10)
}

export async function readAgents($: EngineInterface): Promise<AgentLinkAgent[]> {
  return read($, agentsAtom)
}

export async function readJournal($: EngineInterface): Promise<AgentLinkEntry[]> {
  return read($, journalAtom)
}

export async function readPosts($: EngineInterface): Promise<AgentLinkPost[]> {
  return read($, boardAtom)
}

/** Reads `$.agent.list()` and stores it when it changed; answers the fresh list. */
export async function refreshAgents($: EngineInterface): Promise<AgentLinkAgent[]> {
  const fresh = (await $.agent.list()).map(toSnapshot)
  const current = await read($, agentsAtom)
  if (JSON.stringify(current) !== JSON.stringify(fresh)) await update($, agentsAtom, () => fresh)
  return fresh
}

/** Adds one line to the journal, keeping it bounded. */
export async function appendEntry($: EngineInterface, entry: Omit<AgentLinkEntry, 'id'>): Promise<void> {
  const line: AgentLinkEntry = { id: randomId(), ...entry }
  await update($, journalAtom, list => keepLast([...list, line], MAX_ENTRIES))
}

/** Whether the journal already holds this message text from the last `windowMs`. */
export async function hasRecentMessage($: EngineInterface, text: string, at: number, windowMs: number): Promise<boolean> {
  const journal = await read($, journalAtom)
  return journal.some(entry => entry.kind === 'message' && entry.excerpt === text && at - entry.at <= windowMs)
}

/** Stores a post under the next sequence number, keeping the board bounded; answers the stored post. */
export async function addPost($: EngineInterface, draft: Omit<AgentLinkPost, 'seq'>): Promise<AgentLinkPost> {
  const seq = await update($, seqAtom, n => n + 1)
  const post: AgentLinkPost = { seq, ...draft }
  await update($, boardAtom, list => keepLast([...list, post], MAX_ENTRIES))
  return post
}

export async function readHandoffs($: EngineInterface): Promise<AgentLinkHandoff[]> {
  return read($, handoffsAtom)
}

/** Notes one relay of the session, keeping the list bounded. */
export async function recordHandoff($: EngineInterface, handoff: AgentLinkHandoff): Promise<void> {
  await update($, handoffsAtom, list => keepLast([...list, handoff], MAX_HANDOFFS))
}

export async function clearBoardAndJournal($: EngineInterface): Promise<void> {
  await update($, boardAtom, () => [])
  await update($, journalAtom, () => [])
}

/** The status line: shown only while the session has at least one agent. */
export async function refreshStatus($: EngineInterface): Promise<void> {
  const agents = await read($, agentsAtom)
  if (agents.length === 0) {
    $.ui.status(undefined)
    return
  }
  const journal = await read($, journalAtom)
  const active = agents.filter(isActive).length
  const messages = journal.filter(entry => entry.kind === 'message').length
  const posts = journal.length - messages
  $.ui.status(`🔗 ${active}/${agents.length} agents · ${messages} msgs · ${posts} posts`)
}
