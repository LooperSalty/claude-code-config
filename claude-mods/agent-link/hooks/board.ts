import type { EngineInterface } from 'claude-code'

import type { AgentLinkPost } from '../types'
import {
  MAIN_LABEL,
  addPost,
  appendEntry,
  excerpt,
  keepLast,
  labelOf,
  readAgents,
  readPosts,
  refreshStatus,
} from './state'

export const MAX_POST_CHARS = 2000
export const MAX_TOPIC_CHARS = 40
export const DEFAULT_TOPIC = 'general'
export const DEFAULT_READ_LIMIT = 30
export const MAX_READ_LIMIT = 100

export type BoardArgs = Readonly<Record<string, unknown>>

function asText(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

function topicOf(value: unknown): string {
  const topic = asText(value).toLowerCase().slice(0, MAX_TOPIC_CHARS)
  return topic === '' ? DEFAULT_TOPIC : topic
}

/** A `since` given as milliseconds since the epoch or as an ISO date; null when absent, NaN when bad. */
function sinceOf(value: unknown): number | null {
  if (value === undefined || value === null || value === '') return null
  if (typeof value === 'number') return value
  if (typeof value === 'string') return /^\d+$/.test(value) ? Number(value) : Date.parse(value)
  return Number.NaN
}

function limitOf(value: unknown): number {
  const asNumber = typeof value === 'number' ? Math.floor(value) : DEFAULT_READ_LIMIT
  return Math.min(MAX_READ_LIMIT, Math.max(1, asNumber))
}

function clockOf(at: number): string {
  return new Date(at).toISOString().slice(11, 19)
}

function formatPost(post: AgentLinkPost): string {
  return `#${post.seq} ${clockOf(post.at)} [${post.topic}] ${post.author}: ${post.text}`
}

export async function postToBoard(
  $: EngineInterface,
  agentId: string | undefined,
  args: BoardArgs,
): Promise<string> {
  const text = asText(args.text)
  if (text === '') return 'Erreur : `text` est requis pour post.'
  if (text.length > MAX_POST_CHARS) {
    return `Erreur : texte trop long (${text.length} > ${MAX_POST_CHARS} caractères). Résume.`
  }
  const topic = topicOf(args.topic)
  const at = await $.clock.now()
  const author = labelOf(await readAgents($), agentId)
  const post = await addPost($, { at, authorId: agentId ?? MAIN_LABEL, author, topic, text })
  await appendEntry($, { at, kind: 'post', from: author, to: 'board', topic, excerpt: excerpt(text) })
  await refreshStatus($)
  return `Posté #${post.seq} sur [${topic}] (horodatage ${at}).`
}

export async function readBoard($: EngineInterface, args: BoardArgs): Promise<string> {
  const since = sinceOf(args.since)
  if (Number.isNaN(since)) return 'Erreur : `since` doit être un nombre de millisecondes ou une date ISO.'
  const topic = asText(args.topic).toLowerCase()
  const now = await $.clock.now()
  const board = await readPosts($)
  const matching = board.filter(
    post => (topic === '' || post.topic === topic) && (since === null || post.at > since),
  )
  const shown = keepLast(matching, limitOf(args.limit))
  const scope = topic === '' ? '' : ` [${topic}]`
  const footer = `now=${now} (passe since=${now} pour ne lire que la suite)`
  if (shown.length === 0) return `Board vide${scope}. ${footer}`
  return [`${shown.length}/${matching.length} posts${scope} :`, ...shown.map(formatPost), footer].join('\n')
}

export async function handleBoard(
  $: EngineInterface,
  agentId: string | undefined,
  args: BoardArgs,
): Promise<string> {
  if (args.action === 'post') return postToBoard($, agentId, args)
  if (args.action === 'read') return readBoard($, args)
  return 'Erreur : `action` doit valoir "post" ou "read".'
}
