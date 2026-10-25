/** One agent of the session as the mod last saw it in `$.agent.list()`. */
export type AgentLinkAgent = {
  id: string
  name?: string
  type: string
  description: string
  status: string
  parentId?: string
}

/** One post of the shared board. */
export type AgentLinkPost = {
  seq: number
  at: number
  authorId: string
  author: string
  topic: string
  text: string
}

/** One line of the exchange journal: a message between agents, or a board post. */
export type AgentLinkEntry = {
  id: string
  at: number
  kind: 'message' | 'post'
  from: string
  to: string
  excerpt: string
  topic?: string
  isDelivered?: boolean
}

/** One relay: a compaction the mod answered with a fresh context and a word-for-word archive. */
export type AgentLinkHandoff = {
  at: number
  trigger: string
  /** `main`, or the id of the subagent whose loop was relayed. */
  loop: string
  /** The archive's first file; further parts sit beside it as `-p2.md`, `-p3.md`… */
  archive: string
  parts: number
  messages: number
}

declare module 'claude-code' {
  interface PluginState {
    'agent-link': {
      agents: AgentLinkAgent[]
      board: AgentLinkPost[]
      journal: AgentLinkEntry[]
      seq: number
      handoffs: AgentLinkHandoff[]
    }
  }
}
