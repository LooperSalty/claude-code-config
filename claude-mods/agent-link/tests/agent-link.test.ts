import { describe, expect, mock, test } from 'claude-code/testing'
import type { AgentInfo, AgentSpawnInput, On, PromptComposeInput, SessionMessage } from 'claude-code'

const SCOUT: AgentInfo = {
  id: 'agent-aaa111',
  name: 'scout',
  type: 'Explore',
  description: 'Cherche les routes API',
  status: 'running',
}
const WRITER: AgentInfo = {
  id: 'agent-bbb222',
  type: 'general-purpose',
  description: 'Écrit les tests',
  status: 'completed',
  parentId: 'agent-aaa111',
}

const SPAWN: AgentSpawnInput = {
  tool_use_id: 'toolu_spawn',
  prompt: 'Cherche les routes API.',
  description: 'Routes API',
  subagentType: 'general-purpose',
  provider: { plugin: 'engine', tier: 'core' },
  parentModel: 'claude-opus-5-5',
  background: false,
  fork: false,
}

const COMPOSE: PromptComposeInput = {
  model: 'claude-opus-5-5',
  promptModel: 'claude-opus-5-5',
  surfaces: ['terminal'],
  tools: ['Read', 'mcp__agent-link__roster', 'mcp__agent-link__board'],
  outputStyle: null,
  traits: [],
}

/** The world beneath the plugin: a clock, the agent list, and a status line that records. */
function world(on: On, agents: readonly AgentInfo[]): string[] {
  const statuses: string[] = []
  mock.clock(on, { now: Date.parse('2026-10-25T12:00:00Z') })
  on('agent.list', () => ({ value: [...agents] }))
  on('ui.status', ($, e) => {
    statuses.push(e.text ?? '')
    return { value: undefined }
  })
  on('ui.log', () => ({ value: undefined }))
  return statuses
}

describe('session start', () => {
  test('registers both tools in the prompt and the /agents-chat command', async ($, on) => {
    world(on, [])
    const tools: { name: string; isDeferred?: boolean }[] = []
    const commands: string[] = []
    on('session.start', ($, e) => ({ cwd: e.cwd }))
    on('tool.register', ($, e) => {
      tools.push({ name: e.name, isDeferred: e.isDeferred })
      return { value: { tool: `mcp__agent-link__${e.name}` } }
    })
    on('command.register', ($, e) => {
      commands.push(e.name)
      return { value: { command: e.name } }
    })
    await $.session.start({ cwd: '/work', surface: 'terminal', isInteractive: true })
    expect(tools).toEqual([
      { name: 'roster', isDeferred: false },
      { name: 'board', isDeferred: false },
    ])
    expect(commands).toEqual(['agents-chat', 'relais'])
  })
})

describe('roster', () => {
  test('lists active agents with their SendMessage address and says who calls', async ($, on) => {
    const statuses = world(on, [SCOUT, WRITER])
    const ran = await $.tool.call({ tool: 'mcp__agent-link__roster' })
    expect(ran.deny).toBeUndefined()
    const text = String(ran.result)
    expect(text).toContain('Tu es : le fil principal')
    expect(text).toContain('SendMessage to="scout"')
    expect(text).not.toContain('agent-bbb222')
    expect(statuses.at(-1)).toContain('🔗 1/2 agents')
  })

  test('includes ended agents on request', async ($, on) => {
    world(on, [SCOUT, WRITER])
    const ran = await $.tool.call({ tool: 'mcp__agent-link__roster', includeEnded: true })
    const text = String(ran.result)
    expect(text).toContain('id=agent-bbb222')
    expect(text).toContain('parent=scout')
  })
})

describe('board', () => {
  test('posts, then reads all, by topic and since a timestamp', async ($, on) => {
    world(on, [SCOUT])
    const first = await $.tool.call({ tool: 'mcp__agent-link__board', action: 'post', topic: 'API', text: 'GET /users ok' })
    expect(String(first.result)).toContain('Posté #1 sur [api]')
    const stamp = Number(/horodatage (\d+)/.exec(String(first.result))?.[1])
    await $.tool.call({ tool: 'mcp__agent-link__board', action: 'post', topic: 'bugs', text: 'NPE dans login' })

    const all = String((await $.tool.call({ tool: 'mcp__agent-link__board', action: 'read' })).result)
    expect(all).toContain('2/2 posts')
    expect(all).toContain('[api] main: GET /users ok')

    const bugs = String((await $.tool.call({ tool: 'mcp__agent-link__board', action: 'read', topic: 'bugs' })).result)
    expect(bugs).toContain('1/1 posts [bugs]')
    expect(bugs).not.toContain('GET /users')

    const after = String((await $.tool.call({ tool: 'mcp__agent-link__board', action: 'read', since: stamp })).result)
    expect(after).toContain('Board vide')
  })

  test('refuses an empty post and an unknown action', async ($, on) => {
    world(on, [])
    const empty = await $.tool.call({ tool: 'mcp__agent-link__board', action: 'post', text: '  ' })
    expect(String(empty.result)).toContain('`text` est requis')
    const unknown = await $.tool.call({ tool: 'mcp__agent-link__board', action: 'delete' })
    expect(String(unknown.result)).toContain('"post" ou "read"')
  })
})

describe('collaboration note', () => {
  test('is appended to a new subagent, not to a fork or a resume', async ($, on) => {
    world(on, [SCOUT])
    const prompts: string[] = []
    on('agent.spawn', ($, e) => {
      prompts.push(e.prompt)
      return { model: 'haiku', agentId: 'agent-new' }
    })
    await $.agent.spawn(SPAWN)
    await $.agent.spawn({ ...SPAWN, tool_use_id: 'toolu_fork', fork: true })
    await $.agent.spawn({ ...SPAWN, tool_use_id: 'toolu_resume', resume: { agentId: 'agent-aaa111' } })

    expect(prompts[0]).toContain('[agent-link] Coordination')
    expect(prompts[0]).toContain('mcp__agent-link__board')
    expect(prompts[1]).toBe(SPAWN.prompt)
    expect(prompts[2]).toBe(SPAWN.prompt)
  })

  test('adds a session section to the main prompt only when the tools are offered', async ($, on) => {
    on('prompt.compose', () => ({ sections: [{ id: 'intro', text: 'Intro', scope: 'shared' as const }] }))
    const withTools = await $.prompt.compose(COMPOSE)
    expect(withTools.sections.map(section => section.id)).toEqual(['intro', 'agent-link:collab'])
    expect(withTools.sections.at(-1)?.scope).toBe('session')

    const without = await $.prompt.compose({ ...COMPOSE, tools: ['Read'] })
    expect(without.sections.map(section => section.id)).toEqual(['intro'])
  })
})

describe('journal and pane', () => {
  test('logs a sent message and a board post, and draws them in the Agents pane', async ($, on) => {
    world(on, [SCOUT])
    on('session.send', () => ({ isDelivered: true as const }))
    await $.tool.call({ tool: 'mcp__agent-link__roster' })
    await $.session.send({ to: 'main', text: 'Routes trouvées :\n12 endpoints', origin: { kind: 'model' }, agentId: SCOUT.id })
    await $.tool.call({ tool: 'mcp__agent-link__board', action: 'post', topic: 'api', text: '12 endpoints' })

    for (const surface of ['terminal', 'desktop'] as const) {
      const ui = await $.ui.mount({
        plugin: 'agent-link',
        surface,
        component: 'Pane',
        requestId: 'agents-chat',
        props: {
          title: 'Agents',
          isFocused: false,
          bodyColumns: 80,
          placement: 'dock',
          scroll: { offset: 0, bodyRows: 20 },
          view: {},
        },
      })
      expect(await ui.find({ type: 'Button', key: 'agent-agent-aaa111' })).toBeDefined()
      expect(await ui.find({ type: 'Text', text: 'scout → main : Routes trouvées : 12 endpoints' })).toBeDefined()
      expect(await ui.find({ type: 'Text', text: '[api] scout : 12 endpoints' })).toBeUndefined()
      expect(await ui.find({ type: 'Text', text: '[api] main : 12 endpoints' })).toBeDefined()
    }
  })
})

const TRANSCRIPT: SessionMessage[] = [
  { role: 'user', text: 'Ajoute un endpoint /health <system-reminder>bruit injecté</system-reminder>', toolUses: [] },
  {
    role: 'assistant',
    text: 'Je modifie le serveur.',
    toolUses: [
      {
        tool_use_id: 'toolu_1',
        tool: 'Edit',
        input: { file_path: 'src/server.ts', old_string: 'a', new_string: 'b' },
        text: 'Fichier modifié',
      },
    ],
  },
  { role: 'user', text: '', toolUses: [], toolResults: [{ tool_use_id: 'toolu_1', text: 'Fichier modifié', isError: false }] },
  { role: 'assistant', text: 'Endpoint ajouté.', toolUses: [] },
]

const ARCHIVE = 'C:/Users/dev/.claude/agent-link/handoffs/sess-1/2026-10-25T12-00-00-000Z-main-r1.md'
const PRESENTATION = { isFullscreen: false, columns: 120 }

type Write = { path: string; text: string }

/** The world a relay needs: clock, store, home folder, session facts, and a file system that records. */
function handoffWorld(on: On, store: Readonly<Record<string, unknown>> = {}): Write[] {
  const writes: Write[] = []
  mock.clock(on, { now: Date.parse('2026-10-25T12:00:00Z') })
  mock.store(on, store)
  mock.env(on, { USERPROFILE: 'C:\\Users\\dev' })
  on('session.id', () => ({ value: 'sess-1' }))
  on('session.cwd', () => ({ value: '/work' }))
  on('fs.write', ($, e) => {
    writes.push({ path: e.path.replace(/\\/g, '/'), text: e.text })
    return { value: undefined }
  })
  on('ui.toast', () => ({ value: undefined }))
  on('ui.log', () => ({ value: undefined }))
  return writes
}

/** The engine's own summarizer, beneath the plugin: records whether it ran. */
function summarizer(on: On): { runs: number } {
  const seen = { runs: 0 }
  on('session.compact', () => {
    seen.runs += 1
    return { messages: [{ role: 'user' as const, text: 'résumé', toolUses: [] }] }
  })
  return seen
}

describe('session relay', () => {
  test('answers an automatic compaction with a fresh context and a word-for-word archive', async ($, on) => {
    const writes = handoffWorld(on)
    const engine = summarizer(on)
    const compacted = await $.session.compact({ trigger: 'auto', messages: TRANSCRIPT })

    expect(engine.runs).toBe(0)
    expect(compacted.messages?.length).toBe(1)
    const brief = compacted.messages?.[0]?.text ?? ''
    expect(brief.startsWith('[agent-link · relais de session]')).toBe(true)
    expect(brief).toContain(`Archive : ${ARCHIVE}`)
    expect(brief).toContain('### Demande 1\nAjoute un endpoint /health')
    expect(brief).not.toContain('bruit injecté')
    expect(brief).toContain('- src/server.ts')
    expect(brief).toContain('Endpoint ajouté.')

    expect(writes.map(write => write.path)).toEqual([ARCHIVE])
    const archive = writes[0]?.text ?? ''
    expect(archive).toContain('# Archive agent-link · session sess-1')
    expect(archive).toContain('### → Edit (toolu_1)')
    expect(archive).toContain('"new_string": "b"')
    expect(archive).not.toContain('### ← résultat (toolu_1)')
  })

  test('skips the precompute while on, and leaves every compaction to the engine when off', async ($, on) => {
    handoffWorld(on)
    const engine = summarizer(on)
    const skipped = await $.session.compact({ trigger: 'precompute', messages: TRANSCRIPT })
    expect(skipped.skip).toContain('relais actif')

    await $.command.run({ command: 'relais', args: 'off', origin: { kind: 'composer' }, presentation: PRESENTATION })
    const native = await $.session.compact({ trigger: 'auto', messages: TRANSCRIPT })
    expect(engine.runs).toBe(1)
    expect(native.messages?.[0]?.text).toBe('résumé')
  })

  test('in auto mode relays the threshold but leaves /compact to the summary', async ($, on) => {
    handoffWorld(on, { handoffMode: 'auto' })
    const engine = summarizer(on)
    const manual = await $.session.compact({ trigger: 'manual', messages: TRANSCRIPT })
    expect(manual.messages?.[0]?.text).toBe('résumé')
    const auto = await $.session.compact({ trigger: 'auto', messages: TRANSCRIPT })
    expect(auto.messages?.[0]?.text).toContain('[agent-link · relais de session]')
    expect(engine.runs).toBe(1)
  })

  test('a second relay carries the earlier requests and files, and points to the older archive', async ($, on) => {
    const writes = handoffWorld(on)
    const first = await $.session.compact({ trigger: 'auto', messages: TRANSCRIPT })
    const fresh = first.messages?.[0]
    expect(fresh).toBeDefined()
    const later: SessionMessage[] = [
      { role: 'user', text: fresh?.text ?? '', toolUses: [] },
      { role: 'user', text: 'Ajoute aussi un test pour /health', toolUses: [] },
    ]
    const second = await $.session.compact({ trigger: 'manual', messages: later })
    const brief = second.messages?.[0]?.text ?? ''
    expect(brief).toContain('### Demande 1\nAjoute un endpoint /health')
    expect(brief).toContain('### Demande 2\nAjoute aussi un test pour /health')
    expect(brief).toContain('- src/server.ts')
    expect(brief).toContain(`Archives plus anciennes : ${ARCHIVE}`)
    expect(writes.map(write => write.path)).toEqual([ARCHIVE, ARCHIVE.replace('-r1.md', '-r2.md')])
  })

  test('/relais reports the mode and the relays, and the main prompt points to the archive', async ($, on) => {
    handoffWorld(on)
    on('prompt.compose', () => ({ sections: [{ id: 'intro', text: 'Intro', scope: 'shared' as const }] }))
    const before = await $.prompt.compose({ ...COMPOSE, tools: ['Read'] })
    expect(before.sections.map(section => section.id)).toEqual(['intro'])

    await $.session.compact({ trigger: 'auto', messages: TRANSCRIPT })
    const status = await $.command.run({ command: 'relais', args: '', origin: { kind: 'composer' }, presentation: PRESENTATION })
    expect(status.text).toContain('Relais : on')
    expect(status.text).toContain(`main · auto · 4 messages → ${ARCHIVE}`)

    const after = await $.prompt.compose({ ...COMPOSE, tools: ['Read'] })
    expect(after.sections.map(section => section.id)).toEqual(['intro', 'agent-link:handoff'])
    expect(after.sections.at(-1)?.text).toContain(ARCHIVE)

    const unknown = await $.command.run({ command: 'relais', args: 'peut-être', origin: { kind: 'composer' }, presentation: PRESENTATION })
    expect(unknown.text).toContain('Option inconnue')
  })
})
