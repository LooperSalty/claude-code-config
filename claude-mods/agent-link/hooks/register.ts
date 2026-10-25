import type { EngineInterface, Register } from 'claude-code'

import { hookCollab } from './collab'
import { hookHandoff, registerHandoffCommand } from './handoff'
import { hookJournal } from './journal'
import { hookPane, registerCommand } from './pane'
import { refreshAgents, refreshStatus } from './state'
import { hookTools, registerTools } from './tools'

/** How often agent statuses are re-read for the pane and the status line. */
const POLL_MS = 3000

async function setUp($: EngineInterface): Promise<void> {
  try {
    await registerTools($)
    await registerCommand($)
    await registerHandoffCommand($)
  } catch (error) {
    $.ui.log(`agent-link: setup failed: ${String(error)}`)
  }
}

/** Runs every few seconds: a failure goes to the debug log, never the transcript. */
async function poll($: EngineInterface): Promise<void> {
  try {
    await refreshAgents($)
    await refreshStatus($)
  } catch (error) {
    $.ui.log(`poll failed: ${String(error)}`, { to: 'debug' })
  }
}

export const register: Register = on => {
  on('session.start', async ($, e, next) => {
    await setUp($)
    $.clock.every(POLL_MS, () => void poll($))
    return next(e)
  })

  hookTools(on)
  hookCollab(on)
  hookJournal(on)
  hookPane(on)
  hookHandoff(on)
}
