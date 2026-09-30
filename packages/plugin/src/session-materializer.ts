import { createAssistantMessage, createUserMessage } from '@deepseek-ai/dsh-llm'
import { rewriteCitationMarkers, type DeepSeekRemoteMessage, type DeepSeekRemoteSessionHistory } from '@dsh-deepseek-web/compat'
import { PROVIDER, type ModelId } from './config.ts'
import { formatSearchTimeline } from './search-timeline.ts'
import { extractImportedUserText } from './prompt.ts'
import {
  SESSION_FORMAT_VERSION,
  PluginSessionError,
  type LiveSessionLike,
  type SessionEventLike,
  type SessionPersistenceLike,
} from './dsh-session.ts'
import { ERROR_CODES } from './errors.ts'
import { historyPrefixHash, hashText, readReplay, wrapReplay, type DeepSeekWebReplayV1 } from './replay.ts'

export const CONSERVATIVE_SYSTEM_HASH = hashText('')
export const CONSERVATIVE_TOOLS_HASH = hashText('[]')
const MAX_INSPECT_CANDIDATES = 256
const MAX_TITLE_BYTES = 80

export interface MaterializeRemoteSessionInput {
  accountHash: string
  remote: DeepSeekRemoteSessionHistory
  cachedSessionId?: string
}

export type MaterializeReconcile = 'created' | 'reused' | 'appended' | 'conflict'

export interface MaterializeRemoteSessionResult {
  sessionId: string
  created: boolean
  reconciled: MaterializeReconcile
}

export interface RemoteTurn {
  users: DeepSeekRemoteMessage[]
  assistant: DeepSeekRemoteMessage
}

export function pairRemoteTurns(messages: readonly DeepSeekRemoteMessage[]): RemoteTurn[] {
  const turns: RemoteTurn[] = []
  let users: DeepSeekRemoteMessage[] = []
  for (const message of messages) {
    if (message.role === 'user') {
      users.push(message)
      continue
    }
    if (message.role !== 'assistant' || message.responseMessageId === undefined || message.responseMessageId.length === 0) {
      throw new PluginSessionError('remote assistant is missing a continuation id', ERROR_CODES.REMOTE_HISTORY_INVALID)
    }
    turns.push({ users, assistant: message })
    users = []
  }
  return turns
}

export function projectCanonicalMessage(message: DeepSeekRemoteMessage): {
  role: 'user' | 'assistant'
  source: { kind: string }
  content: Array<{ type: string; text: string }>
} {
  const text = message.role === 'assistant'
    ? rewriteCitationMarkers(message.text, message.citations ?? [])
    : extractImportedUserText(message.text)
  const content: Array<{ type: string; text: string }> = []
  if (message.role === 'assistant' && message.reasoning !== undefined && message.reasoning.length > 0) {
    content.push({ type: 'reasoning', text: message.reasoning })
  }
  if (message.role === 'assistant') {
    const search = formatSearchTimeline(message.citations ?? [])
    if (search.length > 0) content.push({ type: 'text', text: search })
  }
  content.push({ type: 'text', text })
  return {
    role: message.role,
    source: { kind: message.role === 'user' ? 'user' : 'model' },
    content,
  }
}

/** Historical remotes may still record "expert"; it normalises onto the single live model type. */
function modelOf(_message: DeepSeekRemoteMessage, _fallback?: string): ModelId {
  return 'default'
}

export function buildRemoteSessionSeed(input: {
  sessionId: string
  accountHash: string
  remote: DeepSeekRemoteSessionHistory
  now?: number
  prefixMessages?: unknown[]
  startTurn?: number
  startSeq?: number
  headerAlreadyLogged?: boolean
  lastModel?: 'default' | 'expert'
}): { events: SessionEventLike[]; messages: unknown[] } {
  const turns = pairRemoteTurns(input.remote.messages)
  const events: SessionEventLike[] = []
  const messages: unknown[] = [...(input.prefixMessages ?? [])]
  let seq = input.startSeq ?? 0
  let loggedHeader = input.headerAlreadyLogged === true
  let lastModel = input.lastModel
  const now = eventTime(input.now, Date.now())
  const startTurn = input.startTurn ?? 1
  const push = (type: string, data: unknown, extra?: { surfaceOp?: 'append'; time?: number }): SessionEventLike => {
    const event: SessionEventLike = {
      type,
      seq,
      time: eventTime(extra?.time, now + seq),
      data,
      ...(extra?.surfaceOp === undefined ? {} : { surfaceOp: extra.surfaceOp }),
    }
    seq += 1
    events.push(event)
    return event
  }

  const remoteTitle = truncateTitle(input.remote.session.title)
  if (startTurn === 1 && remoteTitle !== undefined) {
    push('session/title', {
      title: remoteTitle,
      messageSeqs: [],
      source: { kind: 'user' },
    }, { time: now })
  }

  for (let index = 0; index < turns.length; index++) {
    const turn = turns[index]!
    const turnNumber = startTurn + index
    const model = modelOf(turn.assistant, input.remote.session.modelType)
    const stamp = eventTime(turn.assistant.createdAt ?? turn.users[0]?.createdAt, now + seq)
    push('turn/start', { turn: turnNumber }, { time: stamp })
    push('step/start', { turn: turnNumber, step: 1 }, { time: stamp })
    for (const user of turn.users) {
      const projected = projectCanonicalMessage(user)
      const userMessage = createUserMessage({
        content: projected.content,
        source: { kind: 'user' },
      })
      messages.push(userMessage)
      push('user/message', userMessage, { surfaceOp: 'append', time: user.createdAt ?? stamp })
    }
    let reason: 'initial' | 'resume' | 'change' = 'resume'
    if (!loggedHeader) reason = 'initial'
    else if (lastModel !== undefined && lastModel !== model) reason = 'change'
    loggedHeader = true
    lastModel = model
    push('request/header', {
      header: { config: { provider: PROVIDER, model } },
      reason,
    }, { time: stamp })
    const projectedAssistant = projectCanonicalMessage(turn.assistant)
    const prefixHash = historyPrefixHash(messages, messages.length)
    const thinking = turn.assistant.thinkingEnabled ?? true
    const nativeSearch = turn.assistant.searchEnabled ?? false
    const replay: DeepSeekWebReplayV1 = {
      kind: 'deepseek-web',
      version: 1,
      dshSessionId: input.sessionId,
      affinity: { accountHash: input.accountHash, model },
      remote: {
        chatSessionId: input.remote.session.chatSessionId,
        responseMessageId: turn.assistant.responseMessageId!,
      },
      history: { inputPrefixHash: prefixHash },
      contract: {
        systemHash: CONSERVATIVE_SYSTEM_HASH,
        toolsHash: CONSERVATIVE_TOOLS_HASH,
        thinking,
        nativeSearch,
      },
      ...(turn.assistant.citations !== undefined && turn.assistant.citations.length > 0
        ? { provider: { citations: turn.assistant.citations } }
        : {}),
    }
    const assistantMessage = createAssistantMessage({
      content: projectedAssistant.content,
      source: {
        provider: PROVIDER,
        model,
        replayState: wrapReplay(replay),
      },
    })
    messages.push(assistantMessage)
    push('assistant/message', {
      turn: turnNumber,
      step: 1,
      message: assistantMessage,
    }, { surfaceOp: 'append', time: stamp })
    push('step/end', { turn: turnNumber, step: 1 }, { time: stamp })
    push('turn/end', { turn: turnNumber, reason: { kind: 'completed' } }, { time: stamp })
  }
  return { events, messages }
}

export function derivedMessagesFromEvents(events: readonly SessionEventLike[]): unknown[] {
  const messages: unknown[] = []
  for (const event of events) {
    if (event.type === 'user/message') messages.push(event.data)
    if (event.type === 'assistant/message') {
      const data = event.data as { message?: unknown }
      if (data.message !== undefined) messages.push(data.message)
    }
  }
  return messages
}

export function remoteFingerprint(messages: readonly DeepSeekRemoteMessage[]): string {
  return pairRemoteTurns(messages).map(turn => {
    const users = turn.users.map(user => `U:${extractImportedUserText(user.text)}`).join('+')
    return `${users}|A:${turn.assistant.responseMessageId}:${turn.assistant.text}`
  }).join('\n')
}

export function localFingerprint(messages: readonly unknown[]): string {
  const lines: string[] = []
  let users: string[] = []
  for (const message of messages) {
    const record = message as {
      role?: string
      content?: unknown
      source?: { replayState?: unknown }
    }
    const text = record.role === 'user' ? extractImportedUserText(textOf(record.content)) : textOf(record.content)
    if (record.role === 'user') {
      users.push(`U:${text}`)
      continue
    }
    if (record.role !== 'assistant') continue
    const replay = readReplay(record.source?.replayState)
    lines.push(`${users.join('+')}|A:${replay?.remote.responseMessageId ?? ''}:${text}`)
    users = []
  }
  return lines.join('\n')
}

function textOf(content: unknown): string {
  if (typeof content === 'string') return content
  if (!Array.isArray(content)) return ''
  return content.map(block => {
    const record = block as { type?: string; text?: string }
    return record.type === 'text' ? record.text ?? '' : ''
  }).join('')
}

export function replayFromEvents(events: readonly SessionEventLike[]): DeepSeekWebReplayV1 | undefined {
  for (let index = events.length - 1; index >= 0; index--) {
    const event = events[index]
    if (event?.type !== 'assistant/message') continue
    const message = (event.data as { message?: { source?: { replayState?: unknown } } }).message
    const replay = readReplay(message?.source?.replayState)
    if (replay) return replay
  }
  return undefined
}

export function sameDir(left: string, right: string): boolean {
  if (left === right) return true
  const normalize = (value: string) => value.replace(/[\\/]+$/, '').replaceAll('\\', '/')
  const a = normalize(left)
  const b = normalize(right)
  return process.platform === 'win32' ? a.toLowerCase() === b.toLowerCase() : a === b
}

export function isReusableMirrorHeader(
  header: { origin?: string; cwd?: string; parentSession?: string },
  mirrorCwd?: string,
  cwdAliases: readonly string[] = [],
): boolean {
  if (header.origin === 'subagent' || header.parentSession !== undefined) return false
  if (mirrorCwd === undefined || mirrorCwd.length === 0) return header.cwd === undefined
  if (header.cwd === undefined || header.cwd.length === 0) return false
  if (sameDir(header.cwd, mirrorCwd)) return true
  return cwdAliases.some(alias => alias.length > 0 && sameDir(header.cwd!, alias))
}

export async function findExistingMirror(
  persistence: SessionPersistenceLike,
  accountHash: string,
  chatSessionId: string,
  cachedId?: string,
  mirrorCwd?: string,
  excludeIds: ReadonlySet<string> = new Set(),
  cwdAliases: readonly string[] = [],
): Promise<string | undefined> {
  if (cachedId !== undefined && !excludeIds.has(cachedId)) {
    try {
      const inspected = await persistence.inspect(cachedId)
      if (
        isReusableMirrorHeader(inspected.meta, mirrorCwd, cwdAliases)
        && mirrorReplayMatches(inspected, accountHash, chatSessionId)
      ) {
        return inspected.meta.id
      }
    } catch {
      /* fall through to a durable scan */
    }
  }
  const headers = await persistence.list()
  const candidates = headers
    .filter(header => !excludeIds.has(header.id) && isReusableMirrorHeader(header, mirrorCwd, cwdAliases))
    .sort((left, right) => right.createdAt - left.createdAt)
    .slice(0, MAX_INSPECT_CANDIDATES)
  for (const header of candidates) {
    try {
      const inspected = await persistence.inspect(header.id)
      if (mirrorReplayMatches(inspected, accountHash, chatSessionId)) return inspected.meta.id
    } catch {
      continue
    }
  }
  return undefined
}

function mirrorReplayMatches(
  inspected: { meta: { id: string }; events: readonly SessionEventLike[] },
  accountHash: string,
  chatSessionId: string,
): boolean {
  const replay = replayFromEvents(inspected.events)
  return replay?.kind === 'deepseek-web'
    && replay.affinity.accountHash === accountHash
    && replay.remote.chatSessionId === chatSessionId
    && (replay.dshSessionId === undefined || replay.dshSessionId === inspected.meta.id)
}

/**
 * Mapping policy: a DeepSeek Web conversation is never used as a DSH session
 * id. Each mirror is a new UUID. Reopen looks up an existing mirror whose
 * latest DeepSeek replay matches accountHash + remote conversation id and
 * whose replay.dshSessionId is either missing or equal to that header id.
 * Production mirrors persist a plugin data-directory cwd so Harness persona
 * `{{cwd}}` and tools have a working directory. That directory is registered
 * as the official **DeepSeek Chat** workspace so continuations are not
 * projected to Ungrouped. Legacy no-cwd rows are not reused once a cwd is
 * configured. Archived Harness session ids are skipped so a popover reopen
 * can mint a live mirror. Process cache is a hint only; persistence remains
 * the restart source of truth. Safe
 * append applies only when the local fingerprint is a strict prefix of the
 * remote history. Divergence returns conflict and opens the existing local
 * copy without merging. When the mirror is already live in SessionStore,
 * later prefix extensions append through that session so the in-memory log
 * stays the authority.
 */
export function appendEventsToLiveSession(
  session: LiveSessionLike,
  events: readonly SessionEventLike[],
): void {
  for (const event of events) {
    if (event.surfaceOp === undefined && event.sourceEventSeqs === undefined) {
      session.append(event.type, event.data)
      continue
    }
    session.append(event.type, event.data, {
      ...event.surfaceOp === undefined ? {} : { surfaceOp: event.surfaceOp },
      ...event.sourceEventSeqs === undefined ? {} : { sourceEventSeqs: event.sourceEventSeqs },
    })
  }
}

export class RemoteSessionMaterializer {
  constructor(
    private readonly persistence: () => SessionPersistenceLike | undefined,
    private readonly liveSession: (id: string) => LiveSessionLike | undefined = () => undefined,
    private readonly options: {
      cwd?: () => string | undefined
      cwdAliases?: () => readonly string[]
      ensureCwd?: (path: string) => Promise<void>
      excludeSessionIds?: () => readonly string[]
      attach?: (sessionId: string) => Promise<void>
    } = {},
  ) {}

  async materialize(input: MaterializeRemoteSessionInput): Promise<MaterializeRemoteSessionResult> {
    const persistence = this.persistence()
    if (persistence === undefined) {
      throw new PluginSessionError(
        'session persistence is unavailable; cannot open a DeepSeek Web conversation',
        ERROR_CODES.MIRROR_UNAVAILABLE,
      )
    }
    const cwd = this.options.cwd?.()
    const excluded = new Set(this.options.excludeSessionIds?.() ?? [])
    const existing = await findExistingMirror(
      persistence,
      input.accountHash,
      input.remote.session.chatSessionId,
      input.cachedSessionId,
      cwd,
      excluded,
      this.options.cwdAliases?.() ?? [],
    )
    if (existing !== undefined) {
      const reused = await this.reconcile(persistence, existing, input)
      await this.attachQuietly(reused.sessionId)
      return reused
    }
    if (pairRemoteTurns(input.remote.messages).length === 0) {
      throw new PluginSessionError(
        'that DeepSeek Web conversation has no completed turns to open',
        ERROR_CODES.REMOTE_HISTORY_INVALID,
      )
    }
    const sessionId = crypto.randomUUID()
    const seed = buildRemoteSessionSeed({
      sessionId,
      accountHash: input.accountHash,
      remote: input.remote,
    })
    if (cwd !== undefined && cwd.length > 0) await this.options.ensureCwd?.(cwd)
    await persistence.create({
      version: SESSION_FORMAT_VERSION,
      id: sessionId,
      createdAt: Date.now(),
      ...(cwd !== undefined && cwd.length > 0 ? { cwd } : {}),
    })
    if (seed.events.length > 0) await persistence.append(sessionId, seed.events)
    await this.attachQuietly(sessionId)
    return { sessionId, created: true, reconciled: 'created' }
  }

  private async attachQuietly(sessionId: string): Promise<void> {
    try {
      await this.options.attach?.(sessionId)
    } catch {
      /* membership can still follow from matching cwd once the workspace exists */
    }
  }

  private async reconcile(
    persistence: SessionPersistenceLike,
    sessionId: string,
    input: MaterializeRemoteSessionInput,
  ): Promise<MaterializeRemoteSessionResult> {
    const inspected = await persistence.inspect(sessionId)
    const localMessages = derivedMessagesFromEvents(inspected.events)
    const local = localFingerprint(localMessages)
    const remote = remoteFingerprint(input.remote.messages)
    if (local === remote) {
      return { sessionId, created: false, reconciled: 'reused' }
    }
    if (remote.startsWith(local) && (local.length === 0 || remote.charAt(local.length) === '\n')) {
      const localTurns = local === '' ? 0 : local.split('\n').length
      const extra = pairRemoteTurns(input.remote.messages).slice(localTurns)
      if (extra.length === 0) return { sessionId, created: false, reconciled: 'reused' }
      const synthetic: DeepSeekRemoteMessage[] = []
      for (const turn of extra) synthetic.push(...turn.users, turn.assistant)
      const lastModel = lastAssistantModel(localMessages)
      const nextTurn = buildRemoteSessionSeed({
        sessionId,
        accountHash: input.accountHash,
        remote: { ...input.remote, messages: synthetic },
        prefixMessages: localMessages,
        startTurn: localTurns + 1,
        startSeq: inspected.events.length === 0 ? 0 : inspected.events[inspected.events.length - 1]!.seq + 1,
        headerAlreadyLogged: localTurns > 0,
        lastModel,
      }).events
      const live = this.liveSession(sessionId)
      if (live !== undefined) appendEventsToLiveSession(live, nextTurn)
      else await persistence.append(sessionId, nextTurn)
      return { sessionId, created: false, reconciled: 'appended' }
    }
    return { sessionId, created: false, reconciled: 'conflict' }
  }
}

function eventTime(value: number | undefined, fallback: number): number {
  const raw = value ?? fallback
  const n = Number.isFinite(raw) ? Math.trunc(raw) : Math.trunc(fallback)
  return Number.isSafeInteger(n) && n >= 0 ? n : 0
}

function truncateTitle(value: string | undefined): string | undefined {
  const trimmed = value?.trim() ?? ''
  if (trimmed.length === 0) return undefined
  let bytes = 0
  let end = 0
  for (const char of trimmed) {
    const size = Buffer.byteLength(char, 'utf8')
    if (bytes + size > MAX_TITLE_BYTES) break
    bytes += size
    end += char.length
  }
  const next = trimmed.slice(0, end).trimEnd()
  return next.length === 0 ? undefined : next
}

function lastAssistantModel(messages: readonly unknown[]): ModelId | undefined {
  for (let index = messages.length - 1; index >= 0; index--) {
    const message = messages[index] as { role?: string; source?: { model?: string } }
    if (message.role !== 'assistant') continue
    return 'default'
  }
  return undefined
}
