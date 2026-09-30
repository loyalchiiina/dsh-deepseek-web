import {
  CitationStreamGate,
  CompatDeepSeekWebClient,
  DeepSeekWebError,
  ERROR_CODES,
  rewriteCitationMarkers,
  type DeepSeekCitation,
  type DeepSeekStreamDelta,
  type DeepSeekTurn,
  type DeepSeekWebClient,
} from '@dsh-deepseek-web/compat'
import { LlmAdapter, LlmError, ReasoningEffortId } from '@deepseek-ai/dsh-llm'
import type {
  GenerateOptions,
  LlmModelInfo,
  LlmProviderInfo,
  LlmResolvedModelInfo,
  StreamChunk,
} from '@deepseek-ai/dsh-llm'
import { nativeSearchActive, PROVIDER, type ModelId, type ResolvedConfig } from './config.ts'
import { formatSearchTimeline } from './search-timeline.ts'
import { flattenText, messageHasImage, toolCallBlocks, toolResultFromMessage } from './messages.ts'
import { resolveSessionTitle } from './title.ts'
import { buildFullPrompt, buildIncrementalPrompt, extractImportedUserText, filterToolsForNativeSearch, type PromptMessage } from './prompt.ts'
import { isCleanMode } from './clean-turn.ts'
import {
  createLiveCursor,
  hashText,
  historyPrefixHash,
  planConversation,
  wrapReplay,
  type DeepSeekWebReplayV1,
  type LiveRemoteCursor,
} from './replay.ts'
import {
  promisedContinuationPrompt,
  promisedToolContinuation,
  projectToolResult,
  repairPrompt,
  resolveToolProtocol,
  visibleTextBeforeToolProtocol,
  toBridgeTools,
  type BridgeToolCall,
} from './tool-bridge/index.ts'

class AsyncMutex {
  private chain = Promise.resolve()
  runExclusive<T>(fn: () => Promise<T>): Promise<T> {
    const run = this.chain.then(fn, fn)
    this.chain = run.then(() => undefined, () => undefined)
    return run
  }
}

export class SessionLockMap {
  private readonly locks = new Map<string, AsyncMutex>()
  async runExclusive<T>(sessionId: string | undefined, signal: AbortSignal | undefined, fn: () => Promise<T>): Promise<T> {
    if (signal?.aborted) throw new LlmError('aborted', 'TIMEOUT')
    const key = sessionId ?? '_none'
    let lock = this.locks.get(key)
    if (lock === undefined) {
      lock = new AsyncMutex()
      this.locks.set(key, lock)
    }
    try {
      return await lock.runExclusive(fn)
    } finally {
      /* keep the mutex; TTL cleanup is process-local and cheap */
    }
  }
}

class ChunkQueue {
  private readonly items: StreamChunk[] = []
  private waiter?: (result: IteratorResult<StreamChunk>) => void
  private done = false
  private error?: unknown

  push(chunk: StreamChunk): void {
    if (this.waiter !== undefined) {
      const waiter = this.waiter
      this.waiter = undefined
      waiter({ value: chunk, done: false })
      return
    }
    this.items.push(chunk)
  }

  close(): void {
    this.done = true
    this.waiter?.({ value: undefined as never, done: true })
    this.waiter = undefined
  }

  fail(error: unknown): void {
    this.error = error
    this.done = true
    this.waiter?.({ value: undefined as never, done: true })
    this.waiter = undefined
  }

  async *iterate(): AsyncIterable<StreamChunk> {
    while (true) {
      if (this.items.length > 0) {
        yield this.items.shift()!
        continue
      }
      if (this.error !== undefined) throw this.error
      if (this.done) return
      const result = await new Promise<IteratorResult<StreamChunk>>(resolve => {
        this.waiter = resolve
      })
      if (result.done) {
        if (this.error !== undefined) throw this.error
        return
      }
      yield result.value
    }
  }
}

class StreamSink {
  private nextIndex = 0
  private reasoningIndex: number | undefined
  private textIndex: number | undefined
  private reasoning = ''
  private rawReasoning = ''
  private reasoningFed = 0
  private rawText = ''
  private fed = 0
  private visibleEmitted = false
  private receivedDelta = false
  private reasoningClosed = false
  private textClosed = false
  private searchClosed = false
  private gate = new CitationStreamGate()
  private citations: DeepSeekCitation[] = []

  constructor(
    private readonly emit: (chunk: StreamChunk) => void,
    private readonly hasTools: boolean,
  ) {}

  get streamedVisible(): boolean {
    return this.visibleEmitted
  }

  push(delta: DeepSeekStreamDelta): void {
    this.receivedDelta = true
    this.citations = delta.citations
    if (delta.reasoning.length > 0) {
      this.rawReasoning += delta.reasoning
      const visibleReasoning = visibleTextBeforeToolProtocol(this.rawReasoning, this.hasTools)
      const unfedReasoning = visibleReasoning.slice(this.reasoningFed)
      this.reasoningFed = visibleReasoning.length
      if (unfedReasoning.length > 0) {
        this.ensure('reasoning')
        this.reasoning += unfedReasoning
        this.emit({ type: 'reasoning-delta', index: this.reasoningIndex!, text: unfedReasoning })
      }
    }
    if (delta.text.length > 0) this.rawText += delta.text
    const visiblePrefix = visibleTextBeforeToolProtocol(this.rawText, this.hasTools)
    const unfed = visiblePrefix.slice(this.fed)
    this.fed = visiblePrefix.length
    const visible = this.gate.push(unfed, this.citations)
    if (visible.length === 0) return
    this.maybeEmitSearchCard()
    this.ensure('text')
    this.visibleEmitted = true
    this.emit({ type: 'text-delta', index: this.textIndex!, text: visible })
  }

  flushTurn(turn: DeepSeekTurn): void {
    this.citations = turn.citations
    if (this.receivedDelta) return
    this.push({ reasoning: turn.reasoning, text: turn.text, citations: turn.citations })
  }

  resetBufferedText(): void {
    this.rawText = ''
    this.fed = 0
    this.rawReasoning = this.reasoning
    this.reasoningFed = this.reasoning.length
    this.citations = []
    this.gate = new CitationStreamGate()
  }

  finish(calls: BridgeToolCall[], replay: DeepSeekWebReplayV1): void {
    this.maybeEmitSearchCard()
    if (calls.length > 0) {
      this.closeReasoning()
      this.closeText()
      for (const call of calls) {
        const argsJson = JSON.stringify(call.arguments)
        const index = this.nextIndex
        this.nextIndex += 1
        this.emit({ type: 'block-start', index, blockType: 'tool-call' })
        this.emit({ type: 'tool-call-delta', index, id: call.id as never, name: call.name, argumentsDelta: argsJson })
        this.emit({
          type: 'block-end',
          index,
          block: { type: 'tool-call', id: call.id as never, name: call.name, arguments: argsJson },
        })
      }
      this.emit({ type: 'finish', reason: { kind: 'tool-calls' } as never, replayState: wrapReplay(replay) })
      return
    }
    const flushed = this.gate.push('', this.citations, true)
    if (flushed.length > 0) {
      this.ensure('text')
      this.visibleEmitted = true
      this.emit({ type: 'text-delta', index: this.textIndex!, text: flushed })
    }
    this.closeReasoning()
    this.closeText()
    this.emit({ type: 'finish', reason: { kind: 'stop' } as never, replayState: wrapReplay(replay) })
  }

  private maybeEmitSearchCard(): void {
    if (this.searchClosed || this.textIndex !== undefined) return
    const markdown = formatSearchTimeline(this.citations)
    if (markdown.length === 0) return
    this.searchClosed = true
    this.closeReasoning()
    const index = this.nextIndex
    this.nextIndex += 1
    this.emit({ type: 'block-start', index, blockType: 'text' })
    this.emit({ type: 'text-delta', index, text: markdown })
    this.emit({ type: 'block-end', index, block: { type: 'text', text: markdown } })
  }

  private ensure(kind: 'reasoning' | 'text'): void {
    if (kind === 'reasoning') {
      if (this.reasoningIndex !== undefined || this.reasoningClosed) return
      this.reasoningIndex = this.nextIndex
      this.nextIndex += 1
      this.emit({ type: 'block-start', index: this.reasoningIndex, blockType: 'reasoning' })
      return
    }
    if (this.textIndex !== undefined) return
    this.textIndex = this.nextIndex
    this.nextIndex += 1
    this.emit({ type: 'block-start', index: this.textIndex, blockType: 'text' })
  }

  private closeReasoning(): void {
    if (this.reasoningIndex === undefined || this.reasoningClosed) return
    this.reasoningClosed = true
    this.emit({ type: 'block-end', index: this.reasoningIndex, block: { type: 'reasoning', text: this.reasoning } })
  }

  private closeText(): void {
    if (this.textIndex === undefined || this.textClosed) return
    this.textClosed = true
    this.emit({ type: 'block-end', index: this.textIndex, block: { type: 'text', text: this.gate.text } })
  }
}

export interface AdapterDeps {
  options: () => ResolvedConfig
  resolveCredential: () => Promise<string>
  client?: DeepSeekWebClient
}

interface AcceptedTurn {
  reasoning: string
  text: string
  calls: BridgeToolCall[]
  replay: DeepSeekWebReplayV1
}

export class DeepSeekWebAdapter extends LlmAdapter {
  private readonly locks = new SessionLockMap()
  private readonly accountCache = new Map<string, { hash: string; at: number }>()
  private readonly remotes = new Map<string, LiveRemoteCursor>()
  private readonly client: DeepSeekWebClient

  constructor(private readonly deps: AdapterDeps) {
    super()
    this.client = deps.client ?? new CompatDeepSeekWebClient()
  }

  clearAccountCache(): void {
    this.accountCache.clear()
    this.remotes.clear()
  }

  override providerInfo(provider: string): LlmProviderInfo {
    return { id: provider, name: 'DeepSeek Web' }
  }

  override async listModels(provider: string): Promise<readonly LlmModelInfo[]> {
    // DeepSeek's web protocol only carries a single "default" model type now
    // (the bundled WASM core never mentions "expert"); the former Expert
    // selection degraded into "same model, native search forced off".
    // Capability is expressed through the thinking / search settings instead,
    // so only one model is advertised. Sessions that recorded "expert" remain
    // readable (see replay.ts and session-materializer.ts).
    return [{ provider, id: 'default', name: 'DeepSeek Web', inputModalities: ['text'] }]
  }

  override async resolveModel(provider: string, model: string): Promise<LlmResolvedModelInfo> {
    const models = await this.listModels(provider)
    const found = models.find(item => item.id === model) ?? models[0]!
    return { ...found, reasoning: deepThinkReasoning(this.deps.options().thinking !== 'disabled') }
  }

  override async *stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    const config = this.deps.options()
    if (messageHasImage(options.messages)) {
      throw new LlmError(
        'DeepSeek Web does not support image understanding; ImageBlock is rejected',
        ERROR_CODES.UNSUPPORTED_IMAGE,
      )
    }
    const credential = await this.deps.resolveCredential()
    const powHeaderPromise = options.purpose === 'session-title'
      ? Promise.resolve(undefined as string | undefined)
      : this.client.solveCompletionPow?.(credential, options.signal) ?? Promise.resolve(undefined)
    const account = await this.resolveAccount(credential, options.signal)
    const queue = new ChunkQueue()
    const work = options.purpose === 'compaction' || options.purpose === 'session-title'
      ? this.oneShot(options, config, credential, account.hash, chunk => queue.push(chunk), powHeaderPromise)
      : this.locks.runExclusive(options.sessionId, options.signal, () => {
        return this.executeConversation(options, config, credential, account.hash, chunk => queue.push(chunk), powHeaderPromise)
      })
    const running = work.then(() => queue.close(), error => queue.fail(error))
    try {
      yield* queue.iterate()
    } finally {
      await running
    }
  }

  private async resolveAccount(credential: string, signal?: AbortSignal): Promise<{ hash: string }> {
    const cached = this.accountCache.get(credential)
    if (cached && Date.now() - cached.at < 600_000) return { hash: cached.hash }
    try {
      const account = await this.client.currentUser(credential, signal)
      this.accountCache.set(credential, { hash: account.accountHash, at: Date.now() })
      return { hash: account.accountHash }
    } catch (error) {
      this.accountCache.delete(credential)
      throw toLlm(error)
    }
  }

  private async oneShot(
    options: GenerateOptions,
    config: ResolvedConfig,
    credential: string,
    accountHash: string,
    emit: (chunk: StreamChunk) => void,
    powHeaderPromise: Promise<string | undefined>,
  ): Promise<void> {
    const sink = new StreamSink(emit, false)
    if (options.purpose === 'session-title') {
      const live = options.sessionId === undefined ? undefined : this.remotes.get(options.sessionId)
      const title = resolveSessionTitle(live?.sessionTitle, options.messages)
      sink.push({ reasoning: '', text: title, citations: [] })
      sink.finish([], {
        kind: 'deepseek-web',
        version: 1,
        dshSessionId: options.sessionId,
        affinity: { accountHash, model: modelOf(options.model) },
        remote: {
          chatSessionId: live?.chatSessionId ?? 'local-title',
          responseMessageId: live?.parentMessageId ?? '0',
        },
        history: { inputPrefixHash: live?.sentHash ?? hashText('') },
        contract: {
          systemHash: live?.systemHash ?? hashText(options.system ?? ''),
          toolsHash: live?.toolsHash ?? hashText('[]'),
          thinking: false,
          nativeSearch: false,
        },
      })
      return
    }
    const session = await this.client.createSession(credential, options.signal)
    const last = options.messages.at(-1) as { content?: unknown } | undefined
    const prompt = flattenText(last?.content) || 'Summarize.'
    const remote = await this.completeChecked({
      credential,
      chatSessionId: session.chatSessionId,
      prompt,
      modelType: modelOf(options.model),
      thinkingEnabled: false,
      searchEnabled: false,
      signal: options.signal,
      onDelta: delta => sink.push(delta),
      powHeader: await powHeaderPromise,
    })
    sink.flushTurn(remote)
    sink.finish([], this.accept(remote, options, config, accountHash, session.chatSessionId, false, hashText(options.system ?? ''), hashText('[]'), false).replay)
  }

  private async executeConversation(
    options: GenerateOptions,
    config: ResolvedConfig,
    credential: string,
    accountHash: string,
    emit: (chunk: StreamChunk) => void,
    powHeaderPromise: Promise<string | undefined>,
  ): Promise<void> {
    const model = modelOf(options.model)
    const clean = isCleanMode(options.sessionId)
    const cleanText = lastHumanUserText(options.messages)
    const nativeSearch = nativeSearchActive(config, model)
    const tools = clean ? [] : toBridgeTools(options.tools ?? [])
    const promptTools = filterToolsForNativeSearch(tools, nativeSearch)
    const sink = new StreamSink(emit, promptTools.length > 0)
    const systemHash = hashText(clean ? '' : (options.system ?? ''))
    const toolsHash = hashText(clean ? '[]' : JSON.stringify(tools))
    const live = options.sessionId === undefined ? undefined : this.remotes.get(options.sessionId)
    const plan = planConversation({
      messages: options.messages,
      sessionId: options.sessionId,
      accountHash,
      model,
      provider: PROVIDER,
      systemHash,
      toolsHash,
      live,
    })
    let chatSessionId: string
    let parentMessageId: string | undefined
    let prompt: string
    let rebuilt = false
    if (plan.kind === 'continue') {
      chatSessionId = plan.chatSessionId
      parentMessageId = plan.parentMessageId
      prompt = clean
        ? cleanText
        : buildIncrementalPrompt({
          toolsUpdate: plan.contractUpdate?.tools ? tools : undefined,
          systemUpdate: plan.contractUpdate?.system ? options.system : undefined,
          delta: toPromptMessages(plan.deltaMessages, config.maxToolResultBytes),
          nativeSearch,
        })
    } else {
      if (options.sessionId !== undefined) this.remotes.delete(options.sessionId)
      const session = await this.client.createSession(credential, options.signal)
      chatSessionId = session.chatSessionId
      prompt = clean
        ? cleanText
        : buildFullPrompt({
          system: options.system,
          messages: toPromptMessages(options.messages, config.maxToolResultBytes),
          tools,
          maxCalls: config.maxToolCallsPerTurn,
          nativeSearch,
        })
      rebuilt = plan.kind === 'rebuild'
    }
    try {
      const remote = await this.completeWithRepair({
        credential,
        chatSessionId,
        parentMessageId,
        prompt,
        modelType: model,
        thinkingEnabled: thinkingEnabled(options, config),
        searchEnabled: nativeSearch,
        signal: options.signal,
        tools: promptTools.map(tool => tool.name),
        nativeSearch,
        config,
        sink,
        powHeader: await powHeaderPromise,
      })
      sink.flushTurn(remote.turn)
      sink.finish(remote.calls, this.accept(remote.turn, options, config, accountHash, chatSessionId, remote.calls, systemHash, toolsHash).replay)
    } catch (error) {
      if (rebuilt) throw toLlm(error)
      if (error instanceof DeepSeekWebError && (error.code === ERROR_CODES.REMOTE_SESSION || error.code === ERROR_CODES.PROTOCOL)) {
        if (options.sessionId !== undefined) this.remotes.delete(options.sessionId)
        const session = await this.client.createSession(credential, options.signal)
        const remote = await this.completeWithRepair({
          credential,
          chatSessionId: session.chatSessionId,
          prompt: clean
            ? cleanText
            : buildFullPrompt({
              system: options.system,
              messages: toPromptMessages(options.messages, config.maxToolResultBytes),
              tools,
              maxCalls: config.maxToolCallsPerTurn,
              nativeSearch,
            }),
          modelType: model,
          thinkingEnabled: thinkingEnabled(options, config),
          searchEnabled: nativeSearch,
          signal: options.signal,
          tools: promptTools.map(tool => tool.name),
          nativeSearch,
          config,
          sink,
        })
        sink.flushTurn(remote.turn)
        sink.finish(remote.calls, this.accept(remote.turn, options, config, accountHash, session.chatSessionId, remote.calls, systemHash, toolsHash).replay)
        return
      }
      throw toLlm(error)
    }
  }

  private async completeChecked(request: Parameters<DeepSeekWebClient['complete']>[0]): Promise<DeepSeekTurn> {
    try {
      return await this.client.complete(request)
    } catch (error) {
      throw toLlm(error)
    }
  }

  private async completeWithRepair(input: {
    credential: string
    chatSessionId: string
    parentMessageId?: string
    prompt: string
    modelType: 'default' | 'expert'
    thinkingEnabled: boolean
    searchEnabled: boolean
    signal?: AbortSignal
    tools: string[]
    nativeSearch: boolean
    config: ResolvedConfig
    sink: StreamSink
    powHeader?: string
  }): Promise<{ turn: DeepSeekTurn; calls: BridgeToolCall[] }> {
    let parent = input.parentMessageId
    let prompt = input.prompt
    let powHeader = input.powHeader
    for (let attempt = 0; attempt <= input.config.maxProtocolRepairAttempts; attempt++) {
      const turn = await this.completeChecked({
        credential: input.credential,
        chatSessionId: input.chatSessionId,
        parentMessageId: parent,
        prompt,
        modelType: input.modelType,
        thinkingEnabled: input.thinkingEnabled,
        searchEnabled: input.searchEnabled,
        signal: input.signal,
        onDelta: delta => input.sink.push(delta),
        powHeader,
      })
      powHeader = undefined
      const parsed = resolveToolProtocol({ text: turn.text, reasoning: turn.reasoning }, input.tools, {
        maxBytes: input.config.maxToolProtocolBytes,
        maxCalls: input.config.maxToolCallsPerTurn,
        responseMessageId: turn.responseMessageId,
      })
      if (parsed.kind === 'calls') return { turn: { ...turn, text: '' }, calls: parsed.calls }
      if (parsed.kind === 'none') {
        const promisedSource = turn.text.trim().length > 0 ? turn.text : turn.reasoning
        if (
          !input.sink.streamedVisible
          && input.tools.length > 0
          && promisedToolContinuation(promisedSource, input.nativeSearch)
          && attempt < input.config.maxProtocolRepairAttempts
        ) {
          input.sink.resetBufferedText()
          parent = turn.responseMessageId
          prompt = promisedContinuationPrompt(input.nativeSearch)
          continue
        }
        return { turn, calls: [] }
      }
      if (input.sink.streamedVisible) return { turn, calls: [] }
      input.sink.resetBufferedText()
      parent = turn.responseMessageId
      prompt = repairPrompt(parsed.reason, input.nativeSearch)
    }
    throw new LlmError('DeepSeek Web tool protocol remained invalid', ERROR_CODES.TOOL_PROTOCOL)
  }

  private accept(
    turn: DeepSeekTurn,
    options: GenerateOptions,
    config: ResolvedConfig,
    accountHash: string,
    chatSessionId: string,
    calls: BridgeToolCall[] | false,
    systemHash: string,
    toolsHash: string,
    bindSession = true,
  ): AcceptedTurn {
    const parsedCalls = Array.isArray(calls) ? calls : []
    const text = rewriteCitationMarkers(turn.text, turn.citations)
    const model = modelOf(options.model)
    const replay: DeepSeekWebReplayV1 = {
      kind: 'deepseek-web',
      version: 1,
      dshSessionId: options.sessionId,
      affinity: { accountHash, model },
      remote: { chatSessionId, responseMessageId: turn.responseMessageId },
      history: { inputPrefixHash: historyPrefixHash(options.messages, options.messages.length) },
      contract: {
        systemHash,
        toolsHash,
        thinking: thinkingEnabled(options, config),
        nativeSearch: nativeSearchActive(config, model),
      },
      ...(turn.citations.length > 0 ? { provider: { citations: turn.citations } } : {}),
    }
    if (bindSession && options.sessionId !== undefined) {
      this.remotes.set(options.sessionId, createLiveCursor({
        sessionId: options.sessionId,
        chatSessionId,
        parentMessageId: turn.responseMessageId,
        accountHash,
        model,
        messages: options.messages,
        systemHash,
        toolsHash,
        sessionTitle: turn.sessionTitle
          ?? (options.sessionId === undefined ? undefined : this.remotes.get(options.sessionId)?.sessionTitle),
      }))
    }
    return { reasoning: turn.reasoning, text, calls: parsedCalls, replay }
  }
}

const OFF_DEEPTHINK = ReasoningEffortId('off')
const ON_DEEPTHINK = ReasoningEffortId('on')

/** chat.deepseek.com DeepThink is always available; plugin config only sets the default. */
function deepThinkReasoning(enabled: boolean): NonNullable<LlmResolvedModelInfo['reasoning']> {
  return {
    efforts: [
      { id: OFF_DEEPTHINK, name: 'Off' },
      { id: ON_DEEPTHINK, name: 'DeepThink' },
    ],
    defaultEffort: enabled ? ON_DEEPTHINK : OFF_DEEPTHINK,
  }
}

function thinkingEnabled(options: GenerateOptions, config: ResolvedConfig): boolean {
  if (config.thinking === 'disabled') return false
  const effort = options.reasoningEffort
  if (effort === undefined) return true
  if (effort === OFF_DEEPTHINK || effort === 'none') return false
  return true
}

/** "expert" is no longer a distinct wire model; every selection collapses onto "default". */
function modelOf(_model: string): ModelId {
  return 'default'
}

function toLlm(error: unknown): LlmError {
  if (error instanceof LlmError) return error
  if (error instanceof DeepSeekWebError) return new LlmError(error.message, error.code)
  return new LlmError(error instanceof Error ? error.message : String(error), ERROR_CODES.HTTP)
}

function lastHumanUserText(messages: readonly unknown[]): string {
  for (let index = messages.length - 1; index >= 0; index--) {
    const record = messages[index] as { role?: string; source?: { kind?: string }; content?: unknown }
    if (record.role !== 'user') continue
    if (record.source?.kind === 'tool' || record.source?.kind === 'plugin') continue
    const text = extractImportedUserText(flattenText(record.content)).replace(/^USER:\n/, '').trim()
    if (text.length > 0) return text
  }
  return ''
}

function toPromptMessages(messages: readonly unknown[], maxToolResultBytes: number): PromptMessage[] {
  const out: PromptMessage[] = []
  for (const message of messages) {
    const result = toolResultFromMessage(message)
    if (result) {
      out.push({
        role: 'tool',
        sourceKind: 'tool',
        text: '',
        toolResults: [projectToolResult(result.id, result.content, result.isError, maxToolResultBytes)],
      })
      continue
    }
    const record = message as { role?: string; content?: unknown; source?: { kind?: string } }
    const calls = toolCallBlocks(record.content).map(call => ({
      id: call.id,
      name: call.name,
      arguments: JSON.parse(call.arguments || '{}') as Record<string, unknown>,
    }))
    out.push({
      role: record.role ?? 'user',
      sourceKind: record.source?.kind,
      text: flattenText(record.content),
      toolCalls: calls.length > 0 ? calls : undefined,
    })
  }
  return out
}

