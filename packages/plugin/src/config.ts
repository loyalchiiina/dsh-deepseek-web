export const PROVIDER = 'deepseek-web'
export const CREDENTIAL_REF_DEFAULT = 'DEEPSEEK_WEB_TOKEN'
export const SETTINGS_NS = 'llm-deepseek-web'

/**
 * Wire model type. DeepSeek's web protocol only carries "default" now, so
 * "expert" is retained purely for reading historical sessions and stale
 * configs; every live request normalises onto "default".
 */
export type ModelId = 'default' | 'expert'
export type ThinkingMode = 'enabled' | 'disabled'
export type NativeSearchMode = 'off' | 'on'
export type BrowserPreference = 'auto' | 'chrome' | 'edge' | 'brave' | 'chromium' | 'managed'

export interface Config {
  tokenEnv?: string
  defaultModel?: ModelId
  thinking?: ThinkingMode
  /** Always on for default; Expert cannot search on the wire. Settings cannot turn this off. */
  nativeSearch?: NativeSearchMode
  streamIdleTimeoutMs?: number
  maxToolProtocolBytes?: number
  maxToolCallsPerTurn?: number
  maxProtocolRepairAttempts?: number
  maxToolResultBytes?: number
  browser?: {
    preferred?: BrowserPreference
    allowManagedDownload?: boolean
  }
}

export interface ResolvedConfig {
  tokenEnv: string
  defaultModel: ModelId
  thinking: ThinkingMode
  nativeSearch: NativeSearchMode
  streamIdleTimeoutMs: number
  maxToolProtocolBytes: number
  maxToolCallsPerTurn: number
  maxProtocolRepairAttempts: number
  maxToolResultBytes: number
  browser: {
    preferred: BrowserPreference
    allowManagedDownload: boolean
  }
}

export const DEFAULT_CONFIG: ResolvedConfig = {
  tokenEnv: CREDENTIAL_REF_DEFAULT,
  defaultModel: 'default',
  thinking: 'enabled',
  nativeSearch: 'on',
  streamIdleTimeoutMs: 180_000,
  maxToolProtocolBytes: 128 * 1024,
  maxToolCallsPerTurn: 16,
  maxProtocolRepairAttempts: 2,
  maxToolResultBytes: 64 * 1024,
  browser: {
    preferred: 'auto',
    allowManagedDownload: true,
  },
}

export function nativeSearchActive(_config: Pick<ResolvedConfig, 'nativeSearch'>, model: string): boolean {
  return model !== 'expert'
}

export function resolveConfig(raw: Config = {}): ResolvedConfig {
  const maxToolCallsPerTurn = raw.maxToolCallsPerTurn ?? DEFAULT_CONFIG.maxToolCallsPerTurn
  if (!Number.isInteger(maxToolCallsPerTurn) || maxToolCallsPerTurn < 1 || maxToolCallsPerTurn > 32) {
    throw new Error('dsh-deepseek-web: maxToolCallsPerTurn must be an integer in 1..32')
  }
  return {
    tokenEnv: raw.tokenEnv ?? DEFAULT_CONFIG.tokenEnv,
    defaultModel: raw.defaultModel ?? DEFAULT_CONFIG.defaultModel,
    thinking: raw.thinking ?? DEFAULT_CONFIG.thinking,
    nativeSearch: 'on',
    streamIdleTimeoutMs: raw.streamIdleTimeoutMs ?? DEFAULT_CONFIG.streamIdleTimeoutMs,
    maxToolProtocolBytes: raw.maxToolProtocolBytes ?? DEFAULT_CONFIG.maxToolProtocolBytes,
    maxToolCallsPerTurn,
    maxProtocolRepairAttempts: raw.maxProtocolRepairAttempts ?? DEFAULT_CONFIG.maxProtocolRepairAttempts,
    maxToolResultBytes: raw.maxToolResultBytes ?? DEFAULT_CONFIG.maxToolResultBytes,
    browser: {
      preferred: raw.browser?.preferred ?? DEFAULT_CONFIG.browser.preferred,
      allowManagedDownload: raw.browser?.allowManagedDownload ?? DEFAULT_CONFIG.browser.allowManagedDownload,
    },
  }
}
