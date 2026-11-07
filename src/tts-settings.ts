// TTS (speech synthesis) settings store for the `tts` plugin.
//
// Keychain: the key lives in the plugin-namespaced account `tts:api-key`
// (SDK secrets prefix the plugin id). Edits are delivered in-band to the
// other window on the plugin event bus for live sync. Unlike the pre-split
// voice plugin there are no legacy-account migrations — the split was a
// clean cut.

import type { PluginContext } from "../../../sdk/js/@companion/plugin";

// ─── TTS settings ───

export type TtsSettings = {
  voice: string;
  lang: string;
  speed: number;
  steps: number;
  volume: number;
  /** TTS engine: on-device model or the cloud provider from Connection. */
  ttsProvider?: "local" | "cloud";
  /** Provider voice id used when ttsProvider is "cloud" (e.g. "alloy"). */
  cloudVoice?: string;
  /** Cloud synthesis endpoint (OpenAI-compatible /v1/audio/speech). */
  cloud: SpeechEndpoint;
};

export type SpeechEndpoint = {
  url: string;
  model: string;
  /** Audio container requested from /v1/audio/speech. "pcm" is raw 16-bit
   *  LE samples and gets wrapped into a WAV for decoding. */
  responseFormat?: string;
};

export const DEFAULT_TTS_SETTINGS: TtsSettings = {
  voice: "M1",
  lang: "en",
  speed: 1.05,
  steps: 8,
  volume: 1.0,
  ttsProvider: "local",
  cloudVoice: "alloy",
  cloud: { url: "https://api.openai.com", model: "gpt-4o-mini-tts", responseFormat: "mp3" },
};

const TTS_STORE_KEY = "companion:tts";

/** Keychain account under this plugin's namespace (`tts:api-key`). */
export const TTS_KEY_NAME = "api-key";

export function loadTtsSettings(): TtsSettings {
  try {
    const raw = localStorage.getItem(TTS_STORE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as Partial<TtsSettings>;
      return {
        ...DEFAULT_TTS_SETTINGS,
        ...parsed,
        cloud: { ...DEFAULT_TTS_SETTINGS.cloud, ...(parsed.cloud as Partial<SpeechEndpoint> | undefined) },
      };
    }
  } catch {
    /* ignore */
  }
  return { ...DEFAULT_TTS_SETTINGS };
}

// ─── Store: snapshots + subscribe/emit + cross-window sync ───
//
// Both windows run this plugin's frontend; the provider and the settings
// sections stay in sync through this store. Saves write localStorage AND
// emit plugin events so the other window picks the change up live.

type Listener = () => void;

export type SpeechKeys = { api: string };

let ctxRef: PluginContext | null = null;
let ttsSnapshot: TtsSettings = DEFAULT_TTS_SETTINGS;
let keysSnapshot: SpeechKeys = { api: "" };
let keysLoaded = false;
let keysError: string | null = null;
const listeners = new Set<Listener>();

function notify(): void {
  for (const l of [...listeners]) {
    try {
      l();
    } catch {
      /* listener errors must not break the store */
    }
  }
}

function normalizeTts(value: unknown): TtsSettings {
  const parsed = (value ?? {}) as Partial<TtsSettings>;
  return {
    ...DEFAULT_TTS_SETTINGS,
    ...parsed,
    cloud: { ...DEFAULT_TTS_SETTINGS.cloud, ...(parsed.cloud ?? {}) },
  };
}

/** Read the key from the namespaced keychain account. Throws on keychain
 *  failure so the UI can show why the key is missing. */
async function loadKey(): Promise<string> {
  const ctx = ctxRef;
  if (!ctx) return "";
  return (await ctx.secrets.get(TTS_KEY_NAME)) ?? "";
}

async function refreshKey(): Promise<void> {
  const ctx = ctxRef;
  if (!ctx) return;
  try {
    keysSnapshot = { api: (await loadKey()) ?? "" };
    keysLoaded = true;
    keysError = null;
  } catch (e) {
    // Leave any previously delivered key untouched — an empty keychain read
    // must not clobber a live key.
    keysLoaded = true;
    keysError = `Keychain unavailable — keys won't survive a restart: ${
      e instanceof Error ? e.message : String(e)
    }`;
    ctx.log.warn(`tts keychain load failed: ${keysError}`);
  }
  notify();
}

/** Idempotent. Must run before the provider/sections touch the store. */
export function initTtsSettings(ctx: PluginContext): void {
  if (ctxRef) return;
  ctxRef = ctx;
  ttsSnapshot = loadTtsSettings();

  // Cross-window sync: the other window's saves land here as events.
  ctx.events.on("tts-settings-changed", (payload) => {
    ttsSnapshot = normalizeTts(payload);
    notify();
  });
  // In-band key delivery from the settings window: the main window never
  // touches the keychain at runtime after the initial warm-up.
  ctx.events.on("tts-keys-changed", (payload) => {
    const k = (payload ?? {}) as Partial<SpeechKeys>;
    if (typeof k.api === "string") keysSnapshot = { api: k.api };
    keysLoaded = true;
    keysError = null;
    notify();
  });

  // Warm the keychain so cloud speech works even if the settings window is
  // never opened.
  void refreshKey();
}

export function subscribeTtsSettings(listener: Listener): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

/** Stable snapshot reference (useSyncExternalStore-friendly). */
export function getTtsSettings(): TtsSettings {
  return ttsSnapshot;
}

export function saveTtsSettings(s: TtsSettings): void {
  ttsSnapshot = normalizeTts(s);
  try {
    localStorage.setItem(TTS_STORE_KEY, JSON.stringify(ttsSnapshot));
  } catch {
    /* ignore */
  }
  ctxRef?.events.emit("tts-settings-changed", ttsSnapshot);
  notify();
}

// ─── Cloud key access (keychain-backed) ───

export function getTtsKey(): string {
  return keysSnapshot.api;
}

export function isTtsKeyLoaded(): boolean {
  return keysLoaded;
}

export function getTtsKeyError(): string | null {
  return keysError;
}

/** Persist the key. Empty value removes it (SDK secrets.set deletes empty
 *  values). Delivers in-band immediately so the other window updates live. */
export async function setTtsKey(value: string): Promise<void> {
  const ctx = ctxRef;
  keysSnapshot = { api: value };
  keysError = null;
  notify();
  ctx?.events.emit("tts-keys-changed", { api: value });
  if (!ctx) return;
  try {
    await ctx.secrets.set(TTS_KEY_NAME, value);
  } catch (e) {
    keysError = `Keychain write failed — the key still works this session but won't survive a restart: ${
      e instanceof Error ? e.message : String(e)
    }`;
    notify();
  }
}
