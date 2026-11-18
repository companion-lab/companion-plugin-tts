import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { PluginContext } from "@companion-lab/plugin";

type TtsSettingsModule = typeof import("./tts-settings");
type TtsSettings = import("./tts-settings").TtsSettings;

// The store keeps module-level state (ctxRef, snapshots, listeners) and
// initTtsSettings is idempotent by design, so every test imports a fresh
// module instance via vi.resetModules + dynamic import.
let mod: TtsSettingsModule;

// Map-backed localStorage fake: node has no built-in and the store reads it
// synchronously in load/save.
function makeLocalStorageFake() {
  const store = new Map<string, string>();
  return {
    getItem: (key: string): string | null => (store.has(key) ? (store.get(key) as string) : null),
    setItem: (key: string, value: string): void => void store.set(key, String(value)),
    removeItem: (key: string): void => void store.delete(key),
    clear: (): void => void store.clear(),
  };
}

// Plain fake ctx: the store only touches events.on/emit, secrets.get/set and
// log.warn. `emit` is a recording spy only — on the real host the bus is a
// Tauri broadcast, so incoming cross-window events are simulated explicitly
// with `fire`, which invokes registered handlers directly.
function makeCtx(
  opts: {
    secret?: string | null;
    get?: () => Promise<string | null>;
    setFails?: Error;
  } = {},
) {
  const handlers = new Map<string, Array<(payload: unknown) => void>>();
  const on = vi.fn((event: string, cb: (payload: unknown) => void) => {
    const list = handlers.get(event) ?? [];
    list.push(cb);
    handlers.set(event, list);
    return () => {
      handlers.set(
        event,
        (handlers.get(event) ?? []).filter((h) => h !== cb),
      );
    };
  });
  const emit = vi.fn((_event: string, _payload?: unknown) => {});
  const fire = (event: string, payload?: unknown) => {
    for (const cb of [...(handlers.get(event) ?? [])]) cb(payload);
  };
  const ctx = {
    id: "tts",
    log: { info: vi.fn(), warn: vi.fn(), error: vi.fn() },
    events: { on, emit },
    secrets: {
      get: vi.fn(() =>
        opts.get ? opts.get() : Promise.resolve(opts.secret !== undefined ? opts.secret : null),
      ),
      set: vi.fn(() => (opts.setFails ? Promise.reject(opts.setFails) : Promise.resolve())),
      remove: vi.fn(async () => {}),
    },
  };
  return {
    ctx: ctx as unknown as PluginContext,
    on,
    emit,
    fire,
    secrets: ctx.secrets,
    log: ctx.log,
  };
}

beforeEach(async () => {
  vi.stubGlobal("localStorage", makeLocalStorageFake());
  vi.resetModules();
  mod = await import("./tts-settings");
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("DEFAULT_TTS_SETTINGS", () => {
  it("matches the shipped defaults", () => {
    expect(mod.DEFAULT_TTS_SETTINGS).toEqual({
      voice: "M1",
      lang: "en",
      speed: 1.05,
      steps: 8,
      volume: 1.0,
      ttsProvider: "local",
      cloudVoice: "alloy",
      cloud: { url: "https://api.openai.com", model: "gpt-4o-mini-tts", responseFormat: "mp3" },
    });
  });

  it("uses the plugin-namespaced keychain account", () => {
    // SDK secrets prefix the plugin id; the store only ever sees "api-key".
    expect(mod.TTS_KEY_NAME).toBe("api-key");
  });
});

describe("loadTtsSettings", () => {
  it("returns the defaults when nothing is stored", () => {
    expect(mod.loadTtsSettings()).toEqual(mod.DEFAULT_TTS_SETTINGS);
  });

  it("merges a stored blob over defaults, filling a partial cloud endpoint", () => {
    localStorage.setItem(
      "companion:tts",
      JSON.stringify({ voice: "F2", cloud: { url: "https://gw.example.com" } }),
    );
    expect(mod.loadTtsSettings()).toEqual({
      ...mod.DEFAULT_TTS_SETTINGS,
      voice: "F2",
      cloud: {
        url: "https://gw.example.com",
        model: "gpt-4o-mini-tts",
        responseFormat: "mp3",
      },
    });
  });

  it("falls back to the defaults on malformed JSON", () => {
    localStorage.setItem("companion:tts", "{oops");
    expect(mod.loadTtsSettings()).toEqual(mod.DEFAULT_TTS_SETTINGS);
  });
});

describe("settings store", () => {
  it("starts on defaults until initialized", () => {
    expect(mod.getTtsSettings()).toEqual(mod.DEFAULT_TTS_SETTINGS);
  });

  it("init adopts the stored blob as the first snapshot", () => {
    localStorage.setItem("companion:tts", JSON.stringify({ voice: "stored-voice" }));
    const { ctx } = makeCtx();
    mod.initTtsSettings(ctx);
    expect(mod.getTtsSettings()).toEqual({
      ...mod.DEFAULT_TTS_SETTINGS,
      voice: "stored-voice",
    });
  });

  it("save → get round-trips, persists to localStorage and emits the sync event", () => {
    const { ctx, emit } = makeCtx();
    mod.initTtsSettings(ctx);
    const saved: TtsSettings = {
      ...mod.DEFAULT_TTS_SETTINGS,
      voice: "F1",
      speed: 0.9,
      cloud: { url: "https://x.example.com", model: "m", responseFormat: "wav" },
    };
    mod.saveTtsSettings(saved);
    expect(mod.getTtsSettings()).toEqual({
      voice: "F1",
      lang: "en",
      speed: 0.9,
      steps: 8,
      volume: 1.0,
      ttsProvider: "local",
      cloudVoice: "alloy",
      cloud: { url: "https://x.example.com", model: "m", responseFormat: "wav" },
    });
    expect(JSON.parse(localStorage.getItem("companion:tts") as string)).toEqual(
      mod.getTtsSettings(),
    );
    expect(emit).toHaveBeenCalledWith("tts-settings-changed", mod.getTtsSettings());
  });

  it("normalizes partial saves back to full defaults", () => {
    const { ctx } = makeCtx();
    mod.initTtsSettings(ctx);
    mod.saveTtsSettings({ voice: "F1" } as TtsSettings);
    expect(mod.getTtsSettings()).toEqual({ ...mod.DEFAULT_TTS_SETTINGS, voice: "F1" });
  });

  it("applies settings-changed events from the other window and notifies listeners", () => {
    const { ctx, fire } = makeCtx();
    mod.initTtsSettings(ctx);
    const listener = vi.fn();
    mod.subscribeTtsSettings(listener);
    fire("tts-settings-changed", {
      voice: "synced",
      cloud: { url: "https://other.example.com" },
    });
    expect(mod.getTtsSettings()).toEqual({
      ...mod.DEFAULT_TTS_SETTINGS,
      voice: "synced",
      cloud: {
        url: "https://other.example.com",
        model: "gpt-4o-mini-tts",
        responseFormat: "mp3",
      },
    });
    expect(listener).toHaveBeenCalledTimes(1);
  });

  it("unsubscribing stops notifications", () => {
    const { ctx } = makeCtx();
    mod.initTtsSettings(ctx);
    const listener = vi.fn();
    const unsubscribe = mod.subscribeTtsSettings(listener);
    mod.saveTtsSettings({ ...mod.DEFAULT_TTS_SETTINGS, voice: "a" });
    expect(listener).toHaveBeenCalledTimes(1);
    unsubscribe();
    mod.saveTtsSettings({ ...mod.DEFAULT_TTS_SETTINGS, voice: "b" });
    expect(listener).toHaveBeenCalledTimes(1);
  });

  it("a throwing listener must not break the other listeners", () => {
    const { ctx } = makeCtx();
    mod.initTtsSettings(ctx);
    const bad = vi.fn(() => {
      throw new Error("boom");
    });
    const good = vi.fn();
    mod.subscribeTtsSettings(bad);
    mod.subscribeTtsSettings(good);
    mod.saveTtsSettings({ ...mod.DEFAULT_TTS_SETTINGS, voice: "c" });
    expect(bad).toHaveBeenCalledTimes(1);
    expect(good).toHaveBeenCalledTimes(1);
  });

  it("ignores a second init (idempotent)", () => {
    const a = makeCtx();
    const b = makeCtx();
    mod.initTtsSettings(a.ctx);
    mod.initTtsSettings(b.ctx);
    expect(b.on).not.toHaveBeenCalled();
    mod.saveTtsSettings({ ...mod.DEFAULT_TTS_SETTINGS, voice: "x" });
    expect(a.emit).toHaveBeenCalled();
    expect(b.emit).not.toHaveBeenCalled();
  });
});

describe("keychain handling", () => {
  it("warms the key from the keychain on init", async () => {
    const { ctx, secrets } = makeCtx({ secret: "sk-live-123" });
    mod.initTtsSettings(ctx);
    await vi.waitFor(() => expect(mod.isTtsKeyLoaded()).toBe(true));
    expect(mod.getTtsKey()).toBe("sk-live-123");
    expect(mod.getTtsKeyError()).toBeNull();
    expect(secrets.get).toHaveBeenCalledWith("api-key");
  });

  it("marks keys loaded with an error when the keychain read fails at init", async () => {
    const { ctx, log } = makeCtx({ get: () => Promise.reject(new Error("locked")) });
    mod.initTtsSettings(ctx);
    await vi.waitFor(() => expect(mod.getTtsKeyError()).not.toBeNull());
    expect(mod.isTtsKeyLoaded()).toBe(true);
    expect(mod.getTtsKey()).toBe("");
    expect(mod.getTtsKeyError()).toBe(
      "Keychain unavailable — keys won't survive a restart: locked",
    );
    expect(log.warn).toHaveBeenCalledWith(expect.stringContaining("tts keychain load failed"));
  });

  it("an empty keychain read must not clobber a live key delivered meanwhile", async () => {
    // The warm-up read is still pending when the user saves a key in-band;
    // when the read later fails, refreshKey must leave the snapshot alone.
    let rejectGet!: (e: Error) => void;
    const { ctx } = makeCtx({
      get: () => new Promise<string | null>((_resolve, reject) => (rejectGet = reject)),
    });
    mod.initTtsSettings(ctx);
    await mod.setTtsKey("live-key");
    rejectGet(new Error("locked"));
    await vi.waitFor(() => expect(mod.getTtsKeyError()).not.toBeNull());
    expect(mod.getTtsKey()).toBe("live-key");
    expect(mod.getTtsKeyError()).toContain("Keychain unavailable");
  });

  it("accepts in-band key delivery from the settings window", () => {
    const { ctx, fire } = makeCtx();
    mod.initTtsSettings(ctx);
    fire("tts-keys-changed", { api: "delivered" });
    expect(mod.getTtsKey()).toBe("delivered");
    expect(mod.isTtsKeyLoaded()).toBe(true);
    expect(mod.getTtsKeyError()).toBeNull();
  });

  it("ignores in-band payloads without a string api but still marks keys loaded", () => {
    const { ctx, fire } = makeCtx();
    mod.initTtsSettings(ctx);
    fire("tts-keys-changed", { api: 42 });
    expect(mod.getTtsKey()).toBe("");
    expect(mod.isTtsKeyLoaded()).toBe(true);
  });

  it("setTtsKey delivers in-band immediately and persists to the keychain", async () => {
    const { ctx, emit } = makeCtx();
    mod.initTtsSettings(ctx);
    // Let the init warm-up settle so it cannot race the snapshot below.
    await vi.waitFor(() => expect(mod.isTtsKeyLoaded()).toBe(true));
    await mod.setTtsKey("sk-new");
    expect(mod.getTtsKey()).toBe("sk-new");
    expect(emit).toHaveBeenCalledWith("tts-keys-changed", { api: "sk-new" });
    expect(ctx.secrets.set).toHaveBeenCalledWith("api-key", "sk-new");
    expect(mod.getTtsKeyError()).toBeNull();
  });

  it("setTtsKey('') writes the empty value (SDK secrets.set deletes it)", async () => {
    const { ctx } = makeCtx();
    mod.initTtsSettings(ctx);
    await vi.waitFor(() => expect(mod.isTtsKeyLoaded()).toBe(true));
    await mod.setTtsKey("");
    expect(ctx.secrets.set).toHaveBeenCalledWith("api-key", "");
    expect(mod.getTtsKey()).toBe("");
  });

  it("keeps the in-session key and records an error when the keychain write fails", async () => {
    const { ctx, emit } = makeCtx({ setFails: new Error("denied") });
    mod.initTtsSettings(ctx);
    await vi.waitFor(() => expect(mod.isTtsKeyLoaded()).toBe(true));
    await mod.setTtsKey("sk-x");
    expect(mod.getTtsKey()).toBe("sk-x");
    expect(mod.getTtsKeyError()).toBe(
      "Keychain write failed — the key still works this session but won't survive a restart: denied",
    );
    // In-band delivery still happened before the write failed.
    expect(emit).toHaveBeenCalledWith("tts-keys-changed", { api: "sk-x" });
  });
});
