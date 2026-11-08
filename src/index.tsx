// Speech Synthesis feature plugin frontend (the `tts` plugin — carved from
// the pre-split voice plugin): the Speech Synthesis + Connection settings
// sections, the TtsProvider (local Supertonic + cloud), and the main-window
// activation (eager engine load, preview listeners, mic-active reaction —
// the STT half of that reaction routes through ctx.services.stt()).
//
// Runs in BOTH windows — each window gets its own provider instance kept in
// sync via localStorage + plugin events. The debug/mic listeners only arm in
// the main window.

import { definePlugin, type PluginContext, type TtsSettingsLike } from "@companion-lab/plugin";
import { currentWindowLabel, tauriInvoke, tauriListen } from "./tauri";
import { initTtsSettings, getTtsSettings } from "./tts-settings";
import { createTtsProvider } from "./tts-provider";
import { TtsSettingsSection } from "./TtsSettingsSection";
import { TtsCloudSection } from "./ConnectionSection";

// Same payload shape the host's TtsSettingsSection emitted on the global
// "tts-debug-play" channel.
type TtsDebugPlayPayload = {
  text: string;
  lang: string;
  voice: string;
  speed: number;
  steps: number;
  volume: number;
  ttsProvider?: "local" | "cloud";
  cloudVoice?: string;
};

export default definePlugin((ctx: PluginContext) => {
  ctx.log.info("activating tts plugin");

  // Settings store first: provider + sections read it (localStorage
  // "companion:tts", keychain via ctx.secrets → tts:api-key).
  initTtsSettings(ctx);

  // Settings: the Speech Synthesis tab is here; the shared Connection tab is
  // registered by this plugin AND the stt plugin with the same id — the
  // host's registry takes whichever registers first and both cloud cards
  // render (sections are keyed by section id, not tab owner).
  ctx.settings.registerTab({
    id: "tts",
    label: "Speech Synthesis",
    group: "Companion",
    order: 10,
    desc: "Speech voice, language, speed, volume, and quality.",
    icon: ({ className }: { className?: string }) => (
      <svg className={className} width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" /><path d="M15.54 8.46a5 5 0 0 1 0 7.07" /></svg>
    ),
  });
  ctx.settings.registerTab({
    id: "connection",
    label: "Connection",
    group: "Companion",
    order: 15,
    desc: "Speech endpoints and API keys.",
    icon: ({ className }: { className?: string }) => (
      <svg className={className} width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12.55a11 11 0 0 1 14.08 0" /><path d="M1.42 9a16 16 0 0 1 21.16 0" /><path d="M8.53 16.11a6 6 0 0 1 6.95 0" /><line x1="12" y1="20" x2="12.01" y2="20" /></svg>
    ),
  });
  ctx.settings.registerSection({
    id: "tts.settings",
    tabId: "tts",
    slot: "top",
    order: 100,
    component: function TtsSharedSection() {
      return <TtsSettingsSection ctx={ctx} />;
    },
  });
  ctx.settings.registerSection({
    id: "tts.cloud",
    tabId: "connection",
    slot: "top",
    order: 100,
    component: function TtsCloudSectionEntry() {
      return <TtsCloudSection ctx={ctx} />;
    },
  });

  // Capability provider.
  const ttsProvider = createTtsProvider(ctx);
  ctx.services.registerTts(ttsProvider);

  // ─── Main-window activation ───
  if (currentWindowLabel() !== "main") return;

  const onViseme = (name: string, weight?: number) => ctx.avatar.setViseme(name, weight);

  // Eager engine load, honoring the perf load_mode: skipped entirely
  // under "lazy" (synthesize() lazy-loads backend-side anyway), deferred past
  // startup under "eager" so the ~383MB ONNX load never contends with the
  // avatar's own model load (both route through the plugin host). Best-effort
  // either way; the delay only buys the first speak a cold engine.
  const EAGER_TTS_LOAD_DELAY_MS = 3000;
  void (async () => {
    try {
      // Perf config is the vrm plugin's backend state — a cross-plugin call
      // must name the host command (this plugin's ctx.backend.call routes to
      // the tts backend only).
      const perf = await tauriInvoke<{ load_mode?: "eager" | "lazy" }>(
        "plugin_call",
        { plugin_id: "vrm", method: "vrm.get_performance_config" },
      );
      if (perf?.load_mode === "lazy") return;
    } catch {
      // Perf config unreachable: keep the historical eager default.
    }
    window.setTimeout(() => {
      ctx.backend.call("tts.load_engine", {}).catch((e) => {
        ctx.log.warn(`eager TTS load skipped: ${e instanceof Error ? e.message : String(e)}`);
      });
    }, EAGER_TTS_LOAD_DELAY_MS);
  })();

  // Preview──────── from the settings window (plugin-namespaced channel).
  ctx.events.on("tts-debug-play", (raw) => {
    const payload = (raw ?? {}) as TtsDebugPlayPayload;
    const { text, ...overrides } = payload;
    if (!text) return;
    void ctx.services.stt()?.setSuppress(true);
    ttsProvider
      .speak(
        text,
        { ...getTtsSettings(), ...overrides } as TtsSettingsLike,
        onViseme,
        () => {
          window.setTimeout(() => void ctx.services.stt()?.setSuppress(false), 700);
          ctx.events.emit("tts-debug-done");
        },
        { noFallback: true },
      )
      .catch((e) => {
        window.setTimeout(() => void ctx.services.stt()?.setSuppress(false), 700);
        ctx.events.emit("tts-debug-error", {
          message: e instanceof Error ? e.message : String(e),
        });
      });
  });

  ctx.events.on("tts-debug-stop", () => {
    ttsProvider.stop();
  });

  // Mic lifecycle (chat panel, global channel — NOT plugin-namespaced, so
  // it needs the raw tauri listen). Mirrors the pre-split voice plugin's
  // onMicActiveChanged: announce + avatar reaction, with STT suppression
  // routed through the services facade.
  void tauriListen<boolean>("mic-active-changed", (active) => {
    void ctx.services.stt()?.setSuppress(true);
    if (active) {
      ctx.avatar.setEmotion("happy");
      ctx.avatar.triggerMotion("nod");
      ttsProvider.speak("Listening", getTtsSettings(), onViseme, () => {
        window.setTimeout(() => void ctx.services.stt()?.setSuppress(false), 700);
      });
    } else {
      ctx.avatar.triggerMotion("shake");
      ttsProvider.speak("Mic off", getTtsSettings(), onViseme, () => {
        window.setTimeout(() => void ctx.services.stt()?.setSuppress(false), 700);
      });
    }
  }).catch((e) => {
    ctx.log.warn(`mic-active-changed listener unavailable: ${e instanceof Error ? e.message : String(e)}`);
  });
});
