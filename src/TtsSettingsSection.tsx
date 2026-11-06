// Speech Synthesis tab section — carved from the pre-split voice plugin's
// TtsSettingsSection (TTS direction only; the recognition/wake-word cards
// moved to the stt plugin's Speech Recognition section).
// - State comes from this plugin's TTS settings store (localStorage
//   "companion:tts").
// - The Test/preview button emits the plugin-namespaced "tts-debug-play"
//   event; the plugin's own main-window activation speaks it with noFallback
//   and reports back on "tts-debug-done"/"tts-debug-error".
// - UI kit components come from ctx.ui (the host's shared kit).

import { useEffect, useState, useSyncExternalStore, type ChangeEvent } from "react";
import type { PluginContext } from "@companion-lab/plugin";
import {
  getTtsSettings,
  saveTtsSettings,
  subscribeTtsSettings,
  type TtsSettings,
} from "./tts-settings";

const VOICES = ["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5", "F6", "F7"];
// OpenAI-compatible speech voices.
const CLOUD_VOICES = [
  "alloy", "ash", "ballad", "coral", "echo", "fable",
  "nova", "onyx", "sage", "shimmer", "verse",
];
const LANGUAGES = [
  "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es",
  "et", "fi", "fr", "hi", "hr", "hu", "id", "it", "lt", "lv",
  "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk",
  "vi", "na",
];

function EngineSwitch({ value, onPick }: {
  value: "local" | "cloud";
  onPick: (v: "local" | "cloud") => void;
}) {
  return (
    <div className="preset-bar">
      {(["local", "cloud"] as const).map((p) => (
        <button
          key={p}
          type="button"
          className={`preset-btn ${value === p ? "preset-btn-active" : ""}`}
          onClick={() => onPick(p)}
        >
          {p === "local" ? "Local" : "Cloud"}
        </button>
      ))}
    </div>
  );
}

export function TtsSettingsSection({ ctx }: { ctx: PluginContext }) {
  const tts = useSyncExternalStore(subscribeTtsSettings, getTtsSettings);
  const [debugText, setDebugText] = useState(
    "Hello, I am your companion. Nice to meet you!",
  );
  const [debugStatus, setDebugStatus] = useState<"idle" | "playing" | "error">("idle");
  const [debugError, setDebugError] = useState<string | null>(null);

  const ttsProvider = tts.ttsProvider ?? "local";
  const isCloud = ttsProvider === "cloud";

  useEffect(() => {
    const offDone = ctx.events.on("tts-debug-done", () => setDebugStatus("idle"));
    // Cloud synthesis failures in the preview are reported instead of
    // silently playing the local engine.
    const offError = ctx.events.on("tts-debug-error", (payload) => {
      const message =
        (payload as { message?: string } | null)?.message ?? "Preview failed";
      setDebugStatus("idle");
      setDebugError(message);
    });
    return () => { offDone(); offError(); };
  }, [ctx]);

  const onChange = (next: TtsSettings) => saveTtsSettings(next);

  const debugPlay = () => {
    if (!debugText.trim()) return;
    debugStop();
    setDebugStatus("playing");
    setDebugError(null);
    // Same payload shape the host's TtsSettingsSection emitted on the global
    // "tts-debug-play" channel; namespaced to this plugin now.
    ctx.events.emit("tts-debug-play", {
      text: debugText,
      lang: tts.lang,
      voice: tts.voice,
      speed: tts.speed,
      steps: tts.steps,
      volume: tts.volume,
      ttsProvider: tts.ttsProvider,
      cloudVoice: tts.cloudVoice,
    });
  };

  const debugStop = () => {
    ctx.events.emit("tts-debug-stop");
    setDebugStatus("idle");
  };

  const { FormField, SectionCard, TextArea, TextInput, Select, RangeSlider, Button } =
    ctx.ui as Record<string, any>;

  return (
    <>
      <SectionCard
        title="Voice Synthesis"
        badge={isCloud ? `cloud • ${tts.cloudVoice || "server default"}` : `${tts.voice} • ${tts.lang.toUpperCase()}`}
      >
        <FormField
          label="Engine"
          hint={isCloud
            ? "Cloud synthesizes via the provider configured in the Connection tab."
            : "Local runs the on-device model — private and offline."}
        >
          <EngineSwitch value={ttsProvider} onPick={(p: "local" | "cloud") => onChange({ ...tts, ttsProvider: p })} />
        </FormField>

        {isCloud ? (
          <FormField
            label="Cloud Voice"
            hint="Voice id is provider-specific — OpenAI uses alloy, ash, coral…; other servers use their own ids (e.g. female, male-french). Free text."
          >
            <TextInput
              list="cloud-voice-options"
              value={tts.cloudVoice ?? ""}
              onChange={(e: ChangeEvent<HTMLInputElement>) =>
                onChange({ ...tts, cloudVoice: e.target.value })}
              placeholder="alloy"
            />
            <datalist id="cloud-voice-options">
              {CLOUD_VOICES.map((v) => (
                <option key={v} value={v} />
              ))}
            </datalist>
          </FormField>
        ) : (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <FormField label="Voice Model">
              <Select
                value={tts.voice}
                onChange={(e: ChangeEvent<HTMLSelectElement>) => onChange({ ...tts, voice: e.target.value })}
              >
                {VOICES.map((v) => (
                  <option key={v} value={v}>Voice {v}</option>
                ))}
              </Select>
            </FormField>

            <FormField label="Language">
              <Select
                value={tts.lang}
                onChange={(e: ChangeEvent<HTMLSelectElement>) => onChange({ ...tts, lang: e.target.value })}
              >
                {LANGUAGES.map((l) => (
                  <option key={l} value={l}>{l.toUpperCase()}</option>
                ))}
              </Select>
            </FormField>
          </div>
        )}

        <RangeSlider
          label="Speaking Speed"
          valueLabel={`${tts.speed.toFixed(2)}×`}
          min={0.7}
          max={2.0}
          step={0.05}
          value={tts.speed}
          onChange={(e: ChangeEvent<HTMLInputElement>) => onChange({ ...tts, speed: parseFloat(e.target.value) })}
        />

        {!isCloud && (
          <RangeSlider
            label="Inference Steps"
            valueLabel={`${tts.steps} steps`}
            min={5}
            max={12}
            step={1}
            value={tts.steps}
            onChange={(e: ChangeEvent<HTMLInputElement>) => onChange({ ...tts, steps: parseInt(e.target.value) })}
          />
        )}
      </SectionCard>

      <SectionCard title="Test Speech Output">
        <FormField label="Preview text" hint="Test current voice, engine, and speed in real time">
          <TextArea
            value={debugText}
            onChange={(e: ChangeEvent<HTMLTextAreaElement>) => setDebugText(e.target.value)}
            rows={2}
            style={{ minHeight: 60 }}
            placeholder="Type text to test..."
          />
        </FormField>

        <div style={{ display: "flex", gap: 8, marginTop: 10, alignItems: "center" }}>
          <Button
            variant="primary"
            onClick={debugPlay}
            disabled={debugStatus === "playing"}
            style={{ flex: 1 }}
          >
            {debugStatus === "playing" ? "Synthesizing & Playing..." : "Play Preview"}
          </Button>

          <Button
            variant="danger"
            onClick={debugStop}
            disabled={debugStatus !== "playing"}
          >
            Stop
          </Button>
        </div>

        {debugError && <div className="error-banner">{debugError}</div>}
      </SectionCard>
    </>
  );
}
