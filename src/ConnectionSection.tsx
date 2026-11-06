// Connection tab section for the tts plugin — carved from the pre-split
// voice plugin's SpeechSettingsSection (Cloud TTS card only; the STT plugin
// owns the Cloud STT card on the same tab). Edits persist through this
// plugin's settings store (localStorage "companion:tts" → cloud endpoint)
// and the key goes to the plugin-namespaced keychain account (tts:api-key),
// delivered in-band to the other window.

import { type ChangeEvent } from "react";
import { useSyncExternalStore } from "react";
import type { PluginContext } from "../../sdk/js/@companion/plugin";
import {
  getTtsKey,
  getTtsKeyError,
  getTtsSettings,
  isTtsKeyLoaded,
  saveTtsSettings,
  setTtsKey,
  subscribeTtsSettings,
  type SpeechEndpoint,
  type TtsSettings,
} from "./tts-settings";

type Preset = {
  id: string;
  label: string;
  url: string;
  model: string;
  keyPlaceholder?: string;
  note: string;
};

// OpenAI-compatible endpoints per direction. TTS and STT are independent:
// each has its own URL, model, and API key, so they can mix providers.
const TTS_PRESETS: Preset[] = [
  {
    id: "openai",
    label: "OpenAI",
    url: "https://api.openai.com",
    model: "gpt-4o-mini-tts",
    note: "OpenAI's speech API serves the tts models for synthesis.",
  },
];

// OpenAI-compatible /v1/audio/speech containers. Many OpenAI-compatible
// servers only implement a subset (e.g. wav/pcm), hence the selector.
const TTS_FORMATS = ["mp3", "wav", "pcm", "opus", "aac", "flac"] as const;

type DirectionProps = {
  title: string;
  presetHint: string;
  urlHint: string;
  modelHint: string;
  modelPlaceholder: string;
  endpoint: SpeechEndpoint;
  presets: Preset[];
  keyValue: string;
  loaded: boolean;
  /** When set, offers a response-format selector below the model field. */
  formats?: readonly string[];
  /** Host UI kit (ctx.ui) — passed down because DirectionSection renders
   *  outside the section component where the kit is destructured. */
  ui: Record<string, any>;
  onEndpoint: (next: SpeechEndpoint) => void;
  onKey: (value: string) => void;
};

function DirectionSection({
  title,
  presetHint,
  urlHint,
  modelHint,
  modelPlaceholder,
  endpoint,
  presets,
  keyValue,
  loaded,
  formats,
  ui,
  onEndpoint,
  onKey,
}: DirectionProps) {
  const { FormField, SectionCard, TextInput, Select, Button } = ui;
  const activePreset =
    presets.find((p) => endpoint.url === p.url && endpoint.model === p.model)?.id ?? "custom";
  const active = presets.find((p) => p.id === activePreset);

  return (
    <SectionCard title={title} badge={keyValue ? "key secured" : "no key"}>
      <FormField label="Endpoint preset" hint={presetHint}>
        <div className="preset-bar">
          {[...presets.map((p) => ({ id: p.id, label: p.label })), { id: "custom", label: "Custom" }].map((p) => (
            <button
              key={p.id}
              type="button"
              className={`preset-btn ${activePreset === p.id ? "preset-btn-active" : ""}`}
              onClick={() => {
                if (p.id === "custom") return;
                const preset = presets.find((x) => x.id === p.id);
                if (preset) onEndpoint({ ...endpoint, url: preset.url, model: preset.model });
              }}
            >
              {p.label}
            </button>
          ))}
        </div>
      </FormField>

      {active && (
        <div className="st-field-hint" style={{ marginBottom: 14 }}>
          {active.note}
        </div>
      )}

      <FormField label="Endpoint URL" hint={urlHint}>
        <TextInput
          value={endpoint.url}
          onChange={(e: ChangeEvent<HTMLInputElement>) => onEndpoint({ ...endpoint, url: e.target.value })}
          placeholder="https://api.openai.com"
        />
      </FormField>

      <FormField
        label={`API Key ${keyValue ? "✓ Secured" : "Not set"}`}
        hint="Isolated in your OS keychain — never logged, never committed"
      >
        <div style={{ display: "flex", gap: 6 }}>
          <TextInput
            type="password"
            value={loaded ? keyValue : ""}
            onChange={(e: ChangeEvent<HTMLInputElement>) => onKey(e.target.value)}
            placeholder={active?.keyPlaceholder ?? "sk-..."}
            disabled={!loaded}
          />
          <Button
            variant="ghost"
            onClick={() => onKey("")}
            disabled={!loaded || !keyValue}
          >
            Clear
          </Button>
        </div>
      </FormField>

      <FormField label="Model" hint={modelHint}>
        <TextInput
          value={endpoint.model}
          onChange={(e: ChangeEvent<HTMLInputElement>) => onEndpoint({ ...endpoint, model: e.target.value })}
          placeholder={modelPlaceholder}
        />
      </FormField>

      {formats && (
        <FormField
          label="Response format"
          hint="Audio container the server should return. pcm is raw 16-bit mono at 24 kHz; use wav for servers that reject mp3."
        >
          <Select
            value={endpoint.responseFormat ?? "mp3"}
            onChange={(e: ChangeEvent<HTMLSelectElement>) =>
              onEndpoint({ ...endpoint, responseFormat: e.target.value })}
          >
            {formats.map((f) => (
              <option key={f} value={f}>{f}</option>
            ))}
          </Select>
        </FormField>
      )}
    </SectionCard>
  );
}

export function TtsCloudSection({ ctx }: { ctx: PluginContext }) {
  // The store warms the keychain at activation; keys land here when the
  // async read completes.
  useSyncExternalStore(subscribeTtsSettings, getTtsSettings);
  const settings = getTtsSettings();
  const key = getTtsKey();
  const loaded = isTtsKeyLoaded();
  const chainError = getTtsKeyError();

  const { FormField, SectionCard, TextInput, Select, Button } = ctx.ui as Record<string, any>;

  return (
    <>
      <DirectionSection
        title="Cloud TTS"
        presetHint="Used when the Speech Synthesis tab's engine is switched to Cloud."
        urlHint="OpenAI-compatible base; /v1/audio/speech is appended automatically"
        modelHint="Model id for /v1/audio/speech (empty = cloud TTS unavailable)"
        modelPlaceholder="gpt-4o-mini-tts"
        endpoint={settings.cloud}
        presets={TTS_PRESETS}
        formats={TTS_FORMATS}
        keyValue={key}
        loaded={loaded}
        ui={ctx.ui}
        onEndpoint={(cloud) => saveTtsSettings({ ...settings, cloud })}
        onKey={(v) => void setTtsKey(v)}
      />
      {chainError && <div className="error-banner">{chainError}</div>}
    </>
  );
}
