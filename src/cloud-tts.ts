// Cloud speech synthesis against OpenAI-compatible audio APIs — carved from
// the pre-split voice plugin's speech-cloud.ts (TTS direction only; the STT
// plugin owns cloudTranscribe). The endpoint lives in the plugin's own
// settings store (`companion:tts` → `cloud`); the key lives in the
// plugin-namespaced keychain account (`tts:api-key`).

import { doFetch, throwIfHttpError } from "./fetch";
import { getTtsKey, getTtsSettings } from "./tts-settings";

/** "https://api.groq.com/openai/v1/" -> "https://api.groq.com" (host port). */
export function normalizeSpeechBase(url: string): string {
  return url.trim().replace(/\/+$/, "").replace(/\/v1$/, "");
}

export async function cloudSynthesize(
  text: string,
  opts: { voice: string; speed: number },
): Promise<ArrayBuffer> {
  const tts = getTtsSettings().cloud;
  const ttsKey = getTtsKey();
  const base = normalizeSpeechBase(tts.url);
  // Fail with an actionable message instead of a provider 404 when the
  // Connection tab was never filled in.
  if (!tts.model) {
    throw new Error(
      "No cloud TTS model set — configure the Cloud TTS endpoint in Settings → Connection",
    );
  }
  const res = await doFetch(`${base}/v1/audio/speech`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      // Keyless endpoints (self-hosted servers) omit auth entirely.
      ...(ttsKey ? { Authorization: `Bearer ${ttsKey}` } : {}),
    },
    body: JSON.stringify({
      model: tts.model,
      input: text,
      // Omit voice entirely for servers that pick their own default.
      ...(opts.voice ? { voice: opts.voice } : {}),
      speed: opts.speed,
      response_format: tts.responseFormat ?? "mp3",
    }),
  });
  await throwIfHttpError(res);
  const audio = await res.arrayBuffer();
  // Raw PCM has no container, so decodeAudioData can't read it. Wrap the
  // 16-bit little-endian samples into a WAV header for playback.
  if ((tts.responseFormat ?? "mp3") === "pcm") {
    return wrapPcm16Wav(audio, PCM_SAMPLE_RATE);
  }
  return audio;
}

/** Sample rate of OpenAI-compatible "pcm" speech output (16-bit mono LE). */
export const PCM_SAMPLE_RATE = 24_000;

/** Wraps raw 16-bit little-endian PCM bytes in a WAV container so the audio
 *  pipeline can decode them — pcm alone is just samples. */
export function wrapPcm16Wav(pcm: ArrayBuffer, sampleRate: number): ArrayBuffer {
  const view = new DataView(pcm);
  const count = Math.floor(pcm.byteLength / 2);
  const samples = new Float32Array(count);
  for (let i = 0; i < count; i++) samples[i] = view.getInt16(i * 2, true) / 0x8000;
  return encodeWavPcm16(samples, sampleRate).buffer as ArrayBuffer;
}

/** Mono 16-bit PCM WAV container. */
export function encodeWavPcm16(samples: Float32Array, sampleRate: number): Uint8Array {
  const dataBytes = samples.length * 2;
  const buf = new ArrayBuffer(44 + dataBytes);
  const view = new DataView(buf);
  const writeStr = (off: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i));
  };
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + dataBytes, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, dataBytes, true);
  let off = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    off += 2;
  }
  return new Uint8Array(buf);
}
