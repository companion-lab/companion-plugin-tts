// TTS provider — port of the host's app/src/audio/tts.ts (the whole playback
// engine: AudioContext graph, viseme analysis, sentence-chunked streaming,
// speak/speakLoop/stop, SpeechSynthesis fallback) wrapped in the SDK's
// TtsProvider shape.
//
// Backend wiring: the former `invoke("synthesize", …)` command is now
// ctx.backend.call("tts.synthesize", …) (same JSON payload, JSON byte-array
// result). Cloud synthesis ports cloudSynthesize from speech-cloud.ts.
//
// STT suppression: with TTS and STT in separate plugins, playback
// suppresses the STT engine through the host's services facade
// (ctx.services.stt()?.setSuppress) and releases it after a 700ms grace
// tail; the grace timer is cancelled whenever new playback starts.

import type { PluginContext, TtsProvider, TtsStream } from "@companion-lab/plugin";
import { cloudSynthesize } from "./cloud-tts";
import { DEFAULT_TTS_SETTINGS, getTtsSettings, type TtsSettings } from "./tts-settings";

// ─── Audio graph ───

let audioCtx: AudioContext | null = null;
let masterGain: GainNode | null = null;

function getAudioCtx(): AudioContext {
  if (!audioCtx) audioCtx = new AudioContext({ sampleRate: 44100 });
  if (audioCtx.state === "suspended") audioCtx.resume();
  return audioCtx;
}

let currentVolume = 1.0;

function getMasterGain(): GainNode {
  const ctx = getAudioCtx();
  if (!masterGain) {
    masterGain = ctx.createGain();
    masterGain.gain.value = currentVolume;
    masterGain.connect(ctx.destination);
  }
  return masterGain;
}

export function setTtsVolume(v: number): void {
  currentVolume = Math.min(1, Math.max(0, v));
  if (!audioCtx || !masterGain) return;
  masterGain.gain.setTargetAtTime(currentVolume, audioCtx.currentTime, 0.03);
}

/** Maps smoothed relative loudness (0 = silence, 1 = recent peak of this
 *  utterance) to a viseme with a continuous openness weight. The mouth
 *  follows the voice's envelope like a volume meter — a discrete open/close
 *  state machine read as mechanical chomping. Shape climbs a ladder with
 *  loudness: pucker "ou" → rounded "oh" → open "aa", so the O family is
 *  always present; a high zero-crossing rate (fricatives) picks the tight
 *  "ih" instead. */
export function amplitudeToViseme(
  level: number,
  zcr: number,
): { name: string; weight: number } {
  if (!Number.isFinite(level) || level < 0.12) return { name: "neutral", weight: 0 };
  let o = Math.min(1, Math.max(0, (level - 0.12) / 0.55));
  o = o * o * (3 - 2 * o); // smoothstep, so the mouth eases in and out
  const name = zcr > 0.2 ? "ih" : level > 0.8 ? "aa" : level > 0.4 ? "oh" : "ou";
  return { name, weight: Math.max(0.25, o) };
}

// ─── STT suppression during playback (was external in App.tsx) ───

const SUPPRESS_GRACE_MS = 700;
let suppressReleaseTimer: number | null = null;
// Whether the mute request is currently outstanding. suppressStt() fires per
// streamed token, so once the mute is engaged only the pending release timer
// needs cancelling — not another IPC round-trip.
let suppressEngaged = false;

function suppressStt(): void {
  if (suppressReleaseTimer !== null) {
    window.clearTimeout(suppressReleaseTimer);
    suppressReleaseTimer = null;
  }
  if (suppressEngaged) return;
  suppressEngaged = true;
  // Cross-plugin via the host services registry (nil until the stt plugin
  // registers its provider in this window).
  void synthCtxRef?.services.stt()?.setSuppress(true);
}

function releaseSttWithGrace(): void {
  if (suppressReleaseTimer !== null) window.clearTimeout(suppressReleaseTimer);
  suppressReleaseTimer = window.setTimeout(() => {
    suppressReleaseTimer = null;
    suppressEngaged = false;
    void synthCtxRef?.services.stt()?.setSuppress(false);
  }, SUPPRESS_GRACE_MS);
}

// ─── Playback ───

// Active one-shot playback (speak()/fallbackSpeak) so stop() can cut it off.
// StreamTts manages its own queue but shares this node.
let activeSource: AudioBufferSourceNode | null = null;

// Bumped by stop() so a speak() still waiting on synthesis knows its audio is
// unwanted and skips playback (but still runs onEnd, so UI state resets).
let speakGeneration = 0;

// Stops the in-flight buffer. onended still fires, so playNext()/speak()
// resolve normally and their onEnd callbacks run.
function haltPlayback(): void {
  const src = activeSource;
  activeSource = null;
  if (!src) return;
  try {
    src.stop();
  } catch {
    /* already stopped or never started */
  }
}

function playAudioBuffer(
  audioBuf: AudioBuffer,
  onViseme: (name: string, weight?: number) => void,
): Promise<void> {
  return new Promise((resolve) => {
    const ctx = getAudioCtx();
    const source = ctx.createBufferSource();
    source.buffer = audioBuf;
    activeSource = source;

    const master = getMasterGain();
    const analyser = ctx.createAnalyser();
    // Small FFT keeps the time-domain window tight (~12ms) so viseme flips
    // track the waveform instead of trailing it.
    analyser.fftSize = 512;
    source.connect(analyser);
    analyser.connect(master);

    const timeData = new Float32Array(analyser.fftSize);
    let raf = 0;
    let lastTick = 0;
    // Loudness envelope of the current utterance: instant attack, slow
    // release (~1s decay at 30Hz), used as the normalization reference.
    let peak = 0;
    // Smoothed normalized level: opening is instantaneous so the mouth pops
    // with the syllable, closing eases over ~2-3 frames so the flap follows
    // the voice without jitter or mechanical chomping.
    let levelSm = 0;

    // Throttle viseme analysis to ~30Hz on its own clock instead of full rAF:
    // same visual result, half the FFT work, and it stops competing with the
    // 3D render loop for frame budget.
    const tick = (now: number) => {
      if (now - lastTick >= 33) {
        lastTick = now;
        analyser.getFloatTimeDomainData(timeData);

        let sum = 0;
        let crossings = 0;
        for (let i = 0; i < timeData.length; i++) {
          sum += timeData[i] * timeData[i];
          if (i > 0 && timeData[i] >= 0 !== timeData[i - 1] >= 0) crossings++;
        }
        const rms = timeData.length ? Math.sqrt(sum / timeData.length) : 0;
        const zcr = timeData.length > 1 ? crossings / (timeData.length - 1) : 0;
        peak = Math.max(rms, peak * 0.97);
        const level = peak > 0 ? rms / peak : 0;
        levelSm = level >= levelSm ? level : levelSm * 0.65 + level * 0.35;

        const viseme = amplitudeToViseme(levelSm, zcr);
        onViseme(viseme.name, viseme.weight);
      }
      raf = requestAnimationFrame(tick);
    };

    source.onended = () => {
      cancelAnimationFrame(raf);
      if (activeSource === source) activeSource = null;
      try {
        source.disconnect();
        analyser.disconnect();
      } catch {
        /* already disconnected */
      }
      onViseme("neutral");
      // Arm the grace release: the next buffer's suppressStt() cancels it
      // (covering inter-chunk gaps); after the final buffer it actually
      // fires and unmutes the mic. StreamTts never goes through speak(),
      // so this is the only release path for streamed replies.
      releaseSttWithGrace();
      resolve();
    };

    // Suppression wraps every buffer: chunked streams stay covered across
    // the inter-chunk gaps because the next start cancels the grace timer.
    suppressStt();
    raf = requestAnimationFrame(tick);
    source.start(0);
  });
}

// ─── Synthesis ───

type SynthOpts = {
  /** Debug preview mode: surface cloud failures to the caller instead of
   *  degrading to the local engine (which reads as "the switch did nothing"). */
  noFallback?: boolean;
};

async function synthesizeChunk(
  ctx: PluginContext,
  text: string,
  settings: TtsSettings,
  opts: SynthOpts = {},
): Promise<AudioBuffer | null> {
  const trimmed = text.trim();
  if (!trimmed) return null;

  if (settings.ttsProvider === "cloud") {
    try {
      // OpenAI-compatible speech API accepts 0.25–4.0.
      const speed = Math.min(4, Math.max(0.25, settings.speed || 1));
      const mp3 = await cloudSynthesize(trimmed, {
        // Empty voice = server default; providers with a fixed id set
        // (e.g. OpenAI) validate it and error loudly.
        voice: settings.cloudVoice || "",
        speed,
      });
      return await getAudioCtx().decodeAudioData(mp3);
    } catch (e) {
      // Cloud failed (no key, network, bad model) — degrade to the local
      // engine instead of dropping the reply on the floor.
      console.error("Cloud TTS failed, falling back to local engine:", e);
      if (opts.noFallback) throw e instanceof Error ? e : new Error(String(e));
    }
  }

  // Former `invoke("synthesize", …)` — the voice backend's method has the
  // same snake_case payload and returns a JSON array of wav bytes.
  const wavBytes = await ctx.backend.call<number[]>("tts.synthesize", {
    text: trimmed,
    lang: settings.lang,
    voice: settings.voice,
    speed: settings.speed,
    steps: settings.steps,
  });

  const buf = new Uint8Array(wavBytes).buffer;
  return getAudioCtx().decodeAudioData(buf);
}

// ─── StreamTts (sentence chunking) — host port ───

const SENTENCE_RE = /[^.!?。！？\n]*[.!?。！？\n]/g;
const MIN_CHUNK = 20;
const MAX_CHUNK = 120;

let generation = 0;

// synthesizeChunk needs the PluginContext; the provider factory captures it
// here so the ported class/method bodies stay 1:1 with the host.
let synthCtxRef: PluginContext | null = null;

type VisemeFn = (name: string, weight?: number) => void;

export class StreamTts {
  private buffer = "";
  private queue: AudioBuffer[] = [];
  private playing = false;
  private speaking = false;
  private flushed = false;
  private stopped = false;
  private gen: number;
  private settings: TtsSettings;
  private onViseme: VisemeFn;
  private onAllDone: () => void;
  private onSpeakingChange: (speaking: boolean) => void;

  constructor(
    settings: TtsSettings,
    onViseme: VisemeFn,
    onAllDone: () => void,
    onSpeakingChange: (speaking: boolean) => void = () => {},
  ) {
    this.gen = ++generation;
    this.settings = settings;
    this.onViseme = onViseme;
    this.onAllDone = onAllDone;
    this.onSpeakingChange = onSpeakingChange;
  }

  push(token: string) {
    if (this.stopped) return;
    // Mute the STT capture from the first token: the reply may start
    // playing any moment, and trailing room/user audio must not reach the
    // recognizer around the playback windows. releaseSttWithGrace() (armed
    // by the last buffer's onended) lifts it once the reply finishes.
    suppressStt();
    this.buffer += token;
    this.drain();
  }

  isSpeaking(): boolean {
    return this.speaking;
  }

  private setSpeaking(v: boolean) {
    if (this.speaking === v) return;
    this.speaking = v;
    this.onSpeakingChange(v);
  }

  flush() {
    this.flushed = true;
    this.enqueueRemaining();
  }

  stop() {
    this.stopped = true;
    this.buffer = "";
    this.queue = [];
    this.flushed = false;
    haltPlayback();
    this.setSpeaking(false);
    releaseSttWithGrace();
  }

  private drain() {
    // Enqueue each sentence separately so the first chunk reaches the
    // synthesizer while later ones are still streaming. Batching the whole
    // buffer into one synth call meant waiting on the entire response.
    let consumed = 0;
    for (const m of this.buffer.matchAll(SENTENCE_RE)) {
      const sentence = m[0];
      if (sentence) {
        consumed += sentence.length;
        this.enqueue(sentence);
      }
    }
    if (consumed) this.buffer = this.buffer.slice(consumed);

    if (this.buffer.length >= MAX_CHUNK) {
      const cut = this.buffer.lastIndexOf(" ", MAX_CHUNK);
      if (cut > MIN_CHUNK) {
        const chunk = this.buffer.slice(0, cut);
        this.buffer = this.buffer.slice(cut + 1);
        this.enqueue(chunk);
      }
    }
  }

  private enqueueRemaining() {
    const remaining = this.buffer.trim();
    this.buffer = "";
    if (remaining) this.enqueue(remaining);
  }

  private enqueue(text: string) {
    const myGen = this.gen;
    synthesizeChunk(synthCtxRef!, text, this.settings)
      .then((buf) => {
        if (this.stopped || myGen !== generation) return;
        if (!buf) return;
        this.queue.push(buf);
        if (!this.playing) this.playNext();
      })
      .catch((e) => {
        console.error("TTS chunk error:", e);
        if (this.flushed && this.queue.length === 0 && !this.buffer) {
          this.setSpeaking(false);
          releaseSttWithGrace();
          this.onAllDone();
        }
      });
  }

  private playNext() {
    if (this.stopped || this.gen !== generation) return;

    if (this.queue.length === 0) {
      this.playing = false;
      if (this.flushed && !this.buffer) {
        this.setSpeaking(false);
        releaseSttWithGrace();
        this.onAllDone();
      }
      return;
    }

    this.playing = true;
    this.setSpeaking(true);
    const buf = this.queue.shift()!;
    playAudioBuffer(buf, this.onViseme).then(() => this.playNext());
  }
}

// ─── speak / speakLoop / stop / fallback — host port ───

export async function speak(
  text: string,
  settings: TtsSettings,
  onViseme: VisemeFn,
  onEnd: () => void,
  opts: SynthOpts = {},
): Promise<void> {
  if (!text) {
    onEnd();
    return;
  }
  const myGen = ++speakGeneration;
  // Suppress recognition for synthesis + playback, released after the grace
  // tail (App.tsx used to do this around every speak call).
  suppressStt();
  const finish = () => {
    releaseSttWithGrace();
    onEnd();
  };

  try {
    const buf = await synthesizeChunk(synthCtxRef!, text, settings, opts);
    if (myGen !== speakGeneration) {
      releaseSttWithGrace();
      onEnd();
      return;
    }
    if (!buf) {
      releaseSttWithGrace();
      onEnd();
      return;
    }
    setTtsVolume(settings.volume ?? 1.0);
    await playAudioBuffer(buf, onViseme);
    finish();
  } catch (err) {
    if (opts.noFallback) {
      releaseSttWithGrace();
      throw err;
    }
    console.error("TTS error:", err);
    if (myGen !== speakGeneration) {
      releaseSttWithGrace();
      onEnd();
      return;
    }
    // The fallback keeps recognition suppressed until it finishes.
    fallbackSpeak(text, settings.volume, onViseme, finish);
  }
}

/**
 * Repeat `text` gaplessly until stop() (or any other speak) bumps the
 * generation. Synthesizes once and reuses the decoded buffer each pass, so
 * looping costs no extra inference.
 */
export async function speakLoop(
  text: string,
  settings: TtsSettings,
  onViseme: VisemeFn,
  onEnd: () => void,
): Promise<void> {
  if (!text) {
    onEnd();
    return;
  }
  const myGen = ++speakGeneration;
  suppressStt();

  let buf: AudioBuffer | null;
  try {
    buf = await synthesizeChunk(synthCtxRef!, text, settings);
  } catch (err) {
    console.error("TTS loop error:", err);
    releaseSttWithGrace();
    if (myGen === speakGeneration) {
      fallbackSpeak(text, settings.volume, onViseme, onEnd);
      return;
    }
    onEnd();
    return;
  }
  if (myGen !== speakGeneration || !buf) {
    releaseSttWithGrace();
    onEnd();
    return;
  }

  setTtsVolume(settings.volume ?? 1.0);
  try {
    // haltPlayback() ends the current buffer via onended, which resolves the
    // awaited promise; the generation check then breaks the loop.
    while (myGen === speakGeneration) {
      await playAudioBuffer(buf, onViseme);
      onViseme("neutral");
    }
  } finally {
    releaseSttWithGrace();
  }
  onEnd();
}

/** Cut off whatever is speaking right now (alarm, debug, or streamed chat). */
export function stop(): void {
  speakGeneration++;
  haltPlayback();
  if ("speechSynthesis" in window) speechSynthesis.cancel();
}

function fallbackSpeak(
  text: string,
  volume: number,
  onViseme: VisemeFn,
  onEnd: () => void,
) {
  if (!("speechSynthesis" in window)) {
    onEnd();
    return;
  }
  const u = new SpeechSynthesisUtterance(text);
  u.volume = Math.min(1, Math.max(0, volume));
  let raf = 0;
  u.onstart = () => {
    let i = 0;
    const cycle = ["aa", "ih", "ou", "ee", "oh", "neutral", "aa", "oh"];
    const tick = () => {
      onViseme(cycle[i % cycle.length]);
      i++;
      raf = window.setTimeout(tick, 80) as unknown as number;
    };
    tick();
  };
  const cleanup = () => {
    if (raf) {
      clearTimeout(raf);
      raf = 0;
    }
    onViseme("neutral");
    onEnd();
  };
  u.onend = cleanup;
  u.onerror = cleanup;
  speechSynthesis.cancel();
  speechSynthesis.speak(u);
}

// ─── Provider ───

/**
 * Fill missing fields from the plugin's own settings store, then defaults.
 * Plugin consumers (alarm/reminder banners) call speak/speakLoop with an
 * empty settings object — the host equivalents always passed the full
 * snapshot, so the store snapshot is the natural fill here.
 */
function resolveSettings(settings: Record<string, unknown>): TtsSettings {
  const passed: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(settings)) {
    if (v !== undefined) passed[k] = v;
  }
  return { ...DEFAULT_TTS_SETTINGS, ...getTtsSettings(), ...passed } as TtsSettings;
}

export function createTtsProvider(ctx: PluginContext): TtsProvider {
  synthCtxRef = ctx;
  return {
    id: "tts",
    speak: (text, settings, onViseme, onEnd, opts) =>
      speak(text, resolveSettings(settings), onViseme, onEnd, opts),
    speakLoop: (text, settings, onViseme, onEnd) =>
      speakLoop(text, resolveSettings(settings), onViseme, onEnd),
    createStream: (settings, onViseme, onAllDone, onSpeakingChange) => {
      const stream = new StreamTts(
        resolveSettings(settings),
        onViseme,
        onAllDone,
        onSpeakingChange,
      );
      const wrapped: TtsStream = {
        push: (t) => stream.push(t),
        flush: () => stream.flush(),
        stop: () => stream.stop(),
      };
      return wrapped;
    },
    stop: () => stop(),
    setVolume: (v) => setTtsVolume(v),
  };
}
