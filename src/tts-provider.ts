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
import {
  DEFAULT_TTS_SETTINGS,
  getTtsSettings,
  subscribeTtsSettings,
  type TtsSettings,
} from "./tts-settings";

// ─── Audio graph ───

let audioCtx: AudioContext | null = null;
let masterGain: GainNode | null = null;

// WKWebView can leave the context unable to produce sound after the Mac
// idles: state "interrupted" (a WebKit-only state, not in the TS union)
// after sleep or audio-route changes, or an output device WebKit keeps
// latched after it disappears while state still reads "running". Nothing in
// the graph throws in those cases — playback just goes quiet and onended
// may never fire — so recovery is proactive: resume on any non-running
// state, and rebuild the whole graph when resume won't take (a fresh
// context re-acquires the current default output device).

const delay = (ms: number) =>
  new Promise<void>((resolve) => window.setTimeout(resolve, ms));

// A wedged backend hangs its callers forever (tts.synthesize runs behind an
// in-process Mutex and plugin_call has no IPC timeout), which would leave
// enqueue()'s promise unsettled and the stream stalled with zero logs.
// Bound the waits so a stuck chunk errors out and the queue keeps moving.
const SYNTH_TIMEOUT_MS = 30_000;
const CLOUD_TIMEOUT_MS = 20_000;
// WebKit's resume() can return a promise that never settles on a wedged
// ("interrupted") context. Bound it, or recoveryPass pins forever and every
// later utterance hangs in ensureRunning() before the playback watchdog can
// arm — silence until the app restarts.
const RESUME_TIMEOUT_MS = 2_000;
// decodeAudioData on a half-dead context can likewise never settle, which
// would leave a chunk's promise hanging and stall the stream queue mid-reply.
const DECODE_TIMEOUT_MS = 15_000;

function withTimeout<T>(p: Promise<T>, ms: number, label: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = window.setTimeout(
      () => reject(new Error(`${label} timed out after ${ms}ms`)),
      ms,
    );
    p.then(
      (v) => {
        window.clearTimeout(timer);
        resolve(v);
      },
      (e) => {
        window.clearTimeout(timer);
        reject(e);
      },
    );
  });
}

let recoveryPass: Promise<AudioContext | null> | null = null;

function logAudio(msg: string): void {
  console.warn(`[tts audio] ${msg}`);
  synthCtxRef?.log.warn(`[tts audio] ${msg}`);
}

function createCtx(): AudioContext {
  const ctx = new AudioContext({ sampleRate: 44100 });
  // Auto-recover whenever the context drops out of running mid-session —
  // otherwise nothing would resume it until the next utterance happened to
  // check (and an "interrupted" state would never resume at all, which is
  // exactly the "no sound after idle" wedge).
  ctx.addEventListener("statechange", () => {
    const st = ctx.state as string;
    if (st !== "running" && st !== "closed") scheduleRecovery();
  });
  return ctx;
}

function getAudioCtx(): AudioContext {
  if (!audioCtx) audioCtx = createCtx();
  return audioCtx;
}

function scheduleRecovery(): void {
  if (recoveryPass) return;
  recoveryPass = recover().finally(() => {
    recoveryPass = null;
  });
}

/** Resume (with one retry) and, if the context still won't run, rebuild the
 *  graph. Returns null when even a fresh context can't run — callers skip
 *  the buffer instead of letting the playback promise hang forever. */
async function recover(depth = 0): Promise<AudioContext | null> {
  const ctx = getAudioCtx();
  const before = ctx.state as string;
  if (before === "running") return ctx;
  for (let attempt = 0; attempt < 2; attempt++) {
    if (attempt > 0) await delay(250);
    try {
      await withTimeout(ctx.resume(), RESUME_TIMEOUT_MS, "ctx.resume()");
    } catch (e) {
      logAudio(`resume() failed from ${before}: ${e}`);
    }
    if ((ctx.state as string) === "running") {
      logAudio(`context resumed from ${before}`);
      return ctx;
    }
  }
  if (depth === 0) {
    logAudio(`context stuck in ${ctx.state} after resume — rebuilding audio graph`);
    rebuildGraph();
    return recover(1);
  }
  logAudio(`context still in ${ctx.state} after rebuild — skipping playback this pass`);
  return null;
}

/** Drop the graph and build a fresh one: closes the old context and resets
 *  the cached gain node (recreated lazily at the current volume). */
function rebuildGraph(): void {
  const old = audioCtx;
  audioCtx = null;
  masterGain = null;
  if (old) {
    try {
      void old.close();
    } catch {
      /* already closed */
    }
  }
  audioCtx = createCtx();
}

// Every new utterance starts on a fresh graph: WebKit can latch onto an
// output device that has since disappeared (sleep/wake, AirPods/monitor
// switch) and keep rendering into the void with state "running" —
// undetectable in-page, so preempt it. (An earlier 5-min idle timer lost to
// anyone speaking more often than every 5 min: each attempt refreshed the
// timestamp, so a latched graph was never rebuilt until an app restart.)
async function preparePlayback(fresh = false): Promise<AudioContext | null> {
  // Never yank the graph out from under a buffer that is still sounding.
  if (fresh && !activeSource) {
    logAudio("fresh graph for new utterance — rebuilding audio graph");
    rebuildGraph();
  }
  return ensureRunning();
}

async function ensureRunning(): Promise<AudioContext | null> {
  if (recoveryPass) await recoveryPass.catch(() => {});
  const ctx = getAudioCtx();
  if ((ctx.state as string) === "running") return ctx;
  return recover();
}

let currentVolume = 1.0;

function getMasterGain(ctx: AudioContext): GainNode {
  // Cached across plays, but it must belong to the *current* context — a
  // graph rebuild swaps audioCtx, and connecting nodes across two contexts
  // throws (or silently renders into a disconnected graph).
  if (!masterGain || masterGain.context !== ctx) {
    masterGain = ctx.createGain();
    masterGain.gain.value = currentVolume;
    masterGain.connect(ctx.destination);
  }
  return masterGain;
}

export function setTtsVolume(v: number): void {
  currentVolume = Math.min(1, Math.max(0, v));
  if (!audioCtx || !masterGain || masterGain.context !== audioCtx) return;
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

/** Plays one decoded buffer. Resolves true when source.start() actually
 *  scheduled audio on the graph, false when the buffer was skipped (no
 *  usable context) or start threw — StreamTts uses this to tell a reply
 *  that was heard from one that vanished silently. */
async function playAudioBuffer(
  audioBuf: AudioBuffer,
  onViseme: (name: string, weight?: number) => void,
  opts: { freshGraph?: boolean } = {},
): Promise<boolean> {
  // Revive/rebuild the context first so playback never starts into a dead
  // graph (which schedules nothing audible and never fires onended).
  let prepared: AudioContext | null = null;
  try {
    prepared = await preparePlayback(opts.freshGraph ?? false);
    // A concurrent recovery may have closed the context during the await
    // above; re-check and give it one more preparation pass before use.
    if (prepared && (prepared.state as string) !== "running") {
      prepared = await preparePlayback(false);
    }
  } catch (e) {
    logAudio(`playback prep failed: ${e}`);
  }
  if (!prepared) {
    // Unrecoverable right now: resolve instead of hanging so the stream
    // queue keeps moving; the next buffer gets a fresh recovery attempt.
    logAudio("skipping buffer — audio context unavailable");
    releaseSttWithGrace();
    return false;
  }
  const ctx = prepared;

  return new Promise<boolean>((resolve) => {
    // Graph nodes are built strictly on the prepared context — a graph
    // rebuild racing this buffer must never mix nodes across two contexts.
    let source: AudioBufferSourceNode;
    let analyser: AnalyserNode;
    try {
      source = ctx.createBufferSource();
      source.buffer = audioBuf;
      activeSource = source;

      const master = getMasterGain(ctx);
      analyser = ctx.createAnalyser();
      // Small FFT keeps the time-domain window tight (~12ms) so viseme flips
      // track the waveform instead of trailing it.
      analyser.fftSize = 512;
      source.connect(analyser);
      analyser.connect(master);
    } catch (e) {
      // Cross-context race or a dying context: resolve instead of throwing
      // so the StreamTts queue keeps moving (a rejection here used to wedge
      // playNext() with playing=true and the mic left suppressed).
      logAudio(`playback graph failed: ${e}`);
      onViseme("neutral");
      releaseSttWithGrace();
      resolve(false);
      return;
    }

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
    // Loudest frame seen this buffer — feeds the silence detector in
    // finish() (the analyser sits before the volume gain, so the slider
    // can't mask it).
    let maxRms = 0;

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
        maxRms = Math.max(maxRms, rms);
        const zcr = timeData.length > 1 ? crossings / (timeData.length - 1) : 0;
        peak = Math.max(rms, peak * 0.97);
        const level = peak > 0 ? rms / peak : 0;
        levelSm = level >= levelSm ? level : levelSm * 0.65 + level * 0.35;

        const viseme = amplitudeToViseme(levelSm, zcr);
        onViseme(viseme.name, viseme.weight);
      }
      raf = requestAnimationFrame(tick);
    };

    // Watchdog: if the context dies mid-utterance, onended never fires and
    // this promise would hang forever — wedging the StreamTts queue with
    // playing=true and leaving the mic suppressed. Resolve on a deadline
    // instead; the next buffer self-heals through preparePlayback().
    let settled = false;
    let watchdog = 0;
    // Whether source.start() succeeded — the resolution value that lets
    // StreamTts distinguish a heard buffer from a silently skipped one.
    let startOk = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      window.clearTimeout(watchdog);
      cancelAnimationFrame(raf);
      if (activeSource === source) activeSource = null;
      try {
        source.disconnect();
        analyser.disconnect();
      } catch {
        /* already disconnected */
      }
      // Silence detector: a non-silent buffer that completed with no signal
      // at all means the graph either never pulled the source or pulled it
      // into nowhere — the "state running but silent" wedge. Rebuild now so
      // the next utterance starts on a context bound to a live device.
      if (audioBuf.duration >= 0.3 && maxRms < 0.001) {
        logAudio(
          `${audioBuf.duration.toFixed(1)}s buffer played with zero signal — output device likely latched, rebuilding audio graph`,
        );
        rebuildGraph();
      }
      onViseme("neutral");
      // Arm the grace release: the next buffer's suppressStt() cancels it
      // (covering inter-chunk gaps); after the final buffer it actually
      // fires and unmutes the mic. StreamTts never goes through speak(),
      // so this is the only release path for streamed replies.
      releaseSttWithGrace();
      resolve(startOk);
    };

    source.onended = finish;

    watchdog = window.setTimeout(
      () => {
        if (settled) return;
        logAudio(
          `onended never fired (context ${ctx.state}) — watchdog released the queue`,
        );
        try {
          source.stop();
        } catch {
          /* never started or already stopped */
        }
        finish();
      },
      Math.max(2000, audioBuf.duration * 1000 + 1500),
    );

    // Suppression wraps every buffer: chunked streams stay covered across
    // the inter-chunk gaps because the next start cancels the grace timer.
    suppressStt();
    raf = requestAnimationFrame(tick);
    try {
      source.start(0);
      startOk = true;
    } catch (e) {
      // Never let a start failure escape the promise — finish() releases
      // the mic and resolves the queue either way.
      logAudio(`source.start failed: ${e}`);
      finish();
    }
  });
}

// ─── Synthesis ───

type SynthOpts = {
  /** Debug preview mode: surface cloud failures to the caller instead of
   *  degrading to the local engine (which reads as "the switch did nothing"). */
  noFallback?: boolean;
};

/** decodeAudioData bounded: a decode queued on a wedged context can never
 *  settle, which would hang the chunk's promise and stall the stream queue
 *  mid-reply (with the mic left suppressed). On failure the graph is rebuilt
 *  — the context is the usual culprit, and a rebuild is harmless when the
 *  audio data itself was bad. */
async function decodeWithRebuild(bytes: ArrayBuffer): Promise<AudioBuffer> {
  try {
    return await withTimeout(
      getAudioCtx().decodeAudioData(bytes),
      DECODE_TIMEOUT_MS,
      "decodeAudioData",
    );
  } catch (e) {
    logAudio(`decodeAudioData failed: ${e} — rebuilding audio graph`);
    rebuildGraph();
    throw e;
  }
}

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
      const mp3 = await withTimeout(
        cloudSynthesize(trimmed, {
          // Empty voice = server default; providers with a fixed id set
          // (e.g. OpenAI) validate it and error loudly.
          voice: settings.cloudVoice || "",
          speed,
        }),
        CLOUD_TIMEOUT_MS,
        "cloud TTS",
      );
      return await decodeWithRebuild(mp3);
    } catch (e) {
      // Cloud failed (no key, network, bad model) — degrade to the local
      // engine instead of dropping the reply on the floor.
      console.error("Cloud TTS failed, falling back to local engine:", e);
      if (opts.noFallback) throw e instanceof Error ? e : new Error(String(e));
    }
  }

  // Former `invoke("synthesize", …)` — the voice backend's method has the
  // same snake_case payload and returns a JSON array of wav bytes.
  const wavBytes = await withTimeout(
    ctx.backend.call<number[]>("tts.synthesize", {
      text: trimmed,
      lang: settings.lang,
      voice: settings.voice,
      speed: settings.speed,
      steps: settings.steps,
    }),
    SYNTH_TIMEOUT_MS,
    "tts.synthesize",
  );

  const buf = new Uint8Array(wavBytes).buffer;
  return decodeWithRebuild(buf);
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
  // Silence accounting: lets the all-done path log a prominent summary when
  // an utterance ended without playing anything, instead of the pet just
  // going quiet with nothing but per-chunk console noise to go on.
  private failedChunks = 0;
  private playedBuffers = 0;
  // The stream's first buffer starts on a fresh graph (per-utterance
  // rebuild); later chunks must not swap contexts mid-reply when the queue
  // briefly drains between chunks.
  private firstPlayback = true;
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

  /** Prominent log when an utterance finished without playing any buffer —
   *  "the pet just went quiet" becomes diagnosable from the log. */
  private summarizeIfSilent(): void {
    if (this.playedBuffers > 0) return;
    logAudio(
      `reply finished with NO audio (0 buffers played, ${this.failedChunks} chunk(s) failed synthesis)`,
    );
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
        this.failedChunks++;
        console.error("TTS chunk error:", e);
        if (this.flushed && this.queue.length === 0 && !this.buffer) {
          this.summarizeIfSilent();
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
        this.summarizeIfSilent();
        this.setSpeaking(false);
        releaseSttWithGrace();
        this.onAllDone();
      }
      return;
    }

    this.playing = true;
    this.setSpeaking(true);
    // Same per-play volume application as speak()/speakLoop(): a preview with
    // an override must not leave its volume on the master gain for streams.
    setTtsVolume(this.settings.volume ?? 1.0);
    const buf = this.queue.shift()!;
    const freshGraph = this.firstPlayback;
    this.firstPlayback = false;
    playAudioBuffer(buf, this.onViseme, { freshGraph }).then(
      (started) => {
        if (started) this.playedBuffers++;
        this.playNext();
      },
      (e) => {
        // playAudioBuffer resolves on its own internal errors; this
        // rejection handler is pure insurance so one failure can never
        // wedge the queue or leave the mic suppressed.
        console.error("playAudioBuffer rejected:", e);
        this.onViseme("neutral");
        releaseSttWithGrace();
        this.playNext();
      },
    );
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
    await playAudioBuffer(buf, onViseme, { freshGraph: true });
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
  let firstPass = true;
  try {
    // haltPlayback() ends the current buffer via onended, which resolves the
    // awaited promise; the generation check then breaks the loop.
    while (myGen === speakGeneration) {
      await playAudioBuffer(buf, onViseme, { freshGraph: firstPass });
      firstPass = false;
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
  // Seed the gain from the store and keep it live: the settings window only
  // saves — playback happens on this window's graph, so a slider drag should
  // reach it here even mid-utterance. Inert in windows that never play.
  setTtsVolume(getTtsSettings().volume ?? 1.0);
  subscribeTtsSettings(() => setTtsVolume(getTtsSettings().volume ?? 1.0));
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
