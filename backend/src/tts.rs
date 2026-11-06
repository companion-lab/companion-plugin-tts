//! Port of `app/src-tauri/src/tts.rs` (Supertonic TTS, ONNX Runtime).
//!
//! Engine code (config/voice-style loading, text processing, chunking,
//! inference, WAV encoding) is verbatim. The tauri command layer is replaced
//! by plain functions the plugin router calls:
//! - `tts_voices`      → method `tts.voices`
//! - `tts_languages`   → method `tts.languages`
//! - `synthesize`      → method `tts.synthesize` (same args, same WAV-byte
//!   array result)
//! - `ensure_loaded` / `unload_engine` / `is_engine_loaded` → the engine
//!   lifecycle perf.rs drives (`tts.load_engine`, `tts.release_memory`,
//!   `tts.engine_status`). The perf-state read of `tts_threads` becomes an
//!   argument supplied by the plugin.

use std::collections::HashMap;
use std::fs;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};

use anyhow::{bail, Context, Result};
use ndarray::{Array, Array3};
use ort::session::Session;
use ort::value::Value;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use regex::Regex;
use serde::{Deserialize, Serialize};
use unicode_normalization::UnicodeNormalization;

pub const AVAILABLE_LANGS: &[&str] = &[
    "ar", "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hi", "hr", "hu", "id",
    "it", "ja", "ko", "lt", "lv", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk",
    "vi", "na",
];

// --- Config ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub ae: AEConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AEConfig {
    pub sample_rate: i32,
    pub base_chunk_size: i32,
}

fn load_cfgs(onnx_dir: &Path) -> Result<Config> {
    let cfg_path = onnx_dir.join("tts.json");
    let file = fs::File::open(&cfg_path)
        .with_context(|| format!("failed to open {}", cfg_path.display()))?;
    let reader = BufReader::new(file);
    let cfgs: Config = serde_json::from_reader(reader)?;
    Ok(cfgs)
}

// --- Voice Style ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceStyleData {
    pub style_ttl: StyleComponent,
    pub style_dp: StyleComponent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleComponent {
    pub data: Vec<Vec<Vec<f32>>>,
    pub dims: Vec<usize>,
}

pub struct Style {
    pub ttl: Array3<f32>,
    pub dp: Array3<f32>,
}

fn load_voice_style(path: &Path) -> Result<Style> {
    let file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let reader = BufReader::new(file);
    let data: VoiceStyleData = serde_json::from_reader(reader)?;

    let ttl_dims = &data.style_ttl.dims;
    let dp_dims = &data.style_dp.dims;

    let ttl_dim1 = ttl_dims[1];
    let ttl_dim2 = ttl_dims[2];
    let dp_dim1 = dp_dims[1];
    let dp_dim2 = dp_dims[2];

    let mut ttl_flat = vec![0.0f32; ttl_dim1 * ttl_dim2];
    let mut idx = 0;
    for batch in &data.style_ttl.data {
        for row in batch {
            for &val in row {
                ttl_flat[idx] = val;
                idx += 1;
            }
        }
    }

    let mut dp_flat = vec![0.0f32; dp_dim1 * dp_dim2];
    idx = 0;
    for batch in &data.style_dp.data {
        for row in batch {
            for &val in row {
                dp_flat[idx] = val;
                idx += 1;
            }
        }
    }

    Ok(Style {
        ttl: Array3::from_shape_vec((1, ttl_dim1, ttl_dim2), ttl_flat)?,
        dp: Array3::from_shape_vec((1, dp_dim1, dp_dim2), dp_flat)?,
    })
}

// --- Unicode Text Processor ---

struct UnicodeProcessor {
    indexer: Vec<i64>,
}

impl UnicodeProcessor {
    fn new(path: &Path) -> Result<Self> {
        let file = fs::File::open(path)?;
        let reader = BufReader::new(file);
        let indexer: Vec<i64> = serde_json::from_reader(reader)?;
        Ok(UnicodeProcessor { indexer })
    }

    fn call(&self, text_list: &[String], lang_list: &[String]) -> Result<(Vec<Vec<i64>>, Array3<f32>)> {
        let mut processed_texts = Vec::new();
        for (text, lang) in text_list.iter().zip(lang_list.iter()) {
            processed_texts.push(preprocess_text(text, lang)?);
        }

        let text_ids_lengths: Vec<usize> = processed_texts.iter().map(|t| t.chars().count()).collect();
        let max_len = *text_ids_lengths.iter().max().unwrap_or(&0);

        let mut text_ids = Vec::new();
        for text in &processed_texts {
            let mut row = vec![0i64; max_len];
            for (j, ch) in text.chars().enumerate() {
                let val = ch as usize;
                if val < self.indexer.len() {
                    row[j] = self.indexer[val];
                }
            }
            text_ids.push(row);
        }

        let text_mask = get_text_mask(&text_ids_lengths);
        Ok((text_ids, text_mask))
    }
}

fn preprocess_text(text: &str, lang: &str) -> Result<String> {
    if !AVAILABLE_LANGS.contains(&lang) {
        bail!("Invalid language: {lang}");
    }

    static EMOJI_RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"[\x{1F600}-\x{1F64F}\x{1F300}-\x{1F5FF}\x{1F680}-\x{1F6FF}\x{1F700}-\x{1F77F}\x{1F780}-\x{1F7FF}\x{1F800}-\x{1F8FF}\x{1F900}-\x{1F9FF}\x{1FA00}-\x{1FA6F}\x{1FA70}-\x{1FAFF}\x{2600}-\x{26FF}\x{2700}-\x{27BF}\x{1F1E6}-\x{1F1FF}]+").unwrap()
    });
    static SPACE_COMMA_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r" ,").unwrap());
    static SPACE_DOT_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r" \.").unwrap());
    static SPACE_BANG_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r" !").unwrap());
    static SPACE_Q_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r" \?").unwrap());
    static SPACE_SEMI_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r" ;").unwrap());
    static SPACE_COLON_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r" :").unwrap());
    static SPACE_QUOTE_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r" '").unwrap());
    static WS_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\s+").unwrap());
    static END_PUNCT_RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r#"[.!?;:,'"\)\]}…。」』】〉》›»]$"#).unwrap()
    });

    let mut text: String = text.nfkd().collect();

    // Remove emojis
    text = EMOJI_RE.replace_all(&text, "").to_string();

    let replacements = [
        ("\u{2013}", "-"),
        ("\u{2011}", "-"),
        ("\u{2014}", "-"),
        ("_", " "),
        ("\u{201C}", "\""),
        ("\u{201D}", "\""),
        ("\u{2018}", "'"),
        ("\u{2019}", "'"),
        ("\u{00B4}", "'"),
        ("\u{0060}", "'"),
        ("[", " "),
        ("]", " "),
        ("|", " "),
        ("/", " "),
        ("#", " "),
        ("\u{2192}", " "),
        ("\u{2190}", " "),
    ];
    for (from, to) in &replacements {
        text = text.replace(from, to);
    }

    for symbol in &["\u{2665}", "\u{2606}", "\u{2661}", "\u{00A9}", "\\"] {
        text = text.replace(symbol, "");
    }

    text = text.replace("@", " at ");
    text = text.replace("e.g.,", "for example, ");
    text = text.replace("i.e.,", "that is, ");

    text = SPACE_COMMA_RE.replace_all(&text, ",").to_string();
    text = SPACE_DOT_RE.replace_all(&text, ".").to_string();
    text = SPACE_BANG_RE.replace_all(&text, "!").to_string();
    text = SPACE_Q_RE.replace_all(&text, "?").to_string();
    text = SPACE_SEMI_RE.replace_all(&text, ";").to_string();
    text = SPACE_COLON_RE.replace_all(&text, ":").to_string();
    text = SPACE_QUOTE_RE.replace_all(&text, "'").to_string();

    while text.contains("\"\"") {
        text = text.replace("\"\"", "\"");
    }
    while text.contains("''") {
        text = text.replace("''", "'");
    }

    text = WS_RE.replace_all(&text, " ").to_string();
    text = text.trim().to_string();

    if !text.is_empty() {
        if !END_PUNCT_RE.is_match(&text) {
            text.push('.');
        }
    }

    text = format!("<{lang}>{text}</{lang}>");
    Ok(text)
}

// --- Mask helpers ---

fn length_to_mask(lengths: &[usize], max_len: usize) -> Array3<f32> {
    let bsz = lengths.len();
    let mut mask = Array3::<f32>::zeros((bsz, 1, max_len));
    for (i, &len) in lengths.iter().enumerate() {
        for j in 0..len.min(max_len) {
            mask[[i, 0, j]] = 1.0;
        }
    }
    mask
}

fn get_text_mask(text_ids_lengths: &[usize]) -> Array3<f32> {
    let max_len = *text_ids_lengths.iter().max().unwrap_or(&0);
    length_to_mask(text_ids_lengths, max_len)
}

fn sample_noisy_latent(
    duration: &[f32],
    sample_rate: i32,
    base_chunk_size: i32,
    chunk_compress: i32,
    latent_dim: i32,
) -> (Array3<f32>, Array3<f32>) {
    let bsz = duration.len();
    let max_dur = duration.iter().cloned().fold(0.0f32, f32::max);
    let wav_len_max = (max_dur * sample_rate as f32) as usize;
    let wav_lengths: Vec<usize> = duration.iter().map(|&d| (d * sample_rate as f32) as usize).collect();

    let chunk_size = (base_chunk_size * chunk_compress) as usize;
    let latent_len = (wav_len_max + chunk_size - 1) / chunk_size;
    let latent_dim_val = (latent_dim * chunk_compress) as usize;

    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut rng = StdRng::from_entropy();

    // Fill the underlying slice directly instead of per-element indexing;
    // samples are i.i.d. so the fill order is irrelevant.
    let mut noisy_latent = Array3::<f32>::zeros((bsz, latent_dim_val, latent_len));
    for x in noisy_latent.as_slice_mut().unwrap().iter_mut() {
        *x = normal.sample(&mut rng);
    }

    let latent_lengths: Vec<usize> = wav_lengths
        .iter()
        .map(|&len| (len + chunk_size - 1) / chunk_size)
        .collect();
    let latent_mask = length_to_mask(&latent_lengths, latent_len);

    for b in 0..bsz {
        for d in 0..latent_dim_val {
            for t in 0..latent_len {
                noisy_latent[[b, d, t]] *= latent_mask[[b, 0, t]];
            }
        }
    }

    (noisy_latent, latent_mask)
}

// --- Text Chunking ---

const ABBREVIATIONS: &[&str] = &[
    "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "Sr.", "Jr.", "St.", "Ave.", "Rd.", "Blvd.", "Dept.",
    "Inc.", "Ltd.", "Co.", "Corp.", "etc.", "vs.", "i.e.", "e.g.", "Ph.D.",
];

fn chunk_text(text: &str, max_len: usize) -> Vec<String> {
    let text = text.trim();
    if text.is_empty() {
        return vec![String::new()];
    }

    static PARA_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\n\s*\n").unwrap());
    let paragraphs: Vec<&str> = PARA_RE.split(text).collect();
    let mut chunks = Vec::new();

    for para in paragraphs {
        let para = para.trim();
        if para.is_empty() {
            continue;
        }
        if para.len() <= max_len {
            chunks.push(para.to_string());
            continue;
        }

        let sentences = split_sentences(para);
        let mut current = String::new();
        let mut current_len = 0;

        for sentence in sentences {
            let sentence = sentence.trim();
            if sentence.is_empty() {
                continue;
            }
            let sentence_len = sentence.len();
            if current_len + sentence_len + 1 > max_len && !current.is_empty() {
                chunks.push(current.trim().to_string());
                current.clear();
                current_len = 0;
            }
            if !current.is_empty() {
                current.push(' ');
                current_len += 1;
            }
            current.push_str(sentence);
            current_len += sentence_len;
        }
        if !current.is_empty() {
            chunks.push(current.trim().to_string());
        }
    }

    if chunks.is_empty() {
        vec![String::new()]
    } else {
        chunks
    }
}

fn split_sentences(text: &str) -> Vec<String> {
    static SENT_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"([.!?])\s+").unwrap());
    let matches: Vec<_> = SENT_RE.find_iter(text).collect();
    if matches.is_empty() {
        return vec![text.to_string()];
    }

    let mut sentences = Vec::new();
    let mut last_end = 0;

    for m in &matches {
        let before_punc = text[last_end..m.start()].trim();
        let punct = &text[m.start()..m.start() + 1];
        let combined = format!("{before_punc}{punct}");
        let is_abbrev = ABBREVIATIONS.iter().any(|a| combined.ends_with(a));

        if !is_abbrev {
            sentences.push(text[last_end..m.end()].to_string());
            last_end = m.end();
        }
    }

    if last_end < text.len() {
        sentences.push(text[last_end..].to_string());
    }

    if sentences.is_empty() {
        vec![text.to_string()]
    } else {
        sentences
    }
}

// --- WAV Encoding ---

fn encode_wav(audio_data: &[f32], sample_rate: i32) -> Vec<u8> {
    let num_channels: u16 = 1;
    let bits_per_sample: u16 = 16;
    let byte_rate = sample_rate as u32 * num_channels as u32 * bits_per_sample as u32 / 8;
    let block_align = num_channels * bits_per_sample / 8;
    let data_size = (audio_data.len() * bits_per_sample as usize / 8) as u32;
    let total_size = 36 + data_size;

    let mut buf = Vec::with_capacity(44 + data_size as usize);

    // RIFF header
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&total_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    // fmt chunk
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&num_channels.to_le_bytes());
    buf.extend_from_slice(&(sample_rate as u32).to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&bits_per_sample.to_le_bytes());
    // data chunk
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());

    buf.resize(44 + data_size as usize, 0);
    for (i, &sample) in audio_data.iter().enumerate() {
        let clamped = sample.max(-1.0).min(1.0);
        let val = (clamped * 32767.0) as i16;
        let bytes = val.to_le_bytes();
        let off = 44 + i * 2;
        buf[off] = bytes[0];
        buf[off + 1] = bytes[1];
    }

    buf
}

// --- TextToSpeech Engine ---

pub struct TextToSpeech {
    cfgs: Config,
    text_processor: UnicodeProcessor,
    dp_session: Session,
    text_enc_session: Session,
    vector_est_session: Session,
    vocoder_session: Session,
}

impl TextToSpeech {
    fn new(
        cfgs: Config,
        text_processor: UnicodeProcessor,
        dp_session: Session,
        text_enc_session: Session,
        vector_est_session: Session,
        vocoder_session: Session,
    ) -> Self {
        TextToSpeech {
            cfgs,
            text_processor,
            dp_session,
            text_enc_session,
            vector_est_session,
            vocoder_session,
        }
    }

    fn infer(
        &mut self,
        text_list: &[String],
        lang_list: &[String],
        style: &Style,
        total_step: usize,
        speed: f32,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let bsz = text_list.len();
        let sample_rate = self.cfgs.ae.sample_rate;
        let base_chunk_size = self.cfgs.ae.base_chunk_size;
        let chunk_compress = 6i32; // from config ttl.chunk_compress_factor
        let latent_dim = 24i32; // from config ttl.latent_dim

        let (text_ids, text_mask) = self.text_processor.call(text_list, lang_list)?;

        let text_ids_shape = (bsz, text_ids[0].len());
        let mut flat = Vec::new();
        for row in &text_ids {
            flat.extend_from_slice(row);
        }
        let text_ids_array = Array::from_shape_vec(text_ids_shape, flat)?;
        let text_ids_value = Value::from_array(text_ids_array)?;
        let text_mask_value = Value::from_array(text_mask.clone())?;
        let style_dp_value = Value::from_array(style.dp.clone())?;

        // Duration prediction
        let dp_outputs = self.dp_session.run(ort::inputs! {
            "text_ids" => &text_ids_value,
            "style_dp" => &style_dp_value,
            "text_mask" => &text_mask_value
        })?;

        let (_, duration_data) = dp_outputs["duration"].try_extract_tensor::<f32>()?;
        let mut duration: Vec<f32> = duration_data.to_vec();
        for dur in duration.iter_mut() {
            *dur /= speed;
        }

        // Text encoding
        let style_ttl_value = Value::from_array(style.ttl.clone())?;
        let text_enc_outputs = self.text_enc_session.run(ort::inputs! {
            "text_ids" => &text_ids_value,
            "style_ttl" => &style_ttl_value,
            "text_mask" => &text_mask_value
        })?;

        let (text_emb_shape, text_emb_data) = text_enc_outputs["text_emb"].try_extract_tensor::<f32>()?;
        let text_emb = Array3::from_shape_vec(
            (text_emb_shape[0] as usize, text_emb_shape[1] as usize, text_emb_shape[2] as usize),
            text_emb_data.to_vec(),
        )?;

        // Sample noisy latent
        let (mut xt, latent_mask) = sample_noisy_latent(
            &duration,
            sample_rate,
            base_chunk_size,
            chunk_compress,
            latent_dim,
        );

        let total_step_array = Array::from_elem(bsz, total_step as f32);
        let total_step_value = Value::from_array(total_step_array)?;

        // Loop-invariant inputs are built once; only xt changes per step.
        let static_text_emb_value = Value::from_array(text_emb)?;
        let static_latent_mask_value = Value::from_array(latent_mask)?;
        let static_text_mask_value = Value::from_array(text_mask)?;

        // Denoising loop
        for step in 0..total_step {
            let current_step_array = Array::from_elem(bsz, step as f32);

            let xt_value = Value::from_array(xt.clone())?;
            let current_step_value = Value::from_array(current_step_array)?;

            let outputs = self.vector_est_session.run(ort::inputs! {
                "noisy_latent" => &xt_value,
                "text_emb" => &static_text_emb_value,
                "style_ttl" => &style_ttl_value,
                "latent_mask" => &static_latent_mask_value,
                "text_mask" => &static_text_mask_value,
                "current_step" => &current_step_value,
                "total_step" => &total_step_value
            })?;

            let (shape, data) = outputs["denoised_latent"].try_extract_tensor::<f32>()?;
            xt = Array3::from_shape_vec(
                (shape[0] as usize, shape[1] as usize, shape[2] as usize),
                data.to_vec(),
            )?;
        }

        // Vocoder
        let final_latent_value = Value::from_array(xt)?;
        let vocoder_outputs = self.vocoder_session.run(ort::inputs! {
            "latent" => &final_latent_value
        })?;

        let (_, wav_data) = vocoder_outputs["wav_tts"].try_extract_tensor::<f32>()?;
        Ok((wav_data.to_vec(), duration))
    }

    fn synthesize(
        &mut self,
        text: &str,
        lang: &str,
        style: &Style,
        total_step: usize,
        speed: f32,
    ) -> Result<(Vec<f32>, f32)> {
        let sample_rate = self.cfgs.ae.sample_rate;
        let max_len = if lang == "ko" || lang == "ja" { 120 } else { 300 };
        let chunks = chunk_text(text, max_len);

        let mut wav_cat: Vec<f32> = Vec::new();
        let mut dur_cat: f32 = 0.0;

        for (i, chunk) in chunks.iter().enumerate() {
            let (wav, duration) = self.infer(&[chunk.clone()], &[lang.to_string()], style, total_step, speed)?;
            let dur = duration[0];
            let wav_len = (sample_rate as f32 * dur) as usize;
            let wav_chunk = &wav[..wav_len.min(wav.len())];

            if i == 0 {
                wav_cat.extend_from_slice(wav_chunk);
                dur_cat = dur;
            } else {
                let silence_len = (0.3 * sample_rate as f32) as usize;
                wav_cat.extend(&vec![0.0f32; silence_len]);
                wav_cat.extend_from_slice(wav_chunk);
                dur_cat += 0.3 + dur;
            }
        }

        Ok((wav_cat, dur_cat))
    }
}

// --- Global Engine State ---

pub struct TtsState {
    engine: Mutex<Option<TtsEngine>>,
    pub(crate) assets_dir: PathBuf,
}

struct TtsEngine {
    tts: TextToSpeech,
    styles: HashMap<String, Style>,
}

pub fn init_state(assets_dir: PathBuf) -> TtsState {
    TtsState {
        engine: Mutex::new(None),
        assets_dir,
    }
}

/// Load the engine on demand. The engine lock is held for the whole load,
/// which serializes concurrent first-callers and any synthesizer.
pub fn ensure_loaded(state: &TtsState, intra_threads: usize) -> Result<()> {
    let mut engine_lock = state.engine.lock().unwrap();
    if engine_lock.is_some() {
        return Ok(());
    }
    let engine = build_engine(&state.assets_dir, intra_threads)?;
    *engine_lock = Some(engine);
    Ok(())
}

pub fn unload_engine(state: &TtsState) {
    let mut engine_lock = state.engine.lock().unwrap();
    *engine_lock = None;
}

pub fn is_engine_loaded(state: &TtsState) -> bool {
    state.engine.lock().map(|e| e.is_some()).unwrap_or(false)
}

fn build_engine(assets_dir: &Path, tts_threads: usize) -> Result<TtsEngine> {
    let onnx_dir = assets_dir.join("onnx");

    let cfgs = load_cfgs(&onnx_dir)?;

    let dp_path = onnx_dir.join("duration_predictor.onnx");
    let text_enc_path = onnx_dir.join("text_encoder.onnx");
    let vector_est_path = onnx_dir.join("vector_estimator.onnx");
    let vocoder_path = onnx_dir.join("vocoder.onnx");

    // ponytail: fixed intra/inter split; upgrade to per-model tuning if profiles diverge
    let cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let intra = if tts_threads == 0 {
        let auto = cpus.clamp(1, 8);
        if cpus > 4 { auto - 1 } else { auto }
    } else {
        tts_threads.clamp(1, 32)
    };

    // ort BuilderResult carries the builder (raw ptr, !Send) so `?` into anyhow fails;
    // stringify instead.
    let build_session = |path: &std::path::Path| -> Result<Session> {
        Session::builder()
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_intra_threads(intra)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_inter_threads(1)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .commit_from_file(path)
            .map_err(|e| anyhow::anyhow!("{e}"))
    };

    let dp_session = build_session(&dp_path)?;
    let text_enc_session = build_session(&text_enc_path)?;
    let vector_est_session = build_session(&vector_est_path)?;
    let vocoder_session = build_session(&vocoder_path)?;

    let unicode_indexer_path = onnx_dir.join("unicode_indexer.json");
    let text_processor = UnicodeProcessor::new(&unicode_indexer_path)?;

    let tts = TextToSpeech::new(
        cfgs,
        text_processor,
        dp_session,
        text_enc_session,
        vector_est_session,
        vocoder_session,
    );

    let voice_dir = assets_dir.join("voice_styles");
    let mut styles = HashMap::new();
    if let Ok(entries) = fs::read_dir(&voice_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                if let Ok(style) = load_voice_style(&path) {
                    styles.insert(name.to_string(), style);
                }
            }
        }
    }

    println!("TTS engine loaded with {} voices (cpus={cpus}, intra={intra})", styles.len());

    Ok(TtsEngine { tts, styles })
}

// --- Command layer (former tauri commands, see lib.rs for the wire methods) ---

pub fn tts_voices() -> Vec<String> {
    vec![
        "M1".into(),
        "M2".into(),
        "M3".into(),
        "M4".into(),
        "M5".into(),
        "F1".into(),
        "F2".into(),
        "F3".into(),
        "F4".into(),
        "F5".into(),
        "F6".into(),
        "F7".into(),
    ]
}

pub fn tts_languages() -> Vec<String> {
    AVAILABLE_LANGS.iter().map(|s| s.to_string()).collect()
}

/// Former `synthesize` command body. `intra_threads` replaces the read of
/// `perf.config.tts_threads` (0 = auto).
pub fn synthesize(
    state: &TtsState,
    intra_threads: usize,
    text: &str,
    lang: &str,
    voice: &str,
    speed: f32,
    steps: usize,
) -> Result<Vec<u8>, String> {
    ensure_loaded(state, intra_threads).map_err(|e| e.to_string())?;
    let mut engine_lock = state.engine.lock().map_err(|e| e.to_string())?;
    let engine = engine_lock.as_mut().ok_or("TTS engine not loaded")?;

    let style = engine.styles.get(voice).ok_or_else(|| {
        let available: Vec<_> = engine.styles.keys().cloned().collect();
        format!("Voice '{voice}' not found. Available: {available:?}")
    })?;

    let (wav, _duration) = engine
        .tts
        .synthesize(text, lang, style, steps, speed)
        .map_err(|e| e.to_string())?;

    Ok(encode_wav(&wav, engine.tts.cfgs.ae.sample_rate))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preprocess_wraps_and_punctuates() {
        assert_eq!(preprocess_text("Hello   world", "en").unwrap(), "<en>Hello world.</en>");
        // Existing terminal punctuation is not doubled.
        assert_eq!(preprocess_text("Hi there!", "en").unwrap(), "<en>Hi there!</en>");
        assert!(preprocess_text("Hi", "xx").is_err(), "unknown language rejected");
    }

    #[test]
    fn chunk_text_respects_max_len_and_keeps_short_text() {
        assert_eq!(chunk_text("  ", 10), vec![String::new()]);
        assert_eq!(chunk_text("short", 10), vec!["short"]);
        // Punctuated text splits at sentence boundaries under max_len.
        let text = "One two three. Four five six. Seven eight nine. Ten eleven twelve.";
        let chunks = chunk_text(text, 20);
        assert!(chunks.len() > 1, "chunks: {chunks:?}");
        assert!(chunks.iter().all(|c| c.len() <= 20), "chunks: {chunks:?}");
        // Rejoining the chunks (with spaces) preserves the words.
        assert_eq!(chunks.join(" "), text);
        // Host behavior: a single sentence with no punctuation is never
        // split mid-sentence and may exceed max_len.
        let unpunctuated = "one two three four five six seven eight nine ten";
        assert_eq!(chunk_text(unpunctuated, 12), vec![unpunctuated]);
    }

    #[test]
    fn split_sentences_keeps_abbreviations_together() {
        let sentences = split_sentences("Dr. Smith went to Washington. He arrived.");
        assert_eq!(sentences.len(), 2);
        assert!(sentences[0].starts_with("Dr. Smith"));
        assert_eq!(sentences[1], "He arrived.");
    }

    #[test]
    fn encode_wav_produces_valid_header_and_size() {
        let samples = vec![0.0f32, 0.5, -0.5, 1.0, -1.0, 0.25];
        let wav = encode_wav(&samples, 22050);
        assert_eq!(wav.len(), 44 + samples.len() * 2);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[12..16], b"fmt ");
        assert_eq!(&wav[36..40], b"data");
        let data_size = u32::from_le_bytes(wav[40..44].try_into().unwrap());
        assert_eq!(data_size, (samples.len() * 2) as u32);
        let sample_rate = u32::from_le_bytes(wav[24..28].try_into().unwrap());
        assert_eq!(sample_rate, 22050);
        // First sample is 0.0 → two zero bytes.
        assert_eq!(&wav[44..46], &[0, 0]);
    }
}
