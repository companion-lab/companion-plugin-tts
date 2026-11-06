//! TTS feature plugin backend for companion-tauri.
//!
//! Port of the host app's on-device speech engines into plugin backends:
//! `tts` ← `app/src-tauri/src/tts.rs` (Supertonic TTS, ONNX). The STT side
//! lives in its own `stt` plugin (`features/stt/backend`); this crate is
//! synthesis-only.
//!
//! Wire protocol — every former tauri command maps 1:1 onto a method with the
//! same JSON args and result shapes (snake_case payload fields preserved, so
//! the frontend is drop-in):
//!
//! | former tauri command / host internal      | method                    |
//! |-------------------------------------------|---------------------------|
//! | `synthesize`                               | `tts.synthesize`          |
//! | `tts_voices`                               | `tts.voices`              |
//! | `tts_languages`                            | `tts.languages`           |
//! | eager load driven by perf.rs               | `tts.load_engine`         |
//! | perf.rs `release_model_memory` (TTS half)  | `tts.release_memory`      |
//! | perf.rs `engine_status` (tts_loaded)       | `tts.engine_status`       |
//!
//! Concurrency: the ported source is fully synchronous (no async fns, no
//! tokio), so `handle` runs directly on the calling thread.
//!
//! Assets: `init` resolves the assets root (containing `supertonic/` with the
//! ONNX models and voice styles) and stores it; the loader resolves models
//! lazily, so the plugin loads fine before any model has been downloaded.

use companion_plugin_sdk::{export_plugin, Plugin, PluginCtx};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

mod tts;

/// Environment override for the assets root (checked first).
const ENV_TTS_ASSETS: &str = "COMPANION_TTS_ASSETS";

/// Lexically resolve `..` components so returned dev candidates are clean
/// absolute paths (the dev candidates are built by joining `..` segments).
fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::ParentDir => {
                out.pop();
            }
            std::path::Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Resolve the assets root, in order:
/// 1. `COMPANION_TTS_ASSETS` env var, when set (trimmed non-empty);
/// 2. `{plugin_dir()}/assets`, i.e. assets bundled with the plugin package
///    (preferred — the plugin is self-contained; models ship inside
///    `features/tts/assets/`);
/// 3. the dev checkout fallback: `{CARGO_MANIFEST_DIR}/../../assets`;
/// 4. fallback to the dev-checkout path without an existence check (so lazy
///    load errors name a sensible location), or the last-resort relative
///    `assets` when even the manifest dir is unknown.
fn resolve_assets_dir(
    env_value: Option<&str>,
    manifest_dir: Option<&Path>,
    plugin_dir: Option<&Path>,
) -> PathBuf {
    // 1. Explicit override.
    if let Some(value) = env_value.map(str::trim).filter(|v| !v.is_empty()) {
        return PathBuf::from(value);
    }

    // 2. Assets bundled with the plugin package (preferred — the plugin is
    //    self-contained).
    if let Some(dir) = plugin_dir {
        let candidate = dir.join("assets");
        if candidate.exists() {
            return candidate;
        }
    }

    // 3. Dev-checkout candidate (existence-checked) — fallback for builds
    //    that haven't staged assets into the package yet.
    if let Some(manifest) = manifest_dir {
        let candidate = normalize(&manifest.join("../../assets"));
        if candidate.exists() {
            return candidate;
        }
        return candidate;
    }

    // 4. Fallback.
    PathBuf::from("assets")
}

// ─── Payload extraction (tauri's named-argument binding, over JSON) ───

fn req_str<'a>(payload: &'a Value, key: &str) -> Result<&'a str, String> {
    payload
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing or invalid argument: {key}"))
}

fn req_f32(payload: &Value, key: &str) -> Result<f32, String> {
    payload
        .get(key)
        .and_then(Value::as_f64)
        .map(|v| v as f32)
        .ok_or_else(|| format!("missing or invalid argument: {key}"))
}

fn req_usize(payload: &Value, key: &str) -> Result<usize, String> {
    payload
        .get(key)
        .and_then(Value::as_u64)
        .map(|v| v as usize)
        .ok_or_else(|| format!("missing or invalid argument: {key}"))
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

pub struct TtsPlugin {
    ctx: Option<PluginCtx>,
    tts: tts::TtsState,
    /// Intra-op threads for the TTS engine; perf.rs supplies this from
    /// performance.json in the host, so here it is a settable value
    /// (0 = auto). Read by `tts.synthesize` / `tts.load_engine`.
    tts_threads: AtomicUsize,
}

impl Default for TtsPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl TtsPlugin {
    pub fn new() -> Self {
        TtsPlugin {
            ctx: None,
            tts: tts::init_state(PathBuf::new()),
            tts_threads: AtomicUsize::new(0),
        }
    }

    /// Wire the resolved assets root into the engine, mirroring the host's
    /// `tts::init_state(resolve_asset_dir("supertonic"))`.
    fn apply_assets_root(&mut self, root: &Path) {
        self.tts = tts::init_state(root.join("supertonic"));
    }

    fn route(&self, method: &str, payload: Value) -> Result<Value, String> {
        match method {
            "tts.synthesize" => {
                let text = req_str(&payload, "text")?;
                let lang = req_str(&payload, "lang")?;
                let voice = req_str(&payload, "voice")?;
                let speed = req_f32(&payload, "speed")?;
                let steps = req_usize(&payload, "steps")?;
                let intra = self.tts_threads.load(Ordering::Relaxed);
                let wav = tts::synthesize(&self.tts, intra, text, lang, voice, speed, steps)?;
                // Same shape the tauri command returns: a JSON byte array.
                Ok(json!(wav))
            }
            "tts.voices" => Ok(json!(tts::tts_voices())),
            "tts.languages" => Ok(json!(tts::tts_languages())),
            "tts.load_engine" => {
                // perf.rs drives eager loading with perf_config.tts_threads;
                // the optional payload field plays that role here.
                if let Some(threads) = payload.get("tts_threads").and_then(Value::as_u64) {
                    self.tts_threads.store(threads as usize, Ordering::Relaxed);
                }
                let intra = self.tts_threads.load(Ordering::Relaxed);
                tts::ensure_loaded(&self.tts, intra).map_err(|e| e.to_string())?;
                Ok(json!({ "ok": true }))
            }
            "tts.release_memory" => {
                // perf.rs `release_model_memory`, TTS half only (the stt
                // plugin owns its engine's unload).
                tts::unload_engine(&self.tts);
                Ok(json!({ "ok": true }))
            }
            "tts.engine_status" => Ok(json!({
                "tts_loaded": tts::is_engine_loaded(&self.tts),
            })),
            other => Err(format!("unknown method: {other}")),
        }
    }
}

impl Plugin for TtsPlugin {
    fn init(&mut self, ctx: PluginCtx) -> Result<(), String> {
        let env_value = std::env::var(ENV_TTS_ASSETS).ok();
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let assets_root = resolve_assets_dir(
            env_value.as_deref(),
            Some(&manifest_dir),
            ctx.plugin_dir().as_deref(),
        );
        ctx.info(&format!("tts assets dir: {}", assets_root.display()));
        self.apply_assets_root(&assets_root);
        self.ctx = Some(ctx);
        Ok(())
    }

    fn handle(&mut self, method: &str, payload: Value) -> Result<Value, String> {
        self.route(method, payload)
    }

    fn shutdown(&mut self) {
        // Mirror the host's teardown (dropping managed state): release the
        // engine. Best-effort, like the Drop impl in the ported module.
        tts::unload_engine(&self.tts);
        if let Some(ctx) = &self.ctx {
            ctx.info("tts backend shut down");
        }
    }
}

export_plugin!(TtsPlugin::new());

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs;

    fn temp_root(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "tts-plugin-test-{tag}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Repo-shaped fixture: `root/l0/l1` is the (fake) CARGO_MANIFEST_DIR so
    /// `{manifest}/../../assets` resolves to `root/assets`.
    fn fixture(tag: &str, with_assets: bool) -> (PathBuf, PathBuf) {
        let root = temp_root(tag);
        let mut manifest = root.clone();
        for i in 0..2 {
            manifest = manifest.join(format!("l{i}"));
        }
        fs::create_dir_all(&manifest).unwrap();
        if with_assets {
            fs::create_dir_all(root.join("assets")).unwrap();
        }
        (root, manifest)
    }

    #[test]
    fn env_var_wins_over_everything() {
        let (root, manifest) = fixture("env", true);
        let resolved = resolve_assets_dir(Some("/custom/tts-assets"), Some(&manifest), None);
        assert_eq!(resolved, PathBuf::from("/custom/tts-assets"));
        assert!(root.exists(), "fixture sanity");
    }

    #[test]
    fn dev_checkout_assets_win_when_no_plugin_dir() {
        let (root, manifest) = fixture("dev-win", true);
        let resolved = resolve_assets_dir(None, Some(&manifest), None);
        assert_eq!(resolved, root.join("assets"));
    }

    #[test]
    fn falls_back_to_dev_path_without_existence_check() {
        let (_root, manifest) = fixture("fallback", false);
        let resolved = resolve_assets_dir(None, Some(&manifest), None);
        assert!(
            resolved.ends_with("assets"),
            "dev candidate returned even when it does not exist, got {resolved:?}"
        );
    }

    /// The resolved root must feed the engine exactly like the host setup:
    /// TTS weights at `<root>/supertonic`.
    #[test]
    fn apply_assets_root_wires_engine() {
        let mut plugin = TtsPlugin::new();
        plugin.apply_assets_root(Path::new("/assets/root"));
        assert_eq!(
            plugin.tts.assets_dir,
            PathBuf::from("/assets/root/supertonic")
        );
    }

    /// Every wire method routes (1:1 command mapping).
    #[test]
    fn all_wire_methods_route() {
        let mut plugin = TtsPlugin::new();
        let cases: Vec<(&str, Value)> = vec![
            ("tts.synthesize", json!({ "text": "hi", "lang": "en", "voice": "M1", "speed": 1.0, "steps": 4 })),
            ("tts.voices", json!({})),
            ("tts.languages", json!({})),
            ("tts.load_engine", json!({})),
            ("tts.release_memory", json!({})),
            ("tts.engine_status", json!({})),
        ];
        for (method, payload) in cases {
            let result = plugin.handle(method, payload);
            assert!(
                !matches!(&result, Err(e) if e.contains("unknown method")),
                "{method} must be routed, got {result:?}"
            );
        }
        assert!(plugin.handle("bogus.method", json!({})).unwrap_err().contains("unknown method"));
    }

    #[test]
    fn tts_voices_and_languages_shapes() {
        let mut plugin = TtsPlugin::new();
        let voices = plugin.handle("tts.voices", json!({})).unwrap();
        assert_eq!(voices.as_array().unwrap().len(), 12);
        assert_eq!(voices[0], "M1");
        let langs = plugin.handle("tts.languages", json!({})).unwrap();
        assert_eq!(langs.as_array().unwrap().len(), tts::AVAILABLE_LANGS.len());
        assert_eq!(langs[0], "ar");
    }

    #[test]
    fn tts_synthesize_validates_arguments() {
        let mut plugin = TtsPlugin::new();
        let err = plugin.handle("tts.synthesize", json!({})).unwrap_err();
        assert!(err.contains("text"), "{err}");
        let err = plugin
            .handle("tts.synthesize", json!({ "text": "hi", "lang": "en", "voice": "M1" }))
            .unwrap_err();
        assert!(err.contains("speed"), "{err}");
    }

    #[test]
    fn engine_status_and_release_memory_roundtrip() {
        let mut plugin = TtsPlugin::new();
        let status = plugin.handle("tts.engine_status", json!({})).unwrap();
        assert_eq!(status["tts_loaded"], false);
        plugin.handle("tts.release_memory", json!({})).unwrap();
        let status = plugin.handle("tts.engine_status", json!({})).unwrap();
        assert_eq!(status["tts_loaded"], false);
    }

    #[test]
    fn shutdown_is_safe_when_idle() {
        let mut plugin = TtsPlugin::new();
        plugin.shutdown();
        let status = plugin.handle("tts.engine_status", json!({})).unwrap();
        assert_eq!(status["tts_loaded"], false);
    }
}
