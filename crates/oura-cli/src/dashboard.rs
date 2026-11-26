//! `oura dashboard` — a local web health dashboard following
//! `notes/dashboard-v2-brainstorm.md`.
//!
//! Rust owns the DB + every non-model calculation (per-night HRV/RHR/skin-temp,
//! SpO2 % via Oura's calibration, device/data-health, baselines + deltas, the
//! digest) and the HTTP server. The torch models (sleep hypnogram, activity, CVA)
//! run through the Python runners, shelled out exactly like `oura sessions`. All
//! data stays on this machine.

use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::SystemTime;

use anyhow::{anyhow, Context, Result};
use axum::{
    body::{Body, Bytes},
    extract::{DefaultBodyLimit, Path as AxumPath, RawQuery, Request, State},
    http::{header, StatusCode},
    middleware::{self, Next},
    response::Response,
    routing::{get, post},
    Router,
};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncReadExt};
use tokio::net::TcpListener;

use oura_store::Store;

use crate::ingest::{DeviceChunk, EventChunk, EventRow, ReadingChunk, ReadingRow};

use oura_summary::{
    bedtime_overrides_path, feature_modes_path, profile_path, write_bedtime_override, ModelInputs,
    ModelOutputs, ModelRunner,
};
// Re-exported so `dashboard::Demographics` (main.rs) and the profile/feature
// handlers keep working after the summary logic moved into `oura-summary`.
pub use oura_summary::{read_profile, write_feature_mode, write_profile, Demographics};

const INDEX_HTML: &str = include_str!("../../../dashboard/web/index.html");
const STYLES_CSS: &str = include_str!("../../../dashboard/web/styles.css");
const APP_JS: &str = include_str!("../../../dashboard/web/app.js");
const DNA_HTML: &str = include_str!("../../../dashboard/web/dna.html");
const DNA_JS: &str = include_str!("../../../dashboard/web/dna.js");
const DNA_CSS: &str = include_str!("../../../dashboard/web/dna.css");
const BLOOD_HTML: &str = include_str!("../../../dashboard/web/blood.html");
const BLOOD_JS: &str = include_str!("../../../dashboard/web/blood.js");
const BLOOD_CSS: &str = include_str!("../../../dashboard/web/blood.css");

fn read_ring_key(path: Option<&Path>) -> Result<String> {
    let path = path.ok_or_else(|| anyhow!("dashboard was started without --key-file"))?;
    let key = std::fs::read_to_string(path)
        .with_context(|| format!("reading key file {}", path.display()))?;
    validate_ring_key(&key)
}

fn write_ring_key(path: Option<&Path>, key: &str) -> Result<String> {
    let path = path.ok_or_else(|| anyhow!("dashboard was started without --key-file"))?;
    let key = validate_ring_key(key)?;
    std::fs::write(path, format!("{key}\n"))
        .with_context(|| format!("writing key file {}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
    }
    Ok(key)
}

fn validate_ring_key(key: &str) -> Result<String> {
    let key = key.trim();
    if key.len() != 32 || !key.as_bytes().iter().all(u8::is_ascii_hexdigit) {
        return Err(anyhow!("auth key must be exactly 16 bytes of hex"));
    }
    Ok(key.to_ascii_lowercase())
}

// ── python model orchestration (shared with `oura sessions` via `pyrunner`) ────
fn repo_root() -> Option<PathBuf> {
    crate::pyrunner::repo_root(Path::new("tools/run_activity_model.py"))
}

fn python_bin(root: &Path) -> PathBuf {
    crate::pyrunner::venv_python(root)
}

/// Bound model subprocesses and drain stdout concurrently (large results must not
/// fill the pipe while the parent waits). A stalled runner is killed and reaped.
fn model_output(
    mut command: Command,
    input: Option<&[u8]>,
    timeout: std::time::Duration,
) -> Option<Value> {
    use std::io::{Read, Write};
    command
        .stdin(if input.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    let mut child = command.spawn().ok()?;
    let stdout = child.stdout.take()?;
    let reader = std::thread::spawn(move || {
        let mut bytes = Vec::new();
        stdout
            .take(32 * 1024 * 1024)
            .read_to_end(&mut bytes)
            .map(|_| bytes)
    });
    if let Some(input) = input {
        if child
            .stdin
            .take()
            .and_then(|mut pipe| pipe.write_all(input).ok())
            .is_none()
        {
            let _ = child.kill();
            let _ = child.wait();
            let _ = reader.join();
            return None;
        }
    }
    let started = std::time::Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) if started.elapsed() < timeout => {
                std::thread::sleep(std::time::Duration::from_millis(20))
            }
            _ => {
                tracing::warn!("model runner timed out or could not be monitored");
                let _ = child.kill();
                let _ = child.wait();
                break None;
            }
        }
    };
    let bytes = reader.join().ok()?.ok()?;
    if !status?.success() {
        return None;
    }
    serde_json::from_slice(&bytes).ok()
}

fn run_py_json(root: &Path, py: &Path, script: &str, args: &[String]) -> Option<Value> {
    run_py_json_stdin_optional(root, py, script, args, None)
}

fn run_py_json_stdin(
    root: &Path,
    py: &Path,
    script: &str,
    args: &[String],
    input: &[u8],
) -> Option<Value> {
    run_py_json_stdin_optional(root, py, script, args, Some(input))
}

fn run_py_json_stdin_optional(
    root: &Path,
    py: &Path,
    script: &str,
    args: &[String],
    input: Option<&[u8]>,
) -> Option<Value> {
    let mut command = Command::new(py);
    command.current_dir(root).arg(root.join(script)).args(args);
    model_output(command, input, std::time::Duration::from_secs(180))
}

/// The web dashboard's [`ModelRunner`]: shells out to the Python torch runners,
/// exactly as before, and falls back to Rust signal-based sleep staging when the
/// Python environment or private `.pt` model file is unavailable on a user's PC.
struct PythonRunner;
impl ModelRunner for PythonRunner {
    fn run(&self, input: ModelInputs) -> ModelOutputs {
        let ModelInputs {
            db,
            tz,
            demo,
            sleep_ranges,
        } = input;
        let root = repo_root();
        let py = root.as_deref().map(python_bin);
        let sleep_stdin = serde_json::to_vec(sleep_ranges).unwrap_or_default();
        let sleep_args = vec![
            db.display().to_string(),
            tz.to_string(),
            "--json".into(),
            "--batch".into(),
        ];
        let cva_args = vec![
            db.display().to_string(),
            "--json".into(),
            "--sex".into(),
            demo.sex.to_string(),
            "--age".into(),
            demo.age.to_string(),
            "--height".into(),
            demo.height_m.to_string(),
            "--weight".into(),
            demo.weight_kg.to_string(),
            "--ring".into(),
            demo.ring_size.to_string(),
        ];
        let act_args = vec![
            db.display().to_string(),
            "--tz".into(),
            tz.to_string(),
            "--json".into(),
        ];
        let illness_args = vec![db.display().to_string(), tz.to_string(), "--json".into()];

        let (mut sleep_batch, cva, activity, illness) = match (root.as_deref(), py.as_deref()) {
            (Some(r), Some(p)) => std::thread::scope(|s| {
                let sh = s.spawn(|| {
                    run_py_json_stdin(r, p, "tools/run_sleep_model.py", &sleep_args, &sleep_stdin)
                });
                let ch = s.spawn(|| run_py_json(r, p, "tools/run_cva_model.py", &cva_args));
                let ah = s.spawn(|| run_py_json(r, p, "tools/run_activity_model.py", &act_args));
                let ih = s.spawn(|| run_py_json(r, p, "tools/run_illness_model.py", &illness_args));
                (
                    sh.join().ok().flatten(),
                    ch.join().ok().flatten(),
                    ah.join().ok().flatten(),
                    ih.join().ok().flatten(),
                )
            }),
            _ => (None, None, None, None),
        };
        if sleep_batch_needs_fallback(sleep_batch.as_ref(), sleep_ranges.len())
            && !sleep_ranges.is_empty()
        {
            let signals = oura_summary::stage_nights_from_signals(db, sleep_ranges);
            sleep_batch = merge_sleep_fallback(sleep_batch, signals);
        }
        ModelOutputs {
            sleep_batch,
            cva,
            activity,
            illness,
        }
    }
}

/// Whether the Python sleep batch must be topped up from the signal estimator.
///
/// `run_sleep_model.py --batch` returns one entry per sleep range, in order: a staged
/// dict, or `null` for a window it couldn't stage (no torch/models, or NSSA inputs
/// missing), still exiting 0. A top-up is needed when the batch is missing, the wrong
/// length (the per-range contract was broken), or carries any null night. Kept beside
/// [`merge_sleep_fallback`] so the "when to fall back" and the "what to replace" stay
/// in agreement — this predicate is the line whose all-or-nothing form was the bug.
fn sleep_batch_needs_fallback(batch: Option<&Value>, range_count: usize) -> bool {
    match batch.and_then(|v| v.as_array()) {
        None => true,
        Some(a) => a.len() != range_count || a.iter().any(Value::is_null),
    }
}

/// Fill the Python sleep batch's missing per-night entries with the signal-derived
/// fallback. `signals` is authoritative on night count and order (exactly one entry
/// per sleep range), so the merged result has one entry per range: the Python staging
/// where present, otherwise the signal estimate.
///
/// A Python `null` (window the model couldn't stage) is replaced by its signal
/// estimate — otherwise the night keeps its physiology but shows no sleep at all.
/// Nights Python staged are preserved. A Python entry with no signal counterpart
/// (length mismatch — not expected from the current producer) is dropped, since
/// without a range it cannot be mapped onto a night anyway; a missing Python entry is
/// simply the signal estimate. If signals are unavailable, the Python batch is
/// returned untouched.
fn merge_sleep_fallback(python: Option<Value>, signals: Option<Value>) -> Option<Value> {
    let Some(Value::Array(sig)) = signals else {
        return python;
    };
    let py = match python {
        Some(Value::Array(py)) => py,
        _ => Vec::new(),
    };
    let merged = sig
        .into_iter()
        .enumerate()
        .map(|(i, s)| match py.get(i) {
            Some(p) if !p.is_null() => p.clone(),
            _ => s,
        })
        .collect();
    Some(Value::Array(merged))
}

/// `build_summary` for the web dashboard — runs the models via Python.
fn build_summary(db: &Path, tz: f64) -> Result<Value> {
    oura_summary::build_summary(db, tz, &PythonRunner)
}

// ── summary cache ─────────────────────────────────────────────────────────────
// build_summary spawns torch subprocesses (~seconds); without a cache every page
// load re-pays that. We memoise the last result and reuse it until the inputs
// change — the DB (a sync appends events) or profile.json (an edit changes the CVA
// inputs / weight-based kcal). Both are cheap mtime stats, so a sync or profile
// edit transparently invalidates the cache with no explicit wiring.
struct SummaryCache {
    db: PathBuf,
    tz: f64,
    token: CacheToken, // (db, profile.json, feature_modes.json, bedtime_overrides.json) mtimes
    value: Arc<Value>,
}

type CacheToken = (
    Option<SystemTime>,
    Option<SystemTime>,
    Option<SystemTime>,
    Option<SystemTime>,
    i64,
);

/// mtimes of every input the summary depends on plus the store decoder version — any
/// change rebuilds it. Covers a sync (oura.db), a profile edit (profile.json), a
/// feature toggle (feature_modes.json), a manual bedtime adjustment
/// (bedtime_overrides.json), and historical decode migrations.
fn summary_token(db: &Path) -> CacheToken {
    (
        mtime(db),
        mtime(&profile_path(db)),
        mtime(&feature_modes_path(db)),
        mtime(&bedtime_overrides_path(db)),
        oura_store::storage::DECODER_VERSION,
    )
}

fn summary_cache() -> &'static Mutex<Option<SummaryCache>> {
    static C: OnceLock<Mutex<Option<SummaryCache>>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(None))
}

fn mtime(p: &Path) -> Option<SystemTime> {
    std::fs::metadata(p).and_then(|m| m.modified()).ok()
}

/// Cached `build_summary`: recompute only when oura.db, profile.json, or
/// feature_modes.json changes.
fn cached_summary(db: &Path, tz: f64) -> Result<Arc<Value>> {
    let _ = oura_store::storage::Store::migrate_if_writable(db);
    let token = summary_token(db);
    if let Some(c) = summary_cache().lock().unwrap().as_ref() {
        if c.db == db && c.tz == tz && c.token == token {
            return Ok(c.value.clone());
        }
    }
    // build outside the lock so concurrent first-loads don't serialise on it
    let value = Arc::new(build_summary(db, tz)?);
    // re-stat the inputs: if any changed while we were building (e.g. a sync landed),
    // the result may predate that change — don't cache it, so the next request
    // rebuilds against the new state instead of serving stale data.
    let token_after = summary_token(db);
    if token_after == token {
        let mut guard = summary_cache().lock().unwrap();
        // don't clobber a fresher entry a concurrent build already stored (compare
        // the DB mtime, which a sync advances)
        let newer_exists = guard
            .as_ref()
            .is_some_and(|c| c.db == db && c.tz == tz && c.token.0 > token.0);
        if !newer_exists {
            *guard = Some(SummaryCache {
                db: db.to_path_buf(),
                tz,
                token,
                value: value.clone(),
            });
        }
    }
    Ok(value)
}

/// Everything a request handler needs. Cheap to clone (the one allocation, the
/// allowed-host list, is shared behind an `Arc`), so axum can hand each handler its
/// own copy.
#[derive(Clone)]
struct AppState {
    db: PathBuf,
    tz: f64,
    name: String,
    key_file: Option<PathBuf>,
    allowed_hosts: Arc<Vec<String>>,
    /// Present only when `--ingest-token-file` was given: enables `/api/ingest/*` for
    /// a pushing agent. `None` for a normal local dashboard, so those routes don't
    /// exist and nothing about local behavior changes.
    ingest: Option<IngestConfig>,
}

/// Server-side ingest state: the bearer token to check and the one writable store the
/// ingest handlers share (SQLite is single-writer, so the `Mutex` serializes writes;
/// summary reads still use their own read-only connections under WAL).
#[derive(Clone)]
struct IngestConfig {
    token: Arc<String>,
    store: Arc<Mutex<Store>>,
}

#[allow(clippy::too_many_arguments)]
pub async fn serve(
    bind: IpAddr,
    port: u16,
    db: PathBuf,
    tz: f64,
    name: String,
    key_file: Option<PathBuf>,
    allowed_hosts: Vec<String>,
    ingest_token: Option<String>,
    seed: Demographics,
) -> Result<()> {
    // seed profile.json from the CLI demographics on first run; afterwards the
    // file is the editable source of truth.
    if !profile_path(&db).exists() {
        let _ = write_profile(&db, &seed.to_json());
    }
    // Remote ingest is opt-in: only when a token is configured do we open the store
    // writable and expose /api/ingest/*. A token read from a file may carry a trailing
    // newline — trim it so it matches the Authorization header byte-for-byte.
    let ingest = match ingest_token {
        Some(token) => {
            let token = token.trim().to_string();
            if token.is_empty() {
                return Err(anyhow!("ingest token is empty"));
            }
            let store = Store::open(&db).context("opening store for ingest")?;
            Some(IngestConfig {
                token: Arc::new(token),
                store: Arc::new(Mutex::new(store)),
            })
        }
        None => None,
    };
    // Loopback Host values are always accepted, so the local UI keeps working
    // exactly as before; `--allowed-host` only *adds* names for a proxy/ingress.
    let mut hosts = vec![format!("127.0.0.1:{port}"), format!("localhost:{port}")];
    hosts.extend(allowed_hosts);
    let state = AppState {
        db,
        tz,
        name,
        key_file,
        allowed_hosts: Arc::new(hosts),
        ingest,
    };
    let listener = TcpListener::bind((bind, port))
        .await
        .context("binding port")?;
    println!("open_oura dashboard running — open http://{bind}:{port} in your browser");
    axum::serve(listener, router(state))
        .await
        .context("serving dashboard")?;
    Ok(())
}

/// One BLE operation at a time: `sync`, `feature` and `live-hr` shell out to our own
/// binary, which owns the ring. Held while the subprocess runs.
static BLE_GATE: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

fn router(state: AppState) -> Router {
    // static assets: on-disk under dashboard/web/ when present (editable without a
    // recompile), else the copy embedded at build time — see `asset`.
    macro_rules! file {
        ($name:expr, $embed:expr, $ctype:expr) => {
            get(|| async { asset_resp($name, $embed, $ctype) })
        };
    }
    const HTML: &str = "text/html; charset=utf-8";
    const CSS: &str = "text/css; charset=utf-8";
    const JS: &str = "text/javascript; charset=utf-8";
    let mut app = Router::new()
        .route("/", file!("index.html", INDEX_HTML, HTML))
        .route("/index.html", file!("index.html", INDEX_HTML, HTML))
        .route("/styles.css", file!("styles.css", STYLES_CSS, CSS))
        .route("/app.js", file!("app.js", APP_JS, JS))
        .route("/dna", file!("dna.html", DNA_HTML, HTML))
        .route("/dna.html", file!("dna.html", DNA_HTML, HTML))
        .route("/dna.js", file!("dna.js", DNA_JS, JS))
        .route("/dna.css", file!("dna.css", DNA_CSS, CSS))
        .route("/blood", file!("blood.html", BLOOD_HTML, HTML))
        .route("/blood.html", file!("blood.html", BLOOD_HTML, HTML))
        .route("/blood.js", file!("blood.js", BLOOD_JS, JS))
        .route("/blood.css", file!("blood.css", BLOOD_CSS, CSS))
        .route("/icons/{*path}", get(icons))
        // Liveness/readiness probe: cheap, never touches the DB or the models, and
        // (via `guard`) answers without a valid Host so a kubelet probe using the
        // pod IP still works.
        .route(
            "/healthz",
            get(|| async { resp(StatusCode::OK, "text/plain; charset=utf-8", b"ok".to_vec()) }),
        )
        .route("/api/blood/report", get(api_blood_report))
        .route("/api/blood/import-dir", post(api_blood_import_dir))
        .route("/api/blood/import", post(api_blood_import))
        .route("/api/dna/files", get(api_dna_files))
        .route("/api/dna/report", get(api_dna_report))
        .route("/api/dna/fetch", post(api_dna_fetch))
        .route("/api/dna/lookup", get(api_dna_lookup))
        .route("/api/summary", get(api_summary))
        .route("/api/hourly-hr", get(api_hourly_hr))
        .route("/api/research-export", get(api_research_export))
        .route("/api/research-marker", post(api_research_marker))
        .route("/api/profile", get(api_get_profile).post(api_post_profile))
        .route("/api/bedtime", post(api_bedtime))
        .route("/api/ring-key", get(api_get_ring_key).post(api_post_ring_key))
        .route("/api/sync", post(api_sync))
        .route("/api/live-hr", post(api_live_hr))
        .route("/api/feature", post(api_feature));
    // Remote ingest routes exist only when a token is configured. They carry whole
    // event/reading chunks, so they get a generous body limit (the rest of the app
    // keeps axum's small default). `guard` authenticates them by bearer token.
    if state.ingest.is_some() {
        let ingest = Router::new()
            .route("/api/ingest/state", get(api_ingest_state))
            .route("/api/ingest/events", post(api_ingest_events))
            .route("/api/ingest/readings", post(api_ingest_readings))
            .route("/api/ingest/device", post(api_ingest_device))
            .layer(DefaultBodyLimit::max(32 * 1024 * 1024));
        app = app.merge(ingest);
    }
    app.fallback(|| async { resp(StatusCode::NOT_FOUND, "text/plain", b"not found".to_vec()) })
        .layer(middleware::from_fn_with_state(state.clone(), guard))
        .with_state(state)
}

/// Host + CSRF guard, run before every route (replaces the inline checks the
/// hand-rolled server did at the top of each request).
async fn guard(State(st): State<AppState>, req: Request, next: Next) -> Response {
    let path = req.uri().path().to_string();
    // Health probe must answer without a valid Host (a kubelet probe uses the pod IP).
    if path == "/healthz" {
        return next.run(req).await;
    }
    // Machine-to-machine ingest: authenticated by a constant-time bearer-token check,
    // not the browser Host/CSRF guards (an agent isn't a browser and sets no Host of
    // ours). Routes only exist when ingest is configured; this is defence in depth.
    if path.starts_with("/api/ingest/") {
        let Some(ingest) = st.ingest.as_ref() else {
            return resp(StatusCode::NOT_FOUND, "text/plain", b"not found".to_vec());
        };
        let presented = req
            .headers()
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .unwrap_or("");
        if !ct_eq(presented.as_bytes(), ingest.token.as_bytes()) {
            return resp(StatusCode::UNAUTHORIZED, "text/plain", b"unauthorized".to_vec());
        }
        return next.run(req).await;
    }
    // Loopback / allowed-host guard (DNS-rebind protection).
    let host = req
        .headers()
        .get(header::HOST)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if !st.allowed_hosts.iter().any(|h| h == host) {
        return resp(StatusCode::FORBIDDEN, "text/plain", b"forbidden".to_vec());
    }
    // CSRF guard for mutating/disclosing endpoints: a same-origin fetch can set this
    // header; an <img>/<form> or cross-origin fetch cannot (no CORS preflight is ever
    // approved). GET /api/ring-key is guarded too — it discloses the ring auth key.
    let csrf_ok = req.headers().contains_key("x-oura-dash");
    let needs_csrf = matches!(
        (req.method().as_str(), path.as_str()),
        ("POST", "/api/profile")
            | ("POST", "/api/research-marker")
            | ("POST", "/api/bedtime")
            | ("POST", "/api/sync")
            | ("POST", "/api/live-hr")
            | ("POST", "/api/feature")
            | ("POST", "/api/ring-key")
            | ("GET", "/api/ring-key")
            | ("POST", "/api/dna/fetch")
            | ("POST", "/api/blood/import-dir")
            | ("POST", "/api/blood/import")
    );
    if needs_csrf && !csrf_ok {
        return resp(StatusCode::FORBIDDEN, "text/plain", b"forbidden".to_vec());
    }
    next.run(req).await
}

/// Vendored SVG icons (Phosphor), served from disk under dashboard/web/icons/.
async fn icons(AxumPath(path): AxumPath<String>) -> Response {
    if path.ends_with(".svg") && !path.contains("..") {
        if let Some(root) = repo_root() {
            if let Ok(bytes) = std::fs::read(root.join("dashboard/web/icons").join(&path)) {
                return resp(StatusCode::OK, "image/svg+xml", bytes);
            }
        }
    }
    resp(StatusCode::NOT_FOUND, "text/plain", b"not found".to_vec())
}

/// Serve a web asset: prefer the on-disk file under `dashboard/web/` (so the UI
/// can be edited and refreshed without recompiling) and fall back to the copy
/// embedded at build time.
fn asset(name: &str, embedded: &'static str) -> Vec<u8> {
    if let Some(root) = repo_root() {
        if let Ok(bytes) = std::fs::read(root.join("dashboard/web").join(name)) {
            return bytes;
        }
    }
    embedded.as_bytes().to_vec()
}

/// Look up a `key` in a `a=b&c=d` query string, percent-decoding the value.
fn query_param(query: &str, key: &str) -> Option<String> {
    query.split('&').find_map(|pair| {
        let (k, v) = pair.split_once('=')?;
        (k == key).then(|| percent_decode(v))
    })
}

/// Minimal application/x-www-form-urlencoded decode: `%XX` escapes and `+` → space.
fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' if i + 2 < bytes.len() => {
                let hex = |c: u8| (c as char).to_digit(16);
                match (hex(bytes[i + 1]), hex(bytes[i + 2])) {
                    (Some(h), Some(l)) => {
                        out.push((h * 16 + l) as u8);
                        i += 3;
                    }
                    _ => {
                        out.push(b'%');
                        i += 1;
                    }
                }
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Build a response with the `no-store` cache policy the dashboard always uses.
fn resp(status: StatusCode, ctype: &str, body: Vec<u8>) -> Response {
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, ctype)
        .header(header::CACHE_CONTROL, "no-store")
        .body(Body::from(body))
        .expect("valid response")
}

fn json_resp(v: &Value) -> Response {
    resp(
        StatusCode::OK,
        "application/json",
        serde_json::to_vec(v).unwrap_or_default(),
    )
}

/// A single newline-delimited JSON line as a complete `application/x-ndjson` body —
/// for the live-HR early replies (ring busy, or no auth key) the page consumes the
/// same way it consumes the stream.
fn ndjson_line(v: &Value) -> Response {
    resp(
        StatusCode::OK,
        "application/x-ndjson",
        format!("{v}\n").into_bytes(),
    )
}

fn asset_resp(name: &str, embedded: &'static str, ctype: &str) -> Response {
    resp(StatusCode::OK, ctype, asset(name, embedded))
}

fn busy_json() -> Value {
    json!({
        "ok": false,
        "error": "Ring is busy; wait for the current operation to finish.",
        "message": "Ring is busy; wait for the current operation to finish."
    })
}

/// The browser's requested tz (query `?tz=`) is honoured only when the server wasn't
/// pinned to a fixed offset (`--tz-offset 0`, which `local_tz_offset_hours()` returns
/// in UTC containers). Mirrors the old inline logic.
fn effective_tz(tz: f64, query: &str) -> f64 {
    if tz == 0.0 {
        query_param(query, "tz")
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|h| h.is_finite() && (-24.0..=24.0).contains(h))
            .unwrap_or(tz)
    } else {
        tz
    }
}

/// Run a blocking, JSON-returning closure off the async executor (CPU/IO-heavy work:
/// genome parsing, PDF scraping, EBI fetch) and wrap the result as a JSON response.
async fn blocking_json<F>(f: F) -> Response
where
    F: FnOnce() -> Value + Send + 'static,
{
    match tokio::task::spawn_blocking(f).await {
        Ok(v) => json_resp(&v),
        Err(e) => json_resp(&json!({ "error": e.to_string() })),
    }
}

/// Constant-time byte equality for the ingest bearer token (the length is allowed to
/// leak; the secret's content is not).
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b) {
        diff |= x ^ y;
    }
    diff == 0
}

// ── remote ingest handlers ───────────────────────────────────────────────────────
// Enabled only when `--ingest-token-file` was set (see `router`). `guard` has already
// checked the bearer token. Each writes through the single shared `Store` behind a
// blocking task so SQLite's single writer never blocks the async runtime.

/// Run a blocking closure with the shared ingest store locked. `None` state → 404.
async fn with_ingest_store<F>(st: &AppState, f: F) -> Response
where
    F: FnOnce(&Store) -> (StatusCode, Value) + Send + 'static,
{
    let Some(ingest) = st.ingest.clone() else {
        return resp(StatusCode::NOT_FOUND, "text/plain", b"not found".to_vec());
    };
    let store = ingest.store.clone();
    match tokio::task::spawn_blocking(move || {
        let guard = store.lock().expect("ingest store mutex poisoned");
        f(&guard)
    })
    .await
    {
        Ok((status, body)) => resp(
            status,
            "application/json",
            serde_json::to_vec(&body).unwrap_or_default(),
        ),
        Err(e) => json_resp(&json!({ "ok": false, "message": e.to_string() })),
    }
}

/// `GET /api/ingest/state` — the server's decoder version and per-serial watermarks.
async fn api_ingest_state(State(st): State<AppState>) -> Response {
    with_ingest_store(&st, |store| {
        let build = || -> anyhow::Result<Value> {
            let mut serials = serde_json::Map::new();
            for serial in store.device_serials()? {
                serials.insert(
                    serial.clone(),
                    json!({
                        "events": store.ingest_watermark(&serial, "events")?,
                        "readings": store.ingest_watermark(&serial, "readings")?,
                    }),
                );
            }
            Ok(json!({
                "decoder_version": store.decoder_version()?,
                "serials": serials,
            }))
        };
        match build() {
            Ok(v) => (StatusCode::OK, v),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                json!({ "ok": false, "message": e.to_string() }),
            ),
        }
    })
    .await
}

/// `POST /api/ingest/events` — insert an event chunk and advance the watermark.
async fn api_ingest_events(State(st): State<AppState>, body: Bytes) -> Response {
    let chunk: EventChunk = match serde_json::from_slice(&body) {
        Ok(c) => c,
        Err(e) => return ingest_bad_request(&e.to_string()),
    };
    with_ingest_store(&st, move |store| {
        if let Some(r) = decoder_guard(store, chunk.decoder_version) {
            return r;
        }
        let rows = match chunk
            .rows
            .into_iter()
            .map(EventRow::into_store)
            .collect::<anyhow::Result<Vec<_>>>()
        {
            Ok(r) => r,
            Err(e) => return ingest_error(&e.to_string()),
        };
        match store.ingest_events(&chunk.serial, &rows) {
            Ok((inserted, watermark)) => ingest_ack(inserted, watermark),
            Err(e) => ingest_error(&e.to_string()),
        }
    })
    .await
}

/// `POST /api/ingest/readings` — insert a readings chunk and advance the watermark.
async fn api_ingest_readings(State(st): State<AppState>, body: Bytes) -> Response {
    let chunk: ReadingChunk = match serde_json::from_slice(&body) {
        Ok(c) => c,
        Err(e) => return ingest_bad_request(&e.to_string()),
    };
    with_ingest_store(&st, move |store| {
        if let Some(r) = decoder_guard(store, chunk.decoder_version) {
            return r;
        }
        let rows: Vec<_> = chunk.rows.into_iter().map(ReadingRow::into_store).collect();
        match store.ingest_readings(&chunk.serial, &rows) {
            Ok((inserted, watermark)) => ingest_ack(inserted, watermark),
            Err(e) => ingest_error(&e.to_string()),
        }
    })
    .await
}

/// `POST /api/ingest/device` — device identity, last-sync time, and feature_modes.json.
async fn api_ingest_device(State(st): State<AppState>, body: Bytes) -> Response {
    let chunk: DeviceChunk = match serde_json::from_slice(&body) {
        Ok(c) => c,
        Err(e) => return ingest_bad_request(&e.to_string()),
    };
    let db = st.db.clone();
    with_ingest_store(&st, move |store| {
        if let Err(e) = store.ingest_device(
            &chunk.serial,
            chunk.hardware_id.as_deref(),
            chunk.firmware.as_deref(),
            chunk.api_version.as_deref(),
            chunk.mac.as_deref(),
            chunk.last_sync_unix,
        ) {
            return ingest_error(&e.to_string());
        }
        // feature_modes.json is a server-written sidecar next to oura.db.
        if let Some(Value::Object(map)) = chunk.feature_modes {
            oura_summary::write_feature_modes(&db, map);
        }
        ingest_ack(0, 0)
    })
    .await
}

fn decoder_guard(store: &Store, client_version: i64) -> Option<(StatusCode, Value)> {
    match store.decoder_version() {
        Ok(server_version) if server_version == client_version => None,
        Ok(server_version) => Some((
            StatusCode::CONFLICT,
            json!({
                "ok": false,
                "message": format!(
                    "decoder version mismatch: server v{server_version}, client v{client_version}"
                ),
            }),
        )),
        Err(e) => Some((
            StatusCode::INTERNAL_SERVER_ERROR,
            json!({ "ok": false, "message": e.to_string() }),
        )),
    }
}

fn ingest_ack(inserted: u32, watermark: i64) -> (StatusCode, Value) {
    (
        StatusCode::OK,
        json!({ "ok": true, "inserted": inserted, "watermark": watermark }),
    )
}

fn ingest_error(message: &str) -> (StatusCode, Value) {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        json!({ "ok": false, "message": message }),
    )
}

fn ingest_bad_request(message: &str) -> Response {
    resp(
        StatusCode::BAD_REQUEST,
        "application/json",
        serde_json::to_vec(&json!({ "ok": false, "message": message })).unwrap_or_default(),
    )
}

// ── route handlers ─────────────────────────────────────────────────────────────
// Each is the body of one arm of the old `handle` match; the host/CSRF checks and
// the BLE gate that used to sit at the top of `handle` now live in `guard` and in
// the three ring handlers respectively.

async fn api_blood_report() -> Response {
    blocking_json(crate::blood::report).await
}

async fn api_blood_import_dir() -> Response {
    blocking_json(crate::blood::import_dir).await
}

async fn api_blood_import(body: String) -> Response {
    let req = serde_json::from_str::<Value>(body.trim_end_matches('\0')).unwrap_or(Value::Null);
    let path = req["path"].as_str().unwrap_or("").to_string();
    blocking_json(move || crate::blood::import_path(&path)).await
}

async fn api_dna_files() -> Response {
    blocking_json(crate::dna::list_files).await
}

async fn api_dna_report(RawQuery(q): RawQuery) -> Response {
    // parsing a genome is CPU/IO-heavy → off the async executor; cached per
    // (genome, selected scores).
    let query = q.unwrap_or_default();
    let file = query_param(&query, "file").unwrap_or_default();
    let scores = query_param(&query, "scores").unwrap_or_default();
    blocking_json(move || crate::dna::report(&file, &scores)).await
}

async fn api_dna_fetch(body: String) -> Response {
    // downloads a PUBLIC PGS Catalog score definition from EBI (no genome data
    // leaves). CSRF-gated like the other action endpoints (see `guard`).
    let req = serde_json::from_str::<Value>(body.trim_end_matches('\0')).unwrap_or(Value::Null);
    let id = req["pgs_id"].as_str().unwrap_or("").to_string();
    blocking_json(move || crate::dna::fetch(&id)).await
}

async fn api_dna_lookup(RawQuery(q): RawQuery) -> Response {
    let query = q.unwrap_or_default();
    let file = query_param(&query, "file").unwrap_or_default();
    let needle = query_param(&query, "q").unwrap_or_default();
    blocking_json(move || crate::dna::lookup(&file, &needle)).await
}

async fn api_summary(State(st): State<AppState>, RawQuery(q): RawQuery) -> Response {
    // building the summary shells out to torch models → off the async executor;
    // cached so only the first load (or post-sync/edit) pays for it.
    let tz = effective_tz(st.tz, q.as_deref().unwrap_or(""));
    let db = st.db.clone();
    match tokio::task::spawn_blocking(move || cached_summary(&db, tz)).await {
        Ok(Ok(v)) => json_resp(&v),
        Ok(Err(e)) => json_resp(&json!({ "error": e.to_string() })),
        Err(e) => json_resp(&json!({ "error": e.to_string() })),
    }
}

async fn api_hourly_hr(State(st): State<AppState>, RawQuery(q): RawQuery) -> Response {
    // HR bars per `minutes` slot (default hourly, the iOS HeartRate screen's data);
    // `days` = 0 → all of them, the day page picks out the day it shows.
    let query = q.unwrap_or_default();
    let tz = effective_tz(st.tz, &query);
    let days = query_param(&query, "days")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let minutes = query_param(&query, "minutes")
        .and_then(|v| v.parse().ok())
        .unwrap_or(60);
    let db = st.db.clone();
    match tokio::task::spawn_blocking(move || {
        oura_summary::hourly_hr::hr_bins(&db, tz, days, minutes)
    })
    .await
    {
        Ok(Ok(v)) => json_resp(&v),
        Ok(Err(e)) => json_resp(&json!({ "error": e.to_string() })),
        Err(e) => json_resp(&json!({ "error": e.to_string() })),
    }
}

async fn api_research_export(State(st): State<AppState>, RawQuery(q): RawQuery) -> Response {
    // ZIP of raw per-beat / per-window CSVs for [start, end) — the same
    // `research_export` the iOS app and `oura research-export` use.
    let query = q.unwrap_or_default();
    let base_tz = st.tz;
    let db = st.db.clone();
    let built = tokio::task::spawn_blocking(move || {
        use oura_summary::research_export as rx;
        let num = |k: &str| query_param(&query, k).and_then(|v| v.parse::<f64>().ok());
        let (Some(start), Some(end)) = (num("start"), num("end")) else {
            return Ok::<_, anyhow::Error>(None); // → 400
        };
        let streams: Vec<_> = query_param(&query, "streams")
            .unwrap_or_default()
            .split(',')
            .filter_map(rx::Stream::from_id)
            .collect();
        let tz = num("tz").unwrap_or(effective_tz(base_tz, &query));
        let export = rx::build(
            &db,
            &rx::Request {
                start_unix: start,
                end_unix: end,
                streams,
                tz_offset_hours: tz,
            },
        )?;
        let name = rx::bundle_name(start, tz);
        Ok(Some((rx::to_zip(&export, &name)?, name)))
    })
    .await;
    match built {
        Ok(Ok(Some((zip, name)))) => Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "application/zip")
            .header(
                header::CONTENT_DISPOSITION,
                format!("attachment; filename=\"{name}.zip\""),
            )
            .header(header::CACHE_CONTROL, "no-store")
            .body(Body::from(zip))
            .expect("valid response"),
        Ok(Ok(None)) => resp(
            StatusCode::BAD_REQUEST,
            "text/plain",
            b"start and end (unix seconds) are required".to_vec(),
        ),
        Ok(Err(e)) => resp(StatusCode::BAD_REQUEST, "text/plain", e.to_string().into_bytes()),
        Err(e) => resp(StatusCode::BAD_REQUEST, "text/plain", e.to_string().into_bytes()),
    }
}

async fn api_research_marker(State(st): State<AppState>, body: String) -> Response {
    let label = serde_json::from_str::<Value>(&body)
        .ok()
        .and_then(|v| v["label"].as_str().map(str::to_owned))
        .unwrap_or_default();
    let now = SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as f64 / 1000.0)
        .unwrap_or(0.0);
    match oura_summary::research_export::add_marker(&st.db, now, &label) {
        Ok(()) => json_resp(&json!({ "ok": true, "unix": now })),
        Err(e) => json_resp(&json!({ "error": e.to_string() })),
    }
}

async fn api_get_profile(State(st): State<AppState>) -> Response {
    json_resp(&read_profile(&st.db).to_json())
}

async fn api_post_profile(State(st): State<AppState>, body: String) -> Response {
    match serde_json::from_str::<Value>(body.trim_end_matches('\0')) {
        Ok(v) => match write_profile(&st.db, &v) {
            Ok(d) => json_resp(&d.to_json()),
            Err(e) => json_resp(&json!({ "error": e.to_string() })),
        },
        Err(e) => json_resp(&json!({ "error": e.to_string() })),
    }
}

async fn api_bedtime(State(st): State<AppState>, body: String) -> Response {
    let req = serde_json::from_str::<Value>(body.trim_end_matches('\0')).unwrap_or(Value::Null);
    let raw_start_ds = req["raw_start_ds"].as_i64();
    let captured_unix = req["captured_unix"].as_i64();
    let reset = req["reset"].as_bool().unwrap_or(false);
    let (start_ds, end_ds) = if reset {
        (None, None)
    } else {
        match (req["start_ds"].as_i64(), req["end_ds"].as_i64()) {
            (Some(s), Some(e)) => (Some(s), Some(e)),
            _ => {
                return json_resp(&json!({ "ok": false, "error": "start_ds and end_ds are required" }))
            }
        }
    };
    let Some(raw_start_ds) = raw_start_ds else {
        return json_resp(&json!({ "ok": false, "error": "raw_start_ds is required" }));
    };
    match write_bedtime_override(&st.db, raw_start_ds, captured_unix, start_ds, end_ds) {
        Ok(v) => json_resp(&v),
        Err(e) => json_resp(&json!({ "ok": false, "error": e.to_string() })),
    }
}

async fn api_get_ring_key(State(st): State<AppState>) -> Response {
    match read_ring_key(st.key_file.as_deref()) {
        Ok(key) => json_resp(&json!({ "ok": true, "key": key })),
        Err(e) => json_resp(&json!({ "ok": false, "message": e.to_string() })),
    }
}

async fn api_post_ring_key(State(st): State<AppState>, body: String) -> Response {
    let req = serde_json::from_str::<Value>(body.trim_end_matches('\0')).unwrap_or(Value::Null);
    match write_ring_key(st.key_file.as_deref(), req["key"].as_str().unwrap_or("")) {
        Ok(_) => json_resp(&json!({ "ok": true })),
        Err(e) => json_resp(&json!({ "ok": false, "message": e.to_string() })),
    }
}

async fn api_sync(State(st): State<AppState>) -> Response {
    let _guard = match BLE_GATE.try_lock() {
        Ok(g) => g,
        Err(_) => return json_resp(&busy_json()),
    };
    let (db, name, key_file) = (st.db.clone(), st.name.clone(), st.key_file.clone());
    match tokio::task::spawn_blocking(move || run_sync(&db, &name, key_file.as_deref())).await {
        Ok(Ok(msg)) => json_resp(&json!({ "ok": true, "message": msg })),
        Ok(Err(e)) => json_resp(&json!({ "ok": false, "message": e.to_string() })),
        Err(e) => json_resp(&json!({ "ok": false, "message": e.to_string() })),
    }
}

async fn api_feature(State(st): State<AppState>, body: String) -> Response {
    let _guard = match BLE_GATE.try_lock() {
        Ok(g) => g,
        Err(_) => return json_resp(&busy_json()),
    };
    let req = serde_json::from_str::<Value>(body.trim_end_matches('\0')).unwrap_or(Value::Null);
    let feature = req["feature"].as_str().unwrap_or("").to_string();
    let mode = req["mode"].as_str().unwrap_or("").to_string();
    let (db, name, key_file) = (st.db.clone(), st.name.clone(), st.key_file.clone());
    match tokio::task::spawn_blocking(move || {
        run_feature(&db, &name, key_file.as_deref(), &feature, &mode)
    })
    .await
    {
        Ok(Ok(msg)) => json_resp(&json!({ "ok": true, "message": msg })),
        Ok(Err(e)) => json_resp(&json!({ "ok": false, "message": e.to_string() })),
        Err(e) => json_resp(&json!({ "ok": false, "message": e.to_string() })),
    }
}

/// Drain the ring by invoking our own binary's `sync` subcommand (reuses all the
/// BLE + cursor logic). Returns the last stdout line on success.
///
/// BLE scanning on macOS is flaky: the ring advertises only periodically (and not at
/// all for a few seconds after a disconnect), so a single short scan often misses a
/// ring that is right there. We use a longer scan window and retry the transient
/// "no matching ring" miss a couple of times before giving up.
fn run_sync(db: &Path, name: &str, key_file: Option<&Path>) -> Result<String> {
    let exe = std::env::current_exe().context("locating oura binary")?;
    let mut last = String::from("sync failed");
    for attempt in 0..3 {
        let mut c = Command::new(&exe);
        c.arg("--db")
            .arg(db)
            .arg("--name")
            .arg(name)
            .arg("--scan-timeout")
            .arg("40"); // global flag; wider than the 25 s default
        if let Some(k) = key_file {
            c.arg("--key-file").arg(k);
        }
        c.arg("sync");
        let out = c.output().context("running `oura sync`")?;
        if out.status.success() {
            let stdout = String::from_utf8_lossy(&out.stdout);
            return Ok(stdout.lines().last().unwrap_or("synced").trim().to_string());
        }
        let stderr = String::from_utf8_lossy(&out.stderr);
        last = stderr
            .lines()
            .last()
            .unwrap_or("sync failed")
            .trim()
            .to_string();
        // only a scan miss is worth retrying; a real error (auth, etc.) is not
        let transient = {
            let l = last.to_lowercase();
            l.contains("no matching")
                || l.contains("not found")
                || l.contains("no device")
                || l.contains("timed out")
                || l.contains("timeout")
        };
        if !transient || attempt == 2 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_secs(2));
    }
    Err(anyhow!("{last}"))
}

/// How long one dashboard live heart-rate session streams (the page can stop it early).
const LIVE_HR_SECONDS: u64 = 60;

/// Stream live heart rate as newline-delimited JSON. A spawned task owns the
/// `oura live-hr` child (BLE stays in the CLI, as for sync) and feeds ndjson lines
/// into the response body through a channel. Lines: `{"status":"live"}` once
/// connected, `{"bpm":..,"ibi_ms":..}` per beat, then `{"done":true}` or
/// `{"error":".."}`. When the browser disconnects, hyper drops the body, the channel
/// receiver drops, the next `send_line` fails, and the child is killed
/// (`kill_on_drop`) — the ring leaves live mode on its own within ~20 s. The BLE gate
/// is held for the whole session by moving the guard into the task.
async fn api_live_hr(State(st): State<AppState>) -> Response {
    let guard = match BLE_GATE.try_lock() {
        Ok(g) => g,
        Err(_) => return ndjson_line(&busy_json()),
    };
    let Some(key_file) = st.key_file.clone() else {
        return ndjson_line(&json!({
            "error": "live heart rate needs the ring's auth key: start the dashboard with --key-file"
        }));
    };
    let (db, name) = (st.db.clone(), st.name.clone());
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(32);
    tokio::spawn(async move {
        let _guard = guard; // held until the session ends
        let _ = live_hr_stream(&tx, &db, &name, &key_file).await;
    });
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/x-ndjson")
        .header(header::CACHE_CONTROL, "no-store")
        .body(Body::from_stream(
            tokio_stream::wrappers::ReceiverStream::new(rx),
        ))
        .expect("valid response")
}

async fn live_hr_stream(
    tx: &tokio::sync::mpsc::Sender<Result<Bytes, std::io::Error>>,
    db: &Path,
    name: &str,
    key_file: &Path,
) -> Result<()> {
    let exe = std::env::current_exe().context("locating oura binary")?;
    let mut child = tokio::process::Command::new(&exe)
        .arg("--db")
        .arg(db)
        .arg("--name")
        .arg(name)
        .arg("--scan-timeout")
        .arg("120") // a worn ring advertises only in sparse bursts
        .arg("--key-file")
        .arg(key_file)
        .arg("live-hr")
        .arg("--seconds")
        .arg(LIVE_HR_SECONDS.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .context("running `oura live-hr`")?;
    let stderr = child.stderr.take().context("child stderr")?;
    let stderr_task = tokio::spawn(read_stderr_tail(stderr));
    let mut lines = tokio::io::BufReader::new(child.stdout.take().context("child stdout")?).lines();
    let session = tokio::time::timeout(std::time::Duration::from_secs(210), async {
        while let Some(line) = lines.next_line().await? {
            let msg = if line.starts_with("Streaming live heart rate") {
                json!({ "status": "live" })
            } else if let Some((bpm, ibi_ms)) = parse_beat_line(&line) {
                json!({ "bpm": bpm, "ibi_ms": ibi_ms })
            } else {
                continue;
            };
            // a send error means the browser hung up → stop; kill_on_drop reaps the child
            send_line(tx, &msg).await?;
        }
        Ok::<_, anyhow::Error>(child.wait().await?.success())
    })
    .await;
    let ok = match session {
        Ok(Ok(ok)) => ok,
        _ => {
            // Retain the BLE gate until the process is killed and reaped.
            let _ = child.kill().await;
            let _ = child.wait().await;
            stderr_task.abort();
            let _ = send_line(tx, &json!({"error": "Live heart rate stopped or timed out."})).await;
            return Ok(());
        }
    };
    let stderr = stderr_task.await.unwrap_or_default();
    let last = if ok {
        json!({ "done": true })
    } else {
        let msg = stderr
            .lines()
            .rfind(|l| !l.trim().is_empty())
            .unwrap_or("live heart rate failed");
        json!({ "error": msg.trim() })
    };
    let _ = send_line(tx, &last).await;
    Ok(())
}

async fn read_stderr_tail(mut reader: impl tokio::io::AsyncRead + Unpin) -> String {
    let mut tail = Vec::new();
    let mut buf = [0u8; 4096];
    while let Ok(n) = reader.read(&mut buf).await {
        if n == 0 {
            break;
        }
        tail.extend_from_slice(&buf[..n]);
        if tail.len() > 8192 {
            tail.drain(..tail.len() - 8192);
        }
    }
    String::from_utf8_lossy(&tail).into_owned()
}

/// Push one ndjson line into the response body's channel. An `Err` means the
/// receiver (the HTTP body) was dropped — i.e. the browser disconnected.
async fn send_line(
    tx: &tokio::sync::mpsc::Sender<Result<Bytes, std::io::Error>>,
    v: &Value,
) -> Result<()> {
    tx.send(Ok(Bytes::from(format!("{v}\n"))))
        .await
        .map_err(|_| anyhow!("live client disconnected"))
}

/// Parse one `oura live-hr` beat line, e.g. `  81 bpm (IBI 733 ms)`.
fn parse_beat_line(line: &str) -> Option<(u16, u16)> {
    let (bpm, ibi) = line.trim().strip_suffix(" ms)")?.split_once(" bpm (IBI ")?;
    Some((bpm.parse().ok()?, ibi.parse().ok()?))
}

/// Toggle an on-ring feature via our `feature-mode` subcommand (BLE, auth-gated).
/// `feature` is whitelisted; `mode` is off|automatic. Writing a feature mode REQUIRES
/// the auth key, so we fail early (with a clear message) if the dashboard was launched
/// without `--key-file`. Scan misses retry; the ring's "already in that mode" rejection
/// (result `0x20`) is reported as a no-op success rather than an error.
fn run_feature(
    db: &Path,
    name: &str,
    key_file: Option<&Path>,
    feature: &str,
    mode: &str,
) -> Result<String> {
    const FEATS: &[&str] = &[
        "daytime_hr",
        "spo2",
        "exercise_hr",
        "real_steps",
        "cva_ppg",
        "ambient",
        "resting_hr",
    ];
    if !FEATS.contains(&feature) {
        return Err(anyhow!("unknown feature"));
    }
    if mode != "off" && mode != "automatic" {
        return Err(anyhow!("invalid mode"));
    }
    let key_file = key_file.ok_or_else(|| {
        anyhow!("can't change ring features: the dashboard was started without --key-file (auth key needed to write feature modes)")
    })?;
    let exe = std::env::current_exe().context("locating oura binary")?;
    let on = if mode == "off" { "off" } else { "on" };
    let mut last = String::from("toggle failed");
    for attempt in 0..3 {
        let mut c = Command::new(&exe);
        c.arg("--db")
            .arg(db)
            .arg("--name")
            .arg(name)
            .arg("--scan-timeout")
            .arg("40")
            .arg("--key-file")
            .arg(key_file)
            .arg("feature-mode")
            .arg(feature)
            .arg("--mode")
            .arg(mode);
        let out = c.output().context("running `oura feature-mode`")?;
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        // 0 = off, 1 = automatic — keep feature_modes.json in step with the toggle so
        // the dashboard shows the new state on the next reload, not only after a sync.
        let mode_int = if mode == "off" { 0 } else { 1 };
        if stdout.contains("SUCCESS") {
            write_feature_mode(db, feature, mode_int);
            return Ok(format!("{feature} turned {on}"));
        }
        if stdout.contains("result 0x20") {
            // ring refuses to set a mode it's already in → it's already in that state
            write_feature_mode(db, feature, mode_int);
            return Ok(format!("{feature} is already {on} (no change needed)"));
        }
        last = stdout
            .lines()
            .chain(stderr.lines())
            .rfind(|l| !l.trim().is_empty())
            .unwrap_or("toggle failed")
            .trim()
            .to_string();
        let transient = {
            let l = last.to_lowercase();
            l.contains("no matching")
                || l.contains("not found")
                || l.contains("no device")
                || l.contains("timed out")
                || l.contains("timeout")
        };
        if !transient || attempt == 2 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_secs(2));
    }
    Err(anyhow!("{last}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::AsyncWriteExt;

    #[test]
    fn fallback_fills_only_null_nights_and_keeps_python_stages() {
        // Python staged nights 0 and 2 but emitted null for night 1 (NSSA could not
        // stage it, or no torch). The fallback must fill exactly night 1 from signals
        // and leave the Python results untouched.
        let python = json!([{"source": "nssa", "stages": [1, 2]}, null, {"source": "nssa"}]);
        let signals = json!([{"source": "signals"}, {"source": "signals", "stages": [2, 3]}, {"source": "signals"}]);
        let out = merge_sleep_fallback(Some(python), Some(signals)).unwrap();
        assert_eq!(
            out,
            json!([
                {"source": "nssa", "stages": [1, 2]},
                {"source": "signals", "stages": [2, 3]},
                {"source": "nssa"},
            ])
        );
    }

    #[test]
    fn fallback_takes_whole_signal_array_when_python_absent_or_empty() {
        let signals = json!([{"source": "signals"}]);
        // Python crashed (no torch) → None.
        assert_eq!(
            merge_sleep_fallback(None, Some(signals.clone())),
            Some(signals.clone())
        );
        // Python returned an empty batch.
        assert_eq!(
            merge_sleep_fallback(Some(json!([])), Some(signals.clone())),
            Some(signals)
        );
    }

    #[test]
    fn fallback_length_follows_signals_one_per_range() {
        // Signals are authoritative on night count/order (one per range). A Python
        // batch shorter than signals must not drop the uncovered nights, and a Python
        // batch longer must not append orphans that map to no range.
        let signals = json!([{"n": 0}, {"n": 1}]);
        // Python shorter (only night 0 staged): night 1 comes from signals.
        assert_eq!(
            merge_sleep_fallback(Some(json!([{"py": 0}])), Some(signals.clone())),
            Some(json!([{"py": 0}, {"n": 1}]))
        );
        // Python longer: the orphan third entry is dropped.
        assert_eq!(
            merge_sleep_fallback(
                Some(json!([null, {"py": 1}, {"orphan": true}])),
                Some(signals)
            ),
            Some(json!([{"n": 0}, {"py": 1}]))
        );
    }

    #[test]
    fn fallback_is_a_noop_when_signals_unavailable() {
        // No events to stage from: keep whatever Python returned rather than blanking.
        let python = json!([null]);
        assert_eq!(
            merge_sleep_fallback(Some(python.clone()), None),
            Some(python)
        );
    }

    #[test]
    fn needs_fallback_fires_on_null_missing_or_wrong_length_only() {
        // The gate that was the bug: a fully-staged batch of the right length needs
        // nothing; a null night, a missing batch, or a per-range count mismatch do.
        assert!(!sleep_batch_needs_fallback(Some(&json!([{"s": 1}, {"s": 2}])), 2));
        assert!(sleep_batch_needs_fallback(Some(&json!([{"s": 1}, null])), 2));
        assert!(sleep_batch_needs_fallback(None, 2));
        assert!(sleep_batch_needs_fallback(Some(&json!([{"s": 1}])), 2)); // too short
        assert!(sleep_batch_needs_fallback(Some(&json!([])), 1)); // empty vs one range
        assert!(!sleep_batch_needs_fallback(Some(&json!([])), 0)); // no ranges, no work
    }


    #[tokio::test]
    async fn live_stderr_is_drained_without_unbounded_retention() {
        let (mut writer, reader) = tokio::io::duplex(1024);
        let task = tokio::spawn(async move {
            writer.write_all(&vec![b'x'; 200000]).await.unwrap();
            writer.write_all(b"\nlast error\n").await.unwrap();
        });
        let tail = tokio::time::timeout(std::time::Duration::from_secs(2), read_stderr_tail(reader)).await.unwrap();
        task.await.unwrap();
        assert_eq!(tail.len(), 8192);
        assert!(tail.ends_with("last error\n"));
    }

    #[test]
    fn stalled_runner_is_killed_and_reaped() {
        let mut command = Command::new("python3");
        command.args(["-c", "import time; time.sleep(30)"]);
        assert!(model_output(command, None, std::time::Duration::from_millis(60)).is_none());
    }
    #[test]
    fn large_runner_output_does_not_deadlock_and_input_reaches_eof() {
        let mut command = Command::new("python3");
        command.args([
            "-c",
            "import json,sys; print(json.dumps({'input':sys.stdin.read(),'data':'x'*200000}))",
        ]);
        let value =
            model_output(command, Some(b"window"), std::time::Duration::from_secs(5)).unwrap();
        assert_eq!(value["input"], "window");
        assert_eq!(value["data"].as_str().unwrap().len(), 200000);
    }
    #[test]
    fn parses_live_hr_beat_lines() {
        assert_eq!(parse_beat_line("  81 bpm (IBI 733 ms)"), Some((81, 733)));
        assert_eq!(
            parse_beat_line("No beats captured. Make sure the ring is worn."),
            None
        );
    }
}
