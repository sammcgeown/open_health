//! `oura push` — the Mac agent's upload side. Reads the local store **read-only** and
//! pushes new rows to a dashboard server's `/api/ingest/*` endpoints over HTTPS with a
//! bearer token. The server owns the resume watermark, so a push is stateless and
//! idempotent: a dropped connection just resends. See docs/kubernetes-deployment.md
//! (Component 1).
//!
//! This command never touches Bluetooth, so it is not behind the `ble` feature.

use std::path::Path;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use serde::Serialize;

use oura_store::Store;

use crate::ingest::{
    DeviceChunk, EventChunk, EventRow, IngestAck, ReadingChunk, ReadingRow, StateResponse,
};

/// Rows per POST. Event bodies are tens of bytes, so ~2000 rows is a small request
/// well under the server's ingest body limit.
const CHUNK_ROWS: usize = 2_000;

#[derive(Debug, Default)]
pub struct PushSummary {
    pub events_inserted: u32,
    pub readings_inserted: u32,
    pub serials: usize,
}

/// Read `db` read-only and push everything past the server's watermark.
pub fn run(db: &Path, server: &str, token: &str) -> Result<PushSummary> {
    let server = server.trim_end_matches('/').to_string();
    let client = Client::new(&server, token);

    let store = Store::open_read_only(db)
        .with_context(|| format!("opening {} read-only", db.display()))?;
    let local_version = store.decoder_version()?;

    let state: StateResponse = client.get_json("/api/ingest/state")?;
    if state.decoder_version != local_version {
        bail!(
            "decoder version mismatch: local store is v{local_version}, server is v{}. \
             Upgrade both the agent and the server to the same build before pushing.",
            state.decoder_version
        );
    }

    let mut summary = PushSummary::default();
    for serial in store.device_serials()? {
        summary.serials += 1;
        let wm = state.serials.get(&serial).copied().unwrap_or_default();

        // If the local DB was rebuilt (local max below the server watermark), restart
        // from 0 — dedup on the server makes the resend harmless.
        let mut after = start_cursor(wm.events, store.max_event_id(&serial)?);
        loop {
            let rows = store.events_since(&serial, after, CHUNK_ROWS)?;
            if rows.is_empty() {
                break;
            }
            let last = rows.last().map(|r| r.source_id).unwrap_or(after);
            let chunk = EventChunk {
                serial: serial.clone(),
                decoder_version: local_version,
                rows: rows.into_iter().map(EventRow::from_store).collect(),
            };
            let ack: IngestAck = client.post_json("/api/ingest/events", &chunk)?;
            ack_ok(&ack, "events")?;
            summary.events_inserted += ack.inserted;
            after = ack.watermark.max(last);
        }

        let mut after = start_cursor(wm.readings, store.max_reading_id(&serial)?);
        loop {
            let rows = store.readings_since(&serial, after, CHUNK_ROWS)?;
            if rows.is_empty() {
                break;
            }
            let last = rows.last().map(|r| r.source_id).unwrap_or(after);
            let chunk = ReadingChunk {
                serial: serial.clone(),
                decoder_version: local_version,
                rows: rows.into_iter().map(ReadingRow::from_store).collect(),
            };
            let ack: IngestAck = client.post_json("/api/ingest/readings", &chunk)?;
            ack_ok(&ack, "readings")?;
            summary.readings_inserted += ack.inserted;
            after = ack.watermark.max(last);
        }
    }

    // Device identity + last-sync + feature_modes.json, once (one ring per Mac in
    // practice; device_info() returns the most-recently-updated device).
    if let Some((serial, hardware_id, firmware, api_version, mac, _updated, last_sync, _cursor)) =
        store.device_info()?
    {
        let feature_modes = read_feature_modes(db);
        let device = DeviceChunk {
            serial,
            hardware_id: non_empty(hardware_id),
            firmware: non_empty(firmware),
            api_version: non_empty(api_version),
            mac: non_empty(mac),
            last_sync_unix: last_sync,
            feature_modes,
        };
        let ack: IngestAck = client.post_json("/api/ingest/device", &device)?;
        ack_ok(&ack, "device")?;
    }

    Ok(summary)
}

/// Resume at the server watermark, unless the local DB is *behind* it (rebuilt): then
/// restart from 0.
fn start_cursor(server_watermark: i64, local_max: i64) -> i64 {
    if server_watermark > local_max {
        0
    } else {
        server_watermark
    }
}

fn ack_ok(ack: &IngestAck, what: &str) -> Result<()> {
    if ack.ok {
        Ok(())
    } else {
        Err(anyhow!(
            "server rejected {what} ingest: {}",
            ack.message.as_deref().unwrap_or("(no message)")
        ))
    }
}

fn non_empty(s: String) -> Option<String> {
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

fn read_feature_modes(db: &Path) -> Option<serde_json::Value> {
    let path = oura_summary::feature_modes_path(db);
    let text = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

/// A tiny bearer-auth JSON client over `ureq` (no `json` feature needed: we serialize
/// with `serde_json` and send/receive strings).
struct Client {
    agent: ureq::Agent,
    base: String,
    bearer: String,
}

impl Client {
    fn new(base: &str, token: &str) -> Self {
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(15))
            .timeout_read(Duration::from_secs(120))
            .build();
        Self {
            agent,
            base: base.to_string(),
            bearer: format!("Bearer {token}"),
        }
    }

    fn get_json<T: serde::de::DeserializeOwned>(&self, path: &str) -> Result<T> {
        let url = format!("{}{path}", self.base);
        let resp = self
            .agent
            .get(&url)
            .set("Authorization", &self.bearer)
            .call()
            .map_err(|e| http_error("GET", path, e))?;
        let body = resp.into_string().context("reading response body")?;
        serde_json::from_str(&body).with_context(|| format!("parsing {path} response"))
    }

    fn post_json<B: Serialize, T: serde::de::DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<T> {
        let url = format!("{}{path}", self.base);
        let payload = serde_json::to_string(body).context("serializing request")?;
        let resp = self
            .agent
            .post(&url)
            .set("Authorization", &self.bearer)
            .set("Content-Type", "application/json")
            .send_string(&payload)
            .map_err(|e| http_error("POST", path, e))?;
        let text = resp.into_string().context("reading response body")?;
        serde_json::from_str(&text).with_context(|| format!("parsing {path} response"))
    }
}

fn http_error(method: &str, path: &str, e: ureq::Error) -> anyhow::Error {
    match e {
        ureq::Error::Status(401, _) => anyhow!(
            "{method} {path} -> 401 Unauthorized: the ingest token is missing or wrong"
        ),
        ureq::Error::Status(409, resp) => {
            let detail = resp.into_string().unwrap_or_default();
            anyhow!("{method} {path} -> 409 Conflict (decoder version mismatch): {detail}")
        }
        ureq::Error::Status(code, resp) => {
            let detail = resp.into_string().unwrap_or_default();
            anyhow!("{method} {path} -> HTTP {code}: {detail}")
        }
        other => anyhow!("{method} {path} failed: {other}"),
    }
}
