//! Wire format shared by the push client (`oura push`) and the dashboard server's
//! `/api/ingest/*` endpoints.
//!
//! Rows travel as JSON. Event bodies are small raw BLE blobs, so they ride as hex
//! (keeps `hex`, already a dependency, as the only codec — no base64 crate). Each row
//! carries its `source_id` (the Mac store's `events.id`/`readings.id`) purely to drive
//! the resume watermark; the row itself is copied verbatim so the server holds a
//! faithful copy. See docs/kubernetes-deployment.md (Components 1–2).

use std::collections::BTreeMap;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use oura_store::storage::{IngestEvent, IngestReading};

/// `GET /api/ingest/state` response: the server's per-serial watermarks plus the
/// server's decoder version (the client refuses to push across a version mismatch).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateResponse {
    pub decoder_version: i64,
    #[serde(default)]
    pub serials: BTreeMap<String, Watermarks>,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Watermarks {
    #[serde(default)]
    pub events: i64,
    #[serde(default)]
    pub readings: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventRow {
    pub source_id: i64,
    pub tag: i64,
    pub name: String,
    pub ring_timestamp: i64,
    /// Raw event body, hex-encoded.
    pub body_hex: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decoded_json: Option<String>,
    pub captured_unix: i64,
}

impl EventRow {
    pub fn from_store(e: IngestEvent) -> Self {
        Self {
            source_id: e.source_id,
            tag: e.tag,
            name: e.name,
            ring_timestamp: e.ring_timestamp,
            body_hex: hex::encode(&e.body),
            decoded_json: e.decoded_json,
            captured_unix: e.captured_unix,
        }
    }

    pub fn into_store(self) -> Result<IngestEvent> {
        Ok(IngestEvent {
            source_id: self.source_id,
            tag: self.tag,
            name: self.name,
            ring_timestamp: self.ring_timestamp,
            body: hex::decode(&self.body_hex).context("decoding event body hex")?,
            decoded_json: self.decoded_json,
            captured_unix: self.captured_unix,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadingRow {
    pub source_id: i64,
    pub kind: String,
    pub value: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
    pub captured_unix: i64,
}

impl ReadingRow {
    pub fn from_store(r: IngestReading) -> Self {
        Self {
            source_id: r.source_id,
            kind: r.kind,
            value: r.value,
            unit: r.unit,
            captured_unix: r.captured_unix,
        }
    }

    pub fn into_store(self) -> IngestReading {
        IngestReading {
            source_id: self.source_id,
            kind: self.kind,
            value: self.value,
            unit: self.unit,
            captured_unix: self.captured_unix,
        }
    }
}

/// A chunk of event rows (`POST /api/ingest/events`). `decoder_version` lets the
/// server reject a mismatched client with 409 before writing anything.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventChunk {
    pub serial: String,
    pub decoder_version: i64,
    pub rows: Vec<EventRow>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadingChunk {
    pub serial: String,
    pub decoder_version: i64,
    pub rows: Vec<ReadingRow>,
}

/// `POST /api/ingest/device`: device identity, the Mac's last-sync time (for the
/// freshness label), and the contents of `feature_modes.json` (written server-side).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceChunk {
    pub serial: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hardware_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub firmware: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mac: Option<String>,
    pub last_sync_unix: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feature_modes: Option<serde_json::Value>,
}

/// The server's reply to an ingest chunk: rows newly inserted and the new watermark.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestAck {
    pub ok: bool,
    #[serde(default)]
    pub inserted: u32,
    #[serde(default)]
    pub watermark: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}
