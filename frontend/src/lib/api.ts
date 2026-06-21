import type { Health, Provider, Source } from "../types";

const BASE = "/api";

export interface IngestResult {
  session_id: string;
  indexed_files: string[];
  total_chunks: number;
}

export async function getHealth(): Promise<Health> {
  const res = await fetch(`${BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed (${res.status})`);
  return res.json();
}

export async function ingestDocuments(params: {
  files: File[];
  sessionId: string;
  provider: Provider;
  apiKey: string;
}): Promise<IngestResult> {
  const form = new FormData();
  params.files.forEach((f) => form.append("files", f));
  form.append("session_id", params.sessionId);
  form.append("provider", params.provider);
  form.append("api_key", params.apiKey);

  const res = await fetch(`${BASE}/documents/ingest`, { method: "POST", body: form });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `Ingest failed (${res.status})`);
  }
  return res.json();
}

export async function listDocuments(sessionId: string): Promise<string[]> {
  const res = await fetch(`${BASE}/documents/${sessionId}`);
  if (!res.ok) return [];
  return (await res.json()).documents ?? [];
}

export async function clearDocuments(sessionId: string): Promise<void> {
  await fetch(`${BASE}/documents/${sessionId}`, { method: "DELETE" });
}

/** Fetch a short-lived presigned S3 URL for the original PDF, or null. */
export async function getDocumentUrl(
  sessionId: string,
  filename: string
): Promise<string | null> {
  const res = await fetch(`${BASE}/documents/${sessionId}/${encodeURIComponent(filename)}/url`);
  if (!res.ok) return null;
  return (await res.json()).url ?? null;
}

export type StreamEvent =
  | { type: "sources"; sources: Source[] }
  | { type: "token"; text: string }
  | { type: "error"; message: string };

/**
 * POST a question and yield Server-Sent Events as they stream in:
 * one `sources` event, then many `token` events, then completion.
 */
export async function* streamQuery(params: {
  question: string;
  sessionId: string;
  provider: Provider;
  model: string;
  apiKey: string;
  history: string;
}): AsyncGenerator<StreamEvent> {
  const res = await fetch(`${BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question: params.question,
      session_id: params.sessionId,
      provider: params.provider,
      model: params.model || null,
      api_key: params.apiKey,
      history: params.history,
    }),
  });

  if (!res.ok || !res.body) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `Query failed (${res.status})`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line.
    let sep: number;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);

      let event = "message";
      let data = "";
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (!data) continue;
      const payload = JSON.parse(data);

      if (event === "sources") yield { type: "sources", sources: payload };
      else if (event === "token") yield { type: "token", text: payload.text };
      else if (event === "error") yield { type: "error", message: payload.message };
      // "done" needs no action — the stream ends.
    }
  }
}
