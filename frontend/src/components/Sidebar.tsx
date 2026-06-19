import { useRef, useState } from "react";
import type { Health, Provider } from "../types";

const OPENAI_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"];
const OLLAMA_MODELS = ["mistral", "llama3", "llama3.2", "phi3", "gemma2"];

interface Props {
  provider: Provider;
  setProvider: (p: Provider) => void;
  model: string;
  setModel: (m: string) => void;
  apiKey: string;
  setApiKey: (k: string) => void;
  documents: string[];
  indexing: boolean;
  onUpload: (files: File[]) => void;
  onClear: () => void;
  health: Health | null;
}

export default function Sidebar(props: Props) {
  const {
    provider,
    setProvider,
    model,
    setModel,
    apiKey,
    setApiKey,
    documents,
    indexing,
    onUpload,
    onClear,
    health,
  } = props;
  const fileRef = useRef<HTMLInputElement>(null);
  const [staged, setStaged] = useState<File[]>([]);

  const models = provider === "openai" ? OPENAI_MODELS : OLLAMA_MODELS;

  function switchProvider(p: Provider) {
    setProvider(p);
    setModel(p === "openai" ? OPENAI_MODELS[0] : OLLAMA_MODELS[0]);
  }

  return (
    <aside className="sidebar">
      <div className="sidebar-logo">🧠 DocMind</div>
      <div className="sidebar-sub">RAG · PDF Chat · Citations</div>

      <div className="section-label">Provider</div>
      <div className="seg">
        <button
          className={provider === "openai" ? "seg-btn active" : "seg-btn"}
          onClick={() => switchProvider("openai")}
        >
          OpenAI
        </button>
        <button
          className={provider === "ollama" ? "seg-btn active" : "seg-btn"}
          onClick={() => switchProvider("ollama")}
        >
          Ollama
        </button>
      </div>

      {provider === "openai" && (
        <input
          className="text-input"
          type="password"
          placeholder="OpenAI API key (sk-...)"
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
        />
      )}

      <select className="text-input" value={model} onChange={(e) => setModel(e.target.value)}>
        {models.map((m) => (
          <option key={m} value={m}>
            {m}
          </option>
        ))}
      </select>

      <div className="section-label">Documents</div>
      <input
        ref={fileRef}
        type="file"
        accept="application/pdf"
        multiple
        hidden
        onChange={(e) => setStaged(Array.from(e.target.files ?? []))}
      />
      <button className="btn ghost" onClick={() => fileRef.current?.click()}>
        + Choose PDFs
      </button>
      {staged.length > 0 && (
        <div className="staged">
          {staged.map((f) => (
            <div key={f.name} className="doc-pill">
              📄 {f.name}
            </div>
          ))}
          <button
            className="btn primary"
            disabled={indexing}
            onClick={() => {
              onUpload(staged);
              setStaged([]);
              if (fileRef.current) fileRef.current.value = "";
            }}
          >
            {indexing ? "Indexing…" : "⚡ Index Documents"}
          </button>
        </div>
      )}

      {documents.length > 0 && (
        <>
          <div className="section-label">Indexed</div>
          {documents.map((d) => (
            <div key={d} className="doc-pill">
              📄 {d}
            </div>
          ))}
        </>
      )}

      <div className="sidebar-spacer" />

      <button className="btn ghost danger" onClick={onClear}>
        🗑 Clear my documents
      </button>

      <div className="sidebar-foot">
        {health ? (
          <span className={health.pinecone_configured ? "dot ok" : "dot bad"}>
            ● {health.pinecone_configured ? "Connected" : "Pinecone not configured"}
          </span>
        ) : (
          <span className="dot">● connecting…</span>
        )}
        <div className="privacy-note">
          Your documents are isolated to this browser session and auto-deleted after
          inactivity. Nothing is stored in a database.
        </div>
      </div>
    </aside>
  );
}
