import { useEffect, useRef, useState } from "react";
import Sidebar from "./components/Sidebar";
import MessageBubble from "./components/MessageBubble";
import {
  clearDocuments,
  getHealth,
  ingestDocuments,
  listDocuments,
  streamQuery,
} from "./lib/api";
import { getSessionId, resetSessionId } from "./lib/session";
import type { ChatMessage, Health, Provider } from "./types";

export default function App() {
  const [sessionId, setSessionId] = useState(getSessionId);
  const [provider, setProvider] = useState<Provider>("openai");
  const [model, setModel] = useState("gpt-4o-mini");
  const [apiKey, setApiKey] = useState("");
  const [documents, setDocuments] = useState<string[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [indexing, setIndexing] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<Health | null>(null);

  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth(null));
    listDocuments(sessionId).then(setDocuments).catch(() => {});
  }, [sessionId]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  async function handleUpload(files: File[]) {
    setError(null);
    setIndexing(true);
    try {
      const result = await ingestDocuments({ files, sessionId, provider, apiKey });
      setDocuments(result.indexed_files);
      setMessages([]);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setIndexing(false);
    }
  }

  async function handleClear() {
    await clearDocuments(sessionId).catch(() => {});
    setSessionId(resetSessionId());
    setDocuments([]);
    setMessages([]);
  }

  async function handleSend() {
    const question = input.trim();
    if (!question || busy) return;
    setError(null);
    setInput("");

    const history = messages
      .slice(-6)
      .map((m) => `${m.role === "user" ? "User" : "Assistant"}: ${m.content}`)
      .join("\n");

    setMessages((prev) => [
      ...prev,
      { role: "user", content: question },
      { role: "assistant", content: "", streaming: true },
    ]);
    setBusy(true);

    try {
      for await (const ev of streamQuery({
        question,
        sessionId,
        provider,
        model,
        apiKey,
        history,
      })) {
        if (ev.type === "sources") {
          setMessages((prev) => updateLast(prev, (m) => ({ ...m, sources: ev.sources })));
        } else if (ev.type === "token") {
          setMessages((prev) =>
            updateLast(prev, (m) => ({ ...m, content: m.content + ev.text }))
          );
        } else if (ev.type === "error") {
          throw new Error(ev.message);
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setMessages((prev) => updateLast(prev, (m) => ({ ...m, streaming: false })));
      setBusy(false);
    }
  }

  const hasDocs = documents.length > 0;

  return (
    <div className="layout">
      <Sidebar
        provider={provider}
        setProvider={setProvider}
        model={model}
        setModel={setModel}
        apiKey={apiKey}
        setApiKey={setApiKey}
        documents={documents}
        indexing={indexing}
        onUpload={handleUpload}
        onClear={handleClear}
        health={health}
      />

      <main className="main">
        <header className="main-head">
          <h1>🧠 DocMind</h1>
          <p>Ask anything about your documents — with citations, confidence, and live answers.</p>
          <span className="provider-badge">
            {provider === "openai" ? "OpenAI" : "Ollama"} · {model}
          </span>
        </header>

        {error && <div className="banner error">{error}</div>}

        <div className="chat" ref={scrollRef}>
          {!hasDocs && messages.length === 0 ? (
            <div className="empty">
              <div className="empty-icon">📂</div>
              <div className="empty-title">No documents indexed yet</div>
              <div className="empty-sub">
                Upload PDFs in the sidebar and click <strong>Index Documents</strong> to start.
              </div>
            </div>
          ) : (
            messages.map((m, i) => <MessageBubble key={i} message={m} />)
          )}
        </div>

        <div className="composer">
          <input
            className="composer-input"
            placeholder={hasDocs ? "Ask a question about your documents…" : "Index documents first…"}
            value={input}
            disabled={!hasDocs || busy}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
          />
          <button className="btn primary send" disabled={!hasDocs || busy} onClick={handleSend}>
            {busy ? "…" : "Send →"}
          </button>
        </div>
      </main>
    </div>
  );
}

function updateLast(
  messages: ChatMessage[],
  fn: (m: ChatMessage) => ChatMessage
): ChatMessage[] {
  if (messages.length === 0) return messages;
  const copy = messages.slice();
  copy[copy.length - 1] = fn(copy[copy.length - 1]);
  return copy;
}
