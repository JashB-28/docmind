import { useEffect, useRef, useState } from "react";
import BotAvatar from "./components/BotAvatar";
import MessageBubble from "./components/MessageBubble";
import Sidebar from "./components/Sidebar";
import {
  clearDocuments,
  getHealth,
  ingestDocuments,
  listDocuments,
  streamQuery,
} from "./lib/api";
import { getSessionId, resetSessionId } from "./lib/session";
import type { ChatMessage, Health, Provider } from "./types";

type Theme = "light" | "dark";

const DEFAULT_MODEL: Record<Provider, string> = {
  openai: "gpt-4o-mini",
  bedrock: "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
  ollama: "mistral",
};

function initialTheme(): Theme {
  const saved = localStorage.getItem("docmind_theme");
  if (saved === "light" || saved === "dark") return saved;
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

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
  const [theme, setTheme] = useState<Theme>(initialTheme);

  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("docmind_theme", theme);
  }, [theme]);

  useEffect(() => {
    getHealth()
      .then((h) => {
        setHealth(h);
        if (h.default_provider) {
          setProvider(h.default_provider);
          setModel(DEFAULT_MODEL[h.default_provider] ?? model);
        }
      })
      .catch(() => setHealth(null));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
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
  const showWelcome = messages.length === 0;

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
        ollamaEnabled={health?.enable_ollama ?? true}
      />

      <main className="main">
        <header className="topbar">
          <div className="brand">
            <BotAvatar size={36} />
            <div>
              <div className="brand-name">DocMind</div>
              <div className="brand-sub">
                {provider === "openai" ? "OpenAI" : provider === "bedrock" ? "Bedrock" : "Ollama"}
                {" · "}
                {model}
              </div>
            </div>
          </div>
          <button
            className="theme-toggle"
            onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
            title={theme === "dark" ? "Switch to light" : "Switch to dark"}
          >
            {theme === "dark" ? "☀" : "☾"}
          </button>
        </header>

        {error && <div className="banner error">{error}</div>}

        <div className="chat" ref={scrollRef}>
          <div className="chat-inner">
            {showWelcome ? (
              <div className="welcome">
                <BotAvatar size={84} />
                <div className="welcome-title">
                  {hasDocs ? "Hey there 👋" : "Welcome to DocMind"}
                </div>
                <div className="welcome-sub">
                  {hasDocs
                    ? "Ask me anything about your documents — I'll answer with citations."
                    : "Upload a PDF in the sidebar and click Index Documents to begin."}
                </div>
              </div>
            ) : (
              messages.map((m, i) => <MessageBubble key={i} message={m} />)
            )}
          </div>
        </div>

        <div className="composer">
          <div className="composer-inner">
            <input
              className="composer-input"
              placeholder={hasDocs ? "Ask a question…" : "Index documents first…"}
              value={input}
              disabled={!hasDocs || busy}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSend()}
            />
            <button
              className="send-btn"
              disabled={!hasDocs || busy}
              onClick={handleSend}
              title="Send"
            >
              {busy ? "…" : "↑"}
            </button>
          </div>
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
