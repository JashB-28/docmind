import { useState } from "react";
import type { ChatMessage } from "../types";
import SourceCard from "./SourceCard";

export default function MessageBubble({ message }: { message: ChatMessage }) {
  const [open, setOpen] = useState(false);
  const isUser = message.role === "user";

  if (isUser) {
    return (
      <div className="msg msg-user">
        <span className="msg-avatar">🙋</span>
        <div className="msg-body">{message.content}</div>
      </div>
    );
  }

  return (
    <div className="msg msg-assistant">
      <span className="msg-avatar">🧠</span>
      <div className="msg-body">
        <div className="msg-text">
          {message.content}
          {message.streaming && <span className="cursor">▋</span>}
        </div>
        {message.sources && message.sources.length > 0 && (
          <div className="sources">
            <button className="sources-toggle" onClick={() => setOpen((v) => !v)}>
              📚 {message.sources.length} source{message.sources.length > 1 ? "s" : ""}
              <span className="chevron">{open ? "▲" : "▼"}</span>
            </button>
            {open && (
              <div className="sources-list">
                {message.sources.map((s) => (
                  <SourceCard key={s.chunk_id} source={s} />
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
