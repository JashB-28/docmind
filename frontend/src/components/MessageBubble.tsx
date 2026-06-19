import { useState } from "react";
import type { ChatMessage } from "../types";
import BotAvatar from "./BotAvatar";
import SourceCard from "./SourceCard";

export default function MessageBubble({ message }: { message: ChatMessage }) {
  const [open, setOpen] = useState(false);

  if (message.role === "user") {
    return (
      <div className="msg msg-user">
        <div className="bubble bubble-user">{message.content}</div>
      </div>
    );
  }

  return (
    <div className="msg msg-assistant">
      <div className="msg-avatar">
        <BotAvatar size={34} />
      </div>
      <div className="bubble bubble-assistant">
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
