export interface Source {
  filename: string;
  page: number | string;
  confidence: number;
  chunk_id: string;
  excerpt: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  streaming?: boolean;
}

export type Provider = "openai" | "ollama" | "bedrock";

export interface Health {
  status: string;
  pinecone_configured: boolean;
  default_llm_backend: string;
  active_sessions: number;
}
