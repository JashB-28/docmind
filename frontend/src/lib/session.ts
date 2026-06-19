// A per-browser session id. It namespaces this user's documents in Pinecone.
// No personal data — just a random handle so one browser's uploads stay isolated.
const KEY = "docmind_session_id";

export function getSessionId(): string {
  let id = localStorage.getItem(KEY);
  if (!id) {
    id = crypto.randomUUID().replace(/-/g, "");
    localStorage.setItem(KEY, id);
  }
  return id;
}

export function resetSessionId(): string {
  const id = crypto.randomUUID().replace(/-/g, "");
  localStorage.setItem(KEY, id);
  return id;
}
