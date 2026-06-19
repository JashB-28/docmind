import type { Source } from "../types";

function confidenceClass(conf: number): string {
  if (conf >= 70) return "conf-high";
  if (conf >= 40) return "conf-mid";
  return "conf-low";
}

function confidenceIcon(conf: number): string {
  if (conf >= 70) return "🟢";
  if (conf >= 40) return "🟡";
  return "🔴";
}

export default function SourceCard({ source }: { source: Source }) {
  return (
    <div className="source-card">
      <div className="source-head">
        <span className="source-filename">📄 {source.filename}</span>
        <span className="source-page"> · Page {source.page}</span>
        <span className={`conf-badge ${confidenceClass(source.confidence)}`}>
          {confidenceIcon(source.confidence)} {source.confidence}%
        </span>
      </div>
      <div className="source-excerpt">{source.excerpt}</div>
    </div>
  );
}
