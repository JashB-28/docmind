import { useId } from "react";

/** The friendly two-eyed DocMind bot — a gradient-ringed squircle with eyes. */
export default function BotAvatar({ size = 40 }: { size?: number }) {
  const id = useId();
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 64 64"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <defs>
        <linearGradient id={`ring-${id}`} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#5ad1ff" />
          <stop offset="50%" stopColor="#6c7cff" />
          <stop offset="100%" stopColor="#b58cff" />
        </linearGradient>
      </defs>
      <rect
        x="4"
        y="4"
        width="56"
        height="56"
        rx="19"
        fill="#23263a"
        stroke={`url(#ring-${id})`}
        strokeWidth="4"
      />
      <ellipse cx="24.5" cy="33" rx="6" ry="8" fill="#fff" />
      <ellipse cx="39.5" cy="33" rx="6" ry="8" fill="#fff" />
    </svg>
  );
}
