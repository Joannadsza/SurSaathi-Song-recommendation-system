export default function Logo({ size = 34 }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 64 64"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <rect width="64" height="64" rx="16" fill="#1B1030" />
      <path
        d="M18 40 V26 Q18 18 26 18 H30 V24 H27 Q23 24 23 28 V40 Z"
        fill="#E8A23D"
      />
      <circle cx="20" cy="44" r="5.5" fill="#E8A23D" />
      <path
        d="M34 44 V22 Q34 16 41 15 L46 14 V20 L42 20.8 Q39 21.4 39 24.5 V44 Z"
        fill="#C4506B"
      />
      <circle cx="36" cy="47" r="5.5" fill="#C4506B" />
    </svg>
  );
}
