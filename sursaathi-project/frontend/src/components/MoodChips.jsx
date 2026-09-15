import { MOOD_COLORS } from "../lib/moods";

export default function MoodChips({ moods, active, onSelect }) {
  return (
    <div className="mood-row">
      {moods.map((m) => {
        const label = m.split(" ").slice(0, -1).join(" ");
        const emoji = m.split(" ").slice(-1)[0];
        const isActive = active === m;
        return (
          <div
            key={m}
            className={`mood-chip${isActive ? " active" : ""}`}
            style={{ "--chip-c": MOOD_COLORS[m] }}
            onClick={() => onSelect(isActive ? null : m)}
          >
            <span className="dot" />
            {label} {emoji}
          </div>
        );
      })}
    </div>
  );
}
