import SongCard from "./SongCard";

export default function SongRow({ title, sub, songs, onRefresh, emptyText }) {
  return (
    <div className="section">
      <div className="section-head">
        <div>
          <h2 className="display">{title}</h2>
          {sub && <div className="sub">{sub}</div>}
        </div>
        {onRefresh && (
          <button className="refresh-btn" onClick={onRefresh}>
            Shuffle ⟳
          </button>
        )}
      </div>
      {songs.length === 0 ? (
        <div className="empty-state">
          <span className="big">Nothing here yet</span>
          {emptyText || "Try a different mood."}
        </div>
      ) : (
        <div className="song-row">
          {songs.map((s) => (
            <SongCard key={s.id} song={s} />
          ))}
        </div>
      )}
    </div>
  );
}
