import { useNavigate } from "react-router-dom";

export default function SongCard({ song }) {
  const navigate = useNavigate();
  return (
    <div className="song-card" onClick={() => navigate(`/song/${song.id}`)}>
      <div className="art">
        <img src={song.thumb} alt={song.name} loading="lazy" />
      </div>
      <div className="body">
        <div className="name">{song.name}</div>
        <div className="singer">{song.singer}</div>
      </div>
    </div>
  );
}
