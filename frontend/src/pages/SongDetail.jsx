import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import SongRow from "../components/SongRow";
import { api } from "../lib/api";
import { MOOD_COLORS } from "../lib/moods";
import { getYoutubeUrl, getSpotifyUrl } from "../lib/links";

export default function SongDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [song, setSong] = useState(null);
  const [recs, setRecs] = useState({ byLyrics: [], byMood: [] });

  useEffect(() => {
    setSong(null);
    api.songDetail(id).then(setSong);
    api.recommendations(id, 4).then(setRecs);
    window.scrollTo({ top: 0 });
  }, [id]);

  if (!song) return <div className="empty-state">Loading...</div>;

  const youtubeUrl = getYoutubeUrl(song);
  const spotifyUrl = getSpotifyUrl(song);
  const youtubeStyle = { background: "#E8433D" };
  const spotifyStyle = { background: "#3FC65B" };
  const actionsStyle = { display: "flex", gap: 10, flexWrap: "wrap", marginTop: 16 };

  return (
    <>
      <div className="detail-top">
        <img src={song.thumb} alt={song.name} />
        <div className="detail-info">
          <div className="back" onClick={() => navigate("/")}>Back</div>
          <h1 className="display">{song.name}</h1>
          <div className="namedev dev-text">{song.nameDev}</div>
          <div className="meta">{song.singer} - {song.year} - {song.theme}</div>

          <div className="tagrow">
            {song.moods.map((m) => (
              <span key={m} className="tag" style={{ borderColor: MOOD_COLORS[m] + "66" }}>{m}</span>
            ))}
          </div>

          <div style={actionsStyle}>
            <a className="karaoke-btn" href={youtubeUrl} target="_blank" rel="noopener noreferrer" style={youtubeStyle}>Play on YouTube</a>
            <a className="karaoke-btn" href={spotifyUrl} target="_blank" rel="noopener noreferrer" style={spotifyStyle}>Play on Spotify</a>
            <button className="karaoke-btn" onClick={() => navigate("/song/" + song.id + "/karaoke")}>Karaoke lyrics</button>
          </div>
        </div>
      </div>

      <div className="lyric-preview">{song.lyrics}</div>

      <SongRow
        title="More like this - lyrics"
        sub="Similar words and themes"
        songs={recs.byLyrics}
        emptyText="No close lyrical matches found."
      />
      <SongRow
        title="More like this - mood"
        sub="Same feeling, different song"
        songs={recs.byMood}
        emptyText="No mood matches found."
      />
    </>
  );
}
