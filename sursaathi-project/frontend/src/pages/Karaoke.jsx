import { useEffect, useRef, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { api } from "../lib/api";
import { getYoutubeUrl } from "../lib/links";

const LINES_PER_STEP = 2;
const STEP_DELAY_MS = 10000;

export default function Karaoke() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [data, setData] = useState(null);
  const [activeGroup, setActiveGroup] = useState(-1);
  const [playing, setPlaying] = useState(false);
  const intervalRef = useRef(null);

  useEffect(() => {
    setData(null);
    setActiveGroup(-1);
    setPlaying(false);
    api.karaoke(id).then(setData);
    window.scrollTo({ top: 0 });
    return () => clearInterval(intervalRef.current);
  }, [id]);

  useEffect(() => {
    clearInterval(intervalRef.current);
    if (playing && data) {
      const totalGroups = Math.ceil(data.lines.length / LINES_PER_STEP);
      intervalRef.current = setInterval(() => {
        setActiveGroup((prev) => {
          const next = prev + 1;
          if (next >= totalGroups) {
            clearInterval(intervalRef.current);
            setPlaying(false);
            return prev;
          }
          return next;
        });
      }, STEP_DELAY_MS);
    }
    return () => clearInterval(intervalRef.current);
  }, [playing, data]);

  if (!data) return <div className="empty-state">Loading...</div>;

  const totalGroups = Math.ceil(data.lines.length / LINES_PER_STEP);
  const isActive = (i) => Math.floor(i / LINES_PER_STEP) === activeGroup;
  const youtubeUrl = getYoutubeUrl(data);
  const youtubeStyle = { background: "#E8433D" };
  const actionsStyle = { display: "flex", gap: 10, marginBottom: 30 };

  const handleToggle = () => {
    if (!playing && activeGroup >= totalGroups - 1) setActiveGroup(-1);
    setPlaying((p) => !p);
  };

  return (
    <div className="karaoke-page">
      <div className="back" onClick={() => navigate("/song/" + id)}>Back to song</div>

      <div className="karaoke-header">
        <h1 className="display">{data.name}</h1>
        <div className="singer">{data.singer}</div>
      </div>

      <div style={actionsStyle}>
        <a className="karaoke-btn" href={youtubeUrl} target="_blank" rel="noopener noreferrer" style={youtubeStyle}>Play on YouTube</a>
        <button className="karaoke-btn" onClick={handleToggle}>{playing ? "Pause" : "Follow along"}</button>
      </div>

      <div className="karaoke-lines">
        {data.lines.map((line, i) => (
          <p
            key={i}
            className={isActive(i) ? "karaoke-line active" : "karaoke-line"}
            onClick={() => setActiveGroup(Math.floor(i / LINES_PER_STEP))}
          >
            {line}
          </p>
        ))}
      </div>
    </div>
  );
}
