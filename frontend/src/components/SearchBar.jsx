import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../lib/api";

export default function SearchBar() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [open, setOpen] = useState(false);
  const boxRef = useRef(null);
  const navigate = useNavigate();

  useEffect(() => {
    const q = query.trim();
    if (!q) {
      setResults([]);
      return;
    }
    const timer = setTimeout(() => {
      api.search(q, 6).then((r) => {
        setResults(r);
        setOpen(true);
      });
    }, 200);
    return () => clearTimeout(timer);
  }, [query]);

  useEffect(() => {
    function onClickOutside(e) {
      if (boxRef.current && !boxRef.current.contains(e.target)) setOpen(false);
    }
    document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, []);

  return (
    <div className="search-shell" ref={boxRef}>
      <div style={{ width: "100%", maxWidth: 460 }}>
        <div className="search-box">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
          <input
            type="text"
            placeholder="Search songs, singers, themes…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onFocus={() => results.length && setOpen(true)}
          />
        </div>
        {open && results.length > 0 && (
          <div className="search-results">
            {results.map((s) => (
              <div
                key={s.id}
                className="search-result-row"
                onClick={() => {
                  setOpen(false);
                  setQuery("");
                  navigate(`/song/${s.id}`);
                }}
              >
                <img src={s.thumb} alt="" />
                <div>
                  <div className="srn">{s.name}</div>
                  <div className="srs">{s.singer}</div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
