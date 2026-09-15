import { useEffect, useState, useCallback } from "react";
import SearchBar from "../components/SearchBar";
import MoodChips from "../components/MoodChips";
import SongRow from "../components/SongRow";
import { api } from "../lib/api";

export default function Home() {
  const [moods, setMoods] = useState([]);
  const [activeMood, setActiveMood] = useState(null);
  const [songs, setSongs] = useState([]);

  useEffect(() => {
    api.moods().then(setMoods);
  }, []);

  const loadFeatured = useCallback(() => {
    api.featured({ mood: activeMood, limit: 4 }).then(setSongs);
  }, [activeMood]);

  useEffect(() => {
    loadFeatured();
  }, [loadFeatured]);

  return (
    <>
      <section className="hero">
        <h1 className="display">SurSaathi</h1>
        <h2 className="display">Har mood ka apna geet</h2>
        <p className="tagline">
          A companion that listens to what you're feeling!
        </p>
      </section>

      <SearchBar />

      <MoodChips moods={moods} active={activeMood} onSelect={setActiveMood} />

      <SongRow
        title={activeMood ? `For a ${activeMood.split(" ")[0]} mood` : "Picked for you"}
        sub={activeMood ? "A few to start with" : "A handful to start exploring"}
        songs={songs}
        onRefresh={loadFeatured}
        emptyText="Try another mood."
      />

      <footer className="site-footer">
        <span className="dev-text" style={{ color: "var(--marigold)" }}>सुर साथी</span>
        {" "}· lyric &amp; mood based Hindi song recommendations
      </footer>
    </>
  );
}
