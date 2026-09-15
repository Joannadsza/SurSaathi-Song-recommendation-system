import { Routes, Route } from "react-router-dom";
import Header from "./components/Header";
import Home from "./pages/Home";
import SongDetail from "./pages/SongDetail";
import Karaoke from "./pages/Karaoke";

export default function App() {
  return (
    <>
      <Header />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/song/:id" element={<SongDetail />} />
        <Route path="/song/:id/karaoke" element={<Karaoke />} />
      </Routes>
    </>
  );
}
