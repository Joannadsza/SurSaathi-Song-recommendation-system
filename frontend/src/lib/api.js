const BASE_URL = "http://localhost:5001/api";

async function get(path) {
  const res = await fetch(`${BASE_URL}${path}`);
  if (!res.ok) throw new Error(`Request failed: ${path}`);
  return res.json();
}

export const api = {
  moods: () => get("/moods"),
  featured: ({ mood, limit = 4 } = {}) => {
    const params = new URLSearchParams();
    if (mood) params.set("mood", mood);
    params.set("limit", limit);
    return get(`/songs/featured?${params.toString()}`);
  },
  search: (q, limit = 8) =>
    get(`/search?q=${encodeURIComponent(q)}&limit=${limit}`),
  songDetail: (id) => get(`/songs/${id}`),
  recommendations: (id, limit = 4) =>
    get(`/songs/${id}/recommendations?limit=${limit}`),
  karaoke: (id) => get(`/songs/${id}/karaoke`),
};
