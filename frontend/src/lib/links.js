export function getYoutubeUrl(song) {
  const match = song.thumb && song.thumb.match(/\/vi\/([^/]+)\//);
  if (match) return `https://www.youtube.com/watch?v=${match[1]}`;
  return `https://www.youtube.com/results?search_query=${encodeURIComponent(
    `${song.name} ${song.singer}`
  )}`;
}

export function getSpotifyUrl(song) {
  return `https://open.spotify.com/search/${encodeURIComponent(
    `${song.name} ${song.singer}`
  )}`;
}