import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";

import SongRow from "../components/SongRow";

import { api } from "../lib/api";

import { MOOD_COLORS } from "../lib/moods";

import {
  getYoutubeUrl,
  getSpotifyUrl
} from "../lib/links";


export default function SongDetail() {

  const { id } = useParams();

  const navigate = useNavigate();


  // ============================================================
  // SONG DATA
  // ============================================================

  const [song, setSong] = useState(null);


  // ============================================================
  // RECOMMENDATIONS
  // ============================================================

  const [recs, setRecs] = useState({
    byLyrics: [],
    byMood: []
  });


  // ============================================================
  // ML MOOD
  // ============================================================

  const [prediction, setPrediction] = useState({
    predictedMood: null,
    confidence: null,
    model: null,
    modelReady: false
  });


  // ============================================================
  // LOADING STATE
  // ============================================================

  const [predictionLoading, setPredictionLoading] =
    useState(true);


  // ============================================================
  // LOAD SONG
  // ============================================================

  useEffect(() => {

    setSong(null);

    setPredictionLoading(true);

    setPrediction({
      predictedMood: null,
      confidence: null,
      model: null,
      modelReady: false
    });


    // ----------------------------------------------------------
    // Load song details
    // ----------------------------------------------------------

    api.songDetail(id)
      .then(setSong);


    // ----------------------------------------------------------
    // Load recommendations
    // ----------------------------------------------------------

    api.recommendations(
      id,
      4
    )
      .then((data) => {

        setRecs({

          byLyrics:
            data.byLyrics || [],

          byMood:
            data.byMood || []

        });


        // ------------------------------------------------------
        // Recommendation API already performs ML prediction.
        // Use its result immediately.
        // ------------------------------------------------------

        setPrediction({

          predictedMood:
            data.predictedMood || null,

          confidence:
            data.confidence || null,

          model:
            data.model || null,

          modelReady:
            data.modelReady || false

        });

        setPredictionLoading(false);

      })
      .catch(() => {

        setPredictionLoading(false);

      });


    window.scrollTo({
      top: 0
    });

  }, [id]);


  // ============================================================
  // LOADING
  // ============================================================

  if (!song) {

    return (
      <div className="empty-state">
        Loading...
      </div>
    );

  }


  // ============================================================
  // EXTERNAL LINKS
  // ============================================================

  const youtubeUrl =
    getYoutubeUrl(song);


  const spotifyUrl =
    getSpotifyUrl(song);


  // ============================================================
  // BUTTON STYLES
  // ============================================================

  const youtubeStyle = {
    background: "#E8433D"
  };


  const spotifyStyle = {
    background: "#3FC65B"
  };


  const actionsStyle = {

    display: "flex",

    gap: 10,

    flexWrap: "wrap",

    marginTop: 16

  };


  // ============================================================
  // ML MOOD DISPLAY
  // ============================================================

  const predictedMood =
    prediction.predictedMood;


  const confidence =
    prediction.confidence;


  return (

    <>

      {/* ======================================================
          SONG HEADER
      ======================================================= */}

      <div className="detail-top">

        <img
          src={song.thumb}
          alt={song.name}
        />


        <div className="detail-info">

          {/* --------------------------------------------------
              BACK BUTTON
          --------------------------------------------------- */}

          <div
            className="back"
            onClick={() => navigate("/")}
          >
            Back
          </div>


          {/* --------------------------------------------------
              SONG NAME
          --------------------------------------------------- */}

          <h1 className="display">
            {song.name}
          </h1>


          {/* --------------------------------------------------
              DEVANAGARI NAME
          --------------------------------------------------- */}

          <div className="namedev dev-text">
            {song.nameDev}
          </div>


          {/* --------------------------------------------------
              SONG INFORMATION
          --------------------------------------------------- */}

          <div className="meta">

            {song.singer}

            {" - "}

            {song.year}

            {" - "}

            {song.theme}

          </div>


          {/* --------------------------------------------------
              EXISTING MOOD TAGS
          --------------------------------------------------- */}

          <div className="tagrow">

            {song.moods.map(
              (mood) => (

                <span
                  key={mood}
                  className="tag"
                  style={{
                    borderColor:
                      (
                        MOOD_COLORS[mood] ||
                        "#ffffff"
                      ) + "66"
                  }}
                >

                  {mood}

                </span>

              )
            )}

          </div>


          {/* ==================================================
              ML PREDICTION
          =================================================== */}

          <div
            style={{
              marginTop: 18,
              padding: 14,
              borderRadius: 12,
              background:
                "rgba(255,255,255,0.05)",
              border:
                "1px solid rgba(255,255,255,0.10)"
            }}
          >

            <div
              style={{
                fontSize: 12,
                opacity: 0.7,
                marginBottom: 5
              }}
            >
              AI MOOD PREDICTION
            </div>


            {predictionLoading ? (

              <div>
                Analysing lyrics...
              </div>

            ) : predictedMood ? (

              <>

                <div
                  style={{
                    fontSize: 18,
                    fontWeight: 600
                  }}
                >
                  {predictedMood}
                </div>


                {confidence !== null && (

                  <div
                    style={{
                      marginTop: 5,
                      fontSize: 13,
                      opacity: 0.75
                    }}
                  >
                    Model confidence:{" "}
                    {confidence}%
                  </div>

                )}

              </>

            ) : (

              <div
                style={{
                  fontSize: 13,
                  opacity: 0.7
                }}
              >
                Mood prediction unavailable.
              </div>

            )}

          </div>


          {/* ==================================================
              ACTION BUTTONS
          =================================================== */}

          <div style={actionsStyle}>

            <a
              className="karaoke-btn"
              href={youtubeUrl}
              target="_blank"
              rel="noopener noreferrer"
              style={youtubeStyle}
            >
              Play on YouTube
            </a>


            <a
              className="karaoke-btn"
              href={spotifyUrl}
              target="_blank"
              rel="noopener noreferrer"
              style={spotifyStyle}
            >
              Play on Spotify
            </a>


            <button
              className="karaoke-btn"
              onClick={() =>
                navigate(
                  "/song/" +
                  song.id +
                  "/karaoke"
                )
              }
            >
              Karaoke lyrics
            </button>

          </div>

        </div>

      </div>


      {/* ======================================================
          LYRICS
      ======================================================= */}

      <div className="lyric-preview">
        {song.lyrics}
      </div>


      {/* ======================================================
          LYRIC-BASED RECOMMENDATIONS
      ======================================================= */}

      <SongRow
        title="More like this - lyrics"
        sub="Similar words and themes"
        songs={recs.byLyrics}
        emptyText="No close lyrical matches found."
      />


      {/* ======================================================
          MOOD-BASED RECOMMENDATIONS
      ======================================================= */}

      <SongRow
        title="More like this - mood"
        sub={
          predictedMood
            ? `Songs matching predicted mood: ${predictedMood}`
            : "Same feeling, different song"
        }
        songs={recs.byMood}
        emptyText="No mood matches found."
      />

    </>

  );

}