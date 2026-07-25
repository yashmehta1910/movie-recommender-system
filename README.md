# 🎬 CineMind AI

A Streamlit movie recommender that combines a content-based similarity model with live [TMDB](https://www.themoviedb.org/) metadata and a Gemini-powered conversational assistant — search, get recommendations, chat with an AI about what you're in the mood for, build a watchlist, and compare movies side by side.

## Features

- **Content-based recommendations** — pick a movie, get 5 similar titles from a precomputed similarity model
- **AI Movie Assistant** — chat naturally ("something darker", "more like this", "romantic comedy under 2 hours") powered by Gemini, with conversational memory and quick-reply suggestions
- **Live TMDB data** — posters, ratings, cast, director, runtime, trailers, and streaming availability, fetched and cached from TMDB
- **Watchlist** — save movies, rate them, export to CSV, and get recommendations based on your taste profile
- **Mood Finder** — browse by mood (Action, Romantic, Emotional, Sci-Fi, etc.)
- **Trending & Categories** — live trending titles this week, plus curated genre shelves
- **Top Picks** — a personalized, auto-refreshing shelf based on your watchlist's genres
- **Compare Movies** — side-by-side comparison of rating, year, and genres
- **Dark / Light theme** toggle, fast-load mode for quicker image loading

## Tech Stack

- [Streamlit](https://streamlit.io/) — UI framework
- [pandas](https://pandas.pydata.org/) — movie data and similarity matrix handling
- [requests](https://docs.python-requests.org/) — TMDB API calls, with retry/backoff via `urllib3`
- [google-genai](https://pypi.org/project/google-genai/) — Gemini API for the AI assistant
- [python-dotenv](https://pypi.org/project/python-dotenv/) — local environment variable loading
- [streamlit-autorefresh](https://pypi.org/project/streamlit-autorefresh/) *(optional)* — powers the auto-refreshing Top Picks shelf
- [aiohttp](https://docs.aiohttp.org/) *(optional)* — enables the faster async TMDB fetch path; the app automatically falls back to a thread pool if it isn't installed

## Project Structure

```
.
├── app.py              # Main Streamlit application
├── movie_list.pkl       # Pickled DataFrame of movies (id, title, tags, genres, etc.)
├── similarity.pkl       # Precomputed similarity matrix for the recommender
├── requirements.txt     # Python dependencies
├── .env                 # Local secrets (TMDB_API_KEY, GEMINI_API_KEY) — never commit this
└── .gitignore
```

> `movie_list.pkl` and `similarity.pkl` are the trained recommender's data files. If you don't have them, you'll need to generate them from a movie dataset (e.g. TMDB 5000) using a content-based similarity pipeline (CountVectorizer/TF-IDF + cosine similarity is the common approach) before the app will start.

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/yashmehta1910/movie-recommender-system
cd YOUR-REPO
```

**2. Create a virtual environment and install dependencies**
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**3. Set up your API keys**

Create a `.env` file in the project root:
```
TMDB_API_KEY=your_tmdb_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

- Get a free TMDB API key at [themoviedb.org/settings/api](https://www.themoviedb.org/settings/api) (required — the app won't start without it).
- Get a Gemini API key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey) (optional — only the AI Movie Assistant page needs it; the rest of the app works without it).

**4. Run it**
```bash
streamlit run app.py
```
The app opens at `http://localhost:8501`.

## Deployment

The easiest option is [Streamlit Community Cloud](https://share.streamlit.io) (free):

1. Push this repo to GitHub (see `.gitignore` below — don't commit `.env`).
2. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
3. Click **Create app** → select this repo, branch, and `app.py` as the entrypoint.
4. In the app's **Secrets** settings, add:
   ```toml
   TMDB_API_KEY = "your_real_key"
   GEMINI_API_KEY = "your_real_key"
   ```
5. Deploy. Future pushes to the branch auto-redeploy.

If `movie_list.pkl` / `similarity.pkl` are large (>50MB), use [Git LFS](https://git-lfs.github.com/) to track them before pushing.

## Notes

- TMDB responses are cached (6–24 hours depending on endpoint) to keep the app fast; a failed request is never cached, so a transient API hiccup retries fresh on the next lookup instead of getting "stuck."
- `TMDB_API_KEY` is required at startup. `GEMINI_API_KEY` is loaded lazily — the app runs fine without it, only the AI Assistant page will show an error until it's set.

---

Made with care by Yash Kumar Mehta