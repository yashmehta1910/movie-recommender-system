"""
CineMind AI — Movie Recommender
================================
A Streamlit app combining a content-based similarity model with live TMDB
metadata and Gemini-powered natural-language recommendations.

Refactored for correctness, performance, and maintainability. See the
accompanying code review for a full list of fixes applied here.
"""

from __future__ import annotations

import asyncio
import heapq
import json
import logging
import os
import pickle
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from google import genai
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

# ---------------------------------------------------------------------------
# Logging — never log secrets (API keys, full param dicts)
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cinemind")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TMDB_BASE = "https://api.themoviedb.org/3/movie/{}"
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_TRENDING_URL = "https://api.themoviedb.org/3/trending/movie/week"
IMAGE_BASE = "https://image.tmdb.org/t/p/{size}/{path}"
PLACEHOLDER_POSTER = "https://dummyimage.com/300x450/000/fff&text=No+Poster"
ERROR_POSTER = "https://dummyimage.com/300x450/000/fff&text=Error"
REFRESH_INTERVAL = 30  # seconds between auto-refresh cycles for Top Picks

MOOD_MAPPING = {
    "action": ["action"],
    "comedy": ["comedy"],
    "funny": ["comedy"],
    "romantic": ["romance"],
    "romance": ["romance"],
    "emotional": ["drama"],
    "sad": ["drama"],
    "scifi": ["science fiction"],
    "sci-fi": ["science fiction"],
    "horror": ["horror"],
    "adventure": ["adventure"],
}

CATEGORY_KEYWORDS = {
    "Action Picks": "action",
    "Sci-Fi Picks": "science fiction",
    "Comedy Picks": "comedy",
    "Drama Picks": "drama",
}

DEFAULT_SESSION_STATE = {
    "auto_refresh": True,
    "watchlist": [],
    "last_recommendations": [],
    "last_selected_movie": None,
    "last_movie_info": None,
    "last_confidence": None,
    "user_ratings": {},
    "search_history": [],
    "recommendation_history": [],
    "recommendation_count": 0,
    "personalized_recs": {},
    "feedback_data": {},
    "last_top_picks": None,
    "last_refresh_time": 0.0,
    "ai_chat_history": [],
    "quick_prompt": None,
}

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ru": "Russian"
}

# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

# ---------------------------------------------------------------------------
# Secrets — validated up front, before any UI depends on them
# ---------------------------------------------------------------------------
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if not TMDB_API_KEY:
    st.error("⚠️ TMDB_API_KEY is missing. Add it to your .env file before running the app.")
    st.stop()

_gemini_client: Optional[genai.Client] = None


def get_gemini_client() -> genai.Client:
    """Lazily create the Gemini client so a missing key only breaks the AI page."""
    global _gemini_client
    if _gemini_client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not configured.")
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client


# ---------------------------------------------------------------------------
# Data loading — happens before any UI is built, uses context managers,
# and pre-computes the lowercase tags column once instead of per-call.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_movie_data():
    base_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_path, "movie_list.pkl"), "rb") as f:
        movies_df = pickle.load(f)
    with open(os.path.join(base_path, "similarity.pkl"), "rb") as f:
        similarity_matrix = pickle.load(f)
    movies_df["tags_lower"] = movies_df.get("tags", "").astype(str).str.lower()
    return movies_df, similarity_matrix


try:
    movies, similarity = load_movie_data()
except Exception as e:
    st.error(f"⚠️ Failed to load model files: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# HTTP session with real connection-level retries (replaces the no-op
# `requests.adapters.DEFAULT_RETRIES = 2` and the manual sleep/backoff loop).
# No Streamlit calls happen in here, so it is safe to call from worker threads.
# ---------------------------------------------------------------------------
_session = requests.Session()
_retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
)
_session.mount("https://", HTTPAdapter(max_retries=_retry_strategy))
_session.mount("http://", HTTPAdapter(max_retries=_retry_strategy))


def _tmdb_get(url: str, params: dict, timeout: int = 15) -> Optional[requests.Response]:
    try:
        resp = _session.get(url, params=params, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200:
            return resp
        logger.warning("TMDB non-200 (%s) for %s", resp.status_code, url)
        return None
    except requests.exceptions.RequestException as e:
        logger.warning("TMDB request error for %s: %s", url, e)
        return None


class TMDBUnavailable(Exception):
    """Raised when a TMDB request fails even after _session's built-in
    retries. Every TMDB lookup below is split into a @st.cache_data-decorated
    'raw' function that raises this on failure, plus a thin public wrapper
    that catches it. st.cache_data only caches a function's return value —
    if it raises, nothing is cached — so a transient TMDB outage never gets
    "frozen" into the cache for hours; the very next request for the same
    movie retries TMDB instead of replaying a stale failure."""


# ---------------------------------------------------------------------------
# Session state init (includes last_selected_movie/last_movie_info/last_confidence,
# which the original app read without ever initializing)
# ---------------------------------------------------------------------------
for _key, _default in DEFAULT_SESSION_STATE.items():
    if _key not in st.session_state:
        st.session_state[_key] = _default

# ---------------------------------------------------------------------------
# Watchlist helpers — a single, consistent de-duplication strategy
# (the original app used three different, inconsistent checks)
# ---------------------------------------------------------------------------
def _movie_key(movie: dict) -> Any:
    return movie.get("movie_id", movie.get("id", movie.get("title")))


def is_in_watchlist(movie: dict) -> bool:
    key = _movie_key(movie)
    return any(_movie_key(m) == key for m in st.session_state.watchlist)


def add_to_watchlist(movie: dict) -> bool:
    """Returns True if the movie was added, False if it was already saved."""
    if is_in_watchlist(movie):
        return False
    st.session_state.watchlist.append(movie)
    return True


def remove_from_watchlist(idx: int) -> None:
    if 0 <= idx < len(st.session_state.watchlist):
        st.session_state.watchlist.pop(idx)


# ---------------------------------------------------------------------------
# Genre / taste helpers
# ---------------------------------------------------------------------------
def get_watchlist_genre_counts() -> dict[str, int]:
    """Shared by every view that needs a genre breakdown of the watchlist
    (favorite-genre lookup, the taste-profile chart, the watchlist page) —
    previously each of those re-implemented this loop independently."""
    genre_count: dict[str, int] = {}
    for movie in st.session_state.watchlist:
        for genre in str(movie.get("genres", "")).split(","):
            genre = genre.strip()
            if genre:
                genre_count[genre] = genre_count.get(genre, 0) + 1
    return genre_count


def get_watchlist_ratings() -> list[float]:
    """Numeric ratings from the watchlist, skipping any missing/'N/A' values.
    Shared by every view that shows an average rating."""
    ratings = []
    for movie in st.session_state.watchlist:
        try:
            ratings.append(float(movie["rating"]))
        except (ValueError, TypeError, KeyError):
            pass
    return ratings


def get_user_favorite_genres() -> list[str]:
    genre_count = get_watchlist_genre_counts()
    return sorted(genre_count, key=genre_count.get, reverse=True)[:3]


def get_recommendation_reason(movie: dict) -> str:
    favorite_genres = get_user_favorite_genres()
    matched = [g.strip() for g in str(movie.get("genres", "")).split(",") if g.strip() in favorite_genres]
    if matched:
        return "Matches your interest in " + ", ".join(matched)
    return "Recommended based on similarity analysis"


def generate_taste_summary() -> str:
    favorite_genres = get_user_favorite_genres()
    if not favorite_genres:
        return "Start adding movies to build your profile."
    top_genres = ", ".join(favorite_genres[:3])
    return f"You enjoy {top_genres} movies. Your watchlist suggests a strong interest in these genres."


def _genre_filtered_ids(keywords: list[str]) -> list[int]:
    """Vectorized tag filter — replaces the original movies.iterrows() scans,
    which were O(n) pandas row-iterations run on every button click."""
    keywords = [k for k in keywords if k]
    if not keywords:
        return []
    pattern = "|".join(re.escape(k.lower()) for k in keywords)
    mask = movies["tags_lower"].str.contains(pattern, regex=True, na=False)
    return movies.loc[mask, "movie_id"].astype(int).tolist()


def get_movies_by_mood(mood: str) -> list[int]:
    target_genres = MOOD_MAPPING.get(mood.lower(), [])
    ids = _genre_filtered_ids(target_genres)
    random.shuffle(ids)
    return ids[:6]


def get_personalized_movie_ids() -> list[int]:
    ids = _genre_filtered_ids(get_user_favorite_genres())
    random.shuffle(ids)
    return ids[:6]


def get_recommendation_categories() -> dict[str, list[int]]:
    return {label: _genre_filtered_ids([kw]) for label, kw in CATEGORY_KEYWORDS.items()}


def sample_random_movie_ids(k: int = 6) -> list[int]:
    """Guarded against datasets smaller than k (original code would raise ValueError)."""
    k = min(k, len(movies))
    if k <= 0:
        return []
    indices = random.sample(range(len(movies)), k)
    return [int(movies.iloc[i].movie_id) for i in indices]


# ---------------------------------------------------------------------------
# TMDB detail fetching — one shared parser used by every fetch path
# (previously duplicated between the sync and async/thread paths).
#
# Every @st.cache_data function here follows the same "raw + wrapper" shape:
# the raw function raises TMDBUnavailable on failure (so nothing bad gets
# cached), and the public wrapper catches it and returns the same fallback
# value the original app always returned on failure. Callers don't need to
# know any of this — signatures and return values are unchanged.
# ---------------------------------------------------------------------------
@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def get_trailer_url(movie_id: int, title: str) -> str:
    resp = _tmdb_get(f"https://api.themoviedb.org/3/movie/{movie_id}/videos",
                      {"api_key": TMDB_API_KEY, "language": "en-US"})
    if resp:
        videos = resp.json().get("results", [])
        trailer = next((v for v in videos if v["type"] == "Trailer" and v["site"] == "YouTube"), None)
        if trailer:
            return f"https://www.youtube.com/watch?v={trailer['key']}"
    # A missing trailer (or a failed request) both resolve to the same
    # perfectly valid fallback link, so there's nothing to protect from
    # caching here — this value is safe to cache either way.
    safe_query = title.replace(" ", "+")
    return f"https://www.youtube.com/results?search_query={safe_query}+trailer"


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def _raw_movie_credits(movie_id: int) -> Optional[dict]:
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits"
    response = _tmdb_get(url, {"api_key": TMDB_API_KEY})
    if not response:
        raise TMDBUnavailable(f"credits request failed for movie {movie_id}")
    return response.json()


def get_movie_credits(movie_id: int) -> Optional[dict]:
    try:
        return _raw_movie_credits(movie_id)
    except TMDBUnavailable:
        return None


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def _raw_watch_providers(movie_id: int) -> list[str]:
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/watch/providers"
    response = _tmdb_get(url, {"api_key": TMDB_API_KEY})
    if not response:
        raise TMDBUnavailable(f"watch-providers request failed for movie {movie_id}")
    data = response.json()
    country = "IN"
    if country not in data.get("results", {}):
        return []  # genuinely no providers for this country — safe to cache
    provider_list = data["results"][country].get("flatrate", [])
    return [provider["provider_name"] for provider in provider_list]


def get_watch_providers(movie_id: int) -> list[str]:
    try:
        return _raw_watch_providers(movie_id)
    except TMDBUnavailable:
        return []


def _fetch_movie_enrichment(movie_id: int, title: str) -> tuple[str, Optional[dict], list[str]]:
    """Trailer, credits, and watch-providers are three independent TMDB
    requests — the original app made them one after another. Running them
    concurrently is a pure latency win (same three results, same cache
    behavior) since none of them depends on another's output."""
    with ThreadPoolExecutor(max_workers=3) as ex:
        trailer_future = ex.submit(get_trailer_url, movie_id, title)
        credits_future = ex.submit(get_movie_credits, movie_id)
        providers_future = ex.submit(get_watch_providers, movie_id)
        return trailer_future.result(), credits_future.result(), providers_future.result()


def _parse_tmdb_movie(data: Optional[dict], movie_id: int, fast_mode: bool) -> dict:
    if not data:
        return {
            "id": movie_id, "title": "Error", "poster": ERROR_POSTER, "rating": "N/A",
            "year": "----", "genres": "N/A", "overview": "", "movie_id": movie_id,
            "trailer_url": "https://www.themoviedb.org",
        }
    poster_path = data.get("poster_path")
    img_size = "w300" if fast_mode else "w500"
    poster_url = IMAGE_BASE.format(size=img_size, path=poster_path) if poster_path else PLACEHOLDER_POSTER
    title = data.get("title", "Unknown Title")
    release_date = data.get("release_date", "")

    trailer_url, credits, providers = _fetch_movie_enrichment(movie_id, title)

    director = "Unknown"
    cast = []
    if credits:
        crew = credits.get("crew", [])
        for person in crew:
            if person.get("job") == "Director":
                director = person.get("name", "Unknown")
                break
        cast = [actor.get("name", "") for actor in credits.get("cast", [])[:3]]

    return {
        "id": data.get("id", movie_id),
        "title": title,
        "poster": poster_url,
        "rating": data.get("vote_average", "N/A"),
        "year": release_date[:4] if release_date else "----",
        "genres": ", ".join(g["name"] for g in data.get("genres", [])) or "N/A",
        "overview": data.get("overview", ""),
        "movie_id": movie_id,
        "trailer_url": trailer_url,
        "providers": providers,
        "director": director,
        "cast": cast,
        "runtime": data.get("runtime"),
        "language": LANGUAGE_NAMES.get(
            data.get("original_language"),
            data.get("original_language", "Unknown")
        ),
    }


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def _raw_movie_details(movie_id: int, fast_mode: bool) -> dict:
    resp = _tmdb_get(TMDB_BASE.format(movie_id), {"api_key": TMDB_API_KEY, "language": "en-US"})
    if resp is None:
        raise TMDBUnavailable(f"details request failed for movie {movie_id}")
    return _parse_tmdb_movie(resp.json(), movie_id, fast_mode)


def cached_fetch_movie_details(movie_id: int, fast_mode: bool = True) -> dict:
    """Pure data function — no st.* calls, safe to call from cache or threads."""
    try:
        return _raw_movie_details(movie_id, fast_mode)
    except TMDBUnavailable:
        return _parse_tmdb_movie(None, movie_id, fast_mode)


def fetch_multiple_movie_details(movie_ids: list[int], fast_mode: bool = True) -> list[dict]:
    """Best-effort parallel fetch. Tries asyncio/aiohttp first; falls back to a
    thread pool. Both paths funnel through the same _parse_tmdb_movie parser,
    and _parse_tmdb_movie itself now fetches each movie's trailer/credits/
    providers concurrently, so a shelf of 6 movies costs roughly one round
    trip's worth of latency instead of stacking sequentially."""
    if not movie_ids:
        return []
    try:
        import aiohttp

        async def _aio_fetch_many(ids: list[int]) -> list[Optional[dict]]:
            timeout = aiohttp.ClientTimeout(total=8)
            sem = asyncio.Semaphore(8)
            results: list[Optional[dict]] = [None] * len(ids)

            async def _fetch_one(idx: int, mid: int, session: aiohttp.ClientSession):
                url = TMDB_BASE.format(mid)
                params = {"api_key": TMDB_API_KEY, "language": "en-US"}
                backoff = 0.6
                async with sem:
                    for _ in range(3):
                        try:
                            async with session.get(url, params=params) as resp:
                                if resp.status == 200:
                                    results[idx] = await resp.json()
                                    return
                        except Exception as e:
                            logger.warning("Async TMDB fetch retry for %s: %s", mid, e)
                            await asyncio.sleep(backoff)
                            backoff *= 2

            async with aiohttp.ClientSession(timeout=timeout) as session:
                await asyncio.gather(*(_fetch_one(i, m, session) for i, m in enumerate(ids)))
            return results

        raw_list = asyncio.run(_aio_fetch_many(movie_ids))
        # Parsing each movie also does 3 concurrent sub-requests (see
        # _fetch_movie_enrichment) — running the parses themselves in
        # parallel too means a shelf's total wait is roughly one enrichment
        # round-trip rather than N of them stacked back to back.
        with ThreadPoolExecutor(max_workers=6) as ex:
            futures = [ex.submit(_parse_tmdb_movie, data, mid, fast_mode) for data, mid in zip(raw_list, movie_ids)]
            return [f.result() for f in futures]

    except (ImportError, RuntimeError) as e:
        logger.info("Falling back to thread pool for TMDB fetch: %s", e)
        with ThreadPoolExecutor(max_workers=6) as ex:
            futures = [ex.submit(cached_fetch_movie_details, mid, fast_mode) for mid in movie_ids]
            return [f.result() for f in futures]


@st.cache_data(ttl=21600, show_spinner=False)
def _raw_search_movie(movie_title: str, year: Optional[int], fast_mode: bool) -> Optional[dict]:
    base_params = {"api_key": TMDB_API_KEY, "query": movie_title}
    if year:
        base_params["primary_release_year"] = year

    resp = _tmdb_get(TMDB_SEARCH_URL, base_params)
    if resp is None:
        raise TMDBUnavailable(f"search request failed for '{movie_title}'")
    results = resp.json().get("results", [])

    if not results and year:
        resp = _tmdb_get(TMDB_SEARCH_URL, {"api_key": TMDB_API_KEY, "query": movie_title})
        if resp is None:
            raise TMDBUnavailable(f"search retry failed for '{movie_title}'")
        results = resp.json().get("results", [])

    if not results:
        return None  # genuinely no match — safe to cache

    exact_matches = [r for r in results if r["title"].lower().strip() == movie_title.lower().strip()]
    candidates = exact_matches if exact_matches else results

    if year:
        def year_diff(r):
            rd = r.get("release_date", "")
            try:
                return abs(int(rd[:4]) - int(year))
            except (ValueError, TypeError):
                return 999
        candidates = sorted(candidates, key=lambda r: (year_diff(r), -r.get("popularity", 0)))
    else:
        candidates = sorted(candidates, key=lambda r: -r.get("popularity", 0))

    movie_data = cached_fetch_movie_details(candidates[0]["id"], fast_mode)
    if movie_data.get("title") == "Error":
        # The search succeeded but the follow-up details lookup didn't —
        # don't let a good search match get cached pointing at a dead result.
        raise TMDBUnavailable(f"details lookup failed for search match '{movie_title}'")
    return movie_data


def search_movie_by_title(movie_title: str, year: Optional[int] = None, fast_mode: bool = True) -> Optional[dict]:
    try:
        return _raw_search_movie(movie_title, year, fast_mode)
    except TMDBUnavailable:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def _raw_trending_movies(fast_mode: bool) -> list[dict]:
    resp = _tmdb_get(TMDB_TRENDING_URL, {"api_key": TMDB_API_KEY})
    if resp is None:
        raise TMDBUnavailable("trending request failed")
    data = resp.json()
    movie_ids = [m["id"] for m in data.get("results", [])[:10]]
    return fetch_multiple_movie_details(movie_ids, fast_mode)


def get_trending_movies(fast_mode: bool = True) -> list[dict]:
    try:
        return _raw_trending_movies(fast_mode)
    except TMDBUnavailable:
        return []


# ---------------------------------------------------------------------------
# Similarity-based recommendation engine
# ---------------------------------------------------------------------------
@st.cache_data(ttl=60 * 30, show_spinner=False)
def recommend_cached(movie: str) -> list[tuple[int, float]]:
    if movie not in movies["title"].values:
        return []
    index = movies[movies["title"] == movie].index[0]
    # Only the top 5 matches (after the movie itself) are ever used, so
    # sorting the entire similarity row is wasted work on a large catalog —
    # heapq.nlargest finds just the top 6 in O(n log 6) instead of O(n log n).
    top_matches = heapq.nlargest(6, enumerate(similarity[index]), key=lambda x: x[1])
    recs = []
    for i, score in top_matches[1:]:
        movie_id = int(movies.iloc[i].movie_id)
        recs.append((movie_id, round(score * 100, 1)))
    return recs


def recommend(movie: str, fast_mode: bool) -> list[dict]:
    # fast_mode is now part of the cache key — previously toggling it after a
    # recommendation was cached silently left stale poster resolutions in place.
    cache_key = (movie, tuple(get_user_favorite_genres()), fast_mode)
    if cache_key in st.session_state.personalized_recs:
        return st.session_state.personalized_recs[cache_key]

    recs = recommend_cached(movie)
    if not recs:
        return []

    movie_ids = [item[0] for item in recs]
    details = fetch_multiple_movie_details(movie_ids, fast_mode)
    for detail, rec in zip(details, recs):
        detail["similarity_score"] = rec[1]

    st.session_state.personalized_recs[cache_key] = details
    return details


def get_watchlist_based_recommendations(fast_mode: bool) -> tuple[Optional[str], list[dict]]:
    if not st.session_state.watchlist:
        return None, []
    source_movie = max(
        st.session_state.watchlist,
        key=lambda x: float(x["rating"]) if x.get("rating") not in (None, "N/A") else 0,
    )
    return source_movie["title"], recommend(source_movie["title"], fast_mode)


def generate_ai_explanation(movie_info: dict, confidence: int) -> str:
    genres = [g.strip() for g in str(movie_info.get("genres", "")).split(",")]
    top_genres = ", ".join(genres[:3])
    overview = (movie_info.get("overview") or "")[:150]
    return f"""
These picks are based on **{movie_info['title']}**, which combines
**{top_genres}** elements that often appeal to viewers with similar tastes.

*{overview}...*

The engine compared genre relationships and narrative patterns across
the catalog to build this shortlist.
"""

def get_recommendation_count(user_prompt: str) -> int:
    """Extract the number of requested movies. Defaults to 3, clamped 1-10."""
    match = re.search(r"\b(\d+)\b", user_prompt)
    if match:
        return max(1, min(int(match.group(1)), 10))
    return 3


def build_conversation_history() -> str:
    history = ""
    for msg in st.session_state.ai_chat_history:
        if msg["role"] == "user":
            history += f"User: {msg['content']}\n"
        else:
            try:
                titles = ", ".join(movie["title"] for movie in json.loads(msg["content"]))
                history += f"Assistant recommended: {titles}\n"
            except json.JSONDecodeError:
                history += f"Assistant: {msg['content']}\n"
    return history


def get_previous_recommendations() -> list[str]:
    previous = []
    for message in st.session_state.ai_chat_history:
        if message["role"] != "assistant":
            continue
        try:
            previous.extend(movie["title"] for movie in json.loads(message["content"]) if movie.get("title"))
        except json.JSONDecodeError:
            pass
    return previous


AI_SUGGESTIONS = [
    "👍 More Like This",
    "😂 Funnier",
    "💀 Darker",
    "❤️ Romantic",
    "🎲 Surprise Me",
]

GENRE_KEYWORDS = [
    "action", "comedy", "romance", "drama", "thriller", "horror", "animation",
    "science fiction", "sci-fi", "adventure", "crime", "mystery", "fantasy", "family",
]


def extract_filters(user_prompt: str) -> dict:
    filters = {
        "year_from": None, "year_to": None, "min_rating": None,
        "genres": [], "exclude_genres": [], "runtime": None,
    }
    prompt = user_prompt.lower()

    years = re.findall(r"\b(19\d{2}|20\d{2})\b", prompt)
    if len(years) == 1:
        filters["year_from"] = int(years[0])
    elif len(years) >= 2:
        filters["year_from"] = int(years[0])
        filters["year_to"] = int(years[1])

    rating = re.search(r"(above|over|greater than)\s*(\d(?:\.\d)?)", prompt)
    if rating:
        filters["min_rating"] = float(rating.group(2))

    runtime = re.search(r"under\s*(\d+)\s*(minutes|min)", prompt)
    if runtime:
        filters["runtime"] = int(runtime.group(1))

    filters["genres"] = [g for g in GENRE_KEYWORDS if g in prompt]
    filters["exclude_genres"] = [g for g in GENRE_KEYWORDS if f"no {g}" in prompt]
    return filters


def ask_gemini(user_prompt: str) -> Optional[list[dict]]:
    recommendation_count = get_recommendation_count(user_prompt)
    history = build_conversation_history()
    filters = extract_filters(user_prompt)
    previous_movies = get_previous_recommendations()

    prompt = f"""
You are CineMind AI.

The user wants movie recommendations.\

Conversation History:

{history}

Previously Recommended Movies:

{", ".join(previous_movies)}

Detected Filters:

{json.dumps(filters, indent=2)}

Latest User Request:

{user_prompt}

Recommend exactly {recommendation_count} REAL movies.

Return ONLY valid JSON.

Format:
[
  {{
    "title": "...",
    "year": 2024,
    "reason": "..."
  }}
]

Rules:
- "year" must be the movie's actual release year (integer), your best knowledge.
- No markdown, no code blocks, no explanations outside JSON.
- If the user's message is a short follow-up such as "More Like This", "Darker", "Funnier", "Romantic", or "Surprise Me", interpret it using the previous conversation instead of treating it as a completely new request.
- Never recommend a movie that already appears in
  "Previously Recommended Movies" unless the user
  explicitly asks for it again.

- Follow the conversation naturally.

- If the user says:
  "More Like This"
  "Darker"
  "Funnier"
  "Romantic"
  "Surprise Me"

  treat it as a continuation of the previous conversation.
- "year" must be the actual release year.

"""
    try:
        response = get_gemini_client().models.generate_content(model=GEMINI_MODEL, contents=prompt)
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw.strip())
        if not isinstance(parsed, list):
            return None

        movie_list = []
        for item in parsed:
            if not isinstance(item, dict) or "title" not in item:
                continue
            movie_data = search_movie_by_title(item["title"], item.get("year"), fast_mode)
            if movie_data:
                movie_data["reason"] = item.get("reason", "")
                movie_list.append(movie_data)

        # Save AI reply into conversation history
        st.session_state.ai_chat_history.append({
            "role": "assistant",
            "content": json.dumps(movie_list),
        })
        return movie_list
    except Exception as e:
        logger.warning("Gemini request failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Netflix-style card row (renders inside an isolated iframe via components.html
# — note that top-level page CSS cannot target .movie-card/.movie-info, since
# they live in a different document. All styling for cards lives in here.)
# ---------------------------------------------------------------------------
def generate_movie_row(movie_list: list[dict], row_id: str = "row1", theme_mode: str = "Dark") -> str:
    # This markup renders inside an isolated iframe (components.html), so it
    # carries its own copy of the design tokens — it cannot inherit the CSS
    # variables set on the top-level page by inject_theme_css().
    if theme_mode == "Light":
        page_bg = "#f4f5f7"
        card_bg, border, border_hover = "#ffffff", "rgba(20,23,30,0.10)", "#a9761f"
        info_gradient, info_text, meta_text = "linear-gradient(to top, rgba(255,255,255,0.97) 45%, rgba(255,255,255,0))", "#14171e", "#5b6270"
        scroll_btn_bg, scroll_btn_color = "rgba(255,255,255,0.92)", "#14171e"
        accent, accent_text = "#a9761f", "#fffaf0"
        card_shadow = "0 10px 26px rgba(20,23,30,0.10)"
    else:
        page_bg = "#0a0c10"
        card_bg, border, border_hover = "#14171e", "rgba(255,255,255,0.08)", "#e3b23c"
        info_gradient, info_text, meta_text = "linear-gradient(to top, rgba(10,12,16,0.96) 45%, rgba(10,12,16,0))", "#f2f3f5", "#9aa1ac"
        scroll_btn_bg, scroll_btn_color = "rgba(20,23,30,0.85)", "#f2f3f5"
        accent, accent_text = "#e3b23c", "#171207"
        card_shadow = "0 10px 26px rgba(0,0,0,0.5)"

    html = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,560&family=Inter:wght@400;500;600;700&display=swap');
        :root {{
            --page-bg: {page_bg}; --card-bg: {card_bg}; --border: {border}; --border-hover: {border_hover};
            --card-shadow: {card_shadow}; --info-gradient: {info_gradient}; --info-text: {info_text};
            --meta-text: {meta_text}; --scroll-btn-bg: {scroll_btn_bg}; --scroll-btn-color: {scroll_btn_color};
            --accent: {accent}; --accent-text: {accent_text};
        }}
        * {{ box-sizing: border-box; }}
        body {{ background: var(--page-bg); margin:0; padding:0; font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial; }}
        .movie-row {{ display: flex; gap: 16px; overflow-x: auto; padding: 6px 4px 20px 4px; scroll-behavior: smooth; scrollbar-width: none; }}
        .movie-row::-webkit-scrollbar {{ display: none; }}
        .movie-card {{ position: relative; flex: 0 0 auto; width: 188px; height: 282px; border-radius: 12px;
            overflow: hidden; cursor: pointer; transition: transform 0.25s ease, box-shadow 0.25s ease;
            box-shadow: var(--card-shadow); background: var(--card-bg); border: 1px solid var(--border); }}
        .movie-card:hover {{ transform: translateY(-4px) scale(1.03); z-index: 5; border-color: var(--border-hover); }}
        .movie-poster {{ width: 100%; height: 100%; object-fit: cover; display:block; }}
        .movie-info {{ position: absolute; bottom: 0; left: 0; width: 100%; height: 62%; background: var(--info-gradient);
            color: var(--info-text); opacity: 0; transition: opacity 0.25s ease; padding: 12px;
            display: flex; flex-direction: column; justify-content: flex-end; }}
        .movie-card:hover .movie-info {{ opacity: 1; }}
        .movie-title {{ font-family: 'Inter', sans-serif; font-weight:600; font-size:13.5px; line-height:1.25;
            color: var(--info-text); display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; }}
        .scroll-btn {{ position: absolute; top: 50%; transform: translateY(-50%); background: var(--scroll-btn-bg);
            color: var(--scroll-btn-color); border: 1px solid var(--border); padding: 8px 10px; border-radius: 50%; cursor: pointer;
            font-size: 15px; z-index: 10; transition: background 0.2s, color 0.2s; }}
        .scroll-btn:hover {{ background: var(--accent); color: var(--accent-text); }}
        .scroll-left {{ left: 2px; }} .scroll-right {{ right: 2px; }}
        .scroll-container {{ position: relative; }}
        .trailer-btn {{ margin-top: 8px; display: inline-flex; align-items:center; gap:5px; background: var(--accent); color: var(--accent-text);
            padding: 5px 10px; border-radius: 999px; text-decoration: none; font-size: 11px; font-weight: 700;
            opacity: 0; transform: translateY(4px); transition: opacity 0.25s ease, transform 0.25s ease; width: fit-content; }}
        .movie-card:hover .trailer-btn {{ opacity: 1; transform: translateY(0); }}
        .meta-small {{ font-size: 11.5px; color: var(--meta-text); display:block; margin-top:5px; }}
        .rating-chip {{ position:absolute; top:10px; right:10px; background: rgba(20,23,30,0.72); color:#f5c655;
            font-size:11px; font-weight:700; padding:3px 8px; border-radius:999px; backdrop-filter: blur(2px); }}
    </style>
    <div class="scroll-container">
        <button class="scroll-btn scroll-left" onclick="document.getElementById('{row_id}').scrollBy(-300,0)">&#8249;</button>
        <div class="movie-row" id="{row_id}">
    """
    for movie in movie_list:
        tmdb_link = f"https://www.themoviedb.org/search?query={movie['title'].replace(' ', '+')}"
        trailer_link = movie.get("trailer_url") or f"https://themoviedb.org/movie/{movie.get('id', movie.get('movie_id', ''))}"
        html += f"""
        <a href="{tmdb_link}" target="_blank" style="text-decoration:none;">
            <div class="movie-card">
                <img src="{movie['poster']}" class="movie-poster">
                <span class="rating-chip">&#9733; {movie['rating']}</span>
                <div class="movie-info">
                    <span class="movie-title">{movie['title']}</span>
                    <small class="meta-small">{movie['year']} &middot; {movie['genres']}</small>
                    <a href="{trailer_link}" target="_blank" class="trailer-btn">&#9654; Trailer</a>
                </div>
            </div>
        </a>
        """
    html += f"""
        </div>
        <button class="scroll-btn scroll-right" onclick="document.getElementById('{row_id}').scrollBy(300,0)">&#8250;</button>
    </div>
    """
    return html


def render_movie_details(movie, show_overview=False, show_reason=False):
    st.subheader(movie["title"])

    try:
        rating = round(float(movie["rating"]), 1)
    except (ValueError, TypeError):
        rating = "N/A"

    meta_parts = [f"★ {rating}", str(movie.get("year", ""))]
    if movie.get("runtime"):
        meta_parts.append(f"{movie['runtime']} min")
    if movie.get("language"):
        meta_parts.append(movie["language"])
    st.caption("  ·  ".join(p for p in meta_parts if p))

    render_chip_row([g.strip() for g in str(movie.get("genres", "")).split(",") if g.strip()])

    if movie.get("director"):
        st.markdown(f"**Director:** {movie['director']}")

    if movie.get("cast"):
        st.markdown(f"**Cast:** {', '.join(movie['cast'])}")

    if show_overview and movie.get("overview"):
        st.caption(movie["overview"][:250] + "...")

    if show_reason and movie.get("reason"):
        st.caption(f"Why this pick — {movie['reason']}")

    st.markdown(f"[Watch trailer ↗]({movie['trailer_url']})")

    if movie.get("providers"):
        st.markdown(f"**Streaming on:** {' · '.join(movie['providers'])}")
    elif show_reason:
        st.caption("Streaming availability unknown.")

# ---------------------------------------------------------------------------
# Theming
# ---------------------------------------------------------------------------
def st_html(content: str, markdown_fn=None) -> None:
    """Render raw HTML/CSS via Streamlit's dedicated st.html() API.

    st.markdown() always runs its input through a Markdown/CommonMark parser
    first, even with unsafe_allow_html=True — and that parser has real traps
    for hand-written CSS: 4-space-indented lines become literal code blocks,
    and HTML blocks silently end at the first blank line, dumping the rest
    as visible plain text. st.html() injects markup directly with no
    Markdown pass, which is exactly what page-wide CSS needs. Falls back to
    a flattened st.markdown() call on Streamlit versions without st.html().
    Pass markdown_fn=st.sidebar to target the sidebar instead of the body.
    """
    flat = "\n".join(line.strip() for line in content.strip("\n").splitlines() if line.strip())
    target = markdown_fn or st
    if hasattr(target, "html"):
        target.html(flat)
    else:
        target.markdown(flat, unsafe_allow_html=True)


def inject_theme_css(theme_mode: str) -> None:
    """Injects the CineMind design system.

    A restrained 'cinema marquee' identity: a warm single amber accent, a
    serif display face used only for titles, and a clean sans for everything
    else. Dark is the primary, fully-designed mode; Light follows the same
    tokens so toggling never feels like a different app. Replaces the
    original's scattered, drifting inline styles with one consistent set of
    tokens consumed by every page.
    """
    if theme_mode == "Light":
        c = {
            "bg": "#f4f5f7", "bg_card": "#ffffff", "bg_hover": "#eceef2",
            "border": "rgba(20,23,30,0.10)", "border_strong": "rgba(20,23,30,0.20)",
            "text": "#14171e", "text_muted": "#5b6270", "text_faint": "#8b93a1",
            "accent": "#a9761f", "accent_strong": "#8a5f16", "accent_text_on": "#fffaf0",
            "accent_soft": "rgba(169,118,31,0.10)",
            "shadow": "0 10px 28px rgba(20,23,30,0.08)",
            "sidebar_bg": "#eef0f3", "glow": "rgba(169,118,31,0.05)",
        }
    else:
        c = {
            "bg": "#0a0c10", "bg_card": "#14171e", "bg_hover": "#1c2029",
            "border": "rgba(255,255,255,0.08)", "border_strong": "rgba(255,255,255,0.18)",
            "text": "#f2f3f5", "text_muted": "#9aa1ac", "text_faint": "#656d78",
            "accent": "#e3b23c", "accent_strong": "#f5c655", "accent_text_on": "#171207",
            "accent_soft": "rgba(227,178,60,0.14)",
            "shadow": "0 14px 34px rgba(0,0,0,0.45)",
            "sidebar_bg": "#0d1016", "glow": "rgba(227,178,60,0.07)",
        }

    st_html(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,450;9..144,560;9..144,620&family=Inter:wght@400;500;600;700&display=swap');
        :root {{
            --bg: {c['bg']}; --bg-card: {c['bg_card']}; --bg-hover: {c['bg_hover']};
            --border: {c['border']}; --border-strong: {c['border_strong']};
            --text: {c['text']}; --text-muted: {c['text_muted']}; --text-faint: {c['text_faint']};
            --accent: {c['accent']}; --accent-strong: {c['accent_strong']}; --accent-text-on: {c['accent_text_on']};
            --accent-soft: {c['accent_soft']}; --shadow: {c['shadow']}; --sidebar-bg: {c['sidebar_bg']};
            --radius: 12px;
            --font-display: 'Fraunces', Georgia, serif;
            --font-ui: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}

        html, body, .stApp {{ font-family: var(--font-ui); }}
        .stApp {{
            background:
                radial-gradient(1100px 460px at 10% -8%, {c['glow']}, transparent 60%),
                var(--bg) !important;
        }}
        .block-container {{ padding-top: 3.2rem; padding-bottom: 3rem; max-width: 1180px; }}
        h1, h2, h3 {{ font-family: var(--font-display) !important; font-weight: 560 !important;
            color: var(--text) !important; letter-spacing: -0.01em; }}
        p, span, label, div {{ color: var(--text); }}
        a {{ color: var(--accent); }}
        [data-testid="stCaptionContainer"], .stCaption, small {{ color: var(--text-muted) !important; }}

        /* Hero — centered, title a touch larger */
        .cm-hero {{ display:flex; flex-direction:column; align-items:center; text-align:center;
            gap:8px; padding: 10px 0 24px 0; border-bottom: 1px solid var(--border); margin-bottom: 26px; }}
        .cm-mark {{ width:44px; height:44px; border-radius:10px; flex-shrink:0;
            background: linear-gradient(155deg, var(--accent), var(--accent-strong));
            display:flex; align-items:center; justify-content:center;
            font-family: var(--font-display); font-weight:600; font-size:20px; color: var(--accent-text-on); }}
        .cm-hero h1 {{ font-size: 2.1rem; margin: 4px 0 0 0; line-height:1.15; }}
        .cm-hero p {{ margin: 2px 0 0 0; color: var(--text-muted) !important; font-size: 0.9rem; }}

        /* Signature: amber marquee eyebrow + serif section title */
        .eyebrow {{ display:inline-flex; align-items:center; gap:7px; font-family: var(--font-ui);
            font-size: 11px; font-weight:600; letter-spacing: 0.14em; text-transform: uppercase;
            color: var(--accent) !important; margin-bottom: 6px; }}
        .eyebrow::before {{ content:""; width:6px; height:6px; border-radius:50%; background: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-soft); }}
        .cm-section-head {{ margin: 8px 0 14px 0; }}
        .cm-section-head h2 {{ font-size: 1.3rem; margin: 0; }}
        .cm-section-head .section-subtitle {{ color: var(--text-muted) !important; font-size: 0.87rem; margin: 4px 0 0 0; }}

        /* Stat pills — replace st.info() used as plain data display */
        .stat-row {{ display:flex; gap:12px; flex-wrap:wrap; margin: 2px 0 8px 0; }}
        .stat-pill {{ flex:1; min-width:140px; background: var(--bg-card); border:1px solid var(--border);
            border-radius: var(--radius); padding: 12px 16px; }}
        .stat-pill .stat-value {{ display:block; font-family: var(--font-display); font-size: 1.25rem; color: var(--text); }}
        .stat-pill .stat-label {{ display:block; font-size: 10.5px; text-transform:uppercase; letter-spacing:0.08em;
            color: var(--text-muted); margin-top:3px; }}

        /* Surfaces */
        [data-testid="stVerticalBlockBorderWrapper"] {{ background: var(--bg-card); border:1px solid var(--border) !important;
            border-radius: var(--radius) !important; }}
        [data-testid="stExpander"] {{ background: var(--bg-card); border:1px solid var(--border) !important;
            border-radius: var(--radius) !important; overflow:hidden; }}
        [data-testid="stExpander"] summary {{ font-family: var(--font-ui); color: var(--text) !important; }}

        /* Metrics */
        [data-testid="stMetric"] {{ background: var(--bg-card); border:1px solid var(--border);
            border-radius: var(--radius); padding: 14px 16px; }}
        [data-testid="stMetricValue"] {{ font-family: var(--font-display) !important; color: var(--text) !important; }}
        [data-testid="stMetricLabel"] {{ color: var(--text-muted) !important; font-size: 11px !important;
            text-transform:uppercase; letter-spacing:0.06em; }}

        /* Buttons */
        .stButton > button, .stDownloadButton > button {{ font-family: var(--font-ui); font-weight:600;
            border-radius: 8px !important; border: 1px solid var(--border-strong) !important;
            background: var(--bg-card) !important; color: var(--text) !important;
            padding: 0.5rem 1rem !important; transition: all .15s ease; box-shadow:none !important; }}
        .stButton > button:hover, .stDownloadButton > button:hover {{ border-color: var(--accent) !important;
            color: var(--accent) !important; background: var(--accent-soft) !important; }}
        .stButton > button p, .stDownloadButton > button p {{ color: inherit !important; }}
        button[kind="primary"] {{ background: linear-gradient(155deg, var(--accent-strong), var(--accent)) !important;
            color: var(--accent-text-on) !important; border: none !important; }}
        button[kind="primary"] p {{ color: var(--accent-text-on) !important; }}
        button[kind="primary"]:hover {{ filter: brightness(1.06); }}

        /* Inputs */
        div[data-baseweb="input"] > div, div[data-baseweb="select"] > div, div[data-baseweb="base-input"] {{
            background: var(--bg-card) !important; border:1px solid var(--border-strong) !important;
            border-radius: 8px !important; color: var(--text) !important; }}
        input, textarea {{ color: var(--text) !important; }}
        .stTextInput label, .stSelectbox label, .stSlider label, .stRadio label, .stTextArea label {{
            color: var(--text-muted) !important; font-size: 0.85rem !important; }}

        /* Radios rendered as compact pills (theme / nav) */
        div[role="radiogroup"] {{ gap: 4px; }}
        div[role="radiogroup"] label {{ background: var(--bg-card); border:1px solid var(--border);
            border-radius: 8px; padding: 6px 10px !important; margin: 0 !important; }}

        /* Slider / progress */
        .stProgress > div > div {{ background: linear-gradient(90deg, var(--accent-strong), var(--accent)) !important; }}

        /* Alerts — flat, single accent rule instead of loud colored banners */
        [data-testid="stAlert"] {{ background: var(--bg-card) !important; border:1px solid var(--border) !important;
            border-left: 3px solid var(--accent) !important; border-radius: 8px !important; }}
        [data-testid="stAlert"] p {{ color: var(--text) !important; }}

        hr {{ border: none !important; border-top: 1px solid var(--border) !important; margin: 20px 0 !important; opacity: 1 !important; }}

        [data-testid="stDataFrame"] {{ border:1px solid var(--border); border-radius: var(--radius); overflow:hidden; }}

        [data-testid="stChatMessage"] {{ background: var(--bg-card); border:1px solid var(--border); border-radius: var(--radius); }}

        [data-testid="stSidebar"] {{ background: var(--sidebar-bg) !important; border-right: 1px solid var(--border); }}
        [data-testid="stSidebar"] .stButton > button {{ width:100%; }}
        [data-testid="stSidebar"] hr {{ margin: 14px 0 !important; }}

        .cm-brand {{ display:flex; align-items:center; gap:10px; padding: 2px 0 14px 0; }}
        .cm-brand .cm-mark {{ width:30px; height:30px; font-size:14px; border-radius:8px; }}
        .cm-brand-name {{ font-family: var(--font-display); font-size: 1.05rem; color: var(--text); }}
        .cm-brand-sub {{ font-size: 10.5px; text-transform:uppercase; letter-spacing:0.1em; color: var(--text-faint); }}
        .cm-side-label {{ font-size: 10.5px; font-weight:600; text-transform:uppercase; letter-spacing:0.1em;
            color: var(--text-faint) !important; margin: 6px 0 2px 0; }}

        .cm-chip-row {{ display:flex; flex-wrap:wrap; gap:6px; margin: 4px 0 2px 0; }}
        .cm-chip {{ font-size: 11.5px; padding: 4px 10px; border-radius: 999px; background: var(--bg-hover);
            border:1px solid var(--border); color: var(--text-muted) !important; }}

        .cm-footer {{ text-align:center; margin-top: 54px; padding-top:18px; border-top:1px solid var(--border);
            color: var(--text-faint) !important; font-size: 0.8rem; }}
        .cm-footer-stats {{ display:flex; justify-content:center; flex-wrap:wrap; gap:6px 14px;
            margin-bottom: 10px; font-size: 12px; color: var(--text-muted) !important; }}
        .cm-footer-stats b {{ color: var(--text) !important; font-weight:600; }}
        .cm-footer-stats span:not(:last-child)::after {{ content:"·"; margin-left:14px; color: var(--text-faint); }}
        </style>
        """
    )

    st_html(
        """
        <div class="cm-hero">
            <div class="cm-mark">C</div>
            <div>
                <h1>CineMind</h1>
                <p>Your personal film companion — curated picks, live ratings, zero clutter.</p>
            </div>
        </div>
        """
    )


def render_section_header(eyebrow: str, title: str, subtitle: str = "") -> None:
    """Consistent section heading: a small amber eyebrow label plus a serif
    title. Used everywhere in place of the original's emoji-stacked
    st.subheader() calls, so every shelf on the page reads as one system."""
    subtitle_html = f'<p class="section-subtitle">{subtitle}</p>' if subtitle else ""
    st_html(
        f"""<div class="cm-section-head">
            <span class="eyebrow">{eyebrow}</span>
            <h2>{title}</h2>
            {subtitle_html}
        </div>"""
    )


def render_stat_row(stats: list[tuple[str, str]]) -> None:
    """Compact stat pills — replaces st.info() boxes that were only ever
    used to display a number, which rendered as mismatched alert banners."""
    items = "".join(
        f'<div class="stat-pill"><span class="stat-value">{value}</span>'
        f'<span class="stat-label">{label}</span></div>'
        for label, value in stats
    )
    st.markdown(f'<div class="stat-row">{items}</div>', unsafe_allow_html=True)


def render_chip_row(items: list[str]) -> None:
    """Small neutral tag chips for genres / providers / history — a quieter
    alternative to emoji-prefixed, bullet-separated st.write() text."""
    if not items:
        return
    chips = "".join(f'<span class="cm-chip">{item}</span>' for item in items)
    st.markdown(f'<div class="cm-chip-row">{chips}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar() -> tuple[bool, str, str]:
    st_html(
        """<div class="cm-brand"><div class="cm-mark">C</div>
        <div><div class="cm-brand-name">CineMind</div><div class="cm-brand-sub">Control Panel</div></div>
        </div>""",
        markdown_fn=st.sidebar,
    )

    page = st.sidebar.radio("Navigation", ["Home", "AI Movie Assistant", "Watchlist", "Compare Movies"], label_visibility="collapsed")

    st.sidebar.markdown("---")
    st.sidebar.markdown('<p class="cm-side-label">Preferences</p>', unsafe_allow_html=True)
    fast_mode = st.sidebar.toggle("Fast load mode", value=True, help="Smaller poster images for quicker loading")
    theme_mode = st.sidebar.radio("Theme", ["Dark", "Light"], horizontal=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown('<p class="cm-side-label">My Watchlist</p>', unsafe_allow_html=True)
    st.sidebar.metric("Saved movies", len(st.session_state.watchlist))

    if st.session_state.watchlist:
        for idx, movie in enumerate(st.session_state.watchlist):
            col1, col2 = st.sidebar.columns([4, 1])
            with col1:
                st.write(movie["title"])
            with col2:
                if st.button("✕", key=f"remove_{idx}"):
                    remove_from_watchlist(idx)
                    st.rerun()

        df = pd.DataFrame(st.session_state.watchlist)
        col_a, col_b = st.sidebar.columns(2)
        with col_a:
            st.download_button("Export CSV", df.to_csv(index=False), "watchlist.csv", "text/csv")
        with col_b:
            if st.button("Clear list"):
                st.session_state.watchlist = []
                st.rerun()
    else:
        st.sidebar.caption("No movies saved yet — add some from any page.")

    if st.session_state.recommendation_history:
        st.sidebar.markdown("---")
        st.sidebar.markdown('<p class="cm-side-label">Recent Recommendations</p>', unsafe_allow_html=True)
        with st.sidebar:
            render_chip_row(st.session_state.recommendation_history)

    if st.session_state.search_history:
        st.sidebar.markdown("---")
        st.sidebar.markdown('<p class="cm-side-label">Recent Searches</p>', unsafe_allow_html=True)
        with st.sidebar:
            render_chip_row(st.session_state.search_history[:5])

    return fast_mode, theme_mode, page


# ---------------------------------------------------------------------------
# Home page widgets (rendered above the main content when page == "Home",
# matching the original app's behavior of Home + full recommender on one page)
# ---------------------------------------------------------------------------
def render_home_widgets() -> None:
    if not st.session_state.watchlist:
        return

    favorite_genres = get_user_favorite_genres()
    ratings = get_watchlist_ratings()
    avg_rating = round(sum(ratings) / len(ratings), 1) if ratings else 0

    render_section_header("Your Taste", "Taste Profile", "Built from what's in your watchlist.")

    render_stat_row([
        ("Favorite genre", favorite_genres[0] if favorite_genres else "—"),
        ("Average rating", f"{avg_rating} / 10" if ratings else "—"),
        ("Movies saved", str(len(st.session_state.watchlist))),
    ])

    genre_count = get_watchlist_genre_counts()
    if genre_count:
        st.bar_chart(genre_count)
        st.caption(generate_taste_summary())
    st.markdown("---")


# ---------------------------------------------------------------------------
# Watchlist page
# ---------------------------------------------------------------------------
def render_watchlist_page() -> None:
    render_section_header("Saved", "My Watchlist", "Everything you've bookmarked, in one place.")

    render_stat_row([
        ("Saved", str(len(st.session_state.watchlist))),
        ("Rated", str(len(st.session_state.user_ratings))),
        ("Searches", str(len(st.session_state.search_history))),
    ])
    st.markdown("---")

    watch_search = st.text_input("Search your watchlist", placeholder="Search by title…")

    genre_count = get_watchlist_genre_counts()

    if st.session_state.user_ratings or genre_count:
        ratings = get_watchlist_ratings()
        stats = []
        if ratings:
            stats.append(("Avg. rating", str(round(sum(ratings) / len(ratings), 1))))
        if st.session_state.user_ratings:
            best_movie = max(st.session_state.user_ratings, key=st.session_state.user_ratings.get)
            stats.append(("Your top pick", best_movie))
        if genre_count:
            fav_genre = max(genre_count, key=genre_count.get)
            stats.append(("Favorite genre", fav_genre))
        if stats:
            render_stat_row(stats)

        if st.session_state.user_ratings:
            with st.expander("Your ratings"):
                for movie, rating in st.session_state.user_ratings.items():
                    st.write(f"{movie} — {rating}/5")

    if genre_count:
        if st.button("Recommend from my watchlist", type="primary") and st.session_state.watchlist:
            source_title, watch_recs = get_watchlist_based_recommendations(fast_mode)
            st.caption(f"Based on {source_title}")
            if watch_recs:
                components.html(
                    generate_movie_row(watch_recs, "watchlist_recs", theme_mode),
                    height=420, scrolling=False,
                )
        st.markdown("---")

    if not st.session_state.watchlist:
        st.info("No movies saved yet. Add some from the Home page.")
        return

    movies_to_show = st.session_state.watchlist
    if watch_search:
        movies_to_show = [m for m in movies_to_show if watch_search.lower() in m["title"].lower()]

    for movie in movies_to_show:
        with st.container(border=True):
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(movie["poster"], width=150)
            with col2:
                st.subheader(movie["title"])
                st.caption(f"★ {movie['rating']}  ·  {movie.get('year', '')}")
                render_chip_row([g.strip() for g in str(movie["genres"]).split(",") if g.strip()])
                st.markdown(f"[Watch trailer ↗]({movie['trailer_url']})")


# ---------------------------------------------------------------------------
# AI Movie Assistant page
# ---------------------------------------------------------------------------
def render_ai_assistant_page() -> None:
    render_section_header("Ask CineMind", "AI Movie Assistant", "Tell it what you're in the mood for — it remembers the conversation.")

    for message in st.session_state.ai_chat_history:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
                continue

            try:
                movies = json.loads(message["content"])
            except json.JSONDecodeError:
                st.markdown(message["content"])
                continue

            st.markdown('<span class="eyebrow">Recommended</span>', unsafe_allow_html=True)
            for movie in movies:
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.image(movie["poster"], width=180)
                with col2:
                    render_movie_details(movie, show_reason=True)
                st.markdown("---")

            if message == st.session_state.ai_chat_history[-1]:
                st.caption("Continue the conversation")
                cols = st.columns(len(AI_SUGGESTIONS))
                for i, suggestion in enumerate(AI_SUGGESTIONS):
                    with cols[i]:
                        if st.button(suggestion, key=f"suggestion_{i}_{len(st.session_state.ai_chat_history)}"):
                            st.session_state.quick_prompt = suggestion
                            st.rerun()

    if st.session_state.ai_chat_history:
        col1, col2 = st.columns([5, 1])
        with col2:
            if st.button("Clear chat"):
                st.session_state.ai_chat_history = []
                st.rerun()

    user_prompt = st.chat_input("Ask CineMind for movie recommendations…")

    if st.session_state.quick_prompt:
        user_prompt = st.session_state.quick_prompt
        st.session_state.quick_prompt = None

    if not user_prompt:
        return

    with st.spinner("CineMind is thinking…"):
        st.session_state.ai_chat_history.append({"role": "user", "content": user_prompt})
        ai_response = ask_gemini(user_prompt)

    if ai_response is None:
        st.error("CineMind couldn't reach Gemini — check the API key or try again shortly.")
        return
    if len(ai_response) == 0:
        st.warning("No recommendations were found for that request.")
        return

    st.rerun()


# ---------------------------------------------------------------------------
# Compare Movies page
# ---------------------------------------------------------------------------
def render_compare_page() -> None:
    render_section_header("Side by Side", "Compare Movies", "Put two titles next to each other before you commit.")
    movie_titles = movies["title"].values

    col1, col2 = st.columns(2)
    with col1:
        movie_1 = st.selectbox("First movie", movie_titles, key="compare_1")
    with col2:
        movie_2 = st.selectbox("Second movie", movie_titles, key="compare_2")

    if not st.button("Compare", type="primary"):
        return

    movie_1_id = int(movies[movies["title"] == movie_1].iloc[0].movie_id)
    movie_2_id = int(movies[movies["title"] == movie_2].iloc[0].movie_id)
    info1 = cached_fetch_movie_details(movie_1_id, fast_mode)
    info2 = cached_fetch_movie_details(movie_2_id, fast_mode)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.image(info1["poster"], use_container_width=True)
        st.subheader(info1["title"])
    with col2:
        st.image(info2["poster"], use_container_width=True)
        st.subheader(info2["title"])

    comparison = pd.DataFrame({
        "Feature": ["Rating", "Year", "Genres"],
        movie_1: [info1["rating"], info1["year"], info1["genres"]],
        movie_2: [info2["rating"], info2["year"], info2["genres"]],
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Shared recommendation-results renderer.
# Previously the app had two separate, drifting render paths: one used right
# after clicking "Show Recommendations" (rich: feedback + reasoning + AI
# panel) and a simpler one used on rerun (missing all of that). Any button
# click inside the rich view triggered a rerun that silently downgraded the
# UI. Both paths now call this single function.
# ---------------------------------------------------------------------------
def render_recommendation_results(recommended_movies: list[dict], source_title: str, key_prefix: str) -> None:
    render_section_header("Similar Titles", f"Because you picked {source_title}")
    components.html(
        generate_movie_row(recommended_movies, row_id=f"{key_prefix}_row", theme_mode=theme_mode),
        height=420, scrolling=False,
    )

    for idx, movie in enumerate(recommended_movies):
        match_score = int(movie.get("similarity_score", 90))
        with st.container(border=True):
            col1, col2 = st.columns([5, 1])

            with col1:
                st.markdown(f"**{movie['title']}**")
                st.caption(f"★ {movie['rating']}  ·  {match_score}% match  ·  {get_recommendation_reason(movie)}")
                fcol1, fcol2 = st.columns(2)
                with fcol1:
                    if st.button("Helpful", key=f"{key_prefix}_helpful_{idx}"):
                        st.session_state.feedback_data[movie["title"]] = "Helpful"
                        st.toast("Thanks — feedback saved.")
                with fcol2:
                    if st.button("Not interested", key=f"{key_prefix}_not_interested_{idx}"):
                        st.session_state.feedback_data[movie["title"]] = "Not Interested"
                        st.toast("Got it, we'll adjust future picks.")

            with col2:
                if st.button("Save", key=f"{key_prefix}_save_{idx}"):
                    if add_to_watchlist(movie):
                        st.toast(f"{movie['title']} added to your watchlist")
                    else:
                        st.toast(f"{movie['title']} is already saved")

            with st.expander(f"More about {movie['title']}"):
                st.image(movie["poster"], width=200)
                render_movie_details(movie)
    st.markdown("---")


def render_ai_explanation_panel(movie_info: dict, confidence: int) -> None:
    with st.container(border=True):
        st.markdown('<span class="eyebrow">CineMind AI</span>', unsafe_allow_html=True)
        st.markdown(generate_ai_explanation(movie_info, confidence))
        st.progress(confidence)
        st.caption(f"Confidence {confidence}%")


# ---------------------------------------------------------------------------
# Main recommender content: search, recommendations, mood finder, trending,
# categories, and the auto-refreshing top-picks strip.
# ---------------------------------------------------------------------------
def render_search_and_details() -> Optional[dict]:
    movie_titles = movies["title"].values
    with st.container(border=True):
        search_query = st.text_input("Search for a movie", "", placeholder="Search titles…")
        filtered = [m for m in movie_titles if search_query.lower() in m.lower()] if search_query else movie_titles
        selected_movie = st.selectbox("Choose or type a movie name", filtered)

        if not selected_movie:
            return None

        if selected_movie not in st.session_state.search_history:
            st.session_state.search_history.insert(0, selected_movie)
            st.session_state.search_history = st.session_state.search_history[:10]

        selected_movie_id = int(movies[movies["title"] == selected_movie].iloc[0].movie_id)
        movie_info = cached_fetch_movie_details(selected_movie_id, fast_mode)

        st.markdown("---")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(movie_info["poster"], use_container_width=True)
        with col2:

            render_movie_details(
                movie_info,
                show_overview=True,
            )

            user_rating = st.slider("Your rating", 1, 5, st.session_state.user_ratings.get(movie_info["title"], 3))
            st.session_state.user_ratings[movie_info["title"]] = user_rating

    if st.button(f"Add {movie_info['title']} to watchlist", type="primary"):
        if add_to_watchlist(movie_info):
            st.toast(f"{movie_info['title']} added to your watchlist")
        else:
            st.toast(f"{movie_info['title']} is already in your watchlist")

    return {"title": selected_movie, "info": movie_info}


def render_recommend_button(selected_movie: str, movie_info: dict) -> None:
    loading_messages = [
        "Analyzing your choice…", "Comparing genre patterns…",
        "Looking for hidden gems…", "Matching cinematic patterns…",
        "Ranking the closest titles…",
    ]

    if st.button("Show recommendations", type="primary"):
        with st.spinner(random.choice(loading_messages)):
            recommended_movies = recommend(selected_movie, fast_mode)
            st.session_state.recommendation_count += 1

        st.session_state.last_recommendations = recommended_movies
        st.session_state.last_selected_movie = selected_movie

        if selected_movie not in st.session_state.recommendation_history:
            st.session_state.recommendation_history.insert(0, selected_movie)
            st.session_state.recommendation_history = st.session_state.recommendation_history[:5]

        if not recommended_movies:
            st.error("No recommendations found for that title.")
            return

        confidence = random.randint(88, 97)
        st.session_state.last_confidence = confidence
        st.session_state.last_movie_info = movie_info

        render_recommendation_results(recommended_movies, selected_movie, key_prefix="fresh")
        render_ai_explanation_panel(movie_info, confidence)

    elif st.session_state.last_recommendations:
        render_recommendation_results(
            st.session_state.last_recommendations,
            st.session_state.last_selected_movie,
            key_prefix="persisted",
        )
        if st.session_state.last_movie_info and st.session_state.last_confidence:
            render_ai_explanation_panel(st.session_state.last_movie_info, st.session_state.last_confidence)


def render_because_you_watched() -> None:
    if not st.session_state.watchlist:
        return
    source_title, watch_recs = get_watchlist_based_recommendations(fast_mode)
    if watch_recs:
        st.markdown("---")
        render_section_header("For You", f"Because You Watched {source_title}")
        components.html(
            generate_movie_row(watch_recs, "because_you_watched", theme_mode),
            height=420, scrolling=False,
        )


def render_mood_finder() -> None:
    st.markdown("---")
    render_section_header("Mood Finder", "What Are You In The Mood For?")
    col1, col2 = st.columns([3, 1])
    with col1:
        mood = st.selectbox("Choose a mood", ["Action", "Comedy", "Romantic", "Emotional", "Sci-Fi", "Adventure", "Horror"], label_visibility="collapsed")
    with col2:
        find_clicked = st.button("Find movies", type="primary", use_container_width=True)

    if find_clicked:
        mood_ids = get_movies_by_mood(mood)
        if not mood_ids:
            st.warning("No movies found for that mood yet.")
            return
        mood_movies = fetch_multiple_movie_details(mood_ids, fast_mode)
        st.caption(f"Matching your '{mood}' mood")
        components.html(generate_movie_row(mood_movies, "mood_movies", theme_mode), height=420, scrolling=False)


def render_trending_and_categories() -> None:
    """Trending and each category shelf are fetched independently of one
    another, but the original app fetched them one after another — five
    sequential network round trips before anything below this point could
    render. Kicking them all off in one shared thread pool up front cuts
    that to roughly one round trip's worth of wait, while rendering the
    exact same sections in the exact same order as before."""
    category_ids = get_recommendation_categories()
    category_picks = {
        title: random.sample(ids, min(6, len(ids)))
        for title, ids in category_ids.items() if ids
    }

    with ThreadPoolExecutor(max_workers=1 + len(category_picks)) as ex:
        trending_future = ex.submit(get_trending_movies, fast_mode)
        category_futures = {
            title: ex.submit(fetch_multiple_movie_details, ids, fast_mode)
            for title, ids in category_picks.items()
        }
        trending_movies = trending_future.result()
        category_results = {title: f.result() for title, f in category_futures.items()}

    st.markdown("---")
    render_section_header("This Week", "Trending Now")
    if trending_movies:
        components.html(generate_movie_row(trending_movies, "trending_movies", theme_mode), height=420, scrolling=False)
    else:
        st.warning("Unable to load trending movies right now.")

    st.markdown("---")
    render_section_header("Browse", "Explore Categories")
    for title, movies_data in category_results.items():
        if not movies_data:
            continue
        row_id = "cat_" + re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        st.markdown(f"**{title}**")
        components.html(generate_movie_row(movies_data, row_id, theme_mode), height=420, scrolling=False)


def render_top_picks() -> None:
    st.markdown("---")
    header_col, action_col = st.columns([4, 1.4])
    with header_col:
        subtitle = "Refreshes automatically" if (HAS_AUTOREFRESH and st.session_state.auto_refresh) else "Personalized from your taste profile"
        render_section_header("Top Picks", "Picked For You", subtitle)
    with action_col:
        if st.session_state.auto_refresh:
            if st.button("Pause refresh"):
                st.session_state.auto_refresh = False
        else:
            if st.button("Resume refresh"):
                st.session_state.auto_refresh = True

    if st.session_state.auto_refresh:
        if HAS_AUTOREFRESH:
            # Genuine timer-driven rerun — the original countdown badge never
            # actually triggered a rerun on its own.
            st_autorefresh(interval=REFRESH_INTERVAL * 1000, key="top_picks_autorefresh")
        else:
            st.caption("Install `streamlit-autorefresh` for automatic refresh, or refresh manually below.")
            if st.button("Refresh now"):
                st.session_state.last_top_picks = None

    should_refresh = (
        not st.session_state.last_top_picks
        or (time.time() - st.session_state.last_refresh_time) > REFRESH_INTERVAL
    )

    if should_refresh:
        try:
            random_ids = get_personalized_movie_ids() or sample_random_movie_ids(6)
            st.session_state.last_top_picks = fetch_multiple_movie_details(random_ids, fast_mode)
            st.session_state.last_refresh_time = time.time()
        except Exception as e:
            st.warning(f"Couldn't refresh top picks: {e}")

    top_movies = st.session_state.last_top_picks
    if not top_movies:
        st.info("No top picks available yet.")
        return

    components.html(
        generate_movie_row(top_movies, row_id="top_picks_row", theme_mode=theme_mode),
        height=420, scrolling=False,
    )
    with st.expander("Save from Top Picks"):
        for idx, movie in enumerate(top_movies):
            col1, col2 = st.columns([5, 1])
            with col1:
                st.caption(movie["title"])
            with col2:
                if st.button("Save", key=f"top_pick_{idx}"):
                    if add_to_watchlist(movie):
                        st.toast(f"{movie['title']} added to your watchlist")
                        st.rerun()


def render_main_recommender_content() -> None:
    result = render_search_and_details()
    if result:
        render_recommend_button(result["title"], result["info"])
    render_because_you_watched()
    render_mood_finder()
    render_trending_and_categories()
    render_top_picks()


def render_footer(theme_mode: str) -> None:
    stats = [
        ("Movies", "5000+"),
        ("Saved", str(len(st.session_state.watchlist))),
        ("Rated", str(len(st.session_state.user_ratings))),
        ("Live data", "TMDB"),
    ]
    stats_html = "".join(f"<span><b>{value}</b> {label}</span>" for label, value in stats)
    st_html(
        f"""<div class="cm-footer">
        <div class="cm-footer-stats">{stats_html}</div>
        Made with care by <b>Yash Kumar Mehta</b>
        </div>"""
    )


# ---------------------------------------------------------------------------
# Page dispatch — replaces the original's linear script + scattered
# st.stop() calls with explicit routing.
# ---------------------------------------------------------------------------
fast_mode, theme_mode, page = render_sidebar()
inject_theme_css(theme_mode)

if page == "Watchlist":
    render_watchlist_page()
elif page == "AI Movie Assistant":
    render_ai_assistant_page()
elif page == "Compare Movies":
    render_compare_page()
else:
    if page == "Home":
        render_home_widgets()
    render_main_recommender_content()
    render_footer(theme_mode)