import pickle
import streamlit as st
import requests
import time
import random
import os

# Configure Streamlit app layout and title
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

# Sidebar controls for user preferences
st.sidebar.title("⚙️ Settings")
fast_mode = st.sidebar.toggle("⚡ Fast Load Mode", value=True)

# Initialize auto-refresh toggle in session state
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = True


# Fetch movie details from TMDB API with caching for better performance
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_movie_details(movie_id, fast_mode=True):
    try:
        api_key = "8265bd1679663a7ea12ac168da84d2e8"  # Replace with personal TMDB key
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
        response = requests.get(url, timeout=5)

        if response.status_code != 200:
            raise Exception(f"API Error: {response.status_code}")

        data = response.json()
        poster_path = data.get('poster_path')
        title = data.get('title', 'Unknown Title')
        rating = data.get('vote_average', 'N/A')
        release_date = data.get('release_date', '')
        year = release_date[:4] if release_date else '----'
        genres = ", ".join([g['name'] for g in data.get('genres', [])]) or 'N/A'

        img_size = "w300" if fast_mode else "w500"
        poster_url = f"https://image.tmdb.org/t/p/{img_size}/{poster_path}" if poster_path else \
                     "https://via.placeholder.com/300x450?text=No+Poster"

        return {
            "title": title,
            "poster": poster_url,
            "rating": rating,
            "year": year,
            "genres": genres
        }

    except Exception:
        return {
            "title": "Error",
            "poster": "https://via.placeholder.com/300x450?text=Error",
            "rating": "N/A",
            "year": "----",
            "genres": "N/A"
        }


# Generate top 5 movie recommendations using cosine similarity
def recommend(movie):
    if movie not in movies['title'].values:
        st.error("❌ Movie not found in the database.")
        return []

    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])

    recommended_movies = []
    for i in distances[1:6]:
        movie_id = movies.iloc[i[0]].movie_id
        details = fetch_movie_details(movie_id, fast_mode)
        recommended_movies.append(details)
    return recommended_movies


# Load pre-trained movie data and similarity model
try:
    base_path = os.path.dirname(__file__)
    movies = pickle.load(open(os.path.join(base_path, 'movie_list.pkl'), 'rb'))
        # --- Rebuild large file from parts ---
    import os

    similarity_path = os.path.join(base_path, 'similarity.pkl')
    if not os.path.exists(similarity_path):
        with open(similarity_path, 'wb') as wfd:
            i = 1
            while os.path.exists(os.path.join(base_path, f"similarity.pkl.part{i}")):
                part_path = os.path.join(base_path, f"similarity.pkl.part{i}")
                with open(part_path, 'rb') as fd:
                    wfd.write(fd.read())
                i += 1
        print("✅ Rebuilt similarity.pkl from parts")

    similarity = pickle.load(open(similarity_path, 'rb'))

except Exception as e:
    st.error(f"⚠️ Failed to load model files: {e}")
    st.stop()


# Apply custom CSS for Netflix-style UI
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #000000 40%, #141414);
        color: white;
        font-family: 'Poppins', sans-serif;
    }
    .header {
        background-color: #111;
        padding: 1rem 2rem;
        border-bottom: 2px solid #e50914;
        text-align: center;
    }
    .header h1 {
        color: #e50914;
        font-size: 2.5rem;
        margin-bottom: 0.2rem;
        letter-spacing: 1px;
        text-shadow: 0 0 15px #e50914;
    }
    .header p {
        color: #ccc;
        font-size: 1rem;
        margin-top: 0;
    }
    h2 {
        color: #e50914;
        text-shadow: 0 0 10px #e50914;
    }
    .footer {
        text-align: center;
        margin-top: 60px;
        padding-top: 10px;
        font-size: 15px;
        color: #888;
        border-top: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)


# Display application header
st.markdown("""
    <div class="header">
        <h1>🎬 Movie Recommender</h1>
        <p>Smart, Fast & Cinematic — Powered by Machine Learning</p>
    </div>
""", unsafe_allow_html=True)


# Dropdown for selecting a movie
movie_list = movies['title'].values
selected_movie = st.selectbox("🎞️ Choose or type a movie name", movie_list)


# Display recommended movies on button click
if st.button('🔍 Show Recommendations'):
    st.info(f"Finding movies similar to **{selected_movie}** using cosine similarity...")
    with st.spinner('🎥 Loading posters...'):
        time.sleep(1.2)
        recommended_movies = recommend(selected_movie)

    if recommended_movies:
        st.success("✅ Found some awesome movies for you!")
        st.subheader(f"🎬 Similar to **{selected_movie}**:")
        st.markdown('<div class="top-picks-grid">', unsafe_allow_html=True)

        for movie in recommended_movies:
            tmdb_link = f"https://www.themoviedb.org/search?query={movie['title'].replace(' ', '+')}"
            st.markdown(f"""
                <a href="{tmdb_link}" target="_blank" style="text-decoration:none;">
                    <div style="display:inline-block;margin:10px;text-align:center;">
                        <img src="{movie['poster']}" width="180" style="border-radius:10px;box-shadow:0 0 10px #e50914;">
                        <div style="color:white;font-size:13px;margin-top:5px;">
                            <b>{movie['title']}</b><br>
                            ⭐ {movie['rating']} | 📅 {movie['year']}<br>
                            🎭 {movie['genres']}
                        </div>
                    </div>
                </a>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# Auto-refreshing "Top Picks" section
st.markdown("<br><hr><br>", unsafe_allow_html=True)
st.subheader("🔥 Top Picks for You (Auto-Refreshing)")

# Control auto-refresh toggle
if st.session_state.auto_refresh:
    if st.button("⏸️ Stop Auto Refresh"):
        st.session_state.auto_refresh = False
else:
    if st.button("▶️ Resume Auto Refresh"):
        st.session_state.auto_refresh = True

placeholder = st.empty()


# Display top picks dynamically with a countdown refresh
if st.session_state.auto_refresh:
    with placeholder.container():
        try:
            random_indices = random.sample(range(len(movies)), 5)
            random_movie_ids = [movies.iloc[i].movie_id for i in random_indices]
            top_movies = [fetch_movie_details(mid, fast_mode) for mid in random_movie_ids]
        except Exception:
            st.warning("⚠️ Network issue while loading posters.")
            top_movies = []

        st.markdown("<div class='top-picks-grid' style='display:flex;justify-content:center;gap:20px;flex-wrap:wrap;'>", unsafe_allow_html=True)
        for m in top_movies:
            tmdb_link = f"https://www.themoviedb.org/search?query={m['title'].replace(' ', '+')}"
            st.markdown(f"""
                <a href="{tmdb_link}" target="_blank" class="top-picks-card">
                    <img src="{m['poster']}" width="200" style="border-radius:10px;box-shadow:0 0 10px #e50914;">
                    <div style="color:white;text-align:center;font-size:13px;margin-top:5px;">
                        <b>{m['title']}</b><br>
                        ⭐ {m['rating']} | 📅 {m['year']}<br>
                        🎭 {m['genres']}
                    </div>
                </a>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Countdown before auto-refresh
        countdown_placeholder = st.empty()
        for remaining in range(5, 0, -1):
            countdown_placeholder.markdown(
                f"<div style='position:fixed;top:20px;right:20px;background:#e50914;color:white;padding:8px 14px;border-radius:50px;font-weight:bold;box-shadow:0 0 10px #e50914;'>⏳ Refreshing in {remaining}s...</div>",
                unsafe_allow_html=True
            )
            time.sleep(1)
        countdown_placeholder.empty()

    st.rerun()
else:
    st.info("⏸️ Auto Refresh is paused. Click ▶️ Resume Auto Refresh to continue.")


# Display footer section
st.markdown("<div class='footer'>Made with ❤️ by <b>Yash Kumar Mehta</b></div>", unsafe_allow_html=True)
