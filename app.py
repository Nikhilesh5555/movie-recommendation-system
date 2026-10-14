"""
CINE•FLOW - Dark Glassmorphism Movie Recommendation System
Built with Streamlit, Custom CSS, and Machine Learning Cosine Similarity.
"""

import html
import re
from pathlib import Path
import pandas as pd
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==============================================================================
# 1. PAGE CONFIGURATION & STATE
# ==============================================================================
st.set_page_config(
    page_title="CINE•FLOW | AI Movie Discovery & Recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session States
if "search_query" not in st.session_state:
    st.session_state["search_query"] = ""
if "selected_genre" not in st.session_state:
    st.session_state["selected_genre"] = None
if "active_nav" not in st.session_state:
    st.session_state["active_nav"] = "Dashboard"

# ==============================================================================
# 2. ULTRA-MODERN DARK GLASSMORPHISM STYLING
# ==============================================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;600;700;800;900&display=swap');

/* Global Reset & Cinematic Backdrop */
:root {
    --glass-bg: rgba(255, 255, 255, 0.05);
    --glass-border: 1px solid rgba(255, 255, 255, 0.1);
    --glass-border-hover: 1px solid rgba(0, 242, 254, 0.45);
    --neon-cyan: #00f2fe;
    --neon-blue: #4facfe;
    --neon-purple: #9b51e0;
    --neon-magenta: #ff007f;
    --text-primary: #ffffff;
    --text-secondary: #94a3b8;
    --card-radius: 16px;
}

.stApp {
    background:
        linear-gradient(180deg, rgba(7, 8, 20, 0.82) 0%, rgba(9, 10, 26, 0.90) 35%, rgba(6, 7, 16, 0.98) 100%),
        radial-gradient(circle at 12% 14%, rgba(0, 242, 254, 0.15) 0%, transparent 40%),
        radial-gradient(circle at 88% 18%, rgba(155, 81, 224, 0.18) 0%, transparent 45%),
        radial-gradient(circle at 50% 85%, rgba(255, 0, 127, 0.08) 0%, transparent 55%),
        url('https://image.tmdb.org/t/p/original/eZ239CUp1d6OryZEBPnO2n87gMG.jpg');
    background-size: cover;
    background-position: center top;
    background-attachment: fixed;
    color: var(--text-primary);
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Hide Default Streamlit Chrome */
header[data-testid="stHeader"] {
    background: transparent !important;
}

#MainMenu, footer {
    visibility: hidden;
}

.block-container {
    max-width: 1560px;
    padding-top: 1.2rem;
    padding-bottom: 3.5rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* ================= TOP HEADER BAR ================= */
.top-header-container {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
    background: rgba(255, 255, 255, 0.035);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: var(--glass-border);
    border-radius: 20px;
    padding: 12px 24px;
    margin-bottom: 26px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), inset 0 1px 1px rgba(255, 255, 255, 0.08);
}

.header-brand-logo {
    font-family: 'Outfit', sans-serif;
    font-size: 26px;
    font-weight: 900;
    letter-spacing: 1.5px;
    background: linear-gradient(135deg, #00f2fe 0%, #4facfe 30%, #9b51e0 70%, #ff007f 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    filter: drop-shadow(0 0 16px rgba(0, 242, 254, 0.4));
}

/* ================= PROMINENT SEARCH INPUT FIELD ================= */
[data-testid="stTextInput"] input {
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 14px !important;
    color: #ffffff !important;
    font-size: 14px !important;
    padding: 12px 18px !important;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3) !important;
    transition: all 0.25s ease !important;
}

[data-testid="stTextInput"] input:focus {
    border-color: #00f2fe !important;
    box-shadow: 0 0 18px rgba(0, 242, 254, 0.35), inset 0 2px 4px rgba(0, 0, 0, 0.3) !important;
    background: rgba(255, 255, 255, 0.08) !important;
}

/* ================= SIDEBAR STYLING ================= */
[data-testid="stSidebar"] {
    background: rgba(9, 10, 24, 0.82) !important;
    backdrop-filter: blur(24px) !important;
    -webkit-backdrop-filter: blur(24px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    box-shadow: 15px 0 40px rgba(0, 0, 0, 0.7) !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
    padding-left: 1.2rem;
    padding-right: 1.2rem;
}

.sidebar-brand-box {
    padding: 8px 10px 20px 10px;
    margin-bottom: 12px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}

.sidebar-section-header {
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
    color: #64748b;
    text-transform: uppercase;
    margin: 18px 0 8px 10px;
}

/* ================= SECTION HEADINGS ================= */
.section-header-wrap {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 10px;
    margin-bottom: 18px;
}

.section-title {
    font-family: 'Outfit', sans-serif;
    font-size: 22px;
    font-weight: 800;
    letter-spacing: -0.3px;
    color: #ffffff;
    display: flex;
    align-items: center;
    gap: 10px;
}

.section-neon-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #00f2fe;
    box-shadow: 0 0 10px #00f2fe, 0 0 20px #00f2fe;
    display: inline-block;
}

/* ================= FROSTED GLASS MOVIE CARDS ================= */
.glass-movie-card {
    background: rgba(255, 255, 255, 0.045);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: var(--glass-border);
    border-radius: var(--card-radius);
    padding: 14px;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.55), inset 0 1px 1px rgba(255, 255, 255, 0.08);
    transition: all 0.35s cubic-bezier(0.2, 0.8, 0.2, 1);
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    height: 100%;
    position: relative;
    overflow: hidden;
}

.glass-movie-card:hover {
    transform: translateY(-8px);
    border-color: rgba(0, 242, 254, 0.45);
    box-shadow:
        0 20px 45px rgba(0, 0, 0, 0.75),
        0 0 25px rgba(0, 242, 254, 0.25),
        0 0 50px rgba(155, 81, 224, 0.18);
}

.card-poster-container {
    position: relative;
    width: 100%;
    aspect-ratio: 2 / 2.95;
    border-radius: 12px;
    overflow: hidden;
    background: #0a0c1a;
    margin-bottom: 12px;
}

.card-poster-image {
    width: 100%;
    height: 100%;
    object-fit: cover;
    transition: transform 0.45s ease;
}

.glass-movie-card:hover .card-poster-image {
    transform: scale(1.06);
}

.card-badge-rating {
    position: absolute;
    top: 8px;
    right: 8px;
    background: rgba(9, 11, 24, 0.8);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 215, 0, 0.4);
    color: #ffd700;
    font-size: 11px;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 20px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
    display: flex;
    align-items: center;
    gap: 4px;
}

.card-movie-title {
    font-size: 15px;
    font-weight: 700;
    color: #ffffff;
    line-height: 1.3;
    margin-bottom: 5px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.card-tags-row {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    margin-bottom: 8px;
}

.glass-tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 10.5px;
    font-weight: 600;
    color: #cbd5e1;
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.12);
}

.card-synopsis {
    font-size: 11.5px;
    line-height: 1.5;
    color: #94a3b8;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
    margin-top: 4px;
}

/* ================= RIGHT PANEL WIDGET ================= */
.widget-box {
    background: rgba(255, 255, 255, 0.045);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: var(--glass-border);
    border-radius: var(--card-radius);
    padding: 20px;
    margin-bottom: 24px;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.55), inset 0 1px 1px rgba(255, 255, 255, 0.08);
}

.widget-title {
    font-family: 'Outfit', sans-serif;
    font-size: 17px;
    font-weight: 800;
    color: #ffffff;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 16px;
    letter-spacing: -0.2px;
}

.top-pick-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px;
    border-radius: 12px;
    margin-bottom: 10px;
    background: rgba(255, 255, 255, 0.025);
    border: 1px solid rgba(255, 255, 255, 0.06);
    transition: all 0.25s ease;
}

.top-pick-item:hover {
    background: rgba(255, 255, 255, 0.06);
    border-color: rgba(0, 242, 254, 0.35);
    transform: translateX(4px);
    box-shadow: 0 4px 16px rgba(0, 242, 254, 0.15);
}

.top-pick-thumb {
    width: 52px;
    height: 70px;
    border-radius: 8px;
    object-fit: cover;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.top-pick-info {
    flex: 1;
    min-width: 0;
}

.top-pick-name {
    font-size: 14px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 3px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.top-pick-cat {
    font-size: 11px;
    font-weight: 600;
    color: #a78bfa;
    margin-bottom: 4px;
}

.top-pick-rating {
    font-size: 11px;
    font-weight: 700;
    color: #ffd700;
    display: flex;
    align-items: center;
    gap: 4px;
}

/* ================= STREAMLIT BUTTON OVERRIDES ================= */
.stButton > button {
    width: 100% !important;
    background: rgba(255, 255, 255, 0.06) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    color: #cbd5e1 !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    transition: all 0.25s ease !important;
}

.stButton > button:hover {
    background: rgba(255, 255, 255, 0.12) !important;
    border-color: rgba(0, 242, 254, 0.45) !important;
    color: #ffffff !important;
    box-shadow: 0 0 16px rgba(0, 242, 254, 0.25) !important;
}

div.stButton[data-testid="stBaseButton-primary"] > button {
    background: linear-gradient(135deg, rgba(0, 242, 254, 0.22) 0%, rgba(155, 81, 224, 0.28) 100%) !important;
    border: 1px solid rgba(0, 242, 254, 0.6) !important;
    color: #ffffff !important;
    box-shadow: 0 0 18px rgba(0, 242, 254, 0.3) !important;
}

/* Filter / Search Header Banner */
.search-banner {
    background: linear-gradient(135deg, rgba(20, 24, 52, 0.85) 0%, rgba(10, 11, 28, 0.95) 100%);
    border: 1px solid rgba(0, 242, 254, 0.35);
    border-radius: 18px;
    padding: 20px 24px;
    margin-bottom: 24px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6), 0 0 20px rgba(0, 242, 254, 0.15);
}
</style>
""",
    unsafe_allow_html=True,
)


# ==============================================================================
# 3. MOVIE DATA & MACHINE LEARNING MODEL
# ==============================================================================
@st.cache_data
def load_movie_dataset():
    """Load and prepare movie data from data/movies.csv."""
    data_path = Path(__file__).parent / "data" / "movies.csv"
    if not data_path.exists():
        return pd.DataFrame(
            [
                {"movieId": 1, "title": "Toy Story (1995)", "genres": "Adventure Animation Children Comedy Fantasy"},
                {"movieId": 2, "title": "Jumanji (1995)", "genres": "Adventure Children Fantasy"},
                {"movieId": 3, "title": "Heat (1995)", "genres": "Action Crime Thriller"},
                {"movieId": 4, "title": "Interstellar (2014)", "genres": "Adventure Drama Sci-Fi"},
                {"movieId": 5, "title": "Dune: Part Two (2024)", "genres": "Action Adventure Sci-Fi"},
            ]
        )

    df = pd.read_csv(data_path)
    df["title"] = df["title"].fillna("Untitled Film")
    df["genres_clean"] = df["genres"].fillna("").str.replace("|", " ", regex=False)
    return df


@st.cache_resource
def build_cosine_similarity_model(genres_list):
    """Compute TF-IDF and Cosine Similarity matrix on movie genres."""
    tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    matrix = tfidf.fit_transform(genres_list)
    similarity = cosine_similarity(matrix)
    return similarity


movies_df = load_movie_dataset()
similarity_matrix = build_cosine_similarity_model(movies_df["genres_clean"])


# ==============================================================================
# 4. CURATED POSTERS & METADATA DATABASE
# ==============================================================================
CURATED_DATA = {
    # Top Rated Masterpieces
    "The Shawshank Redemption": {
        "poster": "https://image.tmdb.org/t/p/w500/9cqNxx0GxF0bflZmeSMuL5tnGzr.jpg",
        "rating": "⭐ 4.9/5",
        "genres": ["Drama", "Crime"],
        "synopsis": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.",
    },
    "Interstellar": {
        "poster": "https://image.tmdb.org/t/p/w500/yQvGrMoipbRoddT0ZR8tPoR7NfX.jpg",
        "rating": "⭐ 4.9/5",
        "genres": ["Sci-Fi", "Adventure"],
        "synopsis": "A team of intrepid explorers travels through a newly discovered wormhole in space in an attempt to ensure humanity's survival.",
    },
    "The Dark Knight": {
        "poster": "https://image.tmdb.org/t/p/w500/qJ2tW6WMUDux911r6m7haRef0WH.jpg",
        "rating": "⭐ 4.9/5",
        "genres": ["Action", "Crime"],
        "synopsis": "Batman raises the stakes in his war on crime, facing the rising chaotic menace known as the Joker across Gotham City.",
    },
    "Dune: Part Two": {
        "poster": "https://image.tmdb.org/t/p/w500/6izwz7rsy95ARzTR3poZ8H6c5pp.jpg",
        "rating": "⭐ 4.8/5",
        "genres": ["Sci-Fi", "Adventure"],
        "synopsis": "Paul Atreides unites with Chani and the Fremen while seeking revenge against the conspirators who destroyed his noble family.",
    },
    # Trending Titles
    "Oppenheimer": {
        "poster": "https://image.tmdb.org/t/p/w500/8Gxv8gSFCU0XGDykEGv7zR1n2ua.jpg",
        "rating": "⭐ 4.8/5",
        "genres": ["Biography", "Drama"],
        "synopsis": "The story of American scientist J. Robert Oppenheimer and his pivotal role in the Manhattan Project during World War II.",
    },
    "Spider-Man: Across the Spider-Verse": {
        "poster": "https://image.tmdb.org/t/p/w500/8Vt6mWEReuy4Of61Lnj5Xj704m8.jpg",
        "rating": "⭐ 4.9/5",
        "genres": ["Animation", "Sci-Fi"],
        "synopsis": "Miles Morales catapults across the Multiverse, where he encounters a society of Spider-People charged with protecting reality.",
    },
    "Blade Runner 2049": {
        "poster": "https://image.tmdb.org/t/p/w500/gajva2L0rPYkEWjzgFlBXCAVBE5.jpg",
        "rating": "⭐ 4.8/5",
        "genres": ["Sci-Fi", "Mystery"],
        "synopsis": "Officer K, a new blade runner, unearths a long-buried secret that has the potential to plunge what's left of society into chaos.",
    },
    "The Creator": {
        "poster": "https://image.tmdb.org/t/p/w500/3dSivDtOuyxLDxPH4v2tcNG1fP7.jpg",
        "rating": "⭐ 4.6/5",
        "genres": ["Sci-Fi", "Action"],
        "synopsis": "Amid a future war between humans and artificial intelligence, an ex-special forces agent is recruited to hunt down an architect of war.",
    },
    "Succession": {
        "poster": "https://image.tmdb.org/t/p/w500/z0XiwdrCQ9yVIr4O0pxzaAYRxdW.jpg",
        "rating": "⭐ 4.9/5",
        "genres": ["Drama", "Business"],
        "synopsis": "The Roy family is known for controlling the biggest media conglomerate in the world, setting off a ruthless succession battle.",
    },
    # Action Picks
    "Mad Max: Fury Road": {
        "poster": "https://image.tmdb.org/t/p/w500/ulcAi4dKpAjHwYGS08vNyx9H6I9.jpg",
        "rating": "⭐ 4.8/5",
        "genres": ["Action", "Adventure"],
        "synopsis": "In a post-apocalyptic wasteland, a woman rebels against a tyrannical ruler in search for her homeland with the aid of Max Rockatansky.",
    },
    "Inception": {
        "poster": "https://image.tmdb.org/t/p/w500/xlaY2zyzMfkhk0HSC5VUwzoZPU1.jpg",
        "rating": "⭐ 4.9/5",
        "genres": ["Action", "Sci-Fi"],
        "synopsis": "A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a CEO.",
    },
    "Gladiator": {
        "poster": "https://image.tmdb.org/t/p/w500/aDb548BOkFfI4nFm0kx8A3Ezh7H.jpg",
        "rating": "⭐ 4.8/5",
        "genres": ["Action", "Drama"],
        "synopsis": "A former Roman General sets out to exact vengeance against the corrupt emperor who murdered his family and sent him into slavery.",
    },
    # Drama Picks
    "The Godfather": {
        "poster": "https://image.tmdb.org/t/p/w500/3bhkrj58Vtu7enYsRolD1fZdja1.jpg",
        "rating": "⭐ 4.9/5",
        "genres": ["Drama", "Crime"],
        "synopsis": "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant youngest son.",
    },
    "Fight Club": {
        "poster": "https://image.tmdb.org/t/p/w500/jSziioSwPVrOy9Yow3XhWIBDjq1.jpg",
        "rating": "⭐ 4.8/5",
        "genres": ["Drama", "Thriller"],
        "synopsis": "An insomniac office worker and a devil-may-care soap maker form an underground fight club that evolves into much more.",
    },
    # Thriller Picks
    "Shutter Island": {
        "poster": "https://image.tmdb.org/t/p/w500/nrmXQ0zcZUL8jFLrakWc90IR8z9.jpg",
        "rating": "⭐ 4.8/5",
        "genres": ["Thriller", "Mystery"],
        "synopsis": "A U.S. Marshal investigates the disappearance of a murderess who escaped from a hospital for the criminally insane.",
    },
    "Se7en": {
        "poster": "https://image.tmdb.org/t/p/w500/191nKfP0ehp3uIvWqgPbFmI4lv9.jpg",
        "rating": "⭐ 4.9/5",
        "genres": ["Thriller", "Crime"],
        "synopsis": "Two detectives, a rookie and a veteran, hunt a serial killer who uses the seven deadly sins as his terrifying motives.",
    },
    "Parasite": {
        "poster": "https://image.tmdb.org/t/p/w500/7IiTTgloJzvGI1TAYymCfbfl3vT.jpg",
        "rating": "⭐ 4.9/5",
        "genres": ["Thriller", "Drama"],
        "synopsis": "Greed and class discrimination threaten the newly formed symbiotic relationship between the wealthy Park family and the destitute Kim clan.",
    },
    "Gone Girl": {
        "poster": "https://image.tmdb.org/t/p/w500/ts996lKsxvjkO2yiYG0ht4qAicO.jpg",
        "rating": "⭐ 4.7/5",
        "genres": ["Thriller", "Mystery"],
        "synopsis": "With his wife's disappearance having become the focus of an intense media circus, a man sees the spotlight turned on him when he's suspected of murder.",
    },
}

# Pre-curated exclusive mappings for the 4 Genres
EXCLUSIVE_GENRE_MOVIES = {
    "Action": [
        "The Dark Knight",
        "Mad Max: Fury Road",
        "Inception",
        "Gladiator",
        "The Creator",
    ],
    "Sci-Fi": [
        "Interstellar",
        "Dune: Part Two",
        "Blade Runner 2049",
        "The Creator",
        "Spider-Man: Across the Spider-Verse",
    ],
    "Drama": [
        "The Shawshank Redemption",
        "Oppenheimer",
        "The Godfather",
        "Fight Club",
        "Succession",
    ],
    "Thriller": [
        "Shutter Island",
        "Se7en",
        "Parasite",
        "Gone Girl",
        "The Dark Knight",
    ],
}


@st.cache_data(ttl=86400)
def fetch_movie_details(title_str):
    """Enrich movie with poster, overview/synopsis, and rating using TMDB API or fallback."""
    for k, v in CURATED_DATA.items():
        if k.lower() in title_str.lower() or title_str.lower() in k.lower():
            return {
                "title": k,
                "poster": v["poster"],
                "rating": v["rating"],
                "genres": v["genres"],
                "synopsis": v["synopsis"],
            }

    api_key = st.secrets.get("TMDB_API_KEY", "")
    clean_query = re.sub(r"\s*\(\d{4}\)\s*$", "", title_str).strip()

    if api_key:
        try:
            r = requests.get(
                "https://api.themoviedb.org/3/search/movie",
                params={"api_key": api_key, "query": clean_query},
                timeout=5,
            )
            if r.status_code == 200:
                results = r.json().get("results", [])
                if results:
                    first = results[0]
                    p_path = first.get("poster_path")
                    poster_url = f"https://image.tmdb.org/t/p/w500{p_path}" if p_path else "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?auto=format&fit=crop&w=500&q=80"
                    vote = first.get("vote_average", 7.5)
                    rating_formatted = f"⭐ {vote/2:.1f}/5" if vote > 0 else "⭐ 4.5/5"
                    overview = first.get("overview", "")
                    if not overview:
                        overview = "A critically acclaimed cinematic production celebrated for its nuanced storytelling, compelling character depth, and atmospheric direction."
                    return {
                        "title": first.get("title", clean_query),
                        "poster": poster_url,
                        "rating": rating_formatted,
                        "genres": ["Cinema", "Featured"],
                        "synopsis": overview,
                    }
        except Exception:
            pass

    return {
        "title": clean_query,
        "poster": "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?auto=format&fit=crop&w=500&q=80",
        "rating": "⭐ 4.6/5",
        "genres": ["Cinema", "Drama"],
        "synopsis": "An extraordinary cinematic voyage delving into powerful character conflicts, stunning worldbuilding, and visual mastery.",
    }


def find_recommendations(query_title, top_n=8):
    """Find ML recommendations using TF-IDF genre cosine similarity."""
    matches = movies_df[movies_df["title"].str.contains(query_title, case=False, regex=False, na=False)]
    if matches.empty:
        genre_matches = movies_df[movies_df["genres_clean"].str.contains(query_title, case=False, regex=False, na=False)]
        if not genre_matches.empty:
            matches = genre_matches

    if matches.empty:
        return []

    movie_idx = matches.index[0]
    matched_title = movies_df.iloc[movie_idx]["title"]

    scores = list(enumerate(similarity_matrix[movie_idx]))
    scores.sort(key=lambda x: x[1], reverse=True)

    recommendations = []
    for idx, score in scores:
        if idx == movie_idx:
            continue
        row = movies_df.iloc[idx]
        genre_tokens = [g for g in row["genres_clean"].split() if g]
        details = fetch_movie_details(row["title"])
        recommendations.append(
            {
                "title": row["title"],
                "genres": genre_tokens[:2] if genre_tokens else ["Cinema"],
                "similarity": round(float(score) * 100, 1),
                "poster": details["poster"],
                "rating": details["rating"],
                "synopsis": details["synopsis"],
            }
        )
        if len(recommendations) >= top_n:
            break

    return matched_title, recommendations


# ==============================================================================
# 5. NAVIGATION SIDEBAR (LEFT PANEL)
# ==============================================================================
with st.sidebar:
    st.markdown(
        """
    <div class="sidebar-brand-box">
        <div class="header-brand-logo" style="font-size:24px;">
            <span>🎬</span> CINE•FLOW
        </div>
        <div style="font-size:11px; font-weight:600; color:#a78bfa; letter-spacing:2px; text-transform:uppercase; margin-top:2px;">
            AI Recommendation Hub
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Main Navigation: Dashboard, Discover, Top Rated
    st.markdown('<div class="sidebar-section-header">MAIN</div>', unsafe_allow_html=True)
    main_items = ["Dashboard", "Discover", "Top Rated"]
    for item in main_items:
        is_active = (st.session_state["active_nav"] == item) and (st.session_state["selected_genre"] is None)
        if st.sidebar.button(
            f"{'⚡' if item=='Dashboard' else '🪐' if item=='Discover' else '⭐'}  {item}",
            key=f"sidebar_nav_{item}",
            type="primary" if is_active else "secondary",
        ):
            st.session_state["active_nav"] = item
            st.session_state["selected_genre"] = None
            st.session_state["search_query"] = ""
            st.rerun()

    # Genres Section: Action, Sci-Fi, Drama, Thriller
    # Strictly NO "Watchlist" and NO "Community" sections.
    st.markdown('<div class="sidebar-section-header">GENRES</div>', unsafe_allow_html=True)
    genre_items = ["Action", "Sci-Fi", "Drama", "Thriller"]
    g_cols = st.columns(2)
    for idx, g in enumerate(genre_items):
        col = g_cols[idx % 2]
        is_g_active = st.session_state.get("selected_genre") == g
        if col.button(
            f"#{g}",
            key=f"side_genre_{g}",
            type="primary" if is_g_active else "secondary",
        ):
            if is_g_active:
                st.session_state["selected_genre"] = None
            else:
                st.session_state["selected_genre"] = g
                st.session_state["search_query"] = ""
            st.rerun()


# ==============================================================================
# 6. TOP HEADER BAR
# ==============================================================================
# Left: Brand Logo CINE•FLOW | Center: Prominent Search Bar | Right: Clean Space
h_col1, h_col2, h_col3 = st.columns([1.1, 3.2, 0.4], gap="medium")

with h_col1:
    st.markdown(
        """
    <div style="padding-top: 6px;">
        <span class="header-brand-logo">🎬 CINE•FLOW</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

with h_col2:
    # Prominent Glassmorphic Search Input Bar
    user_search = st.text_input(
        "Search movies, genres, actors...",
        value=st.session_state.get("search_query", ""),
        placeholder="🔍  Search movies, genres, actors...",
        label_visibility="collapsed",
        key="top_search_bar",
    )
    if user_search != st.session_state["search_query"]:
        st.session_state["search_query"] = user_search
        if user_search:
            st.session_state["selected_genre"] = None
        st.rerun()

with h_col3:
    # Right: Clean space (NO login, NO user profile, NO notification icons)
    st.write("")

st.write("")  # Visual breathing room


# ==============================================================================
# 7. MAIN CONTENT AREA & RIGHT PANEL WIDGET
# ==============================================================================
# Responsive 2-Column Layout: Left (Main Dashboard ~73%), Right (Single Widget ~27%)
col_main, col_widget = st.columns([2.9, 1.15], gap="large")

with col_main:
    search_term = st.session_state.get("search_query", "").strip()
    active_genre = st.session_state.get("selected_genre")

    # --------------------------------------------------------------------------
    # CASE 1: GENRE FILTER ACTIVE (When user clicks Action, Sci-Fi, Drama, or Thriller)
    # --------------------------------------------------------------------------
    if active_genre:
        st.markdown(
            f"""
        <div class="search-banner">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div style="font-size:11px; font-weight:800; color:#00f2fe; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:4px;">
                        Exclusive Genre Filter
                    </div>
                    <div style="font-family:'Outfit',sans-serif; font-size:24px; font-weight:800; color:#ffffff;">
                        Recommended Movies For '{html.escape(active_genre)}'
                    </div>
                </div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Get curated films for this genre
        genre_film_names = EXCLUSIVE_GENRE_MOVIES.get(active_genre, [])
        genre_movie_cards = [fetch_movie_details(title) for title in genre_film_names]

        # Also find additional dataset matches exclusively in this genre
        additional_matches = movies_df[movies_df["genres_clean"].str.contains(active_genre, case=False, na=False)]
        for _, row in additional_matches.head(3).iterrows():
            if row["title"] not in genre_film_names:
                genre_movie_cards.append(fetch_movie_details(row["title"]))

        # Render genre movie cards in rows of 4
        for row_start in range(0, len(genre_movie_cards), 4):
            row_items = genre_movie_cards[row_start : row_start + 4]
            row_cols = st.columns(4, gap="medium")

            for idx_c, movie in enumerate(row_items):
                with row_cols[idx_c]:
                    tags_html = "".join([f'<span class="glass-tag">{html.escape(t)}</span>' for t in movie["genres"][:2]])
                    card_html = f"""
                    <div class="glass-movie-card">
                        <div class="card-poster-container">
                            <img src="{movie['poster']}" class="card-poster-image" alt="{html.escape(movie['title'])}" />
                            <div class="card-badge-rating">{movie['rating']}</div>
                        </div>
                        <div class="card-movie-title" title="{html.escape(movie['title'])}">{html.escape(movie['title'])}</div>
                        <div class="card-tags-row">{tags_html}</div>
                        <div class="card-synopsis">{html.escape(movie['synopsis'])}</div>
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)
            st.write("")

        if st.button("← Back to Dashboard", key="btn_clear_genre_filter"):
            st.session_state["selected_genre"] = None
            st.rerun()

    # --------------------------------------------------------------------------
    # CASE 2: SEARCH QUERY ACTIVE (When User Enters a Search Query and Hits Enter)
    # --------------------------------------------------------------------------
    elif search_term:
        res = find_recommendations(search_term, top_n=8)

        if res:
            matched_film, rec_list = res

            st.markdown(
                f"""
            <div class="search-banner">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div style="font-size:11px; font-weight:800; color:#00f2fe; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:4px;">
                            AI Recommendation Matches
                        </div>
                        <div style="font-family:'Outfit',sans-serif; font-size:24px; font-weight:800; color:#ffffff;">
                            Recommended Movies For '{html.escape(search_term)}'
                        </div>
                    </div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            for row_start in range(0, len(rec_list), 4):
                row_items = rec_list[row_start : row_start + 4]
                row_cols = st.columns(4, gap="medium")

                for idx_c, movie in enumerate(row_items):
                    with row_cols[idx_c]:
                        tags_html = "".join([f'<span class="glass-tag">{html.escape(t)}</span>' for t in movie["genres"]])
                        card_html = f"""
                        <div class="glass-movie-card">
                            <div class="card-poster-container">
                                <img src="{movie['poster']}" class="card-poster-image" alt="{html.escape(movie['title'])}" />
                                <div class="card-badge-rating">{movie['rating']}</div>
                            </div>
                            <div class="card-movie-title" title="{html.escape(movie['title'])}">{html.escape(movie['title'])}</div>
                            <div class="card-tags-row">{tags_html}</div>
                            <div class="card-synopsis">{html.escape(movie['synopsis'])}</div>
                        </div>
                        """
                        st.markdown(card_html, unsafe_allow_html=True)
                st.write("")

            if st.button("← Back to Dashboard", key="btn_clear_search_view"):
                st.session_state["search_query"] = ""
                st.rerun()

        else:
            st.markdown(
                f"""
            <div class="search-banner">
                <div style="font-size:18px; font-weight:700; color:#ffffff; margin-bottom:6px;">
                    No direct genre matches found for "{html.escape(search_term)}"
                </div>
                <div style="font-size:13px; color:#94a3b8;">
                    Try searching for "Interstellar", "Toy Story", "Inception", "Heat", or select a genre from the sidebar.
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )
            if st.button("← Return to Dashboard", key="btn_return_empty_view"):
                st.session_state["search_query"] = ""
                st.rerun()

    # --------------------------------------------------------------------------
    # CASE 3: DEFAULT VIEW (Empty Search / Initial Load)
    # --------------------------------------------------------------------------
    else:
        # ----------------------------------------------------------------------
        # SECTION 1: "Top Rated Movies" (Primary horizontal row at top)
        # ----------------------------------------------------------------------
        st.markdown(
            """
        <div class="section-header-wrap">
            <div class="section-title">
                <span class="section-neon-dot"></span>
                Top Rated Movies
            </div>
            <span style="font-size:12px; font-weight:700; color:#00f2fe; letter-spacing:1px; text-transform:uppercase;">Curated Masterpieces</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

        top_rated_movies = [
            {
                "title": "The Shawshank Redemption",
                "genres": ["Drama", "Crime"],
                "rating": "⭐ 4.9/5",
                "poster": CURATED_DATA["The Shawshank Redemption"]["poster"],
                "synopsis": CURATED_DATA["The Shawshank Redemption"]["synopsis"],
            },
            {
                "title": "Interstellar",
                "genres": ["Sci-Fi", "Adventure"],
                "rating": "⭐ 4.9/5",
                "poster": CURATED_DATA["Interstellar"]["poster"],
                "synopsis": CURATED_DATA["Interstellar"]["synopsis"],
            },
            {
                "title": "The Dark Knight",
                "genres": ["Action", "Crime"],
                "rating": "⭐ 4.9/5",
                "poster": CURATED_DATA["The Dark Knight"]["poster"],
                "synopsis": CURATED_DATA["The Dark Knight"]["synopsis"],
            },
            {
                "title": "Dune: Part Two",
                "genres": ["Sci-Fi", "Adventure"],
                "rating": "⭐ 4.8/5",
                "poster": CURATED_DATA["Dune: Part Two"]["poster"],
                "synopsis": CURATED_DATA["Dune: Part Two"]["synopsis"],
            },
        ]

        top_cols = st.columns(4, gap="medium")

        for i, film in enumerate(top_rated_movies):
            with top_cols[i]:
                tags_str = "".join([f'<span class="glass-tag">{html.escape(t)}</span>' for t in film["genres"]])
                card_markup = f"""
                <div class="glass-movie-card">
                    <div class="card-poster-container">
                        <img src="{film['poster']}" class="card-poster-image" alt="{html.escape(film['title'])}" />
                        <div class="card-badge-rating">{film['rating']}</div>
                    </div>
                    <div class="card-movie-title" title="{html.escape(film['title'])}">{html.escape(film['title'])}</div>
                    <div class="card-tags-row">{tags_str}</div>
                    <div class="card-synopsis">{html.escape(film['synopsis'])}</div>
                </div>
                """
                st.markdown(card_markup, unsafe_allow_html=True)

        st.write("")
        st.write("")

        # ----------------------------------------------------------------------
        # SECTION 2: "Trending This Week" (Grid showing trending posters below)
        # ----------------------------------------------------------------------
        st.markdown(
            """
        <div class="section-header-wrap">
            <div class="section-title">
                <span class="section-neon-dot" style="background:#9b51e0; box-shadow:0 0 10px #9b51e0;"></span>
                Trending This Week
            </div>
            <div style="display:flex; gap:8px;">
                <span class="glass-tag" style="background:rgba(0,242,254,0.15); color:#00f2fe; border-color:rgba(0,242,254,0.35);">🔥 Hot Picks</span>
                <span class="glass-tag">Global Top 5</span>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        trending_movies = [
            {
                "title": "Oppenheimer",
                "genres": ["Biography", "Drama"],
                "rating": "⭐ 4.8/5",
                "poster": CURATED_DATA["Oppenheimer"]["poster"],
                "synopsis": CURATED_DATA["Oppenheimer"]["synopsis"],
            },
            {
                "title": "Spider-Man: Across the Spider-Verse",
                "genres": ["Animation", "Sci-Fi"],
                "rating": "⭐ 4.9/5",
                "poster": CURATED_DATA["Spider-Man: Across the Spider-Verse"]["poster"],
                "synopsis": CURATED_DATA["Spider-Man: Across the Spider-Verse"]["synopsis"],
            },
            {
                "title": "Blade Runner 2049",
                "genres": ["Sci-Fi", "Mystery"],
                "rating": "⭐ 4.8/5",
                "poster": CURATED_DATA["Blade Runner 2049"]["poster"],
                "synopsis": CURATED_DATA["Blade Runner 2049"]["synopsis"],
            },
            {
                "title": "The Creator",
                "genres": ["Sci-Fi", "Action"],
                "rating": "⭐ 4.6/5",
                "poster": CURATED_DATA["The Creator"]["poster"],
                "synopsis": CURATED_DATA["The Creator"]["synopsis"],
            },
            {
                "title": "Succession",
                "genres": ["Drama", "Business"],
                "rating": "⭐ 4.9/5",
                "poster": CURATED_DATA["Succession"]["poster"],
                "synopsis": CURATED_DATA["Succession"]["synopsis"],
            },
        ]

        trend_cols = st.columns(5, gap="small")

        for j, t_film in enumerate(trending_movies):
            with trend_cols[j]:
                trend_tags = "".join([f'<span class="glass-tag">{html.escape(t)}</span>' for t in t_film["genres"][:1]])
                trend_markup = f"""
                <div class="glass-movie-card">
                    <div class="card-poster-container" style="aspect-ratio:2 / 2.9;">
                        <img src="{t_film['poster']}" class="card-poster-image" alt="{html.escape(t_film['title'])}" />
                        <div class="card-badge-rating">{t_film['rating']}</div>
                    </div>
                    <div class="card-movie-title" style="font-size:13.5px;" title="{html.escape(t_film['title'])}">{html.escape(t_film['title'])}</div>
                    <div class="card-tags-row">{trend_tags}</div>
                    <div class="card-synopsis" style="-webkit-line-clamp: 2;">{html.escape(t_film['synopsis'])}</div>
                </div>
                """
                st.markdown(trend_markup, unsafe_allow_html=True)


with col_widget:
    # --------------------------------------------------------------------------
    # SINGLE RIGHT PANEL WIDGET: "Top Picks in Sci-Fi"
    # Glassmorphic list item cards with posters, titles, categories, and ratings.
    # Strictly NO "Recently Viewed" section at the bottom.
    # --------------------------------------------------------------------------
    st.markdown(
        """
    <div class="widget-box">
        <div class="widget-title">
            <span style="color:#00f2fe;">🪐</span> Top Picks in Sci-Fi
        </div>
    """,
        unsafe_allow_html=True,
    )

    scifi_picks = [
        {
            "title": "Interstellar",
            "genre": "Sci-Fi / Adventure",
            "rating": "4.9",
            "poster": CURATED_DATA["Interstellar"]["poster"],
        },
        {
            "title": "Dune: Part Two",
            "genre": "Sci-Fi / Space Epic",
            "rating": "4.8",
            "poster": CURATED_DATA["Dune: Part Two"]["poster"],
        },
        {
            "title": "Blade Runner 2049",
            "genre": "Sci-Fi / Cyberpunk",
            "rating": "4.8",
            "poster": CURATED_DATA["Blade Runner 2049"]["poster"],
        },
        {
            "title": "The Creator",
            "genre": "Sci-Fi / AI Rebellion",
            "rating": "4.6",
            "poster": CURATED_DATA["The Creator"]["poster"],
        },
    ]

    for item in scifi_picks:
        item_html = f"""
        <div class="top-pick-item">
            <img src="{item['poster']}" class="top-pick-thumb" alt="{html.escape(item['title'])}" />
            <div class="top-pick-info">
                <div class="top-pick-name">{html.escape(item['title'])}</div>
                <div class="top-pick-cat">{item['genre']}</div>
                <div class="top-pick-rating">⭐ {item['rating']} / 5.0</div>
            </div>
        </div>
        """
        st.markdown(item_html, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# 8. FOOTER
# ==============================================================================
st.markdown(
    """
<div style="margin-top: 50px; padding: 24px 0; border-top: 1px solid rgba(255, 255, 255, 0.08); text-align: center; color: #64748b; font-size: 12px;">
    <div style="margin-bottom: 6px;">
        <strong style="color: #94a3b8;">CINE•FLOW</strong> — Dark Glassmorphic Movie Recommendation Engine.
    </div>
    <div>Powered by Cosine Similarity & TF-IDF Genre Vectorization. Data provided by MovieLens & TMDB.</div>
</div>
""",
    unsafe_allow_html=True,
)
