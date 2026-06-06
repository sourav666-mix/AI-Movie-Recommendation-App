"""
app.py — CineMatch Streamlit Movie Recommendation Website
=========================================================
Works with Python 3.14+ — no torch required!
Uses only: streamlit, numpy, scikit-learn (all Python 3.14 compatible)

Run:
    streamlit run app.py

Place movie_model.pkl in the same folder before running.
"""

import streamlit as st
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch — AI Movie Recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif !important; }
.stApp { background: #080c14; color: #e8eaf0; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

.stApp::before {
    content: '';
    position: fixed; inset: 0; z-index: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='4'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='.035'/%3E%3C/svg%3E");
    background-size: 180px; pointer-events: none;
}
.spotlight {
    position: fixed; top: -25vh; left: 50%; transform: translateX(-50%);
    width: 80vw; height: 60vw;
    background: radial-gradient(ellipse, rgba(232,200,74,.07) 0%, transparent 68%);
    pointer-events: none; z-index: 0;
    animation: pulse 7s ease-in-out infinite alternate;
}
@keyframes pulse { to { opacity:.45; transform:translateX(-50%) scale(1.08); } }

.navbar {
    display:flex; align-items:center; justify-content:space-between;
    padding:26px 52px; border-bottom:1px solid rgba(255,255,255,.07);
    background:rgba(8,12,20,.8); backdrop-filter:blur(14px);
    position:sticky; top:0; z-index:100;
}
.logo { font-family:'Bebas Neue',sans-serif; font-size:2rem; letter-spacing:.12em; color:#e8c84a; text-shadow:0 0 28px rgba(232,200,74,.25); }
.logo span { color:#e8eaf0; }
.nav-badge { font-size:.7rem; letter-spacing:.2em; text-transform:uppercase; color:#6b7590; border:1px solid rgba(255,255,255,.1); padding:5px 14px; border-radius:99px; }

.hero { text-align:center; padding:80px 24px 52px; }
.eyebrow { display:inline-block; font-size:.7rem; letter-spacing:.28em; text-transform:uppercase; color:#e8c84a; padding:5px 18px; border:1px solid rgba(232,200,74,.28); border-radius:99px; margin-bottom:22px; }
.hero-title { font-family:'Bebas Neue',sans-serif; font-size:clamp(3.2rem,9vw,6.5rem); line-height:.9; letter-spacing:.04em; color:#e8eaf0; }
.hero-title em { font-style:normal; color:#e8c84a; text-shadow:0 0 48px rgba(232,200,74,.3); }
.hero-sub { margin-top:20px; font-size:1.05rem; font-weight:300; color:#6b7590; }

.stTextInput>div>div>input {
    background:#0f1624 !important; border:1.5px solid rgba(255,255,255,.1) !important;
    border-radius:12px !important; color:#e8eaf0 !important;
    font-family:'DM Sans',sans-serif !important; font-size:1.05rem !important;
    padding:16px 20px !important; transition:border-color .25s,box-shadow .25s !important;
}
.stTextInput>div>div>input:focus { border-color:#e8c84a !important; box-shadow:0 0 0 3px rgba(232,200,74,.15) !important; }
.stTextInput>div>div>input::placeholder { color:#6b7590 !important; }

.stRadio>div { flex-direction:row !important; gap:8px; justify-content:center; }
.stRadio>div>label { background:rgba(255,255,255,.05) !important; border:1px solid rgba(255,255,255,.1) !important; border-radius:99px !important; padding:8px 24px !important; color:#6b7590 !important; cursor:pointer !important; font-size:.88rem !important; }
.stRadio>div>label:has(input:checked) { background:#e8c84a !important; color:#000 !important; border-color:#e8c84a !important; font-weight:600 !important; }

.stSelectbox>div>div { background:#0f1624 !important; border:1.5px solid rgba(255,255,255,.1) !important; border-radius:12px !important; color:#e8eaf0 !important; }

.stButton>button {
    width:100%; background:#e8c84a !important; color:#000 !important;
    font-family:'DM Sans',sans-serif !important; font-weight:700 !important;
    font-size:.95rem !important; border:none !important; border-radius:12px !important;
    padding:14px 0 !important; letter-spacing:.04em;
    box-shadow:0 4px 20px rgba(232,200,74,.2) !important;
}
.stButton>button:hover { background:#f5d96a !important; box-shadow:0 6px 28px rgba(232,200,74,.4) !important; transform:translateY(-1px) !important; }

.results-header { display:flex; align-items:baseline; gap:10px; padding-bottom:18px; border-bottom:1px solid rgba(255,255,255,.07); margin-bottom:28px; }
.res-label { font-family:'Bebas Neue',sans-serif; font-size:1.5rem; letter-spacing:.06em; color:#e8eaf0; }
.res-query { font-family:'Bebas Neue',sans-serif; font-size:1.5rem; letter-spacing:.06em; color:#e8c84a; }
.res-count { margin-left:auto; font-size:.75rem; letter-spacing:.12em; text-transform:uppercase; color:#6b7590; }

.movie-card { background:#131c2e; border:1px solid rgba(255,255,255,.07); border-radius:14px; padding:22px 22px 18px; position:relative; overflow:hidden; height:100%; }
.movie-card::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; background:linear-gradient(90deg,#e8c84a,#ff6b4a); }
.card-rank { position:absolute; top:14px; right:16px; font-family:'Bebas Neue',sans-serif; font-size:2.2rem; color:rgba(255,255,255,.06); line-height:1; }
.card-title { font-family:'Bebas Neue',sans-serif; font-size:1.2rem; letter-spacing:.04em; color:#e8eaf0; line-height:1.15; margin-bottom:10px; padding-right:32px; }
.card-tags { font-size:.78rem; font-weight:300; color:#6b7590; line-height:1.55; margin-bottom:14px; }
.score-row { display:flex; align-items:center; gap:10px; }
.score-bg { flex:1; height:3px; background:rgba(255,255,255,.06); border-radius:99px; overflow:hidden; }
.score-fill { height:100%; border-radius:99px; background:linear-gradient(90deg,#e8c84a,#ff6b4a); }
.score-num { font-size:.75rem; letter-spacing:.08em; color:#e8c84a; font-weight:600; min-width:44px; text-align:right; }

.stat-box { background:#131c2e; border:1px solid rgba(255,255,255,.07); border-radius:12px; padding:20px; text-align:center; }
.stat-num { font-family:'Bebas Neue',sans-serif; font-size:2rem; color:#e8c84a; letter-spacing:.04em; }
.stat-label { font-size:.72rem; color:#6b7590; letter-spacing:.14em; text-transform:uppercase; margin-top:4px; }

.search-panel { background:#0f1624; border:1px solid rgba(255,255,255,.08); border-radius:16px; padding:32px 32px 28px; }
.no-model { background:rgba(255,107,74,.06); border:1px solid rgba(255,107,74,.25); border-radius:12px; padding:24px 28px; text-align:center; color:#ff6b4a; font-size:.95rem; line-height:1.7; }
.no-model code { background:rgba(255,255,255,.08); padding:2px 8px; border-radius:4px; font-family:monospace; font-size:.85rem; color:#e8eaf0; }
</style>
""", unsafe_allow_html=True)


# ── Spotlight & Nav ────────────────────────────────────────────────────────────
st.markdown('<div class="spotlight"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="navbar">
  <div class="logo">Cine<span>Match</span></div>
  <div class="nav-badge">✦ Deep Learning AI</div>
</div>
""", unsafe_allow_html=True)


# ── Load model (no torch needed — pure numpy) ──────────────────────────────────
MODEL_PATH = "movie_model.pkl"

@st.cache_resource(show_spinner=False)
def load_model(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    titles     = data["titles"]
    # tags may be missing in older pkl files — fall back to empty strings
    tags       = data.get("tags", [""] * len(titles))
    embeddings = data["embeddings"].astype(np.float32)
    # L2-normalise so dot product == cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms
    return titles, tags, embeddings

model_loaded = os.path.exists(MODEL_PATH)

if model_loaded:
    with st.spinner("Loading model..."):
        TITLES, TAGS, EMBEDDINGS = load_model(MODEL_PATH)


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="eyebrow">✦ Deep Learning Movie AI</div>
  <div class="hero-title">Find Your Next<br><em>Obsession</em></div>
  <div class="hero-sub">Semantic AI that understands what you love — not just keywords.</div>
</div>
""", unsafe_allow_html=True)


# ── No model warning ───────────────────────────────────────────────────────────
if not model_loaded:
    st.markdown("""
    <div style="max-width:680px; margin:0 auto; padding:0 24px;">
      <div class="no-model">
        ⚠️ <strong>movie_model.pkl not found</strong><br><br>
        Run <code>python download_model.py</code> to build the model,<br>
        then place <code>movie_model.pkl</code> in the same folder as this app.
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Stats row ──────────────────────────────────────────────────────────────────
_, sc1, sc2, sc3, _ = st.columns([1, 1, 1, 1, 1])
with sc1:
    st.markdown(f'<div class="stat-box"><div class="stat-num">{len(TITLES):,}</div><div class="stat-label">Movies</div></div>', unsafe_allow_html=True)
with sc2:
    st.markdown(f'<div class="stat-box"><div class="stat-num">{EMBEDDINGS.shape[1]}</div><div class="stat-label">Embedding Dims</div></div>', unsafe_allow_html=True)
with sc3:
    st.markdown('<div class="stat-box"><div class="stat-num">BERT</div><div class="stat-label">Model Type</div></div>', unsafe_allow_html=True)

st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)


# ── Search panel ───────────────────────────────────────────────────────────────
_, main_col, _ = st.columns([1, 4, 1])

with main_col:
    st.markdown('<div class="search-panel">', unsafe_allow_html=True)

    mode = st.radio(
        "Search mode",
        ["🎬  By Movie Title", "✨  By Description"],
        horizontal=True,
        label_visibility="collapsed",
    )
    is_title_mode = mode.startswith("🎬")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    if is_title_mode:
        query = st.selectbox(
            "Movie title",
            options=[""] + sorted(TITLES),
            index=0,
            label_visibility="collapsed",
        )
        st.caption("💡 Select any movie to find semantically similar films")
    else:
        query = st.text_input(
            "Describe what you want",
            placeholder="e.g. heist thriller with unexpected twists, or romantic comedy in Paris…",
            label_visibility="collapsed",
        )
        st.caption("💡 Describe a vibe, genre, mood, or story")

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    n_results = st.slider("Number of recommendations", min_value=3, max_value=20, value=9, step=1)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    search_btn = st.button("Find Films →", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ── Recommend by title (pure numpy — no torch) ────────────────────────────────
def recommend_by_title(movie, top_n):
    lower = [t.lower() for t in TITLES]
    if movie.lower() not in lower:
        return None, []
    idx    = lower.index(movie.lower())
    scores = EMBEDDINGS @ EMBEDDINGS[idx]          # dot product on normalised = cosine
    top_idx = np.argsort(scores)[::-1]
    results = []
    for i in top_idx:
        if i == idx: continue
        results.append({"title": TITLES[i], "score": float(scores[i]), "tag": TAGS[i]})
        if len(results) >= top_n: break
    return TITLES[idx], results

# ── Recommend by free-text (TF-IDF fallback — no torch needed) ────────────────
@st.cache_resource(show_spinner=False)
def build_tfidf(_tags):
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(stop_words="english", max_features=10000)
    mat   = tfidf.fit_transform(_tags)
    return tfidf, mat

def recommend_by_query(query_text, top_n):
    tfidf, mat = build_tfidf(tuple(TAGS))
    qvec   = tfidf.transform([query_text])
    scores = cosine_similarity(qvec, mat).flatten()
    top_idx = np.argsort(scores)[::-1][:top_n]
    return [{"title": TITLES[i], "score": float(scores[i]), "tag": TAGS[i]} for i in top_idx]


# ── Search execution ───────────────────────────────────────────────────────────
if search_btn and query and query.strip():
    st.markdown("<div style='height:48px'></div>", unsafe_allow_html=True)
    _, res_col, _ = st.columns([1, 4, 1])

    with res_col:
        with st.spinner("Finding your next obsession…"):
            if is_title_mode:
                matched, results = recommend_by_title(query, n_results)
                if not results:
                    st.error(f"Movie **'{query}'** not found in dataset.")
                    st.stop()
                verb, label = "Similar to", matched
            else:
                results = recommend_by_query(query, n_results)
                verb, label = "Matches for", f'"{query}"'

        st.markdown(f"""
        <div class="results-header">
          <span class="res-label">{verb} &nbsp;</span>
          <span class="res-query">{label}</span>
          <span class="res-count">{len(results)} films</span>
        </div>
        """, unsafe_allow_html=True)

        max_score = max(r["score"] for r in results) if results else 1
        for row_start in range(0, len(results), 3):
            row_items = results[row_start: row_start + 3]
            cols = st.columns(3)
            for col, r in zip(cols, row_items):
                rank     = row_start + row_items.index(r) + 1
                pct      = round((r["score"] / max_score) * 100) if max_score > 0 else 0
                preview  = r["tag"][:145] + "…" if len(r["tag"]) > 145 else r["tag"]
                score_d  = f"{r['score']*100:.1f}%"
                with col:
                    st.markdown(f"""
                    <div class="movie-card">
                      <div class="card-rank">{str(rank).zfill(2)}</div>
                      <div class="card-title">{r['title']}</div>
                      <div class="card-tags">{preview}</div>
                      <div class="score-row">
                        <div class="score-bg"><div class="score-fill" style="width:{pct}%"></div></div>
                        <div class="score-num">{score_d}</div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

elif search_btn and not query:
    st.warning("Please enter a movie title or description first.")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:80px'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;padding:28px;border-top:1px solid rgba(255,255,255,.07);font-size:.78rem;color:#6b7590;'>
  CineMatch uses deep learning embeddings from <strong style='color:#e8c84a'>all-MiniLM-L6-v2</strong> for title search,
  and TF-IDF for free-text queries — both running fully offline on your laptop.
</div>
""", unsafe_allow_html=True)