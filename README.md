# 🎬 CineMatch — AI Movie Recommendation App

> Semantic AI that understands what you love — not just keywords.

CineMatch is a Streamlit-powered movie recommendation web app that uses deep learning embeddings (`all-MiniLM-L6-v2`) for title-based search and TF-IDF for free-text query matching — all running **fully offline** on your machine. No API keys, no cloud, no fuss.

---

## ✨ Features

- **Search by Movie Title** — Find semantically similar films based on a movie you already love
- **Search by Description** — Describe a vibe, genre, mood, or plot and get matching films
- **Deep Learning Embeddings** — Uses BERT-based `all-MiniLM-L6-v2` for title similarity
- **TF-IDF Fallback** — Fast free-text matching for description queries via scikit-learn
- **Fully Offline** — Runs entirely on your local machine after setup
- **Cinematic UI** — Dark, polished interface built with custom CSS

---

## 🖼️ Preview

<!-- Add a screenshot here -->
> _Add a screenshot of the app here_

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/cinematch.git
   cd cinematch
   ```

2. **Install dependencies**

   ```bash
   pip install streamlit numpy scikit-learn sentence-transformers
   ```

3. **Build the model**

   ```bash
   python download_model.py
   ```

   This will generate `movie_model.pkl` in the project root.

4. **Run the app**

   ```bash
   streamlit run app.py
   ```

5. Open your browser at `http://localhost:8501`

---

## 📁 Project Structure

```
cinematch/
├── app.py                  # Main Streamlit application
├── download_model.py       # Script to build movie_model.pkl
├── movie_model.pkl         # Pre-built model file (generated, not committed)
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 🧠 How It Works

| Mode | Method | Description |
|------|--------|-------------|
| By Movie Title | Cosine similarity on L2-normalised BERT embeddings | Finds movies closest in semantic embedding space |
| By Description | TF-IDF vectorisation + cosine similarity | Matches free-text queries against movie tag corpus |

The model file (`movie_model.pkl`) stores:
- `titles` — list of movie titles
- `tags` — genre/keyword tags per movie
- `embeddings` — pre-computed BERT embeddings (numpy float32 array)

---

## 📦 Requirements

```txt
streamlit
numpy
scikit-learn
sentence-transformers
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| Model path | `movie_model.pkl` | Path to the pickled model file |
| Max recommendations | 20 | Adjustable via the UI slider |
| TF-IDF max features | 10,000 | Set in `build_tfidf()` |

---

## 🙋 FAQ

**Q: The app says `movie_model.pkl not found`. What do I do?**  
Run `python download_model.py` first to generate the model file.

**Q: Does this require a GPU?**  
No. The app runs on CPU using pre-computed embeddings. No PyTorch is required at runtime.

**Q: Can I use my own movie dataset?**  
Yes — modify `download_model.py` to use your own titles, tags, and embeddings, then re-export the `.pkl` file.

---

## 📄 License

[MIT](LICENSE)

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

<div align="center">
  Made with ❤️ and 🎬
</div>
