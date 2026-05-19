# 📡 YouTube Comment Sentiment Analysis Dashboard

A production-grade ML application that fetches live YouTube comments and runs
real-time sentiment analysis using a fine-tuned RoBERTa model, rendering an
interactive analytics dashboard built with Streamlit and Plotly.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit Frontend                       │
│   Sidebar Config ──► Run Button ──► Dashboard (KPIs + Charts)│
└───────────────────────────┬─────────────────────────────────┘
                            │
                ┌───────────▼────────────┐
                │   fetch_comments()     │
                │   Hybrid Data Layer    │
                └───────────┬────────────┘
                            │
           ┌────────────────┴───────────────────┐
           │                                    │
  ┌────────▼──────────┐             ┌───────────▼──────────┐
  │  YouTube Data      │    FAIL     │   Mock Dataset       │
  │  API v3 (Primary) │────────────►│   (Fallback)         │
  │  google-api-python│  quota /    │   In-memory DataFrame │
  │  -client          │  no key /   │   20 curated comments │
  └────────┬──────────┘  network    └───────────┬──────────┘
           │                                    │
           └────────────────┬───────────────────┘
                            │  pd.DataFrame
                            │  [comment, author, likes, published_at]
                ┌───────────▼────────────┐
                │  process_sentiment()   │
                │  HuggingFace Pipeline  │
                │  cardiffnlp/roberta    │
                │  Batched inference     │
                └───────────┬────────────┘
                            │  + [sentiment, confidence]
                ┌───────────▼────────────┐
                │   build_dashboard()    │
                │   KPI Metrics          │
                │   Plotly Pie Chart     │
                │   Box Plot (Likes)     │
                │   Timeline Line Chart  │
                │   Confidence Histogram │
                │   Filterable Explorer  │
                └────────────────────────┘
```

---

## 🛡️ Resilience Design: API Rate-Limiting & Quota Handling

This project demonstrates production-ready systems thinking through a
**Hybrid Resilient Data Model**:

| Failure Mode | Detection | Recovery Strategy |
|---|---|---|
| Missing API key | `st.secrets` + `os.environ` both absent | Immediate fallback to mock data, `st.warning` banner |
| Quota exceeded (`403 quotaExceeded`) | Exception message parsing | Graceful fallback + user-facing explanation |
| Comments disabled on video | `403 commentsDisabled` | Graceful fallback + specific error message |
| Network timeout / 5xx | Generic `Exception` catch | Fallback + truncated error shown to user |
| Invalid / unparseable URL | `_extract_video_id()` returns `None` | `st.error` + fallback |
| Empty API response | `len(records) == 0` guard | Fallback with warning |

The fallback dataset is **schema-identical** to live API data, so the entire
downstream pipeline (sentiment inference, all charts, explorer) remains 100%
functional in demo mode. This is important for portfolio reviewers who may not
have API keys.

---

## 🤖 Model Selection Rationale

**`cardiffnlp/twitter-roberta-base-sentiment-latest`**

- Fine-tuned on **~124 million tweets** — optimal for informal, emoji-rich text.
- Outperforms generic BERT-base on social-media sentiment benchmarks.
- Three-class output: Positive / Neutral / Negative (aligned with YouTube use case).
- Practical token limit of 128 makes it fast on CPU for batch inference.
- Available on HuggingFace Hub with no authentication required.

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-username/yt-sentiment-dashboard.git
cd yt-sentiment-dashboard
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API Key (optional — dashboard works without it)

**Option A — `.env` file (local development):**
```
YOUTUBE_API_KEY=AIza...your_key_here
```

**Option B — Streamlit secrets (Streamlit Cloud deployment):**
Create `.streamlit/secrets.toml`:
```toml
YOUTUBE_API_KEY = "AIza...your_key_here"
```

### 3. Run

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🔑 Obtaining a YouTube Data API v3 Key

1. Go to [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project (or select existing).
3. Enable **YouTube Data API v3** under *APIs & Services → Library*.
4. Create credentials: *APIs & Services → Credentials → Create Credentials → API Key*.
5. (Recommended) Restrict the key to YouTube Data API v3 only.

**Default free quota**: 10,000 units/day. A `commentThreads.list` call costs
**1 unit** per request, so you can fetch up to ~1,000 comment pages per day.

---

## 📁 Project Structure

```
yt-sentiment-dashboard/
├── app.py                  # Single-file Streamlit application
├── requirements.txt        # Pinned dependencies
├── README.md               # This file
└── .streamlit/
    └── secrets.toml        # (gitignored) API key storage
```

---

## ☁️ Deployment on Streamlit Cloud

1. Push to a public GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect the repo.
3. Under *Advanced Settings → Secrets*, add:
   ```
   YOUTUBE_API_KEY = "AIza..."
   ```
4. Deploy. The app will launch in demo mode if the key is absent.

---

## 🧱 Code Quality Standards

- **PEP 8** compliant (verified with `flake8`).
- Full **type hints** on all public functions.
- **Google-style docstrings** with Parameters / Returns sections.
- `@st.cache_resource` on the model loader and API client to prevent redundant
  initialisation across Streamlit reruns.
- Batched inference via HuggingFace pipeline for throughput efficiency.
- Session state (`st.session_state`) used to persist results across sidebar
  interactions without re-running expensive API + ML calls.

---

## 📊 Dashboard Sections

| Section | Chart Type | Key Insight |
|---|---|---|
| KPI Metrics | Metric cards | Total, % Positive, % Negative, Avg confidence |
| Top Liked Comment | Styled callout | Highest-engagement comment |
| Sentiment Distribution | Donut pie chart | Overall tone at a glance |
| Engagement by Sentiment | Box plot + jitter | Do positive comments get more likes? |
| Sentiment Over Time | Line chart (hourly) | Sentiment trend during video virality |
| Confidence Distribution | Overlapping histogram | Model certainty per class |
| Comment Explorer | Styled dataframe | Filter by sentiment + min likes |

---

## 🔮 Potential Extensions

- **Real-time streaming** via YouTube Live Chat API with `asyncio` polling.
- **Multi-video comparison** — analyse sentiment across a playlist.
- **Aspect-based sentiment** — extract topics (audio, visuals, content) separately.
- **Redis caching layer** to persist results across sessions and reduce API calls.
- **PostgreSQL sink** — persist all analysed comments for longitudinal analysis.
- **Alerting** — trigger a Slack webhook when negative sentiment exceeds a threshold.

---

## 📄 License

MIT License. Free for personal and commercial use.
