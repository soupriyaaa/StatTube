from __future__ import annotations

import os
import re
import logging
import warnings
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st


# Logging configuration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


# Constants

MODEL_ID: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
LABEL_MAP: dict[str, str] = {
    "positive": "Positive",
    "negative": "Negative",
    "neutral": "Neutral",
    
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive",
}
SENTIMENT_COLORS: dict[str, str] = {
    "Positive": "#22c55e",
    "Neutral": "#f59e0b",
    "Negative": "#ef4444",
}
DEFAULT_BATCH_SIZE: int = 32
MAX_TOKEN_LENGTH: int = 128  # RoBERTa practical limit for tweets/comments



# Mock / Fallback Dataset

def _build_mock_dataframe() -> pd.DataFrame:
    """
    Return a realistic mock YouTube comment dataset.

    Used when the YouTube Data API key is absent or the quota is exhausted.
    Columns mirror exactly what ``fetch_comments`` produces so the rest of
    the pipeline is completely agnostic of the data source.

    Returns
    -------
    pd.DataFrame
        Columns: comment, author, likes, published_at
    """
    records = [
        {
            "comment": "This video is absolutely incredible! Best content on the platform 🔥🔥",
            "author": "TechEnthusiast_42",
            "likes": 312,
            "published_at": "2024-11-01T14:23:00Z",
        },
        {
            "comment": "Honestly disappointing. Expected much more depth on the topic.",
            "author": "CriticalViewer",
            "likes": 87,
            "published_at": "2024-11-01T15:10:00Z",
        },
        {
            "comment": "The explanation at 4:32 finally made this concept click for me. Thank you!",
            "author": "LearningEveryDay",
            "likes": 204,
            "published_at": "2024-11-01T15:45:00Z",
        },
        {
            "comment": "I've watched this three times already. Pure gold 👏",
            "author": "RepeatWatcher99",
            "likes": 456,
            "published_at": "2024-11-01T16:00:00Z",
        },
        {
            "comment": "This is just recycled content from six months ago. Nothing new here.",
            "author": "SkepticalSam",
            "likes": 33,
            "published_at": "2024-11-01T16:30:00Z",
        },
        {
            "comment": "Okay video. Not great, not terrible. Could use better editing.",
            "author": "AverageJoeViewer",
            "likes": 19,
            "published_at": "2024-11-01T17:00:00Z",
        },
        {
            "comment": "The production quality has improved so much! Keep it up 🎬",
            "author": "ProductionNerd",
            "likes": 178,
            "published_at": "2024-11-01T17:20:00Z",
        },
        {
            "comment": "Clickbait title. The actual content doesn't match at all 😤",
            "author": "DisappointedDave",
            "likes": 65,
            "published_at": "2024-11-01T17:55:00Z",
        },
        {
            "comment": "Subscribed immediately after watching this. Rare quality content.",
            "author": "NewSubscriber_2024",
            "likes": 291,
            "published_at": "2024-11-01T18:10:00Z",
        },
        {
            "comment": "The background music is distracting and too loud throughout.",
            "author": "AudioPurist",
            "likes": 44,
            "published_at": "2024-11-01T18:40:00Z",
        },
        {
            "comment": "Neutral take but I appreciated the balanced perspective on this.",
            "author": "BalancedOpinion",
            "likes": 27,
            "published_at": "2024-11-01T19:00:00Z",
        },
        {
            "comment": "Amazing breakdown! Shared with my entire study group 📚",
            "author": "StudyGroupLeader",
            "likes": 389,
            "published_at": "2024-11-01T19:30:00Z",
        },
        {
            "comment": "The timestamps in the description are missing. Very frustrating.",
            "author": "TimestampFan",
            "likes": 52,
            "published_at": "2024-11-01T19:50:00Z",
        },
        {
            "comment": "Informative as always. Nothing groundbreaking but solid content.",
            "author": "RegularFollower",
            "likes": 88,
            "published_at": "2024-11-01T20:15:00Z",
        },
        {
            "comment": "This changed my perspective completely. Life-changing video honestly!",
            "author": "MindBlownUser",
            "likes": 503,
            "published_at": "2024-11-01T20:45:00Z",
        },
        {
            "comment": "Terrible advice. Do NOT follow what they say at 8:00.",
            "author": "ExpertDebunker",
            "likes": 71,
            "published_at": "2024-11-01T21:00:00Z",
        },
        {
            "comment": "Fine. Just fine. Nothing special to write home about tbh.",
            "author": "MehReviewer",
            "likes": 12,
            "published_at": "2024-11-01T21:20:00Z",
        },
        {
            "comment": "The research behind this is top notch. Love the citations! 🧠",
            "author": "AcademicWatcher",
            "likes": 267,
            "published_at": "2024-11-01T21:45:00Z",
        },
        {
            "comment": "Worst video I've seen this month. Complete waste of 15 minutes.",
            "author": "HarshCritic2024",
            "likes": 23,
            "published_at": "2024-11-01T22:00:00Z",
        },
        {
            "comment": "I like the effort but the pacing is a bit slow in the second half.",
            "author": "ConstructiveFeedback",
            "likes": 39,
            "published_at": "2024-11-01T22:30:00Z",
        },
    ]
    df = pd.DataFrame(records)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    return df



# YouTube API Client

@st.cache_resource(show_spinner=False)
def get_youtube_client():
    """
    Build and return an authenticated YouTube Data API v3 client.

    Resolution order for the API key:
    1. ``st.secrets["YOUTUBE_API_KEY"]``  (Streamlit Cloud / secrets.toml)
    2. ``os.environ["YOUTUBE_API_KEY"]``  (local .env / shell export)

    Returns
    -------
    googleapiclient.discovery.Resource | None
        A configured API resource object, or ``None`` if no key is found.
    """
    api_key: Optional[str] = None

    # 1. Streamlit secrets (preferred for cloud deployment)
    try:
        api_key = st.secrets["YOUTUBE_API_KEY"]
        logger.info("API key loaded from st.secrets.")
    except (KeyError, FileNotFoundError):
        logger.debug("YOUTUBE_API_KEY not found in st.secrets.")

    # 2. Environment variable fallback
    if not api_key:
        api_key = os.environ.get("YOUTUBE_API_KEY")
        if api_key:
            logger.info("API key loaded from environment variable.")

    if not api_key:
        logger.warning("No YouTube API key found. Will use mock data.")
        return None

    try:
        from googleapiclient.discovery import build  # type: ignore

        client = build("youtube", "v3", developerKey=api_key)
        logger.info("YouTube API client initialised successfully.")
        return client
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to build YouTube API client: %s", exc)
        return None



# Video ID Extraction

def _extract_video_id(url: str) -> Optional[str]:
    """
    Parse a YouTube URL and return the video ID string.

    Supports the following URL formats:
    - ``https://www.youtube.com/watch?v=VIDEO_ID``
    - ``https://youtu.be/VIDEO_ID``
    - ``https://www.youtube.com/embed/VIDEO_ID``
    - Raw 11-character video IDs

    Parameters
    ----------
    url : str
        A YouTube URL or raw video ID.

    Returns
    -------
    str | None
        The extracted video ID, or ``None`` if parsing fails.
    """
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"(?:embed\/)([0-9A-Za-z_-]{11})",
        r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    # Accept raw 11-char IDs directly
    if re.fullmatch(r"[0-9A-Za-z_-]{11}", url.strip()):
        return url.strip()

    return None


# ---------------------------------------------------------------------------
# Comment Fetching (Primary + Fallback)
# ---------------------------------------------------------------------------
def fetch_comments(video_url: str, max_results: int = 100) -> pd.DataFrame:
    """
    Fetch top-level comments for a YouTube video with graceful fallback.

    Workflow
    --------
    1. Extract the video ID from ``video_url``.
    2. Attempt to retrieve comments via the YouTube Data API v3.
    3. If the API is unavailable (missing key, quota exceeded, network error),
       log a warning, display an ``st.warning`` banner, and return mock data.

    Parameters
    ----------
    video_url : str
        Full YouTube URL or raw 11-character video ID.
    max_results : int, optional
        Maximum number of top-level comments to retrieve (default: 100).
        Must be in the range [1, 500].

    Returns
    -------
    pd.DataFrame
        Columns:
        - ``comment``      : str   — raw comment text
        - ``author``       : str   — display name of the commenter
        - ``likes``        : int   — number of likes on the comment
        - ``published_at`` : datetime (UTC-aware) — publication timestamp
    """
    youtube = get_youtube_client()

    if youtube is None:
        st.warning(
            "⚠️ **Demo Mode — Mock Data Active**: No YouTube API key detected. "
            "Add `YOUTUBE_API_KEY` to your environment or `secrets.toml` to "
            "enable live data fetching. The dashboard below is fully functional "
            "using a curated sample dataset.",
            icon="🔑",
        )
        logger.info("Returning mock dataset (no API client).")
        return _build_mock_dataframe()

    video_id = _extract_video_id(video_url)
    if not video_id:
        st.error("Could not parse a valid video ID from the provided URL.")
        logger.error("Invalid video URL: %s", video_url)
        return _build_mock_dataframe()

    records: list[dict] = []
    next_page_token: Optional[str] = None
    fetched: int = 0

    try:
        while fetched < max_results:
            page_limit = min(100, max_results - fetched)
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=page_limit,
                pageToken=next_page_token,
                textFormat="plainText",
                order="relevance",
            )
            response = request.execute()

            for item in response.get("items", []):
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                records.append(
                    {
                        "comment": snippet.get("textDisplay", ""),
                        "author": snippet.get("authorDisplayName", "Unknown"),
                        "likes": snippet.get("likeCount", 0),
                        "published_at": snippet.get("publishedAt", ""),
                    }
                )
                fetched += 1

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

        if not records:
            st.warning("No comments found for this video. Using mock data.")
            return _build_mock_dataframe()

        df = pd.DataFrame(records)
        df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
        df["likes"] = pd.to_numeric(df["likes"], errors="coerce").fillna(0).astype(int)
        logger.info("Fetched %d comments from YouTube API.", len(df))
        return df

    except Exception as exc:  # pylint: disable=broad-except
        error_str = str(exc)

        if "quotaExceeded" in error_str or "403" in error_str:
            st.warning(
                "**API Quota Exhausted** — Your YouTube Data API v3 daily quota "
                "has been exceeded. The dashboard is running on cached mock data. "
                "Quota resets at midnight Pacific Time.",
                icon="📊",
            )
            logger.warning("YouTube API quota exceeded. Falling back to mock data.")
        elif "commentsDisabled" in error_str or "403" in error_str:
            st.warning(
                "**Comments Disabled** — Comments are disabled for this video. "
                "Displaying mock data instead.",
                icon="🔒",
            )
            logger.warning("Comments disabled for video: %s", video_id)
        else:
            st.warning(
                f"**API Error** — {error_str[:200]}. Falling back to mock data.",
                icon="🛑",
            )
            logger.error("YouTube API error: %s", exc)

        return _build_mock_dataframe()



# Sentiment Processing

@st.cache_resource(show_spinner=False)
def _load_sentiment_pipeline():
    """
    Load the HuggingFace sentiment pipeline (cached across Streamlit reruns).

    The model ``cardiffnlp/twitter-roberta-base-sentiment-latest`` is a
    RoBERTa-base model fine-tuned on ~124M tweets and optimised for
    social-media text including slang, emojis, and informal language.

    Returns
    -------
    transformers.Pipeline
        A ``text-classification`` pipeline ready for inference.
    """
    from transformers import pipeline  # type: ignore

    logger.info("Loading sentiment model: %s", MODEL_ID)
    sentiment_pipe = pipeline(
        task="text-classification",
        model=MODEL_ID,
        tokenizer=MODEL_ID,
        truncation=True,
        max_length=MAX_TOKEN_LENGTH,
        top_k=1,
    )
    logger.info("Sentiment model loaded successfully.")
    return sentiment_pipe


def process_sentiment(df: pd.DataFrame, batch_size: int = DEFAULT_BATCH_SIZE) -> pd.DataFrame:
    """
    Run batched sentiment inference on comment text and enrich the DataFrame.

    The function handles:
    - Truncation to ``MAX_TOKEN_LENGTH`` tokens to respect RoBERTa's input limit.
    - Batch processing for throughput efficiency.
    - Label normalisation to {Positive, Neutral, Negative}.
    - Empty-comment guards.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with at least a ``comment`` column.
    batch_size : int, optional
        Number of samples per inference batch (default: 32).

    Returns
    -------
    pd.DataFrame
        Original DataFrame with two new columns appended:
        - ``sentiment``  : str   — one of {Positive, Neutral, Negative}
        - ``confidence`` : float — model confidence score in [0, 1]
    """
    df = df.copy()

    # Guard: replace NaN / empty strings with a neutral placeholder
    df["comment"] = df["comment"].fillna("").astype(str)
    texts = df["comment"].tolist()

    if not texts:
        df["sentiment"] = pd.Series(dtype=str)
        df["confidence"] = pd.Series(dtype=float)
        return df

    pipe = _load_sentiment_pipeline()

    sentiments: list[str] = []
    confidences: list[float] = []

    with st.spinner("Running sentiment inference…"):
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            results = pipe(batch)
            for result in results:
                # top_k=1 returns a list of one dict
                top = result[0] if isinstance(result, list) else result
                raw_label: str = top.get("label", "neutral").lower()
                score: float = float(top.get("score", 0.0))

                # Normalise label
                normalised = LABEL_MAP.get(raw_label, "Neutral")
                sentiments.append(normalised)
                confidences.append(round(score, 4))

    df["sentiment"] = sentiments
    df["confidence"] = confidences
    logger.info(
        "Sentiment inference complete. Distribution: %s",
        df["sentiment"].value_counts().to_dict(),
    )
    return df


# ---------------------------------------------------------------------------
# Dashboard Rendering
# ---------------------------------------------------------------------------
def build_dashboard(df: pd.DataFrame) -> None:
    """
    Render the full analytics dashboard from a sentiment-enriched DataFrame.

    Sections rendered
    -----------------
    1. **KPI Metrics Row** — Total Comments, Positive %, Top Liked Comment.
    2. **Sentiment Distribution** — Plotly pie chart.
    3. **Engagement Distribution** — Plotly box / violin chart of likes per sentiment.
    4. **Sentiment Timeline** — Plotly line chart of comment volume over time.
    5. **Interactive Comment Explorer** — Filterable dataframe with confidence scores.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: comment, author, likes, published_at,
        sentiment, confidence.
    """
    if df.empty:
        st.error("No data available to render the dashboard.")
        return

    # ------------------------------------------------------------------ #
    # 1. KPI Metrics
    # ------------------------------------------------------------------ #
    st.markdown("---")
    st.markdown("### Key Metrics")

    total = len(df)
    positive_pct = (df["sentiment"] == "Positive").mean() * 100
    negative_pct = (df["sentiment"] == "Negative").mean() * 100
    avg_conf = df["confidence"].mean() * 100

    top_liked_row = df.loc[df["likes"].idxmax()]
    top_liked_text = top_liked_row["comment"][:120] + (
        "…" if len(top_liked_row["comment"]) > 120 else ""
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Comments", f"{total:,}")
    col2.metric("Positive Sentiment", f"{positive_pct:.1f}%")
    col3.metric("Negative Sentiment", f"{negative_pct:.1f}%")
    col4.metric("Avg. Confidence", f"{avg_conf:.1f}%")

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border-left: 4px solid #f59e0b;
            border-radius: 8px;
            padding: 14px 18px;
            margin: 12px 0 24px 0;
            font-size: 0.9rem;
            color: #e2e8f0;
        ">
            <span style="color:#f59e0b; font-weight:700;">Top Liked Comment
            ({top_liked_row['likes']:,} likes):</span><br/>
            <span style="font-style:italic;">"{top_liked_text}"</span>
            <span style="color:#94a3b8; font-size:0.8rem;">
            — {top_liked_row['author']}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------ #
    # 2. Sentiment Distribution — Pie Chart
    # ------------------------------------------------------------------ #
    st.markdown("### Sentiment Distribution")

    dist = df["sentiment"].value_counts().reset_index()
    dist.columns = ["sentiment", "count"]

    fig_pie = px.pie(
        dist,
        names="sentiment",
        values="count",
        color="sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        hole=0.45,
        title="Overall Sentiment Breakdown",
    )
    fig_pie.update_traces(
        textposition="outside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
    )
    fig_pie.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        margin=dict(t=50, b=20),
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # ------------------------------------------------------------------ #
    # 3. Engagement by Sentiment — Box Plot
    # ------------------------------------------------------------------ #
    st.markdown("### Engagement (Likes) by Sentiment")

    fig_box = px.box(
        df,
        x="sentiment",
        y="likes",
        color="sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        points="all",
        hover_data=["author", "comment"],
        title="Like Distribution per Sentiment Category",
        labels={"likes": "Number of Likes", "sentiment": "Sentiment"},
    )
    fig_box.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        font_color="#e2e8f0",
        showlegend=False,
        xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b"),
        margin=dict(t=50, b=20),
    )
    st.plotly_chart(fig_box, use_container_width=True)

    
    # 4. Sentiment Timeline — Line Chart
    
    st.markdown("### Sentiment Over Time")

    df_time = df.copy()
    df_time["hour"] = df_time["published_at"].dt.floor("h")
    timeline = (
        df_time.groupby(["hour", "sentiment"])
        .size()
        .reset_index(name="count")
    )

    if timeline["hour"].nunique() < 2:
        # Not enough time variation — use bar chart instead
        bar_data = df["sentiment"].value_counts().reset_index()
        bar_data.columns = ["sentiment", "count"]
        fig_time = px.bar(
            bar_data,
            x="sentiment",
            y="count",
            color="sentiment",
            color_discrete_map=SENTIMENT_COLORS,
            title="Comment Count by Sentiment (all comments share same timestamp)",
            text_auto=True,
        )
    else:
        fig_time = px.line(
            timeline,
            x="hour",
            y="count",
            color="sentiment",
            color_discrete_map=SENTIMENT_COLORS,
            markers=True,
            title="Hourly Sentiment Volume",
            labels={"hour": "Time (UTC)", "count": "# Comments"},
        )

    fig_time.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        font_color="#e2e8f0",
        xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b"),
        legend=dict(title="Sentiment"),
        margin=dict(t=50, b=20),
        hovermode="x unified",
    )
    st.plotly_chart(fig_time, use_container_width=True)

    
    # 5. Confidence Distribution — Histogram
    
    st.markdown("### Model Confidence Distribution")

    fig_conf = px.histogram(
        df,
        x="confidence",
        color="sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        nbins=20,
        barmode="overlay",
        opacity=0.75,
        title="Inference Confidence Score Distribution",
        labels={"confidence": "Confidence Score", "count": "# Comments"},
    )
    fig_conf.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        font_color="#e2e8f0",
        xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b"),
        margin=dict(t=50, b=20),
    )
    st.plotly_chart(fig_conf, use_container_width=True)

    
    # 6. Interactive Comment Explorer
    
    st.markdown("### Interactive Comment Explorer")

    sentiment_filter = st.multiselect(
        "Filter by Sentiment",
        options=["Positive", "Neutral", "Negative"],
        default=["Positive", "Neutral", "Negative"],
        key="sentiment_filter",
    )
    min_likes = st.slider(
        "Minimum Likes",
        min_value=0,
        max_value=int(df["likes"].max()),
        value=0,
        key="likes_filter",
    )

    filtered = df[
        (df["sentiment"].isin(sentiment_filter)) & (df["likes"] >= min_likes)
    ][["author", "comment", "sentiment", "confidence", "likes", "published_at"]].sort_values(
        "likes", ascending=False
    )

    def _colour_sentiment(val: str) -> str:
        colours = {"Positive": "#166534", "Negative": "#7f1d1d", "Neutral": "#78350f"}
        return f"background-color: {colours.get(val, '')}; color: white; border-radius: 4px; padding: 2px 6px;"

    st.dataframe(
        filtered.style.applymap(_colour_sentiment, subset=["sentiment"]).format(
            {"confidence": "{:.2%}", "likes": "{:,}"}
        ),
        use_container_width=True,
        height=420,
    )
    st.caption(f"Showing **{len(filtered):,}** of **{total:,}** comments.")



# Streamlit Entry Point

def main() -> None:
    """
    Configure the Streamlit page and orchestrate the full pipeline:
    fetch → infer → render.
    """
    st.set_page_config(
        page_title="YT Sentiment Dashboard",
        page_icon="📡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        h1, h2, h3 {
            font-family: 'Space Mono', monospace !important;
        }
        .stApp {
            background: linear-gradient(160deg, #020817 0%, #0f172a 60%, #020817 100%);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
            border-right: 1px solid #334155;
        }
        .stMetric {
            background: rgba(30,41,59,0.7);
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 14px 10px !important;
        }
        .stMetric label { color: #94a3b8 !important; font-size: 0.8rem !important; }
        .stMetric [data-testid="stMetricValue"] { color: #f1f5f9 !important; }
        div[data-testid="stHorizontalBlock"] > div { gap: 12px; }
        .stButton > button {
            background: linear-gradient(135deg, #3b82f6, #6366f1);
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            padding: 0.6rem 1.4rem;
            transition: opacity 0.2s;
        }
        .stButton > button:hover { opacity: 0.85; }
        .stTextInput input, .stSelectbox select {
            background: #1e293b !important;
            color: #e2e8f0 !important;
            border: 1px solid #475569 !important;
            border-radius: 6px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------ #
    # Header
    # ------------------------------------------------------------------ #
    st.markdown(
        """
        <div style="text-align:center; padding: 2rem 0 1rem 0;">
            <h1 style="font-size:2.4rem; color:#f1f5f9; letter-spacing:-1px;">
                StatTube
            </h1>
            <p style="color:#94a3b8; font-size:1rem; max-width:640px; margin:0 auto;">
                Live comment analysis powered by
                <code style="color:#38bdf8;">cardiffnlp/twitter-roberta-base-sentiment-latest</code>.
                Resilient hybrid architecture with automatic fallback to cached data.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------ #
    # Sidebar Controls
    # ------------------------------------------------------------------ #
    with st.sidebar:
        st.markdown("## Configuration")
        st.markdown("---")

        video_url = st.text_input(
            "YouTube Video URL",
            value="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            help="Paste any YouTube URL. Leave as-is to demo with mock data.",
        )
        max_comments = st.slider(
            "Max Comments to Fetch",
            min_value=10,
            max_value=500,
            value=50,
            step=10,
            help="Higher values consume more API quota.",
        )
        batch_size = st.select_slider(
            "Inference Batch Size",
            options=[8, 16, 32, 64],
            value=32,
            help="Larger batches are faster but use more RAM.",
        )

        st.markdown("---")
        
        

        run_btn = st.button("Analyse Comments", use_container_width=True)

    # ------------------------------------------------------------------ #
    # Pipeline Execution
    # ------------------------------------------------------------------ #
    if run_btn or "df_results" not in st.session_state:
        with st.spinner("Fetching comments…"):
            df_raw = fetch_comments(video_url=video_url, max_results=max_comments)

        df_enriched = process_sentiment(df_raw, batch_size=batch_size)
        st.session_state["df_results"] = df_enriched
        st.session_state["last_run"] = datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )

    if "df_results" in st.session_state:
        df = st.session_state["df_results"]
        last_run = st.session_state.get("last_run", "")
        st.caption(f"Last analysis run: **{last_run}** | Comments analysed: **{len(df):,}**")
        build_dashboard(df)

    # ------------------------------------------------------------------ #
    # Footer
    # ------------------------------------------------------------------ #
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align:center; color:#475569; font-size:0.78rem; padding:1rem 0;">
            Built with Streamlit · HuggingFace Transformers · YouTube Data API v3 · Plotly Express<br/>
            
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
