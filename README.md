# StatTube: YouTube Comment Sentiment Analytics Dashboard

A modular machine learning application that ingests live YouTube video comment streams and executes real-time NLP inference using a fine-tuned RoBERTa transformer architecture, rendered via an interactive data visualization dashboard.

## System Architecture

The application is engineered as a decoupled, single-instance data pipeline:
1. **Ingestion Layer:** Accepts YouTube URLs, validates via regex, and interacts with the Google YouTube Data API v3.
2. **Resilience Strategy:** Implements a schema-identical Mock Dataset Fallback engine. In the event of API quota exhaustion, missing credentials, or timeouts, the system gracefully degrades to a curated in-memory dataset, keeping the downstream ML pipeline and visualization layers operational for portfolio reviewers.
3. **Inference Layer:** Processes text batches using a Hugging Face Transformers pipeline running `cardiffnlp/twitter-roberta-base-sentiment-latest` via PyTorch.
4. **Visualization Layer:** Feeds the augmented Pandas DataFrame into an asynchronous Streamlit UI rendering interactive Plotly charts.

## Resilience Design and Quota Error Handling

| Failure Mode | Detection Mechanism | Recovery Strategy |
|---|---|---|
| Missing API Credentials | `st.secrets` and `os.environ` validation failure | Triggers explicit warning banner; executes immediate fallback to mock dataset. |
| Quota Exceeded (`403 quotaExceeded`) | API Client Exception parsing | Suspends upstream API requests; switches to mock data engine with user notification. |
| Restricted Access (`403 commentsDisabled`) | Status code validation | Captures exception context; gracefully halts execution without application crashes. |
| Network Timeout / HTTP 5xx | Request exception wrapper | Implements standard try-catch blocks; surfaces truncated error logs for clean debugging. |
| Malformed / Unparseable URL | Regex validation extraction failures | Returns explicit error feedback to UI input field; prevents upstream empty payloads. |

## Model Selection Rationale

The pipeline utilizes **`cardiffnlp/twitter-roberta-base-sentiment-latest`** based on specific architectural advantages:
* **Domain Calibration:** Pre-trained on ~124 million social media posts, making it optimized for informal structures, typos, and slang typical of YouTube comments.
* **Benchmark Superiority:** Outperforms rule-based engines (e.g., VADER) and vanilla BERT on social media sentiment benchmarks.
* **Resource Efficiency:** Features a practical 128-token execution window, allowing high-throughput batched inference on CPU environments without requiring dedicated GPU clusters.

## Project Structure

```text
yt-sentiment-dashboard/
├── .streamlit/
│   └── secrets.toml     # Local environment secrets storage (explicitly git-ignored)
├── app.py               # Main Streamlit orchestration and layout configuration
├── requirements.txt     # Pinned Python package dependencies
├── .gitignore           # Global file tracking exclusion definitions
└── README.md            # Technical system documentation