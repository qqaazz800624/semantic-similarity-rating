# Semantic-Similarity Rating (SSR)

A Python package implementing the Semantic-Similarity Rating methodology for converting LLM textual responses to Likert scale probability distributions using semantic similarity against reference statements.

## Overview

The SSR methodology addresses the challenge of mapping rich textual responses from Large Language Models (LLMs) to structured Likert scale ratings. Instead of forcing a single numerical rating, SSR preserves the inherent uncertainty and nuance in textual responses by generating probability distributions over all possible Likert scale points.

This package provides a distilled, reusable implementation of the SSR methodology described in the paper "Measuring Synthetic Consumer Purchase Intent Using Semantic-Similarity Ratings" (2025).

## Installation

### Local Development
To install this package locally for development, run:
```bash
pip install -e .
```

### From GitHub Repository
To install this package into your own project from GitHub, run:
```bash
pip install git+https://github.com/pymc-labs/semantic-similarity-rating.git
```

## Quick Start

```python
import polars as po
import numpy as np
from semantic_similarity_rating import ResponseRater

# Create example reference sentences dataframe
reference_set_1 = [
    "Strongly disagree",
    "Disagree",
    "Neutral",
    "Agree",
    "Strongly agree",
]
reference_set_2 = [
    "Disagree a lot",
    "Kinda disagree",
    "Don't know",
    "Kinda agree",
    "Agree a lot",
]
df = po.DataFrame(
    {
        "id": ["set1"] * 5 + ["set2"] * 5,
        "int_response": [1, 2, 3, 4, 5] * 2,
        "sentence": reference_set_1 + reference_set_2,
    }
)

# Initialize rater
rater = ResponseRater(df)

# Create some example synthetic consumer responses
llm_responses = ["I totally agree", "Not sure about this", "Completely disagree"]

# Get PMFs for synthetic consumer responses
pmfs = rater.get_response_pmfs(
    reference_set_id="set1",      # Reference set to score against, or "mean"
    llm_responses=llm_responses,  # List of LLM responses to score
    temperature=1.0,              # Temperature for scaling the PMF
    epsilon=0.0,                  # Small regularization parameter to prevent division by zero and add smoothing
)

# Get survey response PMF
survey_pmf = rater.get_survey_response_pmf(pmfs)

print(survey_pmf)
```

## Web App (GUI)

A Streamlit web app (`app.py`) lets you configure a survey question and a list of respondent personas, run the simulation, and view results — no code editing required.

### 1. Install the app dependencies

```bash
pip install -e ".[app]"
```

### 2. Get a Gemini API key

Every user signs in on the app's login screen with their **own Google account (email) and Gemini API key** — get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey). The key is kept in the browser session only (never saved to disk), and each user's runs consume their own quota.

(A `.env` file with `GOOGLE_API_KEY` is still used by the `quick_start.py` CLI demo, but the web app no longer reads it.)

### 3. Start the app

```bash
./run_app.sh
```

This activates the `ssr` conda environment, checks your `.env`, and starts the app on port 8501 by default. To use a different port:

```bash
PORT=9000 ./run_app.sh
```

Alternatively, run Streamlit directly:

```bash
streamlit run app.py
```

### 4. Open it in your browser

The script prints two URLs:

```
Local:   http://localhost:8501
Network: http://<your-ip>:8501
```

- Open the **Local** URL yourself.
- Share the **Network** URL with colleagues on the same network so they can use the app from their own browser, running on your machine.

> Note: with the Network URL, anyone on the same network can reach the app, but each user must sign in with their own Gemini API key, so runs consume their own quota. To additionally require a shared password before the login screen, add `APP_PASSWORD=your-password` to `.env`.

### Sharing with colleagues outside your network (e.g. working from home)

The Network URL only works on the same local network. To share the app with a colleague anywhere on the internet, use the Cloudflare-tunnel launcher instead:

```bash
brew install cloudflared   # one-time setup
./run_app_public.sh
```

This starts the app **plus** a Cloudflare quick tunnel and prints a public `https://….trycloudflare.com` URL. Your colleague opens that URL in any browser — nothing to install on her side — and enters the `APP_PASSWORD` from your `.env`.

Notes:

- `APP_PASSWORD` must be set in `.env`; the script refuses to start without it, since the URL is reachable from the whole internet.
- The public URL **changes on every launch** — always share the freshly printed one, and send the URL and password through separate channels.
- The app runs on your machine: keep it on and connected (the script uses `caffeinate` to prevent sleep while serving).
- Stop sharing with `Ctrl+C`.

### Deploying to Streamlit Community Cloud (fixed URL, no machine needed)

For a **permanent URL** that works without keeping your computer on, deploy to [Streamlit Community Cloud](https://share.streamlit.io) (free) straight from this GitHub repo:

1. Push the latest code to GitHub (`git push`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in **with your GitHub account**.
3. Click **Create app** → **Deploy a public app from GitHub**, then select:
   - Repository: `qqaazz800624/semantic-similarity-rating`
   - Branch: `main`
   - Main file path: `app.py`
   - App URL: pick a custom subdomain — this becomes the permanent address, e.g. `https://your-name.streamlit.app`
4. (Optional) Open **Advanced settings → Secrets** and add `APP_PASSWORD = "..."` (TOML format) if you want a shared password gate in front of the login screen. No `GOOGLE_API_KEY` secret is needed — every user signs in with their own key.
5. Click **Deploy**. First build takes several minutes (it downloads PyTorch and the embedding model).
6. Restrict who can open the app: in the app's **Settings → Sharing**, set viewers to **Only specific people** and invite your colleague's email — she signs in with that Google/GitHub account once, no password to pass around.

Notes:

- Dependencies are installed from `requirements.txt`, which pins CPU-only PyTorch wheels to fit the free instance.
- Never commit `.env` — on the cloud, the key lives in the app's Secrets instead.
- Free-tier instances have limited memory; if the app crashes on startup, check the app logs in the Cloud dashboard.

### 5. Use the app

0. On the login screen, enter your Google account (email) and your Gemini API key — the key is verified against the Gemini API before the app unlocks. Use the sidebar's **登出 (Log out)** button to switch accounts.
1. (Optional) In the sidebar, pick the **LLM model** used to simulate respondents. The dropdown is fetched live from the Gemini API, so it always reflects the text-generation models your API key can use. The default `gemini-flash-lite-latest` is the fastest and cheapest; `pro`-series models give higher-quality answers but are slower and cost more.
2. Enter the survey statement.
3. Add/remove respondent personas (each one a free-text description).
4. Click **開始模擬問卷調查 (Run survey)**.
5. Review per-respondent and aggregated Likert-scale results (the model used is shown at the top of the results), and optionally download them as CSV.

## Methodology

The ESR methodology works by:
1. Defining reference statements for each Likert scale point
2. Computing cosine similarities between LLM response embeddings and reference statement embeddings
3. Converting similarities to probability distributions using minimum similarity subtraction and normalization
4. Optionally applying temperature scaling for distribution control

## Core Components

- `ResponseRater`: Main class implementing the SSR methodology
- `get_response_pmfs()`: Convert LLM response embeddings to PMFs using specified reference set

## Citation

```
Maier, B. F., Aslak, U., Fiaschi, L., Pappas, K., Wiecki, T. (2025). Measuring Synthetic Consumer Purchase Intent Using Embeddings-Similarity Ratings.
```

## License

MIT License
