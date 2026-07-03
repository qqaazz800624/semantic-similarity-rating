"""Quick-start demo: LLM personas + Semantic Similarity Rating (SSR).

Pipeline (following Maier et al., arXiv:2510.08338):
1. LLM personas give free-text answers to a survey statement.
2. ResponseRater maps each answer to a PMF over a 5-pt Likert scale
   via embedding similarity to reference (anchor) statements.
3. Individual PMFs are averaged into a survey-level distribution.

Setup:
    Put GOOGLE_API_KEY="your-key" in a .env file (never hardcode keys)
    pip install google-genai tqdm python-dotenv
"""

import os
import time

import polars as po
from dotenv import load_dotenv
from google import genai
from tqdm import tqdm

from semantic_similarity_rating import ResponseRater

load_dotenv()

# --- 1. LLM client (new google-genai SDK; key from environment) ---
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("Set the GOOGLE_API_KEY environment variable first.")
client = genai.Client(api_key=api_key)
MODEL_NAME = "gemini-flash-lite-latest"

# --- 2. Reference Likert anchor sets ---
# The paper averages over multiple differently-worded reference sets to
# reduce sensitivity to any single anchor phrasing (used via id="mean").
reference_sets = {
    "plain": [
        "Strongly disagree",
        "Disagree",
        "Neutral / No opinion",
        "Agree",
        "Strongly agree",
    ],
    "conversational": [
        "No, I completely disagree with that.",
        "I mostly disagree with that.",
        "I'm not sure; I don't have a strong opinion either way.",
        "I mostly agree with that.",
        "Yes, I completely agree with that.",
    ],
    "first_person": [
        "That is definitely not true for me.",
        "That is mostly not true for me.",
        "That is only partly true for me.",
        "That is mostly true for me.",
        "That is definitely true for me.",
    ],
}

df_reference = po.DataFrame(
    {
        "id": [set_id for set_id, s in reference_sets.items() for _ in s],
        "int_response": [1, 2, 3, 4, 5] * len(reference_sets),
        "sentence": [sent for s in reference_sets.values() for sent in s],
    }
)

# --- 3. Personas and survey question ---
survey_question = (
    "I believe working remotely significantly increases my daily productivity."
)

personas = [
    "You are a 25-year-old software engineer who loves cutting-edge tech and "
    "thoroughly enjoys the flexibility of working from home.",
    "You are a 50-year-old traditional banking executive who values face-to-face "
    "communication and worries about management difficulties caused by remote work.",
    "You are a 30-year-old marketing planner who feels that while remote work "
    "offers freedom, it makes team brainstorming sessions highly inefficient.",
]


# --- 4. LLM persona response function (with simple retry) ---
def get_llm_free_text_response(persona: str, question: str, max_retries: int = 3) -> str:
    prompt = f"""
Background Persona: {persona}

Survey Statement: "{question}"

Task: What is your realistic opinion on this statement based on your background?
Please role-play the persona and express your true feelings in exactly one
sentence (around 15-30 words).

Constraint: Output the direct response text only. Do not include any numerical
ratings, labels, or quotes. Must respond in English.
"""
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME, contents=prompt
            )
            return response.text.strip()
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2**attempt)  # exponential backoff: 1s, 2s, ...


# --- 5. Collect free-text responses ---
print("--- Generating LLM Persona Responses ---")
llm_responses = []
for i, persona in enumerate(tqdm(personas, desc="Generating"), 1):
    resp = get_llm_free_text_response(persona, survey_question)
    tqdm.write(f'Respondent {i} Output:\n   "{resp}"\n')
    llm_responses.append(resp)

# --- 6. SSR: responses -> Likert PMFs ---
print("--- Executing Semantic Similarity Rating (SSR) ---")
rater = ResponseRater(df_reference)

# temperature=1.0: keep the raw SSR distribution. T < 1 sharpens the PMF
# toward one-hot, discarding the uncertainty SSR is designed to preserve.
pmfs = rater.get_response_pmfs(
    reference_set_id="mean",  # average PMFs across all reference sets
    llm_responses=llm_responses,
    temperature=1.0,
    epsilon=1e-5,
)

# --- 7. Aggregate into a survey-level distribution ---
survey_pmf = rater.get_survey_response_pmf(pmfs)

# --- 8. Report ---
scale_labels = reference_sets["plain"]

print("\n" + "=" * 70)
print("SSR PIPELINE EXECUTION REPORT")
print("=" * 70)

for i, (resp, pmf) in enumerate(zip(llm_responses, pmfs), 1):
    print(f'\nRespondent {i} Free-Text: "{resp}"')
    print(
        po.DataFrame(
            {
                "Score": [1, 2, 3, 4, 5],
                "Scale Anchor": scale_labels,
                "Probability (PMF)": [f"{p * 100:.2f}%" for p in pmf],
            }
        )
    )

print("\n" + "-" * 70)
print("OVERALL AGGREGATED SURVEY DISTRIBUTION (Survey PMF)")
print("-" * 70)
print(
    po.DataFrame(
        {
            "Score": [1, 2, 3, 4, 5],
            "Scale Anchor": scale_labels,
            "Aggregated Probability": [f"{p * 100:.2f}%" for p in survey_pmf],
        }
    )
)
