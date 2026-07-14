"""Build validation inputs for the SSR survey app from the OpinionQA dataset.

OpinionQA (Santurkar et al. 2023, github.com/tatsu-lab/opinions_qa) packages
Pew American Trends Panel questions together with individual human responses.
This script selects 18 questions with clean 5-point ordinal scales from two
waves, rephrases each as an agree/disagree statement compatible with the SSR
app's Likert anchors, and emits:

  opinionqa_questions.xlsx      -> upload to the app's 問卷題目 section
  opinionqa_personas.xlsx       -> upload to the app's 受訪者設定 section
  opinionqa_human_reference.csv -> weighted human answer distributions,
                                   consumed by compare_results.py

Data setup (once):
  Download the OpinionQA bundle (see README.md) so that <data-dir> contains
  human_resp/W92/responses.csv and human_resp/W54/responses.csv.

Usage:
  python prepare_opinionqa_files.py --data-dir /path/to/opinionqa
"""

import argparse
from pathlib import Path

import pandas as pd

# Score 5 = the "agree" end of the SSR scale = the most positive option.
W92_SCORES = {
    "Very good for society": 5,
    "Somewhat good for society": 4,
    "Neither good nor bad for society": 3,
    "Somewhat bad for society": 2,
    "Very bad for society": 1,
}
W54_SCORES = {
    "Helping a lot": 5,
    "Helping a little": 4,
    "Neither helping nor hurting": 3,
    "Hurting a little": 2,
    "Hurting a lot": 1,
}

# (wave, Pew question key, statement shown to the LLM personas)
# Statements are phrased so that "strongly agree" corresponds to score 5
# under the mappings above.
QUESTIONS = [
    ("W92", "SOCIETY_TRANS_W92",
     "Greater social acceptance of people who are transgender is good for our society."),
    ("W92", "SOCIETY_RHIST_W92",
     "Increased public attention to the history of slavery and racism in America is good for our society."),
    ("W92", "SOCIETY_JBCLL_W92",
     "It is good for our society that good-paying jobs increasingly require a college degree."),
    ("W92", "SOCIETY_RELG_W92",
     "The decline in the share of Americans belonging to an organized religion is good for our society."),
    ("W92", "SOCIETY_WHT_W92",
     "White people declining as a share of the U.S. population is good for our society."),
    ("W92", "SOCIETY_GUNS_W92",
     "An increase in the number of guns in the U.S. is good for our society."),
    ("W92", "SOCIETY_SSM_W92",
     "Same-sex marriages being legal in the U.S. is good for our society."),
    ("W54", "ECON5_a_W54",
     "The country's current economic conditions are helping me and my family."),
    ("W54", "ECON5_b_W54",
     "The country's current economic conditions are helping people who are wealthy."),
    ("W54", "ECON5_c_W54",
     "The country's current economic conditions are helping the middle class."),
    ("W54", "ECON5_d_W54",
     "The country's current economic conditions are helping people who are poor."),
    ("W54", "ECON5_e_W54",
     "The country's current economic conditions are helping older adults."),
    ("W54", "ECON5_f_W54",
     "The country's current economic conditions are helping young adults."),
    ("W54", "ECON5_g_W54",
     "The country's current economic conditions are helping people who are white."),
    ("W54", "ECON5_h_W54",
     "The country's current economic conditions are helping people who are black."),
    ("W54", "ECON5_i_W54",
     "The country's current economic conditions are helping people who are Hispanic."),
    ("W54", "ECON5_j_W54",
     "The country's current economic conditions are helping people without college degrees."),
    ("W54", "ECON5_k_W54",
     "The country's current economic conditions are helping people with college degrees."),
]

# 20 personas roughly mirroring the U.S. adult mix on age, gender,
# race/ethnicity, education, income, region, and political lean.
# They describe demographics only — never opinions on the survey topics.
PERSONAS = [
    "You are a 24-year-old white man from rural Ohio working as a warehouse worker. "
    "You finished high school, earn a modest income, and lean Republican.",
    "You are a 27-year-old Hispanic woman from Los Angeles working as a retail sales "
    "associate while taking community college classes. You lean Democrat.",
    "You are a 22-year-old Black man from Atlanta, a college student working part-time. "
    "You are politically liberal.",
    "You are a 29-year-old white woman from Seattle working as a software product "
    "designer with a bachelor's degree. You are politically progressive.",
    "You are a 35-year-old white man from suburban Dallas, a sales manager with a "
    "bachelor's degree and a comfortable income. You lean Republican.",
    "You are a 38-year-old Hispanic man from Phoenix who runs a small landscaping "
    "business. You finished high school and are politically independent.",
    "You are a 42-year-old Black woman from Chicago working as a registered nurse "
    "with a bachelor's degree. You lean Democrat.",
    "You are a 33-year-old Asian American woman from New Jersey working as a financial "
    "analyst with a master's degree. You are politically moderate, leaning Democrat.",
    "You are a 47-year-old white woman from a small town in Missouri working as a "
    "school administrative assistant. You have some college education and lean Republican.",
    "You are a 44-year-old white man from Denver working as an electrician. You have a "
    "trade certification, a middle income, and are politically independent.",
    "You are a 31-year-old white woman from Nashville, a stay-at-home mother of two. "
    "You have some college education, attend church weekly, and lean Republican.",
    "You are a 49-year-old Native American man from Oklahoma working for a county "
    "public works department. You have some college education and lean Democrat.",
    "You are a 55-year-old white man from rural Pennsylvania who works as a long-haul "
    "truck driver. You finished high school and strongly lean Republican.",
    "You are a 52-year-old Black woman from Baltimore working as a city social services "
    "caseworker with a bachelor's degree. You lean Democrat.",
    "You are a 58-year-old white woman from suburban Minneapolis working as a dental "
    "office manager. You have some college education and are a political moderate.",
    "You are a 61-year-old Hispanic man from Miami, a retired postal worker with a high "
    "school education. You lean Democrat but hold some conservative social views.",
    "You are a 63-year-old white man from Charlotte, a semi-retired accountant with a "
    "bachelor's degree and a comfortable income. You lean Republican.",
    "You are a 68-year-old white woman from rural Iowa, a retired elementary school "
    "teacher with a bachelor's degree. You are a political moderate.",
    "You are a 72-year-old white man from Tampa, a retired factory supervisor with a "
    "high school education and a modest fixed income. You lean Republican.",
    "You are a 66-year-old Black woman from Detroit, a retired hospital administrative "
    "clerk. You have some college education and lean Democrat.",
]

QUESTION_XLSX_HEADER = "問卷題目"
PERSONA_XLSX_HEADER = "受訪者描述"


def weighted_distribution(responses: pd.Series, weights: pd.Series, scores: dict) -> tuple[list[float], int]:
    """Weighted P(score=1..5), dropping Refused/NaN and renormalizing."""
    df = pd.DataFrame({"resp": responses, "w": weights}).dropna()
    df = df[df["resp"].isin(scores)]
    df["score"] = df["resp"].map(scores)
    total = df["w"].sum()
    probs = [float(df.loc[df["score"] == s, "w"].sum() / total) for s in range(1, 6)]
    return probs, len(df)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True,
                        help="Directory containing human_resp/W92 and human_resp/W54")
    parser.add_argument("--out-dir", default=str(Path(__file__).parent),
                        help="Where to write the output files (default: this folder)")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    waves = {}
    for wave, weight_col in (("W92", "WEIGHT_W92"), ("W54", "WEIGHT_W54")):
        keys = [k for w, k, _ in QUESTIONS if w == wave]
        waves[wave] = pd.read_csv(
            data_dir / "human_resp" / wave / "responses.csv",
            usecols=keys + [weight_col],
        )

    ref_rows = []
    for wave, key, statement in QUESTIONS:
        scores = W92_SCORES if wave == "W92" else W54_SCORES
        weight_col = f"WEIGHT_{wave}"
        probs, n = weighted_distribution(waves[wave][key], waves[wave][weight_col], scores)
        ref_rows.append({
            "question": statement,
            "wave": wave,
            "pew_key": key,
            "n_human": n,
            **{f"human_p{s}": round(probs[s - 1], 6) for s in range(1, 6)},
            "human_mean": round(sum(p * s for s, p in enumerate(probs, 1)), 4),
        })

    pd.DataFrame(ref_rows).to_csv(out_dir / "opinionqa_human_reference.csv", index=False)
    pd.DataFrame({QUESTION_XLSX_HEADER: [q for _, _, q in QUESTIONS]}).to_excel(
        out_dir / "opinionqa_questions.xlsx", index=False)
    pd.DataFrame({PERSONA_XLSX_HEADER: PERSONAS}).to_excel(
        out_dir / "opinionqa_personas.xlsx", index=False)

    print(f"Wrote {len(QUESTIONS)} questions, {len(PERSONAS)} personas, "
          f"and human reference distributions to {out_dir}")
    for r in ref_rows:
        print(f"  [{r['wave']}] mean={r['human_mean']:.2f} n={r['n_human']}  {r['question'][:70]}")


if __name__ == "__main__":
    main()
