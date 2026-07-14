# Validating the SSR app against real human survey data (OpinionQA)

This folder contains everything needed to benchmark the app's simulated
respondents against **real human answer distributions** from Pew Research's
American Trends Panel, via the [OpinionQA dataset](https://github.com/tatsu-lab/opinions_qa)
(Santurkar et al. 2023). Metrics follow the SSR paper
([Maier et al., arXiv:2510.08338](https://arxiv.org/abs/2510.08338)).

## Files

| File | Purpose |
|---|---|
| `opinionqa_questions.xlsx` | 18 survey statements — upload in the app's **問卷內容 → 上傳 Excel（多題）** |
| `opinionqa_personas.xlsx` | 20 U.S.-representative personas — upload in **受訪者設定 → 批次匯入** |
| `opinionqa_human_reference.csv` | Weighted human answer distributions (P(1)..P(5), mean, n) per question |
| `compare_results.py` | Compares the app's exported CSV against the human reference |
| `prepare_opinionqa_files.py` | Rebuilds all of the above from the raw OpinionQA download |

## Workflow

1. Open the app and upload `opinionqa_questions.xlsx` (questions) and
   `opinionqa_personas.xlsx` (personas). That's 18 × 20 = **360 LLM calls** —
   a few minutes on a billed API key; a free-tier key may hit rate limits.
2. Run the survey and download the results CSV (`ssr_results.csv`).
3. Compare:

   ```bash
   python compare_results.py ssr_results.csv
   ```

   Reported metrics (SSR-paper style):
   - **KS similarity** per question and averaged — the paper considers > 0.85
     to indicate realistic response distributions;
   - **Pearson r** between simulated and human per-question mean scores;
   - mean absolute error of mean scores (1–5 scale).

## Question selection & mapping

Only questions with clean 5-point ordinal scales were used, rephrased as
agree/disagree statements so the app's Likert anchors apply. Score 5 always
corresponds to the most positive original option:

- **Wave 92 (2021), 7 questions** — "is X good or bad for society"
  (Very good … Very bad) → "X is good for our society."
- **Wave 54 (2019), 11 questions** — "are economic conditions helping or
  hurting X" (Helping a lot … Hurting a lot) → "…conditions are helping X."

Human distributions are survey-weighted (`WEIGHT_W92` / `WEIGHT_W54`);
"Refused" answers are dropped and the distribution renormalized.

## Known caveats — read before quoting numbers

- **Time mismatch.** Human data is from 2019 (W54) and 2021 (W92); the LLM
  answers from its current worldview. The W54 economy questions (pre-COVID
  boom) are especially era-sensitive. For stricter runs, add a sentence like
  "It is 2019; answer based on that time." to every persona and run each wave
  separately.
- **Scale mapping is an approximation.** "Agree" ↔ "Helping a lot" is a
  reasonable but imperfect equivalence; the W92 good/bad-for-society scale is
  the cleaner comparison.
- **20 personas ≈ demographic sketch, not a weighted sample.** They mirror
  the U.S. adult mix roughly (age, gender, race, education, politics, region)
  but are not statistically calibrated. More/finer personas sharpen the test.
- Pew data is for research use; keep the raw OpinionQA download out of git
  (only derived aggregate distributions are committed here).

## Rebuilding from raw data

```bash
# Download the OpinionQA bundle pieces (~63 MB for the two waves used):
B=0x050b7e72abb04d1f9b493c1743e580cf
for W in W92 W54; do
  mkdir -p /path/to/opinionqa/human_resp/$W
  curl -L "https://worksheets.codalab.org/rest/bundles/$B/contents/blob/human_resp/American_Trends_Panel_$W/responses.csv" \
       -o /path/to/opinionqa/human_resp/$W/responses.csv
done

python prepare_opinionqa_files.py --data-dir /path/to/opinionqa
```
