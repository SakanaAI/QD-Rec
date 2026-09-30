"""
Prompt templates for the papers domain: zero-shot and few-shot memo generation,
the judge score, the novelty check and the pairwise duel.
"""

# Zero-shot memo prompts (seed archive, Novelty)

MEMO_SYSTEM_PROMPT = """You are a research advisor who writes memos on whether papers are worth reading."""


MEMO_USER_PROMPT = """## Researcher Profile:
{researcher_profile}

---

## Paper:
Title: {title}

Abstract:
{abstract}

---

Write a concise research memo for this researcher.

The abstract is already a summary, so DO NOT re-summarize the paper. Instead, focus on:
- Why this paper could be interesting for THIS researcher (your "hypothesis" of relevance)
- One concrete next step (experiment, ablation, or extension) if they read it

IMPORTANT:
- Be grounded in the abstract; do not invent unrelated claims.
- Keep it short: 2-6 sentences (one short paragraph).
- Write in plain text; do NOT include LaTeX or math notation.

Format your response as JSON:
{{
  "research_notes": "2-6 sentences: start with your relevance hypothesis for this researcher (no paper summary), and end with one concrete next step (experiment, ablation, or extension) grounded in the abstract."
}}"""


# Few-shot memo prompts (Novelty+Evo, QD-Rec)

MEMO_FEWSHOT_SYSTEM_PROMPT = """You are a research advisor who writes memos on whether papers are worth reading."""


MEMO_FEWSHOT_USER_PROMPT = """## Researcher Profile:
{researcher_profile}

---

## Existing Example Memos (for inspiration only):

{existing_recommendations}

---

## New Paper:
Title: {title}

Abstract:
{abstract}

---

Write a concise research memo for this researcher.

The abstract is already a summary, so DO NOT re-summarize the paper. Instead, focus on:
- Why this paper could be interesting for THIS researcher (your "hypothesis" of relevance)
- One concrete next step (experiment, ablation, or extension) if they read it

Use the existing memos as evolutionary building blocks:
- Mutate: adapt a useful reasoning pattern from ONE example to the new paper and this researcher
- Crossover: combine complementary angles from TWO OR MORE examples into a stronger memo
- Explore: propose a genuinely different angle not covered by any example

Also REPORT which operator you used and the example number(s) you built on, using the
"Example N" numbers shown above (leave the source list empty for Explore).

IMPORTANT:
- Do not copy phrases verbatim from the examples.
- Be grounded in the abstract; do not invent unrelated claims.
- Keep it short: 2-6 sentences (one short paragraph).
- Write in plain text; do NOT include LaTeX or math notation.

Format your response as JSON:
{{
  "operation": "mutate" | "crossover" | "explore",
  "source_examples": [example numbers you built on, e.g. [2] for mutate or [1, 3] for crossover; [] for explore],
  "evolution_note": "ONE sentence on how the example memo(s) shaped this — e.g. which angle came from which Example N. This is the ONLY field that may reference the examples; leave empty for explore.",
  "research_notes": "2-6 sentences: start with your relevance hypothesis for this researcher (no paper summary), and end with one concrete next step (experiment, ablation, or extension) grounded in the abstract. Write the memo as STANDALONE: do NOT reference, number, or allude to the example memos, and do NOT narrate the operator or how it was derived (put any such mention only in evolution_note)."
}}"""


# Judge prompts

JUDGE_SYSTEM_PROMPT = """You are a critical research advisor who evaluates paper recommendations."""

JUDGE_USER_PROMPT = """## Researcher Profile:
{researcher_profile}

---

## Paper Abstract:
{paper_abstract}

---

## Research Notes:
{research_memo}

---

Evaluate this paper recommendation for the researcher.

Assess how valuable this paper + research memo is for this researcher, considering:
- Relevance to the researcher's stated interests
- Technical quality and novelty
- Quality of insights in the research memo
- Potential for inspiring new research directions

Be rigorous and honest. Most recommendations are NOT outstanding. It might be written as though there are deep connections with the researcher's interests, but don't be fooled easily.

Calibration (important):
- Use the full 0-10 scale. Avoid score inflation.
- If the memo is generic (could fit many papers) or offers vague directions, score it lower even if it "sounds reasonable".
- Prefer concrete, falsifiable, technically grounded next steps over broad brainstorming.

Provide your evaluation in this EXACT JSON format:

{{
  "reasoning": "2-3 sentences summarizing your assessment",
  "score": <integer from 0-10>
}}

Scoring guide:
- 0-2: Poor recommendation (irrelevant/low-signal; likely a waste of time)
- 3-4: Weak recommendation (thin connection; generic memo; few concrete next steps)
- 5-6: OK / mixed (some relevance, but limited novelty or actionable direction; unsurprising)
- 7: Good (clear relevance + shows specific, plausible research directions)
- 8: Very good (strong relevance + multiple specific directions; minimal hand-waving)
- 9-10: Exceptional (highly relevant, genuinely novel insight, very interesting)"""


# Novelty check and blind pairwise duel (see evaluation.py)

NOVELTY_CHECK_SYSTEM_PROMPT = """You are a senior research advisor who assesses the novelty of paper recommendations."""

NOVELTY_CHECK_USER_PROMPT = """## Researcher Profile:
{researcher_profile}

---

## Existing Recommendations:

{existing_text}

---

## New Recommendation to Assess:

{new_text}

---

Assess whether the NEW recommendation above offers a meaningfully different reading angle or memo perspective compared to the existing ones.

Consider:
- Does the new memo explore a different relevance hypothesis or research direction for this researcher?
- Focus on the angle of the memo, not just the topic of the paper.
- Evaluate strictly: superficial differences in paper topic with the same reading angle are NOT novel. Only a genuinely different memo framing or research direction counts.

Respond in the following JSON format:

{{
  "novelty_reasoning": "1-2 sentences explaining whether the new recommendation is a genuinely novel reading angle or memo framing compared to the existing ones.",
  "is_novel": true or false
}}"""

PAIRWISE_DUEL_SYSTEM_PROMPT = """You are a senior research advisor who compares paper recommendations and decides which is more valuable."""

PAIRWISE_DUEL_USER_PROMPT = """## Researcher Profile:
{researcher_profile}

---

## Recommendations to Compare:

{duel_text}

---

Which recommendation is more valuable for this researcher? Judge objectively, on memo content alone, by the following criteria:
- Relevance to the researcher's stated interests
- Specificity and depth of the reading hypothesis
- Quality and concreteness of the proposed research directions
- Potential to inspire new work

You must pick exactly one winner; ties are not allowed.

Respond in the following JSON format:

{{
  "comparison_reasoning": "2-3 sentences explaining the key differentiators and your decision.",
  "winner": "A" or "B"
}}"""
