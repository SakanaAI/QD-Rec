"""
Prompt templates for the M&A recommender: the judge's 0-10 score, zero-shot and
few-shot hypothesis generation, the novelty check and the blind pairwise duel.
"""


# Judge (0-10 score)
JUDGE_SYSTEM_PROMPT = """You are a critical M&A advisor scoring acquisition hypotheses."""

JUDGE_USER_PROMPT = """Evaluate the following acquisition hypothesis.

## Acquiring Company:
{acquirer_profile}

## Target Company:
{target_profile}

## Acquisition Hypothesis:
{hypothesis}

---

Score how compelling and plausible the strategic case for the acquisition is, considering:
- Strategic fit with the acquirer's existing business
- Synergy potential (revenue and cost synergies)
- Feasibility (is this acquisition realistic and executable?)
- Risk factors (integration challenges, market risks, regulatory concerns)
- Value creation potential

A hypothesis that argues against the acquisition, concludes the target is a poor fit, or offers only a hollow/implausible rationale is a weak recommendation and must score low, even if its reasoning is accurate. Reserve high scores for genuinely compelling, specific, plausible theses.

Provide your evaluation in this EXACT JSON format:

{{
  "reasoning": "Few sentences summarizing your overall assessment",
  "score": <integer from 0-10>
}}

Scoring guide (strength of the recommendation to acquire):
- 0-2: No real case (argues against, no strategic fit, or hollow/implausible rationale)
- 3-4: Weak case (strained or generic rationale; major concerns)
- 5-6: Moderate case (some plausible merit but notable gaps)
- 7-8: Strong case (specific, plausible fit with clear value creation)
- 9-10: Excellent case (highly compelling, specific, plausible thesis)"""


# Hypothesis generation, zero-shot (seed archive, Novelty)
HYPOTHESIS_SYSTEM_PROMPT = """You are an M&A strategist who generates acquisition hypotheses."""

HYPOTHESIS_USER_PROMPT = """## Acquiring Company:
{acquirer_profile}

---

## Target Company:
{target_profile}

---

Generate an acquisition hypothesis explaining why the acquirer should consider acquiring the target company.

Focus on:
- Strategic fit with the acquirer's existing business
- Specific synergies (revenue growth, cost savings, capability acquisition)
- Market positioning and competitive advantages
- Financial attractiveness (valuation, growth trajectory, deal sizing relative to acquirer)
- Timing considerations (is the target's current valuation/performance favorable?)

Be concrete and specific to the actual companies involved. Reference financial metrics when relevant (e.g., enterprise value, growth rates, returns).

Make the strongest plausible case for why the acquirer should acquire this company; recommend against it only if the strategic fit is genuinely poor.

Provide your response in this EXACT JSON format:

{{
  "hypothesis": "2-6 sentences explaining WHY acquiring this company would benefit the acquirer. Be specific about strategic fit, synergies, and value creation potential."
}}"""


# Hypothesis generation, few-shot (Novelty+Evo, QD-Rec)
HYPOTHESIS_FEWSHOT_SYSTEM_PROMPT = """You are an M&A strategist who generates acquisition hypotheses."""

HYPOTHESIS_FEWSHOT_USER_PROMPT = """## Acquiring Company:
{acquirer_profile}

---

## Existing Acquisition Theses (for inspiration only):

{existing_theses}

---

## Target Company to Evaluate:
{target_profile}

---

Generate an acquisition hypothesis explaining why the acquirer should consider acquiring the target company.

Focus on:
- Strategic fit with the acquirer's existing business
- Specific synergies (revenue growth, cost savings, capability acquisition)
- Market positioning and competitive advantages
- Financial attractiveness (valuation, growth trajectory, deal sizing relative to acquirer)
- Timing considerations (is the target's current valuation/performance favorable?)

Be concrete and specific to the actual companies involved. Reference financial metrics when relevant (e.g., enterprise value, growth rates, returns).

Make the strongest plausible case for why the acquirer should acquire this company; recommend against it only if the strategic fit is genuinely poor.

Use the existing theses as inspiration. You may:
- **Mutate**: Take a strategic angle from ONE thesis and adapt it to this new target
- **Crossover**: Combine elements from TWO OR MORE theses into a novel rationale
- **Explore**: Propose a completely different strategic direction not yet covered by any thesis

Also report which operator you used and the thesis number(s) you built on, using the
"Thesis N" numbers shown above (leave the source list empty for Explore).

Do not copy phrases verbatim from the existing theses.

Provide your response in this EXACT JSON format:

{{
  "operation": "mutate" | "crossover" | "explore",
  "source_examples": [thesis numbers you built on, e.g. [2] for mutate or [1, 3] for crossover; [] for explore],
  "evolution_note": "ONE sentence on how the example thesis/theses shaped this hypothesis, e.g. which angle came from which Thesis N. This is the ONLY field that may reference the examples; leave empty for explore.",
  "hypothesis": "2-6 sentences explaining WHY acquiring this company would benefit the acquirer. Be specific about strategic fit, synergies, and value creation potential. Write it as a STANDALONE rationale, in the same style as one written with no examples in front of you: do NOT reference, number, or allude to the example theses, and do NOT narrate the operator or how it was derived (put any such mention only in evolution_note)."
}}"""


# Novelty check, and the blind pairwise duel run only when the new thesis is not novel

NOVELTY_CHECK_SYSTEM_PROMPT = """You are a senior M&A advisor who assesses the novelty of acquisition theses."""

NOVELTY_CHECK_USER_PROMPT = """## Acquiring Company:
{acquirer_profile}

---

## Existing Acquisition Theses:

{existing_text}

---

## New Thesis to Assess:

{new_text}

---

Assess whether the NEW thesis above represents a meaningfully different strategic direction or value-creation mechanism compared to the existing ones:
- Consider target industry, strategic rationale (vertical/horizontal integration, capability acquisition, market access), and synergy type (revenue, cost, technology, talent).
- Evaluate strictly: minor variations or superficially different targets with the same thesis are NOT novel. Only truly distinct strategic directions count.

Respond in the following JSON format:

{{
  "novelty_reasoning": "1-2 sentences explaining whether the new thesis is a genuinely novel strategic direction compared to the existing ones.",
  "is_novel": true or false
}}"""

PAIRWISE_DUEL_SYSTEM_PROMPT = """You are a senior M&A advisor who compares acquisition theses and decides which is stronger."""

PAIRWISE_DUEL_USER_PROMPT = """## Acquiring Company:
{acquirer_profile}

---

## Acquisition Theses to Compare:

{duel_text}

---

Which thesis is stronger overall for the acquirer? Judge objectively, on the strategic case alone, by the following criteria:
- Strategic fit with the acquirer's existing business
- Synergy potential (revenue and cost synergies)
- Feasibility and execution risk
- Value creation potential

You must pick exactly one winner; ties are not allowed.

Respond in the following JSON format:

{{
  "comparison_reasoning": "2-3 sentences explaining the key differentiators and your decision.",
  "winner": "A" or "B"
}}"""
