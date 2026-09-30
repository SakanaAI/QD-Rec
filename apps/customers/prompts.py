"""
Prompt templates for the Marketing domain: zero-shot and few-shot hypothesis
generation, the judge score, the novelty check and the pairwise duel.
"""

# Zero-shot hypothesis prompts (seed archive, Novelty)

HYPOTHESIS_SYSTEM_PROMPT = """You are a market research expert who generates hypotheses about users' interest in products and services."""

HYPOTHESIS_USER_PROMPT = """## Target product/service:
{target_product}

---

## Persona profile:
{persona_profile}

---

Generate a compelling hypothesis explaining why this persona would be interested in this product.

Focus on the following:
- The person's specific needs or pain points that the product can address
- How the product fits into that person's lifestyle and daily life
- The emotional and practical benefits for this specific persona
- The specific features or attributes that resonate with that person

Refer specifically to the actual persona, citing their background, interests, and situation.

Make the strongest plausible case for why this persona is worth targeting; conclude they are a poor target only if the fit is genuinely poor.

Respond in the following JSON format:

{{
  "hypothesis": "2-6 sentences explaining why this persona would be interested in this product (or, only if the fit is genuinely poor, why they would not)."
}}"""


# Few-shot hypothesis prompts (Novelty+Evo, QD-Rec)

HYPOTHESIS_FEWSHOT_SYSTEM_PROMPT = """You are a market research expert who generates hypotheses about users' interest in products and services."""

HYPOTHESIS_FEWSHOT_USER_PROMPT = """## Target product/service:
{target_product}

---

## Existing example hypotheses (for inspiration only):

{existing_theses}

---

## Persona to evaluate:
{persona_profile}

---

Generate a compelling hypothesis explaining why this persona would be interested in this product.

Focus on the following:
- The person's specific needs or pain points that the product can address
- How the product fits into that person's lifestyle and daily life
- The emotional and practical benefits for this specific persona
- The specific features or attributes that resonate with that person

Refer specifically to the actual persona, citing their background, interests, and situation.

Make the strongest plausible case for why this persona is worth targeting; conclude they are a poor target only if the fit is genuinely poor.

You can use the existing hypotheses as inspiration:
- **Mutate**: Take the reasoning angle of ONE existing hypothesis and adapt it to this persona
- **Crossover**: Combine elements of TWO OR MORE hypotheses to create new logic
- **Explore**: Propose a completely different motivation or use case not covered by any hypothesis

Also report which operation you used, and which of the "Persona N" numbers above you based it on (source_examples). For Explore, set source_examples to an empty list.

Do not copy phrases verbatim from the existing hypotheses.

Respond in the following JSON format:

{{
  "operation": "mutate" | "crossover" | "explore",
  "source_examples": [The persona numbers you based it on. E.g., for mutate [2], for crossover [1, 3], for explore []],
  "evolution_note": "One sentence explaining how this hypothesis was influenced by the existing example hypotheses (which part of which 'Persona N' you incorporated, etc.). This is the only field where you may reference the examples. Empty string for explore.",
  "hypothesis": "2-6 sentences explaining why this persona would be interested in this product (or, only if the fit is genuinely poor, why they would not). Write it as a standalone text, as if you had written it without the example hypotheses at hand: do not reference, number, or hint at the example hypotheses, and do not explain in the body which operation you used or how it was derived (those references go only in evolution_note)."
}}"""


# Judge prompts

JUDGE_SYSTEM_PROMPT = """You are a critical market research analyst scoring persona-targeting recommendations."""

JUDGE_USER_PROMPT = """Evaluate the following persona-targeting recommendation.

## Target product/service:
{target_product}

## Persona profile:
{persona_profile}

## Hypothesis (why this persona would or would not be interested):
{hypothesis}

---

Score how compelling and plausible the case for targeting this persona is, considering:
- Relevance: Does the product actually address this persona's needs?
- Specificity: Is the case specific to this persona, or is it generic?
- Plausibility: Is it believable that this persona would actually buy/use the product?
- Depth of insight: Does it demonstrate a genuine understanding of the persona?
- Value alignment: Does the product align with this persona's values, lifestyle, and means?

A hypothesis that argues the persona would NOT be interested, concludes the fit is poor, or offers only a hollow/implausible rationale is a weak recommendation and must score low, even if its reasoning is accurate. Otherwise evaluate strictly and honestly; not every recommendation is good.

Provide your evaluation in the following JSON format:

{{
  "reasoning": "A concise one-sentence summary of the evaluation",
  "score": <integer from 0 to 10>
}}

Scoring guide (strength of the recommendation to target this persona):
- 0-2: No real case (argues against, no fit, generic reasoning, or not believable)
- 3-4: Weak case (weak fit, serious concerns)
- 5-6: Moderate case (some relevance, but lacking specificity or depth)
- 7-8: Strong case (clear fit, manageable concerns)
- 9-10: Excellent case (compelling, specific, and insightful connection)"""




# Novelty check and blind pairwise duel (see evaluation.py)

NOVELTY_CHECK_SYSTEM_PROMPT = """You are a senior market research expert who assesses the novelty of persona-product hypotheses."""

NOVELTY_CHECK_USER_PROMPT = """## Target product/service:
{target_product}

---

## Existing hypotheses:

{existing_text}

---

## New hypothesis to assess:

{new_text}

---

Assess whether the NEW hypothesis above represents a meaningfully different hypothesis compared to the existing ones:
- Does the new hypothesis explore a different motivation or use case for the product?
- Focus on the novelty of the hypothesis, not the difference in users.
- Evaluate strictly: trivial variations are not novel.

Respond in the following JSON format:

{{
  "novelty_reasoning": "1-2 sentences explaining whether the new hypothesis is novel compared to the existing ones.",
  "is_novel": true or false
}}"""

PAIRWISE_DUEL_SYSTEM_PROMPT = """You are a senior market research expert who compares persona-product hypotheses."""

PAIRWISE_DUEL_USER_PROMPT = """## Target product/service:
{target_product}

---

## Hypotheses to compare:

{duel_text}

---

Which hypothesis is better overall? Judge objectively, on content quality alone, by the following criteria:
- Relevance: How well does the product fit this persona's needs?
- Specificity: Is the hypothesis specific and insightful?
- Plausibility: Would this persona realistically use the product?
- Value alignment: Does the product align with this persona's values, lifestyle, and means?

You must pick exactly one winner; ties are not allowed.

Respond in the following JSON format:

{{
  "comparison_reasoning": "2-3 sentences explaining the key differentiators and your decision.",
  "winner": "A" or "B"
}}"""
