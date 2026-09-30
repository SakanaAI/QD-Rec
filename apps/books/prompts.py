"""
Prompt templates for the Goodreads-Children recommender.

There is no judge prompt, because quality is the reader's star rating. The LLM only
writes memos and judges novelty.
"""

# Memo generation, zero-shot (seed archive, Novelty)

MEMO_SYSTEM_PROMPT = """You are a reading advisor who writes memos on whether a reader would enjoy a book."""

MEMO_USER_PROMPT = """## Reader Profile:
{profile}

---

## Candidate Book:
Title: {title}

Description:
{description}

---

Write a concise memo on why this reader might (or might not) enjoy this book.

The description already summarizes the book, so DO NOT re-summarize it. Instead, focus on:
- Why THIS reader in particular might enjoy (or not enjoy) this book, grounded in the specific tastes, themes, and pet peeves visible in their profile.
- What about the book connects to or clashes with what they have rewarded or panned before.

IMPORTANT:
- Be grounded in the book description and the reader's profile; do not invent plot details not in the description.
- Keep it short: 2-6 sentences (one short paragraph).
- Do not mention or guess a star rating.

Format your response as JSON:
{{
  "notes": "2-6 sentences: start directly with your relevance hypothesis grounded in this reader's demonstrated tastes; do not re-summarize the plot."
}}"""


# Memo generation, few-shot (Novelty+Evo, QD-Rec)

MEMO_FEWSHOT_SYSTEM_PROMPT = """You are a reading advisor who writes memos on whether a reader would enjoy a book."""

MEMO_FEWSHOT_USER_PROMPT = """## Reader Profile:
{profile}

---

## Existing Example Memos (for inspiration only):

{existing_recommendations}

---

## New Candidate Book:
Title: {title}

Description:
{description}

---

Write a concise memo on why this reader might (or might not) enjoy this book.

The description already summarizes the book, so DO NOT re-summarize it. Instead, focus on:
- Why THIS reader in particular might enjoy (or not enjoy) this book, grounded in the specific tastes, themes, and pet peeves visible in their profile.
- What about the book connects to or clashes with what they have rewarded or panned before.

Use the existing memos as evolutionary building blocks:
- Mutate: adapt a useful reasoning angle from ONE example to the new book and this reader
- Crossover: combine complementary angles from TWO OR MORE examples
- Explore: propose a genuinely different angle not covered by any example

Also REPORT which operator you used and the example number(s) you built on, so the
reasoning can be traced. Use the "Example N" numbers shown above; for Explore, leave
the source list empty.

IMPORTANT:
- Do not copy phrases verbatim from the examples.
- Be grounded in the book description and the reader's profile; do not invent plot details not in the description.
- Keep it short: 2-6 sentences (one short paragraph).
- Do not mention or guess a star rating.

Format your response as JSON:
{{
  "operation": "mutate" | "crossover" | "explore",
  "source_examples": [example numbers you built on, e.g. [2] for mutate or [1, 3] for crossover; [] for explore],
  "evolution_note": "ONE sentence on how the example memo(s) shaped this — e.g. which angle came from which Example N. This is the ONLY field that may reference the examples; leave empty for explore.",
  "notes": "2-6 sentences: start directly with your relevance hypothesis grounded in this reader's demonstrated tastes; do not re-summarize the plot. Write it as STANDALONE: do NOT reference, number, or allude to the example memos, and do NOT narrate the operator or how it was derived (put any such mention only in evolution_note)."
}}"""


# Novelty check (replacement is decided by the reader's rating)

NOVELTY_SYSTEM_PROMPT = """You are a reading advisor who assesses whether a new book recommendation adds genuine variety to a reader's shelf."""

NOVELTY_USER_PROMPT = """## Reader Profile:
{profile}

---

## Recommendations on the shelf:

{recommendations_text}

---

The NEW recommendation is: **{new_recommendation_label}**

Assess whether the NEW recommendation is meaningfully different from the existing ones — i.e. whether it adds genuine variety to a diverse shelf for this reader.

Important:
- Judge by the kind of book and the kind of appeal, not by quality.
- Be strict: a superficially different title that offers the same kind of reading experience as an existing one is NOT novel.
- Do not use or guess any ratings/scores; judge by content and appeal only.

Provide your response in this EXACT JSON format:

{{
  "novelty_reasoning": "1-2 sentences explaining why the new recommendation is or isn't novel.",
  "is_novel": true or false
}}"""
