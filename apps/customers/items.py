"""California persona records, item embeddings and prompt text."""

import pandas as pd

from .config import (
    PERSONAS_FILE,
    EMBEDDINGS_FILE,
    PERSONA_FIELDS,
    CONTEXT_FIELDS,
    DEMOGRAPHIC_FIELDS,
)


def load_persona_table(file_path: str = PERSONAS_FILE) -> pd.DataFrame:
    """Load the persona table, with its `uuid` column renamed to `item_id`."""
    df = pd.read_parquet(file_path)
    if "uuid" not in df.columns:
        raise ValueError("persona dataset has no 'uuid' column")
    return df.rename(columns={"uuid": "item_id"})


def load_items(profile: str) -> list[dict]:
    """Every persona of the dataset, in file order (the same pool for every product)."""
    return load_persona_table().to_dict("records")


def load_item_embeddings(profile: str) -> dict:
    """{item_id: embedding} from EMBEDDINGS_FILE (the same store for every product)."""
    df = pd.read_parquet(EMBEDDINGS_FILE)
    if "uuid" not in df.columns:
        raise ValueError("persona embedding store has no 'uuid' column")
    df = df.rename(columns={"uuid": "item_id"})
    item_embeddings = {}
    for _, row in df.iterrows():
        item_id = row.get("item_id")
        emb = row.get("embedding")
        if item_id is None or emb is None:
            continue
        item_embeddings[str(item_id)] = emb
    return item_embeddings


# Section labels used by item_text
_DEMOGRAPHIC_LABELS = {
    "sex": "Sex",
    "age": "Age",
    "marital_status": "Marital status",
    "education_level": "Education",
    "occupation": "Occupation",
    "city": "City",
    "state": "State",
}

_PERSONA_LABELS = {
    "persona": "Persona summary",
    "professional_persona": "Professional",
    "sports_persona": "Sports",
    "arts_persona": "Arts & culture",
    "travel_persona": "Travel",
    "culinary_persona": "Food",
}

_CONTEXT_LABELS = {
    "cultural_background": "Cultural background",
    "skills_and_expertise": "Skills & expertise",
    "hobbies_and_interests": "Hobbies & interests",
    "career_goals_and_ambitions": "Career goals",
}


def item_label(persona_data: dict) -> str:
    occupation = str(persona_data["occupation"]).replace("_", " ")
    return f"{persona_data['age']}-year-old {occupation}, {persona_data['city']}"


def item_text(persona_data: dict) -> str:
    """Profile text of a persona: demographics, then each persona and context field."""
    lines = []

    lines.append("[Demographics]")
    for field in DEMOGRAPHIC_FIELDS:
        value = persona_data.get(field)
        if value and str(value).strip():
            label = _DEMOGRAPHIC_LABELS.get(field, field)
            lines.append(f"{label}: {value}")

    for field in PERSONA_FIELDS:
        value = persona_data.get(field)
        if value and str(value).strip():
            label = _PERSONA_LABELS.get(field, field)
            lines.append(f"\n[{label}]")
            lines.append(str(value))

    for field in CONTEXT_FIELDS:
        value = persona_data.get(field)
        if value and str(value).strip():
            label = _CONTEXT_LABELS.get(field, field)
            lines.append(f"\n[{label}]")
            lines.append(str(value))

    return "\n".join(lines)


def archive_text(persona_data: dict, hypothesis: str) -> str:
    """Persona profile plus hypothesis, embedded for novelty and diversity."""
    persona_text = item_text(persona_data)
    return f"{persona_text}\n\n---\n\nHypothesis:\n{hypothesis}"


def embedding_text(persona_data: dict) -> str:
    """Text of the per-item embedding (create_embeddings.py): item_text."""
    return item_text(persona_data)
