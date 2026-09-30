"""
Researcher profiles for the papers domain.
"""

PROFILES = {
    "OpenEndedness Researcher": {
        "name": "OpenEndedness Researcher",
        "profile": """An ML researcher working on open-ended and quality-diversity methods. Their through-line is divergent search, reaching ambitious capabilities through diverse intermediate stepping stones rather than direct climbs toward a fixed objective, but in practice they bring a handful of separable questions to a paper. Sometimes a paper is interesting for how it measures novelty or behavioral diversity without hand-designed descriptors, through learned representations and embeddings that define what a good behavior space even is; sometimes for its archive and selection mechanics, such as what to retain and how local competition and illumination shape search; sometimes for its operators, such as mutation, crossover, or using large models as variation operators; sometimes for auto-curricula and co-evolution, where agents and the environments that challenge them generate their own next problems; sometimes for what it says about exploration and the deception of objectives, including intrinsic motivation, when direct optimization stalls, and what transfers from reinforcement-learning exploration; and sometimes for recursive self-improvement, where systems rewrite their own code, operators, or objectives so that gains compound, as in self-referential and self-modifying agents. They have a skeptical streak toward pure scaling and benchmark-chasing, and they actively look to import techniques from adjacent areas into open-ended search. They do not expect every paper at a major ML conference to be about open-endedness; they look for the specific angle (a method, a measurement, a failure mode, an assumption) that connects to their interest.""",
    },
    "AI Startup Researcher": {
        "name": "AI Startup Researcher",
        "profile": """An applied ML researcher at an early-stage AI startup who reads research through the lens of practical leverage. They are not only asking, "Could this become a product?" but also, "Could this make a product cheaper, safer, faster, more reliable, more defensible, easier to evaluate, or easier to specialize?" Their interest is in papers that could plausibly affect a 3-to-12-month startup roadmap, whether by enabling a new product surface, improving unit economics, reducing inference cost, increasing reliability, strengthening safety guarantees, opening a vertical market, or creating a data or evaluation moat. When reading papers from major ML conferences in particular, they do not expect every paper to be immediately productizable. They recognize that much of this work is theoretical, exploratory, or several steps removed from company use, so they look for the subset of work that hints at future engineering leverage, changes an important assumption, improves the efficiency/safety/reliability envelope, or suggests a direction that could compound into practical advantage over time.""",
    },
}

DEFAULT_PROFILE = "AI Startup Researcher"


def get_profile(name: str) -> dict:
    if name not in PROFILES:
        available = ", ".join(PROFILES.keys())
        raise ValueError(f"Unknown researcher: {name}. Available: {available}")
    return PROFILES[name]


def get_researcher_name(name: str) -> str:
    return get_profile(name)["name"]


def get_profile_text(name: str) -> str:
    return get_profile(name)["profile"]


def profile_tag(name: str) -> str:
    """The profile's token in archive filenames: the researcher's name with
    spaces as underscores (e.g. ``OpenEndedness_Researcher``)."""
    return get_researcher_name(name).replace(" ", "_")
