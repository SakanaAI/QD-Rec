"""
Product profiles for the Marketing domain. Each profile is a fictional product;
the recommender looks for diverse personas who would be interested in it.
"""

# Selected with --profile
PRODUCTS = {
    "SkillForge": """SkillForge is a fictional online learning platform offered at $89-$249 per month. Centered on business skills, technology, and creative fields, it provides content aimed at practical, job-ready skill acquisition. Its defining feature is that it is designed not as passive video-watching but around the goal of "being able to use what you learn at work."
Courses are organized into three learning tracks. The "Business Track" delivers skills that make you immediately useful in white-collar roles -- data analysis, project management, financial modeling, presentation. The "Tech Track" offers hands-on courses from the basics to applications of Python, SQL, cloud, and AI, understandable even for non-engineers. The "Creative Track" teaches expressive skills usable for side work and career changes, such as UI/UX design, video editing, and writing.
Each course is taught by a working expert and includes a weekly live Q&A session, a Slack community of fellow learners, and assignment submissions with feedback modeled on real work. Progress is visualized on a dashboard, and completion certificates can be posted on LinkedIn or a portfolio. A corporate plan adds team-level enrollment management and progress reporting.
The intended users are diverse: late-20s-to-30s professionals aiming to upskill in their current jobs, women considering reskilling as they plan a return from childcare, seniors preparing for a second career after retirement, and employees with a side-gig mindset who want a source of income beyond their main job.
With the concept of "making re-learning unremarkable," SkillForge is designed so that even busy working adults can steadily build skills with 3-4 hours of study a week. With no enrollment fee and flexible, cancel-anytime plans, it lowers the barrier to staying with learning and aims to support the career autonomy of working people in the US.""",
    "Printer3D": """LAYERCRAFT is a fictional desktop 3D-printer brand in the $500-$900 price range. It is aimed at people who already make, fix, or teach things and want to do so with their own parts and designs, and it positions itself as "the printer that just prints": the machine arrives assembled and calibrated, and the first print starts within fifteen minutes of unboxing.
The lineup comprises three models. "Layercraft Studio" is a compact, fully enclosed printer, quiet enough for an apartment or a classroom. "Layercraft Workshop" has a larger build volume and prints tougher engineering materials for functional parts. "Layercraft Color" holds four spools and switches between them automatically for multi-color prints.
Every model has automatic bed leveling, a companion app with a library of ready-to-print designs, and a 2-year warranty with US-based support. A design marketplace lets owners sell or share their own models, and educators get free lesson plans and a classroom management mode. Filament is sold in recyclable spools at $20-$30 per kilogram.
Owners use it for different things: replacing a broken part on an appliance or a car instead of ordering one; prototyping a product for a small business or a side venture; STEM projects with children or students; props, cosplay and tabletop-gaming pieces; custom organizers, tools and gifts. It is not for everyone: it needs a permanent desk-sized spot, a few evenings to learn basic 3D design and slicing, filament at $20-$30 per kilogram, and patience, since a typical print takes several hours.
LAYERCRAFT frames a 3D printer as a tool for people who make things, like a sewing machine or a table saw: something that pays for itself in the things it makes and the things it fixes. By taking the setup and the failed prints out of the experience, it aims to make owning one ordinary in that kind of household.""",
}

DEFAULT_PROFILE = "SkillForge"


def get_profile_text(name: str) -> str:
    """The product description shown to the proposer and the judge."""
    if name not in PRODUCTS:
        raise KeyError(f"Unknown product {name!r}; valid products: {list(PRODUCTS)}")
    return PRODUCTS[name]


def profile_tag(name: str) -> str:
    """The profile's token in archive filenames: the product name itself."""
    return name
