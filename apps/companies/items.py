"""Company records, item embeddings and prompt text for M&A recommendations."""

import csv
import json

from shared.archive_records import normalize_source_item

from .config import EMBEDDINGS_FILE, UNIVERSE_FILE


def load_items(profile: str) -> list[dict]:
    """Every company of the universe except the acquirer `profile`, in file order.

    The CSV's ID column ``Instrument`` becomes ``item_id`` in loaded items and
    archive records.
    """
    companies: list[dict] = []
    with open(UNIVERSE_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if profile and row.get("Instrument") == profile:
                continue
            companies.append(normalize_source_item(row, "Instrument"))
    return companies


def item_label(row_dict: dict) -> str:
    return row_dict["CompanyName"]


def item_text(row_dict):
    """
    Full text of a company from its universe row: name, location, classification,
    business description and financials. The item ID is left out.
    """
    company = row_dict.get("CompanyName", "Unknown")
    city = row_dict.get("City", "")
    region = row_dict.get("Region of Headquarters", "")
    country = row_dict.get("Country", "")

    location = f"{city}, {region} ({country})"

    sector_eco = row_dict.get("TRBC Economic Sector Name", "")
    sector_biz = row_dict.get("TRBC Business Sector Name", "")
    ind_group = row_dict.get("TRBC Industry Group Name", "")
    ind_name = row_dict.get("TRBC Industry Name", "")
    activity = row_dict.get("TRBC Activity Name", "")

    classification = f"""Classification:
  - Sector: {sector_eco} / {sector_biz}
  - Industry: {ind_group} / {ind_name}
  - Activity: {activity}"""

    desc = row_dict.get("BusinessDescription", "")

    snapshot = row_dict.get("SnapshotDate", "")
    ev = row_dict.get("Enterprise Value (Daily Time Series)", "N/A")

    ev_lag1 = row_dict.get("Enterprise Value (Daily Time Series)_lag_1", "N/A")
    ev_lag2 = row_dict.get("Enterprise Value (Daily Time Series)_lag_2", "N/A")
    ev_lag3 = row_dict.get("Enterprise Value (Daily Time Series)_lag_3", "N/A")
    ev_lag4 = row_dict.get("Enterprise Value (Daily Time Series)_lag_4", "N/A")

    ytd = row_dict.get("YTD Total Return", "N/A")
    ytd_2 = row_dict.get("2 YTD Total Return", "N/A")

    financials = f"""Financial Profile (Snapshot: {snapshot}, Local Currency):
  - Enterprise Value: {ev} (Current)
  - Historical EV (1y/2y/3y/4y ago): {ev_lag1} / {ev_lag2} / {ev_lag3} / {ev_lag4}
  - Performance: {ytd}% YTD, {ytd_2}% 2-Year"""

    full_text = f"""Company: {company}
Location: {location}

{classification}

Business Description:
{desc}

{financials}"""

    return full_text


def archive_text(company_dict, hypothesis):
    """Text embedded as entry["embedding"]: the company's item_text plus the hypothesis."""
    company_text = item_text(company_dict)

    combined_text = f"""{company_text}

Acquisition Hypothesis:
{hypothesis}"""

    return combined_text


def embedding_text(company_dict) -> str:
    """Text of the per-item embedding (create_embeddings.py): the business
    description only, not the full item_text."""
    return company_dict.get("BusinessDescription", "")


def load_item_embeddings(profile: str) -> dict[str, list[float]]:
    """
    {item_id: embedding} from EMBEDDINGS_FILE, the same store for every acquirer.
    The file's ID column is ``Instrument``.
    """
    item_embeddings = {}
    with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_id = row.get("Instrument")
            embedding_str = row.get("embedding")
            if item_id and embedding_str:
                item_embeddings[item_id] = json.loads(embedding_str)
    return item_embeddings
