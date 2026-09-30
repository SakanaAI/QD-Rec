"""
Download NVIDIA Nemotron-Personas-USA from Hugging Face, keep the California
personas and save a random sample of them.

Dataset: https://huggingface.co/datasets/nvidia/Nemotron-Personas-USA

Usage:
    python -m apps.customers.download_personas
"""

import argparse
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' package not installed.")
    print("Please install it with: uv add datasets")
    exit(1)


DEFAULT_OUTPUT_FILE = "data/nemotron_personas_CA.parquet"
DATASET_NAME = "nvidia/Nemotron-Personas-USA"
# Pin the dataset revision so updates cannot change the sampled candidate pool.
DATASET_REVISION = "5b4cd35ab46490c1da1bd2b5a2324d6f871be180"
STATE_FILTER = "CA"  # California subset (config.py PERSONAS_FILE expects this)
DEFAULT_MAX_SAMPLES = 10_000  # CA subset size used in the paper


def download_personas(
    output_file: str = DEFAULT_OUTPUT_FILE,
    max_samples: int | None = DEFAULT_MAX_SAMPLES,
):
    """
    Download the dataset and save a sample of its California personas.

    output_file is relative to this directory; max_samples=None keeps every
    California persona.
    """
    print(f"Downloading {DATASET_NAME} from HuggingFace...")
    print("Note: This is a 1M persona dataset and may take several minutes.\n")

    try:
        dataset = load_dataset(DATASET_NAME, revision=DATASET_REVISION, split="train")
        print(f"Loaded dataset with {len(dataset)} rows")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nPlease ensure you have:")
        print("1. Installed the datasets package: uv add datasets")
        print("2. Network access to HuggingFace")
        print("3. Sufficient disk space for the download")
        return None

    df = dataset.to_pandas()
    print(f"Converted to DataFrame: {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")

    if "state" not in df.columns:
        print("Error: expected a 'state' column in the dataset; found none.")
        return None
    ca_mask = df["state"].isin([STATE_FILTER, "California"])
    df = df[ca_mask].reset_index(drop=True)
    print(f"\nFiltered to state == {STATE_FILTER}: {len(df):,} personas")
    if len(df) == 0:
        print("Error: California filter matched no rows; check the 'state' column values.")
        return None

    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
        print(f"Sampled to {max_samples:,} rows (random_state=42)")

    output_path = Path(__file__).parent / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output_path, index=False)
    print(f"\nSaved {len(df):,} personas to: {output_path}")

    print("\n" + "=" * 60)
    print("Dataset Info:")
    print("=" * 60)
    print(f"Total rows: {len(df):,}")
    print(f"Columns ({len(df.columns)}):")
    for col in df.columns:
        dtype = df[col].dtype
        non_null = df[col].notna().sum()
        print(f"  - {col}: {dtype} ({non_null:,} non-null)")

    print("\n" + "=" * 60)
    print("Sample persona:")
    print("=" * 60)
    if len(df) > 0:
        sample = df.iloc[0]
        for col in df.columns:
            val = str(sample[col])
            if len(val) > 200:
                val = val[:200] + "..."
            print(f"  {col}: {val}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Download NVIDIA Nemotron-Personas-USA dataset (California subset) from HuggingFace"
    )
    parser.parse_args()

    download_personas(
        output_file=DEFAULT_OUTPUT_FILE,
        max_samples=DEFAULT_MAX_SAMPLES,
    )


if __name__ == "__main__":
    main()
