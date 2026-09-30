"""Filesystem helpers shared across apps."""

import json
import os
import re
from datetime import datetime


def seed_suffix(seed: int) -> str:
    """Filename suffix _s<N>, or empty for the default seed 0."""
    return "" if int(seed) == 0 else f"_s{int(seed)}"


def model_tag(model: str) -> str:
    """Filesystem-safe short name for a model id: provider prefix dropped,
    underscores normalised (``provider/model_name`` -> ``model-name``)."""
    return model.split("/")[-1].strip().replace("_", "-")


def find_latest_archive(archives_dir: str, profile_tag: str, suffix: str) -> str | None:
    """Find the newest archive matching a profile token and exact stage/model/seed suffix.

    Rank by timestamp prefix, then modification time. Return None if none match.
    """
    if not os.path.exists(archives_dir):
        return None

    token_re = re.compile(rf"(^|[^A-Za-z0-9]){re.escape(profile_tag)}([^A-Za-z0-9]|$)")
    matching: list[str] = []
    for filename in os.listdir(archives_dir):
        if not filename.endswith(".json"):
            continue
        stem = os.path.splitext(filename)[0]
        if token_re.search(stem) and stem.endswith(f"_{suffix}"):
            matching.append(filename)
    if not matching:
        return None

    def _score(fn: str) -> tuple[int, float, float]:
        """Files with a parseable timestamp prefix first, then by modification time."""
        mtime = os.path.getmtime(os.path.join(archives_dir, fn))
        try:
            ts = datetime.strptime(os.path.splitext(fn)[0][:15], "%Y-%m-%d_%H%M").timestamp()
        except ValueError:
            return (0, 0.0, mtime)
        return (1, ts, mtime)

    return os.path.join(archives_dir, max(matching, key=_score))


def write_json_atomic(path: str, data, **dump_kwargs) -> None:
    """Write through a sibling temporary file, then atomically replace path.

    Keeping both files on the same filesystem makes the replacement atomic.
    """
    tmp = f"{path}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, **dump_kwargs)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        # Clean up on KeyboardInterrupt/SystemExit as well as write errors.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def unique_archive_path(path: str) -> str:
    """Return path or the first unused <stem>-2.json, <stem>-3.json, etc.

    Choose once per run and reuse the result for checkpoints.
    """
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    n = 2
    while os.path.exists(f"{stem}-{n}{ext}"):
        n += 1
    out = f"{stem}-{n}{ext}"
    print(f"[paths] {os.path.basename(path)} already exists - writing {os.path.basename(out)} instead")
    return out
