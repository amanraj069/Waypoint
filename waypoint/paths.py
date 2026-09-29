"""Where everything lives on disk, in one place.

Every other module asks here rather than working out its own path, so moving a
directory is a one-line change. Paths are absolute, anchored at the repository
root, so a command behaves the same whatever directory it is run from.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
WORLDS_DIR = DATA / "worlds"    # saved environments, one JSON per place
PLANS_DIR = DATA / "plans"      # verified plans that fly without an API key
FAULTS_DIR = DATA / "faults"    # fault schedules to replay

RUNS_DIR = ROOT / "runs"        # every flight flown, as a self-contained JSON
