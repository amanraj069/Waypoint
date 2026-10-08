"""The experiment harness: fly batches of missions and measure the closed loop.

`run_batch` flies them (`runner.py`), `load` and `aggregate` turn the files it
wrote into metrics (`metrics.py`), and `write_report` writes CSVs, a summary
and plots (`report.py`). `python -m waypoint.experiments` is the command line;
`experiments/README.md` says how to reproduce each metric.
"""

from waypoint.experiments.metrics import MissionRow, ReplanRow, aggregate, load
from waypoint.experiments.report import write_report
from waypoint.experiments.runner import BatchConfig, run_batch, unverified
from waypoint.experiments.spec import CONDITIONS, Batch, MissionSpec, prepare, random_schedule

__all__ = [
    "CONDITIONS",
    "Batch",
    "BatchConfig",
    "MissionRow",
    "MissionSpec",
    "ReplanRow",
    "aggregate",
    "load",
    "prepare",
    "random_schedule",
    "run_batch",
    "unverified",
    "write_report",
]
