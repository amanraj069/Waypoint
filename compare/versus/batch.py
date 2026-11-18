"""One flight of the comparison, as a job a worker process can run."""

from __future__ import annotations

import gzip
import json
import logging
import tempfile
import traceback
from pathlib import Path


def work(job: tuple) -> dict:
    """One flight of one mission by one system, in its own process."""
    system, mission_id, seed, out = job
    for noisy in ("mcp", "httpx", "openai", "waypoint", "compare"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    from compare.common.llm import default_client  # noqa: PLC0415
    from compare.versus.fly import compact, fly  # noqa: PLC0415
    from compare.versus.scenario import missions, scenario, truths  # noqa: PLC0415
    from compare.versus.score import grade  # noqa: PLC0415
    from waypoint.planner import llm_planner  # noqa: PLC0415

    spec = next(m for m in missions() if m.id == mission_id)
    record: dict = {
        "system": system, "mission_id": mission_id, "mission": spec.mission, "world": spec.world or "open",
        "faults_spec": spec.faults, "seed": seed, "error": None,
    }
    try:
        scn = scenario(spec, seed)
        record["faults"] = [event.model_dump(mode="json") for event in scn.faults.events]
        spent = llm_planner.usage_snapshot()
        with tempfile.TemporaryDirectory() as scratch:
            run, detail = fly(system, scn, None if system == "waypoint" else default_client(), Path(scratch))
        record["usage"] = llm_planner.usage_since(spent)
        record["detail"] = detail
        record["grade"] = grade(run, truths().get(mission_id), scn.constraints)
        record["outcome"] = (run or {}).get("outcome") or detail.get("outcome")
        target = Path(out) / system
        target.mkdir(parents=True, exist_ok=True)
        if run is not None:
            with gzip.open(target / f"{mission_id}.run.json.gz", "wt", encoding="utf-8") as handle:
                json.dump(compact(run), handle)
    except Exception as exc:  # noqa: BLE001 -- one flight must not end the batch
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()[-1500:]
    target = Path(out) / system
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{mission_id}.json").write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    return record
