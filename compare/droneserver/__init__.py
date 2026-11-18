"""DroneServer: an LLM-agnostic, MAVLink-based drone command and control interface over MCP, rebuilt.

Ramos Silva and Burke (UC Irvine), arXiv:2601.15486. Rebuilt from the paper and
the authors' release (https://github.com/PeterJBurke/droneserver, MIT, commit
06b7d1a). The safety pipeline's rules (tiers, bounds, geofence, confirmation
tokens, scopes, rate limits, audit records) are the released modules, vendored
unchanged; the guard, the 98-tool MCP catalogue, the agent loop, the system
prompt, the mission prompts and the flight-evidence verdicts follow the
release; MavSDK and the autopilot are replaced by a simulated vehicle on
Waypoint's simulator. `compare/droneserver/README.md` lists every element and
every difference.

    from compare.common.llm import default_client
    from compare.droneserver import run_trial

    trial = run_trial(default_client(), "T2")
    trial.verdict.passed, trial.verdict.reason
"""

from compare.droneserver.benchmark.missions import MISSIONS, Trial, fresh_vehicle, run_trial
from compare.droneserver.server import DroneServer, control_settings
from compare.droneserver.vehicle import Vehicle

__all__ = ["DroneServer", "MISSIONS", "Trial", "Vehicle", "control_settings", "fresh_vehicle", "run_trial"]
