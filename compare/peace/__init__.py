"""PEACE: a Planner-Executor Agent with Constraint Enforcement for UAVs, rebuilt.

Uysal, Kehrer, Panichella (University of Bern), arXiv:2606.00104, May 2026.
Rebuilt from the paper and the authors' release
(https://github.com/erdemuysalx/PEACE, MIT, commit 4ceda37): the schemas, the
services, the tool set, the planner prompt and the planner-executor control
flow are the released ones; ROS 2 / MAVROS is replaced by `compare.common.flight`.
`compare/peace/README.md` lists every element and every difference.

    from compare.common.flight import make_stack
    from compare.common.llm import default_client
    from compare.peace import PeaceAgent

    agent = PeaceAgent(make_stack(), default_client())
    result = agent.run("Take off to 5 meters then move forward 3 meters")
"""

from compare.peace.agent import MissionResult, PeaceAgent
from compare.peace.config import PeaceSettings
from compare.peace.schemas import MissionPlan, ToolCall

__all__ = ["MissionPlan", "MissionResult", "PeaceAgent", "PeaceSettings", "ToolCall"]
