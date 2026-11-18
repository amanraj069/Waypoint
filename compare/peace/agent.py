"""PEACE's planner-executor (peace/nodes/planner_executor.py), without ROS.

The paper, section III-B: "A single LLM call converts the user query, robot
state, and world model into a structured mission plan: a reasoning string
followed by an ordered list of tool calls. The executor dispatches the tool
calls sequentially and re-invokes the LLM only on tool failure, in which case
a bounded replan is issued."

`PeaceAgent.run(query)` is one mission, with the released control flow:

1. Snapshot the robot state and the world model; build the two messages
   (the static system prompt; the query followed by the live context, with the
   state history when memory is on).
2. One LLM call, constrained to the `MissionPlan` schema. A reply that does not
   validate, or a plan with no steps, fails the mission there.
3. Execute the steps in order, with no LLM in the loop. A step whose result
   contains "failed" or starts with "ERROR" is a tool failure.
4. On a tool failure, while fewer than `agent_max_replan_depth` (2) replans
   have been made: one LLM call with the original query, the completed steps,
   the remaining steps, the failure reason and a fresh world context, asking
   for a revised plan "for the remaining objectives only"; then execute that.
   Otherwise the mission fails.
5. Write one row to the mission log and stop safely (loiter, stream off).

What replaces ROS: the subscriptions are the flight stack's listeners, the
query topic is the `run` call, and the worker pool is the caller's thread.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Optional

from compare.common.flight import FlightStack
from compare.common.llm import ChatClient, LLMError
from compare.common.sensors import Camera, SceneObject, lidar_scan
from compare.peace.config import PeaceSettings
from compare.peace.exceptions import InferenceError
from compare.peace.perception import ObjectDetector, VisionNode, detections_from
from compare.peace.prompts import LANGUAGE_SYSTEM_PROMPT, format_world_context
from compare.peace.schemas import ExecutionStatus, MissionPlan, RobotState, ToolCall, WorldModel
from compare.peace.services.constraint_enforcement import ConstraintEnforcementService
from compare.peace.services.mission_logger import MissionLoggerService
from compare.peace.services.robot_state import RobotStateAggregatorService
from compare.peace.services.world_model import WorldModelAggregatorService
from compare.peace.tools.uav_tools import UAVTools

log = logging.getLogger("compare.peace")


def release_protocol(llm: ChatClient) -> ChatClient:
    """The model as the release calls it (peace/backends/llm.py).

    The release sends Ollama the two messages with `format` set to the
    `MissionPlan` JSON schema, a temperature of 0.7 and a 2000-token cap, and
    sets no reasoning effort. A client configured for the rest of Waypoint
    carries Waypoint's own reasoning effort; this returns a copy without it,
    that passes the schema itself to the model. (A reasoning model takes no
    temperature, and its thinking would count against the cap, so on one those
    two are not sent.)
    """
    import copy  # noqa: PLC0415

    from compare.common.llm import OpenAIChat  # noqa: PLC0415

    if not isinstance(llm, OpenAIChat):
        return llm
    released = copy.copy(llm)
    released.reasoning_effort = None
    released.loose_schema = True
    return released


@dataclass
class MissionResult:
    """What one query came to. `state` is the released terminal state: completed or failed."""

    query: str
    state: str = "failed"
    message: str = ""
    mission_id: str = ""
    #: Every plan the LLM produced for this mission: the first, then each replan.
    plans: list[MissionPlan] = field(default_factory=list)
    #: Every step executed, across replans, with its result string.
    steps: list[dict] = field(default_factory=list)
    replan_count: int = 0
    llm_calls: int = 0
    llm_seconds: float = 0.0
    #: Simulated seconds the mission took, planning hovers included.
    flight_seconds: float = 0.0
    statuses: list[ExecutionStatus] = field(default_factory=list)

    @property
    def completed(self) -> bool:
        return self.state == "completed"

    @property
    def safety_aborts(self) -> list[dict]:
        """The steps the constraint enforcement layer refused."""
        return [s for s in self.steps if str(s["result"]).startswith("ERROR: Safety violation")]


class PeaceAgent:
    """Mission planning and flight execution with the planner-executor pattern."""

    def __init__(
        self,
        stack: FlightStack,
        llm: ChatClient,
        settings: PeaceSettings | None = None,
        *,
        objects: list[SceneObject] | tuple[SceneObject, ...] = (),
        camera: Camera | None = None,
        detector: ObjectDetector | None = None,
        lidar: bool = True,
        log_dir: str | None = None,
        tools_class: type[UAVTools] = UAVTools,
        system_prompt: str = LANGUAGE_SYSTEM_PROMPT,
        extra_context: Callable[[], str] | None = None,
        hold_while_thinking: bool = True,
    ) -> None:
        self.settings = settings or PeaceSettings()
        self.stack = stack
        self.llm = release_protocol(llm)
        self.system_prompt = system_prompt
        self.objects = list(objects)
        self.lidar = lidar
        #: Appended to the world context. PEACE has none; the Waypoint bridge uses it for named places.
        self.extra_context = extra_context
        #: The aircraft keeps flying (hovering) for as long as the LLM takes, as it does in real time.
        self.hold_while_thinking = hold_while_thinking

        self.ego_state_svc = RobotStateAggregatorService(context_window=self.settings.language_context_window)
        self.world_model_svc = WorldModelAggregatorService(
            context_window=self.settings.language_context_window,
            max_detections=self.settings.language_max_detections,
        )
        self.constraint_svc = ConstraintEnforcementService(
            min_altitude=self.settings.action_min_altitude,
            max_altitude=self.settings.action_max_altitude,
            horizontal_limit=self.settings.action_horizontal_limit,
        )
        self.mission_logger = MissionLoggerService(
            output_dir=log_dir or self.settings.log_output_dir,
            enabled=self.settings.log_enable and log_dir is not None,
        )

        # Telemetry: the flight stack's listeners stand where the MAVROS subscriptions were.
        stack.on_state(self._state_cb)
        stack.on_odom(self._odom_cb)
        self.vision = (
            VisionNode(stack, self.objects, self._detections_cb, self.settings, camera=camera, detector=detector)
            if self.objects
            else None
        )

        self.tools = tools_class(stack, self.ego_state_svc, self.world_model_svc, self.constraint_svc, self.settings)
        self._result: MissionResult | None = None

    def close(self) -> None:
        self.tools.stop_safely()
        self.mission_logger.close()

    # -- telemetry callbacks -------------------------------------------------------

    def _state_cb(self, mode: str, armed: bool, connected: bool) -> None:
        self.ego_state_svc.update_flight_mode(mode, armed, connected)

    def _odom_cb(self, x, y, z, vx, vy, vz, yaw) -> None:
        import math  # noqa: PLC0415

        # Odometry carries a quaternion; a multirotor's yaw is the rotation about z.
        self.ego_state_svc.update_odom(x, y, z, vx, vy, vz, 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))

    def _detections_cb(self, message: str) -> None:
        try:
            self.world_model_svc.update_detections(detections_from(message))
        except Exception as e:  # noqa: BLE001
            log.error("Failed to parse detections: %s", e)

    def _scan(self) -> None:
        """The LiDAR callback, taken when the world model is about to be read."""
        if not self.lidar:
            return
        ranges, angle_min, increment = lidar_scan(
            self.stack.world, self.stack.backend.obstacles, self.stack.state.position, self.stack.yaw,
            objects=self.objects,
        )
        self.world_model_svc.update_lidar(ranges, angle_min, increment, 0.1, 30.0)

    def world_context(self, history: list[RobotState] | None = None) -> tuple[RobotState, WorldModel, str]:
        self._scan()
        ego_state = self.ego_state_svc.get_ego_state()
        world_model = self.world_model_svc.get_world_model()
        context = format_world_context(ego_state, world_model, history)
        if self.extra_context is not None:
            extra = self.extra_context()
            if extra:
                context = f"{context}\n\n{extra}"
        return ego_state, world_model, context

    # -- planner-executor ------------------------------------------------------------

    def run(self, user_query: str, metadata: Optional[dict[str, str]] = None) -> MissionResult:
        """One mission: plan in a single LLM call, execute, replan on tool failure, log."""
        metadata = metadata or {}
        result = self._result = MissionResult(query=user_query)
        began = self.stack.now
        mission_id = ""
        ego_state = None
        world_model = None
        started_at = time.time()
        model_name = getattr(self.llm, "name", self.settings.language_model)
        try:
            history = (
                self.ego_state_svc.get_ego_state_history(self.settings.agent_memory_window)
                if self.settings.agent_enable_memory
                else None
            )
            ego_state, world_model, context = self.world_context(history)
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"{user_query}\n\n{context}"},
            ]
            mission_id = str(uuid.uuid4())
            result.mission_id = mission_id
            self._status(mission_id, "executing", "Planning mission")
            raw = self._chat(messages)
            plan = MissionPlan.model_validate_json(raw)
            plan.user_query = user_query
            plan.mission_id = mission_id
            result.plans.append(plan)
            if not plan.steps:
                self._status(mission_id, "failed", "LLM returned empty plan")
                self._log(mission_id, user_query, "", [], "failed", ego_state, world_model, metadata, started_at, 0, model_name)
                return self._done(result, began)
            step_summaries = [
                f"{s.tool}({', '.join(f'{k}={v}' for k, v in s.args.items())})" if s.args else s.tool
                for s in plan.steps
            ]
            self._status(
                mission_id, "executing", "Executing plan",
                reasoning=plan.reasoning, steps=step_summaries, current_index=0, total=len(plan.steps),
            )
            self._execute_plan(plan, ego_state, world_model, 0, metadata, started_at, model_name)
        except InferenceError as e:
            log.error("LLM inference failed: %s", e)
            self._status("", "failed", str(e))
            self._log(mission_id, user_query, "", [], "failed", ego_state, world_model, metadata, started_at, 0, model_name)
        except Exception as e:  # noqa: BLE001 -- a plan that does not validate fails the mission, as released
            log.error("Query processing error: %s: %s", type(e).__name__, e)
            self._status("", "failed", str(e))
            self._log(mission_id, user_query, "", [], "failed", ego_state, world_model, metadata, started_at, 0, model_name)
        return self._done(result, began)

    def _execute_plan(
        self,
        plan: MissionPlan,
        ego_state: RobotState,
        world_model: WorldModel,
        replan_depth: int = 0,
        metadata: Optional[dict[str, str]] = None,
        started_at: float = 0.0,
        model_name: str = "",
    ) -> None:
        """Execute plan steps sequentially, triggering a replan on step failure if allowed."""
        metadata = metadata or {}
        mission_id = plan.mission_id
        steps = list(plan.steps)
        completed: list[ToolCall] = []
        executed_steps: list[dict] = []
        final_state = "failed"
        try:
            for i, step in enumerate(steps):
                self._status(
                    mission_id, "executing", f"Step {i + 1}/{len(steps)}: {step.tool}",
                    current_index=i + 1, total=len(steps),
                )
                result = self.tools.execute(step.tool, step.args)
                log.info("[STEP %d/%d] %s -> %s", i + 1, len(steps), step.tool, result)
                record = {"tool": step.tool, "args": step.args, "result": result, "reasoning": step.reasoning}
                executed_steps.append(record)
                self._result.steps.append({**record, "plan": replan_depth, "t": round(self.stack.now, 2)})
                if "failed" in result.lower() or result.startswith("ERROR"):
                    if replan_depth < self.settings.agent_max_replan_depth:
                        new_plan = self._replan_remaining(plan, completed, result)
                        if new_plan:
                            final_state = "replanning"
                            self._result.replan_count = replan_depth + 1
                            self._execute_plan(
                                new_plan, ego_state, world_model, replan_depth + 1, metadata, started_at, model_name
                            )
                            return
                    self._status(mission_id, "failed", result)
                    return
                completed.append(step)
            self._status(mission_id, "completed", "Mission complete")
            final_state = "completed"
        except Exception as e:  # noqa: BLE001
            log.error("Plan execution error: %s", e)
            self._status(mission_id, "failed", str(e))
        finally:
            # Only the terminal call in the replan recursion writes the row, so each
            # mission produces exactly one log entry.
            if final_state != "replanning":
                self._log(
                    mission_id, plan.user_query or "", plan.reasoning, executed_steps, final_state,
                    ego_state, world_model, metadata, started_at, replan_depth, model_name,
                )
                self.tools.stop_safely()

    def _replan_remaining(
        self, plan: MissionPlan, completed: list[ToolCall], failure_reason: str
    ) -> Optional[MissionPlan]:
        """One replanning LLM call describing the mission state and failure reason.

        Returns a new MissionPlan for the remaining objectives, or None on error.
        """
        _, _, context = self.world_context()

        completed_summary = (
            "\n".join(f"  - {s.tool}({s.args}): {s.reasoning}" for s in completed) if completed else "  None"
        )
        remaining_summary = (
            "\n".join(f"  - {s.tool}({s.args}): {s.reasoning}" for s in plan.steps if s not in completed)
            if plan.steps
            else "  None"
        )

        replan_user = (
            f"Original query: {plan.user_query or ''}\n\n"
            f"Completed steps:\n{completed_summary}\n\n"
            f"Remaining steps (not yet executed):\n{remaining_summary}\n\n"
            f"Failure reason: {failure_reason}\n\n"
            f"Current world context:\n{context}\n\n"
            "Produce a revised MissionPlan for the remaining objectives only. "
            "Return ONLY the JSON mission plan, no additional text."
        )
        try:
            raw = self._chat(
                [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": replan_user},
                ]
            )
            new_plan = MissionPlan.model_validate_json(raw)
            new_plan.mission_id = new_plan.mission_id or str(uuid.uuid4())
            new_plan.user_query = new_plan.user_query or plan.user_query
            log.info("[REPLAN REASONING] %s", new_plan.reasoning)
            self._result.plans.append(new_plan)
            return new_plan
        except Exception as e:  # noqa: BLE001
            log.error("Replanning failed: %s: %s", type(e).__name__, e)
            return None

    # -- helpers ---------------------------------------------------------------------

    def _chat(self, messages: list[dict]) -> str:
        """The released LLM call: the two messages, the output constrained to the MissionPlan schema."""
        try:
            reply = self.llm.chat(
                messages,
                schema=MissionPlan.model_json_schema(),
                temperature=self.settings.language_model_temperature,
                max_tokens=self.settings.language_model_num_predict,
            )
        except LLMError as e:
            raise InferenceError(str(e)) from e
        self._result.llm_calls += 1
        self._result.llm_seconds += reply.seconds
        if self.hold_while_thinking and reply.seconds > 0:
            self.stack.advance(reply.seconds)
        return reply.text

    def _status(
        self, mission_id: str, state: str, message: str, reasoning: str = "",
        steps: list[str] | None = None, current_index: int = 0, total: int = 0,
    ) -> None:
        status = ExecutionStatus(
            mission_id=mission_id, current_waypoint_index=current_index, total_waypoints=total, state=state,
            position_error=0.0, message=message, timestamp=time.time(), reasoning=reasoning, steps=steps or [],
        )
        if self._result is not None:
            self._result.statuses.append(status)
            if state in ("completed", "failed", "aborted"):
                self._result.state = state
                self._result.message = message

    def _log(self, mission_id, query, reasoning, steps, final_state, ego_state, world_model, metadata, started_at, replans, model) -> None:
        self.mission_logger.log_mission(
            mission_id, query, reasoning, steps, final_state, ego_state, world_model,
            metadata=metadata,
            final_ego_state=self.ego_state_svc.get_ego_state(),
            started_at=started_at,
            duration_s=(time.time() - started_at) if started_at else 0.0,
            replan_count=replans,
            model=model,
        )

    def _done(self, result: MissionResult, began: float) -> MissionResult:
        result.flight_seconds = round(self.stack.now - began, 2)
        return result
