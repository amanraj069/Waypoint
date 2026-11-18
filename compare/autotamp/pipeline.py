"""AutoTAMP: translate to STL, plan, check, re-prompt (env_and_optimize_single_agent.py).

The paper, section III-C. One instruction goes through:

1. Translation, two LLM calls: the instruction and the scene to a lifted
   sentence and its atomic propositions (few-shot); then the lifted sentence to
   a pre-order STL list (few-shot, the NL2TL examples).
2. Syntactic checking: "we use a simple rules-based STL syntax checker; any
   errors are provided as feedback when re-prompting the LLM to generate
   corrected STL. We repeat until no errors are found (up to five iterations)."
3. The STL planner turns the formula into a timed-waypoint trajectory.
4. Semantic checking: "we provide the STL planner's generated state sequence as
   context alongside the original instruction and ask the LLM to check whether
   the plan aligns with the instruction's semantics. If it does not, the LLM is
   prompted to modify the STL ... This process terminates in the case of no
   detected errors or no change in STL (up to three iterations)."

`nl_to_trajectory` is the release's `NL2action` for the GPT-4 chat path, call
for call, and `build_spec` its `test` (the STL list to the planner's tree).
The two switches give the paper's ablation rows: No Corrections, Syntax,
Syntax + Semantics (AutoTAMP).

Kept from the release, because they shape the results:

* every `enter(...)` is planned as "eventually enter" (the release's test for
  a preceding temporal operator can never be true), and a `not_enter(...)`
  with no operator before it as "always not enter";
* after the syntax loop the lifted sentence is translated to STL a second
  time, and a list that still miscounts is forced into shape by prepending
  'and's or dropping leading tokens;
* a formula coming back from the semantic check is planned as written:
  implications and negations in it are not rewritten, so such a formula is
  kept as the answer but not re-planned;
* HouseWorld plans end at the test case's end position (a hard goal), in 20
  segments; Chip's Challenge plans use 30 segments and stop at the first
  trajectory found (MIPGap 0.99).
"""

from __future__ import annotations

import ast
import copy
import re
import time
from dataclasses import dataclass, field

from compare.autotamp import prompts
from compare.autotamp.envs import Environment, regions
from compare.autotamp.stl.pwl_plan import Node, PlanStats, plan
from compare.autotamp.syntax import (
    AP_dict_syntactic_correct_check,
    check_syntactic_correct,
    check_syntactic_correct_inverse_order,
    extract_list_for_semantic_check_LLM_response,
    filter_negation_imply,
    force_syntax,
    parse_string,
    state_sequence_text,
    syntax_feedback,
    trajectory_len_cal,
)
from compare.common.llm import ChatClient

SYNTAX_ITERATIONS = 5
SEMANTIC_ITERATIONS = 3
TMAX = 50.0
VMAX = 5.0


def _window(token: str, tmax: float) -> tuple[float, float]:
    """The time window of 'globally [a,b]' and its like; 'infinite' is the horizon."""
    matches = re.findall(r"\[([^\]]+)\]", token)
    time_expressions = [item.strip() for match in matches for item in match.split(",")]
    time_expressions = [int(item) if item.isdigit() else item for item in time_expressions]
    if len(time_expressions) != 2:
        raise ValueError(f"cannot read the time window of {token!r}")
    start = int(time_expressions[0])
    end = tmax if time_expressions[1] == "infinite" else time_expressions[1]
    return start, end


ENTER_SEMANTICS = ("released", "strict")


def build_spec(input_stl_original: list, prop2block: dict, tmax: float = TMAX, enter_semantics: str = "released") -> Node:
    """A pre-order STL token list as the planner's tree (the release's parser in `test`).

    `enter_semantics="released"` is the release's parser, token for token.
    `"strict"` is this rebuild's alternative, for measuring what the release's
    wrapping costs: see `build_spec_strict`.
    """
    if enter_semantics == "strict":
        return build_spec_strict(input_stl_original, prop2block, tmax)
    if enter_semantics != "released":
        raise ValueError(f"enter_semantics must be one of {', '.join(ENTER_SEMANTICS)}")
    input_stl = copy.deepcopy(input_stl_original)

    while not (len(input_stl) == 1 and type(input_stl[0]) != str):  # noqa: E721
        before = [type(item) for item in input_stl]
        for i in range(len(input_stl) - 1, -1, -1):
            if type(input_stl[i]) != str:  # noqa: E721
                pass
            elif input_stl[i][:5] == "enter":
                block = prop2block["prop_" + input_stl[i][10:-1]][-1]
                phi = Node("mu", info={"A": block[0], "b": block[1]})
                input_stl.pop(i)
                input_stl.insert(i, phi)
                # As released: this condition holds for every token, so an enter is always "eventually enter".
                if (
                    i == 0
                    or type(input_stl[i - 1]) != str  # noqa: E721
                    or not (input_stl[i - 1][0:4] == "glob" and input_stl[i - 1][0:4] == "fina")
                ):
                    phi_2 = Node("F", deps=[phi], info={"int": [0, tmax]})
                    input_stl.pop(i)
                    input_stl.insert(i, phi_2)
            elif input_stl[i][:9] == "not_enter":
                block = prop2block["prop_" + input_stl[i][14:-1]][-1]
                phi = Node("negmu", info={"A": block[0], "b": block[1]})
                input_stl.pop(i)
                input_stl.insert(i, phi)
                if i == 0 or type(input_stl[i - 1]) != str:  # noqa: E721
                    phi_2 = Node("A", deps=[phi], info={"int": [0, tmax]})
                    input_stl.pop(i)
                    input_stl.insert(i, phi_2)
            elif input_stl[i] == "and" or input_stl[i] == "or":
                phi = Node(input_stl[i], deps=[input_stl[i + 1], input_stl[i + 2]])
                input_stl.pop(i + 2)
                input_stl.pop(i + 1)
                input_stl.pop(i)
                input_stl.insert(i, phi)
            elif input_stl[i].split(" ")[0] in ("globally", "finally"):
                op = "A" if input_stl[i].split(" ")[0] == "globally" else "F"
                interval = [0, tmax] if input_stl[i] in ("globally", "finally") else list(_window(input_stl[i], tmax))
                phi = Node(op, deps=[input_stl[i + 1]], info={"int": interval})
                input_stl.pop(i + 1)
                input_stl.pop(i)
                input_stl.insert(i, phi)
            elif input_stl[i].split(" ")[0] == "until":
                interval = [0, tmax] if input_stl[i] == "until" else list(_window(input_stl[i], tmax))
                phi = Node("U", deps=[input_stl[i + 1], input_stl[i + 2]], info={"int": interval})
                input_stl.pop(i + 2)
                input_stl.pop(i + 1)
                input_stl.pop(i)
                input_stl.insert(i, phi)
        if [type(item) for item in input_stl] == before and len(input_stl) > 1:
            # The release would loop for ever on a list it cannot reduce; here that is an error.
            raise ValueError(f"the STL list cannot be parsed: {input_stl_original}")
    return input_stl[0]


def build_spec_strict(tokens: list, prop2block: dict, tmax: float = TMAX) -> Node:
    """The same token list read with the operators' textbook meaning. Not the release's behaviour.

    The release plans every `enter(room)` as "eventually enter(room)", wherever
    it stands. Under a temporal operator that empties the formula:
    `until not_enter(A) enter(B)` becomes "A is not entered until (eventually
    B)", which holds from the first instant whenever B is ever entered, so the
    order is not enforced; `globally [0,5] enter(B)` becomes "for 5 s, B is
    eventually entered", which a trajectory satisfies by waiting anywhere and
    entering B at the end, so the stay in the room is not enforced. Here an `enter` or
    `not_enter` that is the operand of globally, finally or until is the bare
    predicate; outside any temporal operator an `enter` is still "eventually"
    and a `not_enter` "always".
    """
    position = 0

    def read(temporal_parent: bool) -> Node:
        nonlocal position
        if position >= len(tokens):
            raise ValueError(f"the STL list ends early: {tokens}")
        token = tokens[position]
        position += 1
        head = token.split(" ")[0]
        if token[:5] == "enter":
            block = prop2block["prop_" + token[10:-1]][-1]
            node = Node("mu", info={"A": block[0], "b": block[1]})
            return node if temporal_parent else Node("F", deps=[node], info={"int": [0, tmax]})
        if token[:9] == "not_enter":
            block = prop2block["prop_" + token[14:-1]][-1]
            node = Node("negmu", info={"A": block[0], "b": block[1]})
            return node if temporal_parent else Node("A", deps=[node], info={"int": [0, tmax]})
        if token in ("and", "or"):
            first = read(temporal_parent)
            return Node(token, deps=[first, read(temporal_parent)])
        if head in ("globally", "finally"):
            interval = [0, tmax] if token == head else list(_window(token, tmax))
            return Node("A" if head == "globally" else "F", deps=[read(True)], info={"int": interval})
        if head == "until":
            interval = [0, tmax] if token == head else list(_window(token, tmax))
            first = read(True)
            return Node("U", deps=[first, read(True)], info={"int": interval})
        raise ValueError(f"the STL list cannot be parsed at {token!r}: {tokens}")

    spec = read(False)
    if position != len(tokens):
        raise ValueError(f"the STL list has tokens left over: {tokens}")
    return spec


def plan_stl(
    domain: str,
    input_stl: list,
    prop2block: dict,
    x0,
    goal,
    *,
    tmax: float = TMAX,
    vmax: float = VMAX,
    time_limit: float = 2000.0,
    stats: PlanStats | None = None,
    enter_semantics: str = "released",
):
    """The release's `test`: parse the STL and plan with the domain's planner settings."""
    spec = build_spec(input_stl, prop2block, tmax, enter_semantics)
    if domain == "chip":
        return plan([x0], [spec], bloat=0.18, MIPGap=0.99, num_segs=30, tmax=tmax, vmax=vmax, time_limit=time_limit, stats=stats)
    if domain == "HouseWorld":
        return plan(
            [x0], [spec], bloat=0.05, size=0.11 / 2, num_segs=20, tmax=tmax, vmax=vmax, hard_goals=[goal],
            time_limit=time_limit, stats=stats,
        )
    raise ValueError("unknown domain")


@dataclass
class Outcome:
    """What one instruction came to."""

    instruction: str
    #: The STL the pipeline ended with (pre-order token list), after any corrections.
    stl: list = field(default_factory=list)
    #: The lifted STL (over prop_i) from the second translation call.
    lifted_stl: list = field(default_factory=list)
    #: 1 when the STL passed the final syntax check and was handed to the planner, as the release marks it.
    mark_syntactic: int = 0
    #: [[x, y, t], ...] of the last trajectory planned; None when no plan was found.
    trajectory: list | None = None
    trajectory_length: float | None = None
    syntactic_iterations: int = 0
    semantic_iterations: int = 0
    llm_calls: int = 0
    time_total: float = 0.0
    time_syntactic: float = 0.0
    time_semantic: float = 0.0
    time_planning: float = 0.0
    plans: list[dict] = field(default_factory=list)
    #: Why the pipeline stopped early, when it did (the release swallows these and moves on).
    error: str | None = None
    transcript: list[dict] = field(default_factory=list)


def lenient_translation(reply: str) -> str:
    """The translation reply from its last "transformed sentence" on, dropping anything a model said first."""
    start = reply.rfind("transformed sentence")
    return reply[start:] if start > 0 else reply


def lenient_list(reply: str) -> str:
    """The first bracketed list in a reply, for a model that wraps its answer in a sentence or an "STL:" label."""
    match = re.search(r"\[[^\[\]]*\]", reply)
    return match.group(0) if match else reply


class _Chat:
    """The release's three chat helpers over one client, counting calls and keeping the transcript."""

    def __init__(self, llm: ChatClient, outcome: Outcome, lenient: bool = True):
        self.llm, self.outcome, self.lenient = llm, outcome, lenient

    def _ask(self, kind: str, messages: list[dict]) -> str:
        reply = self.llm.chat([{"role": "system", "content": prompts.SYSTEM}, *messages])
        self.outcome.llm_calls += 1
        self.outcome.transcript.append({"kind": kind, "prompt": messages[-1]["content"], "reply": reply.text})
        return reply.text

    def first_round(self, kind: str, user_prompt_1: str) -> str:
        return self._ask(kind, [{"role": "user", "content": user_prompt_1}])

    def second_round(self, kind: str, user_prompt_1: str, assistant_feed: str, user_prompt_2: str) -> str:
        return self._ask(
            kind,
            [
                {"role": "user", "content": user_prompt_1},
                {"role": "assistant", "content": assistant_feed},
                {"role": "user", "content": user_prompt_2},
            ],
        )

    def nl2tl_preorder(self, lifted_NL: str) -> list:
        reply = self.first_round("nl2tl", prompts.NL2TL_PREORDER + lifted_NL + "\nSTL: ").strip()
        return ast.literal_eval(lenient_list(reply) if self.lenient else reply)

    def translation(self, reply: str) -> str:
        return lenient_translation(reply) if self.lenient else reply


def nl_to_trajectory(
    llm: ChatClient,
    instruction: str,
    *,
    start_position,
    end_position,
    domain: str = "HouseWorld",
    environment: str = "env2",
    syntactic_correct_loop: bool = True,
    semantic_correct_loop: bool = True,
    time_limit: float = 2000.0,
    tmax: float = TMAX,
    vmax: float = VMAX,
    enter_semantics: str = "released",
    lenient_replies: bool = True,
) -> Outcome:
    """One instruction through AutoTAMP. Never raises: a failure is an Outcome with `error` set.

    `time_limit` is the planner's, in seconds (2000 in the release); a solve
    that runs out returns the best trajectory it has, as the release's does.

    `lenient_replies` reads the model's answer out of a reply that says more
    than the answer. The release evaluates the whole STL reply as a Python
    literal and splits the translation reply at its first colon, which the
    GPT-4 of 2023 satisfied by replying with nothing else; a current model
    often adds "STL:" or a sentence, and every such case would be lost before
    the method is exercised at all. False restores the release's reading.
    """
    outcome = Outcome(instruction=instruction)
    began = time.perf_counter()
    try:
        _run(
            llm, outcome, regions(domain, environment), list(start_position), list(end_position),
            syntactic_correct_loop, semantic_correct_loop, time_limit, tmax, vmax, enter_semantics, lenient_replies,
        )
    except Exception as exc:  # noqa: BLE001 -- the release's `except: pass` around every test case
        outcome.error = f"{type(exc).__name__}: {exc}"
    outcome.time_total = round(time.perf_counter() - began, 3)
    return outcome


def _run(llm, outcome: Outcome, env: Environment, start, end, syntactic, semantic, time_limit, tmax, vmax, enter_semantics, lenient=True) -> None:
    chat = _Chat(llm, outcome, lenient)

    def solve(stl: list) -> list | None:
        stats = PlanStats()
        clock = time.perf_counter()
        PWLs = plan_stl(
            env.domain, stl, env.prop2block, start, end, tmax=tmax, vmax=vmax, time_limit=time_limit, stats=stats,
            enter_semantics=enter_semantics,
        )
        outcome.time_planning += time.perf_counter() - clock
        outcome.plans.append({"stl": list(stl), **vars(stats), "found": PWLs != [None]})
        if PWLs == [None]:
            return None
        outcome.trajectory = [[P[0][0], P[0][1], P[1]] for P in PWLs[0]]
        outcome.trajectory_length = round(trajectory_len_cal(PWLs), 4)
        return outcome.trajectory

    # First round
    user_prompt_1 = prompts.translation_prompt(env.action_list, outcome.instruction)
    AP_and_lifted_NL = chat.translation(chat.first_round("translate", user_prompt_1))
    lifted_NL, AP_dict = parse_string(AP_and_lifted_NL)
    TL_list = chat.nl2tl_preorder(lifted_NL)

    clock = time.perf_counter()
    if syntactic:
        try:
            # The syntax loop: up to five corrections of the atomic propositions.
            iteration = 0
            for iteration in range(SYNTAX_ITERATIONS):  # noqa: B007
                if AP_dict_syntactic_correct_check(AP_dict):
                    break
                AP_and_lifted_NL = chat.translation(
                    chat.second_round("syntax", user_prompt_1, copy.deepcopy(AP_and_lifted_NL), syntax_feedback(AP_dict))
                )
                lifted_NL, AP_dict = parse_string(AP_and_lifted_NL)
            outcome.syntactic_iterations = iteration + 1
            TL_list = chat.nl2tl_preorder(lifted_NL)
            TL_list = force_syntax(TL_list)
        except Exception:  # noqa: BLE001 -- as released: a failed correction keeps what it had
            pass
    outcome.time_syntactic = round(time.perf_counter() - clock, 3)
    outcome.lifted_stl = list(TL_list)

    # transform TL_list into input_stl
    input_stl: list = []
    for item in TL_list:
        if item[0:4] == "prop":
            input_stl += AP_dict[item]
        else:
            input_stl += [item]
    input_stl = filter_negation_imply(input_stl)
    outcome.stl = list(input_stl)

    if not (
        check_syntactic_correct_inverse_order(input_stl) == "correct"
        and "negation" not in input_stl
        and "imply" not in input_stl
    ):
        outcome.mark_syntactic = 0
        return
    outcome.mark_syntactic = 1
    position_time_list = solve(input_stl)
    if position_time_list is None:
        return

    clock = time.perf_counter()
    if semantic:
        output_stl_1 = copy.deepcopy(input_stl)
        for _ in range(SEMANTIC_ITERATIONS):
            outcome.semantic_iterations += 1
            user_prompt_2 = prompts.semantic_check_prompt(
                output_stl_1, state_sequence_text(env.tag_and_shape, position_time_list)
            )
            response = chat.second_round("semantic", user_prompt_1, AP_and_lifted_NL, user_prompt_2)
            output_stl_after_semantic = force_syntax(extract_list_for_semantic_check_LLM_response(response))
            if output_stl_after_semantic == output_stl_1:
                break  # no change in STL: the check found nothing to correct
            output_stl_1 = copy.deepcopy(output_stl_after_semantic)
            input_stl = copy.deepcopy(output_stl_after_semantic)
            outcome.stl = list(input_stl)
            if (
                check_syntactic_correct_inverse_order(input_stl) == "correct"
                and "negation" not in input_stl
                and "imply" not in input_stl
            ):
                replanned = solve(input_stl)
                if replanned is not None:
                    position_time_list = replanned
    outcome.time_semantic = round(time.perf_counter() - clock, 3)
