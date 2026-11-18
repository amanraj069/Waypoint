"""AutoTAMP, rebuilt: the released prompts, checker and planner encoding, on an open solver."""

from __future__ import annotations

import hashlib
import json
import math

import pytest

from compare.autotamp import prompts
from compare.autotamp.checkers import ROOMS, Visit, check_houseworld, check_visits, visits, within_time
from compare.autotamp.envs import regions
from compare.autotamp.experiment import METHODS, PAPER, cases, instructions, run, summarise
from compare.autotamp.pipeline import build_spec, nl_to_trajectory, plan_stl
from compare.autotamp.reference import reference_spec, time_limits
from compare.autotamp.stl.milp import INF, Model
from compare.autotamp.stl.pwl_plan import EPS, M, T_MIN_SEP, Node, PlanStats, plan
from compare.autotamp.syntax import (
    check_syntactic_correct,
    check_syntactic_correct_inverse_order,
    extract_list_for_semantic_check_LLM_response,
    filter_negation_imply,
    force_syntax,
    parse_string,
    state_sequence_text,
    syntax_feedback,
)
from compare.common.llm import ScriptedChat

START, END = [1.3, -1.3], [0.0, 1.3]
ENV = regions("HouseWorld")


def translation(sentence: str, props: dict[str, list[str]]) -> str:
    body = " ; ".join(f"'{name}' : {stl}" for name, stl in props.items())
    return f"transformed sentence: {sentence}.\n{body}"


def trajectory_of(stl, semantics="released", start=START, end=END, limit=20.0):
    out = plan_stl("HouseWorld", stl, ENV.prop2block, start, end, time_limit=limit, enter_semantics=semantics)
    return None if out == [None] else [[p[0][0], p[0][1], p[1]] for p in out[0]]


# -- the released artefacts ---------------------------------------------------------


def test_the_prompts_are_the_released_text():
    digests = {name: hashlib.sha256(text.encode()).hexdigest()[:16] for name, text in
               (("translate", prompts.TRANSLATE), ("nl2tl", prompts.NL2TL_PREORDER), ("end_to_end", prompts.END_TO_END))}
    assert digests == {"translate": "6468b8ce9ba9efab", "nl2tl": "8810bdeff8042d22", "end_to_end": "f46c4430facb9ffe"}
    assert prompts.SYSTEM == "You are a helpful assistant."
    assert prompts.TRANSLATE.startswith("Please help me detect the actions in the sentence and transform the action expression")
    check = prompts.semantic_check_prompt(["finally", "enter(room1)"], "[in white road at time 0.0].")
    assert check.startswith("Based on your predicted STL ['finally', 'enter(room1)'] , the state sequence [[location, time]]")
    assert check.endswith("in the last line output the full final STL beginning with STL: . \nOutput:")


def test_the_houseworld_scene_is_described_to_the_llm_as_released():
    assert ENV.action_list.startswith(
        "available scene objects: [name : room1, color: cyan, function: Kitchen, position and size: (1.2, -0.8, 1.4, -0.7)], "
        "[name : room2, color: red, function: RestRoom2, position and size: (0, 0.9, 1.4, -0.7)]"
    )
    assert ENV.action_list.endswith("\navailable actions: [enter(), not_enter()]")
    rooms = ENV.rooms()
    assert rooms["room5"] == ("blue", "LivingRoom", (-0.5, 0.5, -0.5, 0.5))
    assert len(rooms) == 12 and [rooms[f"room{i}"][1] for i in range(9, 13)] == ["wall"] * 4
    assert ROOMS["Kitchen"] == (-1.2, -0.8, -1.4, -0.7)


def test_the_chip_environments_have_the_released_layouts():
    assert {name: len(regions("chip", name).prop2block) for name in ("env1", "env2", "env3")} == {"env1": 21, "env2": 23, "env3": 32}
    env1 = regions("chip", "env1").rooms()
    assert env1["room1"][1] == "key1" and env1["room6"][1] == "goal" and env1["room8"][1] == "door1"


def test_the_instruction_sets_are_the_released_200_cases():
    assert [len(instructions(family)) for family in range(1, 11)] == [10] * 10
    assert instructions(1)[0] == "Proceed towards the green bedroom, and then make your way to the yellow bathroom."
    all_cases = cases()
    assert len(all_cases) == 200 and len({c["id"] for c in all_cases}) == 200
    assert cases(seed=0)[5]["start"] == all_cases[5]["start"], "the random starts are reproducible"
    nine = [c for c in all_cases if c["family"] == 9][0]
    assert -1.45 <= nine["start"][0] <= -1.25 and 1.05 <= nine["end"][0] <= 1.25
    a = [c for c in all_cases if c["family"] == 1 and c["variant"] == "a"][0]
    assert 1.2 <= a["start"][0] <= 1.41 and -1.41 <= a["start"][1] <= -1.2 and -0.2 <= a["end"][0] <= 0.2


# -- the rules-based syntax checker (paper section III-C) ----------------------------


def test_the_syntax_checker_counts_leaves_against_binary_operators():
    assert check_syntactic_correct(["and", "finally", "enter(room1)", "globally", "not_enter(room5)"]) == "correct"
    assert check_syntactic_correct(["finally [0,5]", "enter(room1)"]) == "correct"
    assert check_syntactic_correct(["enter(room1)", "enter(room2)"]) == 2, "one binary operator short"
    assert check_syntactic_correct(["and", "and", "enter(room1)", "enter(room2)"]) == 0, "one binary operator too many"
    assert check_syntactic_correct(["and", "go(room1)", "enter(room2)"]) == "go(room1)", "not STL at all"
    assert check_syntactic_correct_inverse_order(["and", "enter(room1)", "enter(room2)"]) == "correct"
    assert check_syntactic_correct_inverse_order(["enter(room1)", "and", "enter(room2)"]) == "error"


def test_the_syntax_correction_says_what_is_missing():
    message = syntax_feedback({"prop_1": ["enter(room1)", "enter(room2)"], "prop_2": ["and", "and", "enter(room3)", "enter(room4)"]})
    assert message == (
        "Your predicted prop_1 lacks 1 operators with two leaves, the correct syntactic format should be like "
        "['and', 'enter(room1)', 'enter(room2)']. "
        "Your predicted prop_2 has 1 more operators with two leaves, the correct syntactic format should be like "
        "['and', 'enter(room3)', 'enter(room4)']. Please correct your answer.\nOutput: "
    )
    assert force_syntax(["enter(room1)", "enter(room2)"]) == ["and", "enter(room1)", "enter(room2)"]
    assert force_syntax(["and", "and", "enter(room1)", "enter(room2)"]) == ["and", "enter(room1)", "enter(room2)"]


def test_implications_and_negations_are_rewritten():
    assert filter_negation_imply(["globally", "negation", "enter(room5)"]) == ["globally", "not_enter(room5)"]
    assert filter_negation_imply(["imply", "enter(room1)", "enter(room2)"]) == ["or", "not_enter(room1)", "enter(room2)"]
    assert filter_negation_imply(["negation", "and", "enter(room1)", "enter(room2)"]) == ["or", "not_enter(room1)", "not_enter(room2)"]


def test_the_translation_reply_is_parsed_as_released():
    sentence, props = parse_string(translation("finally prop_1 and globally prop_2", {"prop_1": ["enter(room3)"], "prop_2": ["and", "not_enter(room5)", "not_enter(room6)"]}))
    assert sentence == "finally prop_1 and globally prop_2."
    assert props == {"prop_1": ["enter(room3)"], "prop_2": ["and", "not_enter(room5)", "not_enter(room6)"]}
    assert extract_list_for_semantic_check_LLM_response("thinking...\nSTL: ['finally', 'enter(room1)']") == ["finally", "enter(room1)"]
    assert extract_list_for_semantic_check_LLM_response("no list here") == []


# -- the STL planner (Sun et al. 2022, as shipped with AutoTAMP) -----------------------


def test_the_milp_layer_solves_a_small_problem():
    model = Model()
    x, z = model.add_var(lb=-INF), model.add_var(binary=True)
    model.add_ge(x - 3 + 10 * z)  # x >= 3 unless z
    model.add_ge(x + 5 * z - 1)
    solution = model.minimize(x + 10 * z)
    assert solution.status == "optimal" and abs(solution.value(x) - 3.0) < 1e-6 and round(solution.value(z)) == 0
    infeasible = Model()
    y = infeasible.add_var()
    infeasible.add_ge(-1 - y)
    assert not infeasible.minimize(y).found


def test_the_planner_constants_are_the_released_ones():
    assert (M, EPS, T_MIN_SEP) == (1e3, 1e-2, 1e-2)


def test_a_plan_reaches_a_room_within_the_speed_limit_and_ends_at_the_goal():
    stats = PlanStats()
    out = plan_stl("HouseWorld", ["enter(room3)"], ENV.prop2block, START, END, stats=stats)
    trajectory = [[p[0][0], p[0][1], p[1]] for p in out[0]]
    assert stats.status == "optimal" and stats.num_segs == 20 and len(trajectory) == 21
    assert trajectory[0][:2] == pytest.approx(START) and trajectory[-1][:2] == pytest.approx(END, abs=1e-6)
    assert "MasterBedroom" in {v.room for v in visits(trajectory)}
    for (xa, ya, ta), (xb, yb, tb) in zip(trajectory, trajectory[1:]):
        assert tb - ta >= T_MIN_SEP - 1e-9
        assert abs(xb - xa) + abs(yb - ya) <= 5.0 * (tb - ta) + 1e-6, "the L1 velocity limit, vmax = 5"


def test_always_not_enter_keeps_the_whole_trajectory_out():
    free = trajectory_of(["and", "enter(room3)", "enter(room7)"])
    fenced = trajectory_of(["and", "and", "enter(room3)", "enter(room7)", "globally", "not_enter(room5)"])
    assert "LivingRoom" not in {v.room for v in visits(fenced)}
    assert {"MasterBedroom", "RestRoom"} <= {v.room for v in visits(fenced)}
    assert fenced[-1][2] >= free[-1][2] - 1e-6


def test_an_impossible_formula_returns_no_plan_as_released():
    assert plan_stl("HouseWorld", ["globally", "enter(room1)"], ENV.prop2block, START, END, time_limit=20.0, enter_semantics="strict") == [None]


def test_the_released_parser_plans_every_enter_as_eventually_enter():
    """The release's test for a preceding temporal operator can never be true, so an order is not enforced."""
    spec = build_spec(["until", "not_enter(room2)", "enter(room1)"], ENV.prop2block)
    assert (spec.op, spec.deps[0].op, spec.deps[1].op, spec.deps[1].deps[0].op) == ("U", "negmu", "F", "mu")
    strict = build_spec(["until", "not_enter(room2)", "enter(room1)"], ENV.prop2block, enter_semantics="strict")
    assert (strict.deps[0].op, strict.deps[1].op) == ("negmu", "mu")
    stay = build_spec(["finally", "globally [0,5]", "enter(room4)"], ENV.prop2block)
    assert [stay.op, stay.deps[0].op, stay.deps[0].deps[0].op] == ["F", "A", "F"], "a stay becomes 'always eventually'"
    assert stay.deps[0].info["int"] == [0, 5]

    # "Do not enter the red restroom until the kitchen": from this start the short way to the kitchen crosses it.
    ordered = ["and", "and", "enter(room1)", "enter(room2)", "until", "not_enter(room2)", "enter(room1)"]
    released = [v.room for v in visits(trajectory_of(ordered))]
    strict_rooms = [v.room for v in visits(trajectory_of(ordered, "strict", limit=60.0))]
    assert released.index("RestRoom2") < released.index("Kitchen"), "as released, the order is not enforced"
    assert strict_rooms.index("Kitchen") < strict_rooms.index("RestRoom2")


def test_a_stay_is_enforced_only_under_the_strict_reading():
    stay = ["finally", "globally [0,5]", "enter(room4)"]
    released = trajectory_of(stay)
    assert max(v.stay for v in visits(released) if v.room == "Bedroom") < 2.0, "as released it waits elsewhere, not in the room"
    strict = trajectory_of(stay, "strict", limit=30.0)
    assert max(v.stay for v in visits(strict) if v.room == "Bedroom") >= 5.0 - 1e-6


# -- the pipeline: translate, check, plan, check (paper Fig. 1 and Fig. 3) -------------


def test_no_corrections_is_two_llm_calls_and_one_plan():
    llm = ScriptedChat([translation("finally prop_1", {"prop_1": ["enter(room3)"]}), "['finally', 'prop_1']"])
    outcome = nl_to_trajectory(llm, "Go to the green bedroom.", start_position=START, end_position=END,
                               syntactic_correct_loop=False, semantic_correct_loop=False)
    assert outcome.error is None and outcome.llm_calls == 2 == llm.calls
    assert (outcome.stl, outcome.lifted_stl, outcome.mark_syntactic) == (["finally", "enter(room3)"], ["finally", "prop_1"], 1)
    assert len(outcome.plans) == 1 and outcome.trajectory is not None
    first = llm.requests[0]["messages"]
    assert first[0] == {"role": "system", "content": "You are a helpful assistant."}
    assert first[1]["content"] == prompts.TRANSLATE + "Input:\n" + ENV.action_list + "\nsentence: Go to the green bedroom.\nOutput: "
    assert llm.requests[1]["messages"][1]["content"] == prompts.NL2TL_PREORDER + "finally prop_1.\nSTL: "


def test_the_syntax_loop_reprompts_with_the_checkers_message():
    broken = translation("finally prop_1", {"prop_1": ["enter(room3)", "enter(room7)"]})
    fixed = translation("finally prop_1", {"prop_1": ["and", "enter(room3)", "enter(room7)"]})
    llm = ScriptedChat([broken, "['finally', 'prop_1']", fixed, "['finally', 'prop_1']"])
    outcome = nl_to_trajectory(llm, "Visit the green bedroom and the yellow restroom.", start_position=START, end_position=END,
                               syntactic_correct_loop=True, semantic_correct_loop=False)
    assert outcome.llm_calls == 4 and outcome.syntactic_iterations == 2
    correction = llm.requests[2]["messages"]
    assert [m["role"] for m in correction] == ["system", "user", "assistant", "user"]
    assert correction[2]["content"] == broken
    assert correction[3]["content"].startswith("Your predicted prop_1 lacks 1 operators with two leaves")
    assert outcome.stl == ["finally", "and", "enter(room3)", "enter(room7)"] and outcome.trajectory is not None
    # Without the loop the same translation is syntactically wrong and nothing is planned.
    unchecked = nl_to_trajectory(ScriptedChat([broken, "['finally', 'prop_1']"]), "x", start_position=START, end_position=END,
                                 syntactic_correct_loop=False, semantic_correct_loop=False)
    assert unchecked.mark_syntactic == 0 and unchecked.trajectory is None and not unchecked.plans


def test_the_semantic_check_shows_the_llm_the_planned_state_sequence_and_replans():
    wrong = translation("finally prop_1", {"prop_1": ["enter(room2)"]})  # the red restroom, not the yellow one
    llm = ScriptedChat([
        wrong, "['finally', 'prop_1']", "['finally', 'prop_1']",
        "The instruction asks for the yellow restroom, room7.\nSTL: ['finally', 'enter(room7)']",
        "The trajectory now enters the yellow restroom.\nSTL: ['finally', 'enter(room7)']",
    ])
    outcome = nl_to_trajectory(llm, "Go to the yellow restroom.", start_position=START, end_position=END)
    assert outcome.semantic_iterations == 2 and outcome.llm_calls == 5 and len(outcome.plans) == 2
    assert outcome.stl == ["finally", "enter(room7)"]
    assert "RestRoom" in {v.room for v in visits(outcome.trajectory)}
    check = llm.requests[3]["messages"]
    assert check[2]["content"] == wrong
    assert check[3]["content"].startswith("Based on your predicted STL ['finally', 'enter(room2)'] , the state sequence")
    assert "in red RestRoom2 at time" in check[3]["content"]
    assert "Based on your predicted STL ['finally', 'enter(room7)']" in llm.requests[4]["messages"][3]["content"]


def test_the_semantic_check_stops_after_three_iterations():
    replies = [translation("finally prop_1", {"prop_1": ["enter(room1)"]}), "['finally', 'prop_1']", "['finally', 'prop_1']"]
    replies += [f"STL: ['finally', 'enter(room{n})']" for n in (2, 3, 4, 6)]
    outcome = nl_to_trajectory(ScriptedChat(replies), "Go somewhere.", start_position=START, end_position=END)
    assert outcome.semantic_iterations == 3 and outcome.stl == ["finally", "enter(room4)"] and len(outcome.plans) == 4


def test_a_reply_that_cannot_be_parsed_fails_the_case_without_raising():
    outcome = nl_to_trajectory(ScriptedChat(["I cannot help with that"]), "Go.", start_position=START, end_position=END)
    assert outcome.error and outcome.trajectory is None and outcome.mark_syntactic == 0


def test_a_reply_that_says_more_than_the_answer_is_read_leniently_unless_told_not_to():
    chatty = [
        "I will assume the green room.\n\nOutput:\n" + translation("finally prop_1", {"prop_1": ["enter(room3)"]}),
        "STL: ['finally', 'prop_1']\n\nExplanation: prop_1 must eventually hold.",
    ]
    lenient = nl_to_trajectory(ScriptedChat(list(chatty)), "Go to the green bedroom.", start_position=START, end_position=END,
                               syntactic_correct_loop=False, semantic_correct_loop=False)
    assert lenient.error is None and lenient.stl == ["finally", "enter(room3)"] and lenient.trajectory is not None
    as_released = nl_to_trajectory(ScriptedChat(list(chatty)), "Go to the green bedroom.", start_position=START, end_position=END,
                                   syntactic_correct_loop=False, semantic_correct_loop=False, lenient_replies=False)
    assert as_released.error is not None and as_released.trajectory is None


def test_the_state_sequence_reads_as_released():
    text = state_sequence_text(ENV.tag_and_shape, [[1.3, -1.3, 0.0], [0.5, -1.0, 0.2], [0.5, -1.1, 0.3], [0.0, 1.3, 0.9]])
    assert text == "[in white road at time 0.0, in red RestRoom2 at time 0.2, in red RestRoom2 at time 0.3, in white road at time 0.9]."


# -- this rebuild's success checkers and time limits -----------------------------------


def seen(*rooms) -> list:
    """A sequence of room visits, 0.2 long unless a (room, stay) pair says otherwise."""
    out, t = [], 0.0
    for room in rooms:
        room, stay = room if isinstance(room, tuple) else (room, 0.2)
        out.append(Visit(room, t, t + stay))
        t += stay + 0.1
    return out


def test_checkers_read_each_instruction_family():
    assert check_visits(1, seen("MasterBedroom", "RestRoom")) and not check_visits(1, seen("RestRoom", "MasterBedroom"))
    assert check_visits(2, seen("Kitchen", "RestRoom", "DiningRoom", "Bedroom", "RestRoom2"))
    assert not check_visits(2, seen("RestRoom2", "Kitchen", "RestRoom", "DiningRoom", "Bedroom")), "the red restroom comes last"
    assert check_visits(3, seen("MasterBedroom", "RestRoom")) and not check_visits(3, seen("MasterBedroom", "LivingRoom", "RestRoom"))
    assert check_visits(4, seen("Kitchen", ("Bedroom", 5.0), "RestRoom"))
    assert not check_visits(4, seen("Kitchen", ("Bedroom", 2.0), "RestRoom")), "the five second stay is required"
    assert not check_visits(4, seen("Kitchen", ("Bedroom", 5.0), "ExerciseRoom", "RestRoom"))
    assert check_visits(5, seen("Bedroom", "DiningRoom")) and not check_visits(5, seen("Bedroom"))
    assert check_visits(6, seen("RestRoom2", "MasterBedroom", "LivingRoom", "ExerciseRoom", "RestRoom", "DiningRoom"))
    assert check_visits(7, seen("Kitchen", "RestRoom2", "RestRoom")) and not check_visits(7, seen("RestRoom2", "Kitchen", "RestRoom"))
    assert check_visits(8, seen("Kitchen", ("Bedroom", 10.0), "LivingRoom", "Kitchen"))
    assert not check_visits(8, seen("Kitchen", "LivingRoom", ("Bedroom", 10.0), "LivingRoom", "Kitchen")), "blue before the bedroom"
    every = ["LivingRoom", "Kitchen", "RestRoom2", "MasterBedroom", "Bedroom", "RestRoom", "DiningRoom", "ExerciseRoom"]
    assert check_visits(9, seen(*every)) and check_visits(10, seen(*every))
    assert not check_visits(9, seen("Bedroom", *every)), "the pink room came before the living room"
    assert not check_visits(10, seen(*every, "Kitchen")), "it must end in the smaller blue room"
    assert not check_houseworld(1, None) and not check_houseworld(1, [])
    walked = visits([[1.3, -1.3, 0.0], [0.45, -1.05, 1.0], [-1.0, -1.05, 2.0]])
    assert [v.room for v in walked] == ["RestRoom2", "DiningRoom", "Kitchen"]


def test_reference_trajectories_pass_their_own_checkers_and_set_the_time_limits():
    limits = time_limits()
    assert limits["1a"] == pytest.approx(0.907 / 0.8, abs=0.02)
    assert len(limits) >= 15
    from compare.autotamp.reference import VARIANTS, reference_trajectory

    for family in (1, 3, 5):
        start, end = VARIANTS["a"]
        trajectory = reference_trajectory(family, start, end, time_limit=30.0)
        assert check_houseworld(family, trajectory), family
        assert within_time(trajectory, limits[f"{family}a"])
        assert not within_time(trajectory, trajectory[-1][2] * 0.5)
    assert reference_spec(8).op == "and"


def test_the_experiment_scores_both_houseworlds_and_resumes(tmp_path):
    def brain(messages):
        prompt = messages[-1]["content"]
        if prompt.endswith("\nSTL: "):
            return "['and', 'prop_1', 'prop_2']"
        if "green" in prompt.rsplit("sentence: ", 1)[-1]:
            return translation("prop_1 and prop_2", {"prop_1": ["enter(room3)"], "prop_2": ["enter(room7)"]})
        return "no translation"

    two = cases(families=[1], paraphrases=2, variants=("a",))
    summary = run(two, ["no-corrections", "syntax"], tmp_path, llm=ScriptedChat(brain), planner_time_limit=20.0, progress=lambda _: None)
    assert set(summary["methods"]) == {"no-corrections", "syntax"}
    entry = summary["methods"]["no-corrections"]
    assert entry["cases"] == 2 and entry["planned"] >= 1
    assert entry["houseworld2"]["success"] <= entry["houseworld1"]["success"]
    rows = [json.loads(line) for line in (tmp_path / "cases.jsonl").read_text().splitlines()]
    assert len(rows) == 4 and {"transcript", "trajectory", "time_limit_hw2", "success_hw1", "success_hw2"} <= set(rows[0])
    again = run(two, ["no-corrections", "syntax"], tmp_path, llm=ScriptedChat([]), planner_time_limit=20.0, progress=lambda _: None)
    assert again["methods"]["syntax"]["cases"] == 2, "finished cases are not run again"
    assert summarise(rows)["methods"]["syntax"]["mean_llm_calls"] >= 2
    assert PAPER["GPT-4"]["autotamp"] == (82.5, 82.0) and list(METHODS) == ["no-corrections", "syntax", "autotamp"]
