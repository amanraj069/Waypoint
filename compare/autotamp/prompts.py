"""AutoTAMP's prompts, as released.

`data/prompts.json` holds the three long prompt texts, extracted from the
release's source without retyping:

    translate        the few-shot translation prompt (`part1` in NL2action):
                     natural language plus the scene to a lifted sentence and
                     its atomic propositions in pre-order STL
    nl2tl_preorder   the lifted sentence to a pre-order STL list
                     (`GPT_NL2TL_preorder`, the paper's NL2TL examples)
    end_to_end       the LLM End-to-end Motion Planning baseline's prompt

The corrective re-prompts are short and built in `syntax.py` and `pipeline.py`
with the release's wording.
"""

from __future__ import annotations

import json
from pathlib import Path

_PROMPTS = json.loads((Path(__file__).parent / "data" / "prompts.json").read_text(encoding="utf-8"))

TRANSLATE = _PROMPTS["translate"]
NL2TL_PREORDER = _PROMPTS["nl2tl_preorder"]
END_TO_END = _PROMPTS["end_to_end"]
#: The system message of every chat call in the release.
SYSTEM = "You are a helpful assistant."


def translation_prompt(action_list: str, sentence: str) -> str:
    return TRANSLATE + "Input:\n" + action_list + "\nsentence: " + sentence + "\nOutput: "


def semantic_check_prompt(predicted_stl: list, state_sequence: str) -> str:
    """The autoregressive semantic check (paper Fig. 3): the planner's state sequence back to the LLM."""
    return (
        "Based on your predicted STL " + str(predicted_stl)
        + " , the state sequence [[location, time]] of the generated trajectory is: " + state_sequence
        + "\n \nPlease print the initial instruction again and check whether this state sequence follows the instruction. "
        "Let us do it step by step, first specifically explain the semantic meanings of the instruction, and then list all the available rooms in the given environment, "
        "then determine the rooms planned to visit or avoid and whether the trajectory is consistent. "
        "Next modify or keep the final STL based on above analysis. First output your thinking steps and in the last line output the full final STL beginning with STL: . "
        "\nOutput:"
    )
