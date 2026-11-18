"""AutoTAMP's STL list handling and its rules-based syntax checker (openai_func.py).

An STL formula is a pre-order (prefix) token list, e.g.

    ['and', 'finally', 'enter(room1)', 'globally', 'not_enter(room5)']

with unary operators (negation, globally, finally, optionally with a time
window: 'globally [0,5]') and binary ones (and, or, imply, equal, until).
"The action primitives in this work are 'enter(room_name)' and
'not_enter(room_name)'" (paper, section III-C).

The checker is the release's counting rule: scanning the list, a proposition
adds one and a binary operator takes one away; a well-formed formula ends at
exactly one. When it does not, the correction the release sends back says how
many binary operators are missing or surplus.
"""

from __future__ import annotations

import copy
import math
import re

import numpy as np


def parse_string(input_string: str) -> tuple[str, dict[str, list[str]]]:
    """The translation reply as (lifted sentence, {prop_i: pre-order STL list})."""
    # Split the string into sentence and properties part
    parts = re.split(r"\s*:\s*", input_string, maxsplit=1)

    try:
        sentence = parts[1].split(".", 1)[0].strip() + "."
        properties = parts[1].split(".", 1)[1].strip()
    except Exception:  # noqa: BLE001 -- the release's fallback for a reply without a full stop
        sentence = parts[1].split("\n")[0].strip()
        properties = parts[1].split("\n")[1].strip()

    # Remove white spaces and split properties by ';'
    property_list = re.split(r"\s*;\s*", properties)
    if property_list[-1] == "":
        property_list = property_list[:-1]

    prop_dict = {}
    for prop in property_list:
        key_value = re.split(r"\s*:\s*", prop)
        # Remove the first and the last characters (which are quotes) from the key
        key = key_value[0][1:-1]
        values = re.findall(r"'[^']*'|\"[^\"]*\"", key_value[1])
        values = [v[1:-1] for v in values]
        prop_dict[key] = values

    return sentence, prop_dict


def is_unary_operator(token: str) -> bool:
    return token in ("negation", "globally", "finally") or token.split(" ")[0] in ("globally", "finally", "negation")


def is_operator(token: str) -> bool:
    return token in ("and", "imply", "equal", "or", "until", "negation", "globally", "finally") or token.split(" ")[
        0
    ] in ("globally", "finally", "until")


def check_syntactic_correct(converted_list):
    """'correct', or the proposition count the list ends at (1 is correct), or the token that is not STL."""
    count = 0
    for item in converted_list:
        if is_operator(item):
            if is_unary_operator(item):
                pass
            else:
                count -= 1
        elif item[:4] == "prop" or item[:5] == "enter" or item[:4] == "not_":
            count += 1
        else:
            return item
    if count == 1:
        return "correct"
    return count


def check_syntactic_correct_inverse_order(converted_list_input) -> str:
    converted_list = copy.deepcopy(converted_list_input)
    converted_list.reverse()
    count = 0
    for item in converted_list:
        if is_operator(item):
            if is_unary_operator(item):
                pass
            else:
                count -= 1
                if count == 0:
                    return "error"
        elif item[:4] == "prop" or item[:5] == "enter" or item[:4] == "not_":
            count += 1
            if count == 0:
                return "error"
        else:
            return "error"
    if count == 1:
        return "correct"
    return "error"


def AP_dict_syntactic_correct_check(AP_dict: dict) -> bool:
    return all(check_syntactic_correct(prop_list) == "correct" for prop_list in AP_dict.values())


def syntax_feedback(AP_dict: dict) -> str:
    """The release's syntax correction message: what each malformed proposition lacks or has too much of."""
    message = ""
    for prop_tag, prop_list in AP_dict.items():
        verdict = check_syntactic_correct(prop_list)
        if verdict != "correct" and type(verdict) == int:  # noqa: E721
            if verdict > 1:
                message += (
                    "Your predicted " + prop_tag + " lacks " + str(verdict - 1)
                    + " operators with two leaves, the correct syntactic format should be like "
                    + str(["and"] * (verdict - 1) + prop_list) + ". "
                )
            elif verdict < 1:
                message += (
                    "Your predicted " + prop_tag + " has " + str(1 - verdict)
                    + " more operators with two leaves, the correct syntactic format should be like "
                    + str(prop_list[abs(verdict - 1):]) + ". "
                )
    return message + "Please correct your answer.\nOutput: "


def force_syntax(stl_list: list) -> list:
    """The release's last resort for a list that still miscounts: prepend 'and's, or drop leading tokens."""
    verdict = check_syntactic_correct(stl_list)
    if verdict != "correct" and type(verdict) == int:  # noqa: E721
        if verdict > 1:
            return ["and"] * (verdict - 1) + stl_list
        if verdict < 1:
            return stl_list[abs(verdict - 1):]
    return stl_list


def filter_negation_imply(input_stl_original: list) -> list:
    """Rewrite imply as (or, negation), push negations onto enter / not_enter, De Morgan over and / or."""
    input_stl = copy.deepcopy(input_stl_original)
    # filter imply operation
    for index_negation in range(len(input_stl) - 1, -1, -1):
        if input_stl[index_negation] == "imply":
            input_stl.pop(index_negation)
            input_stl.insert(index_negation, "or")
            input_stl.insert(index_negation + 1, "negation")
    # filter negation operation
    for index_negation in range(len(input_stl) - 1, -1, -1):
        if input_stl[index_negation] == "negation" and input_stl[index_negation + 1][:5] == "enter":
            phi = "not_" + input_stl[index_negation + 1]
            input_stl.pop(index_negation + 1)
            input_stl.pop(index_negation)
            input_stl.insert(index_negation, phi)
        elif input_stl[index_negation] == "negation" and input_stl[index_negation + 1][:9] == "not_enter":
            phi = input_stl[index_negation + 1][4:]
            input_stl.pop(index_negation + 1)
            input_stl.pop(index_negation)
            input_stl.insert(index_negation, phi)

    if "negation" in input_stl:
        for index_negation in range(len(input_stl) - 2):
            if input_stl[index_negation] == "negation" and (
                input_stl[index_negation + 1] == "and" or input_stl[index_negation + 1] == "or"
            ):
                count_operator = 0
                for operator_index in range(index_negation, len(input_stl)):
                    if input_stl[operator_index] == "and":
                        input_stl[operator_index] = "or"
                        count_operator += 1
                    elif input_stl[operator_index] == "or":
                        input_stl[operator_index] = "and"
                        count_operator += 1
                    elif input_stl[operator_index][:5] == "enter":
                        input_stl[operator_index] = "not_" + input_stl[operator_index]
                        count_operator -= 1
                        if count_operator == -1:
                            break
                    elif input_stl[operator_index][:9] == "not_enter":
                        input_stl[operator_index] = input_stl[operator_index][4:]
                        count_operator -= 1
                        if count_operator == -1:
                            break
                input_stl.pop(index_negation)
    return input_stl


def extract_list_for_semantic_check_LLM_response(s: str) -> list[str]:
    """The STL list on the reply's 'STL: [...]' line."""
    match = re.search(r"STL:\s*\[(.*?)\]", s)
    if match:
        return [item.strip().strip("'") for item in match.group(1).split(",")]
    return []


# -- the planner's trajectory as a state sequence, for the semantic check -------------


def trajectory_len_cal(PWLs) -> float:
    trajectory_len = 0.0
    for PWL in PWLs:
        positions = [[P[0][0], P[0][1]] for P in PWL]
        for i in range(len(positions) - 1):
            trajectory_len += float(
                np.sqrt((positions[i][0] - positions[i + 1][0]) ** 2 + (positions[i][1] - positions[i + 1][1]) ** 2)
            )
    return trajectory_len


def state_check(PWL_x_y_t, tag_and_shape):
    for color, block_array in tag_and_shape.items():
        for block_item_list in block_array:
            block_item = block_item_list[:-1]
            if (
                PWL_x_y_t[0] > -block_item[0]
                and PWL_x_y_t[0] < block_item[1]
                and PWL_x_y_t[1] > -block_item[2]
                and PWL_x_y_t[1] < block_item[3]
            ):
                return color, block_item_list[-1], block_item
    return "white", "road", None


def interpolate_list(data, divide_path_time_ratio):
    if (divide_path_time_ratio & (divide_path_time_ratio - 1)) != 0 or divide_path_time_ratio == 0:
        raise ValueError("divide_path_time_ratio must be a power of 2.")
    new_data = list(data)
    for _ in range(int(math.log2(divide_path_time_ratio))):
        temp_data = []
        for i in range(len(new_data) - 1):
            mid_point = [(new_data[i][j] + new_data[i + 1][j]) / 2 for j in range(len(new_data[i]))]
            temp_data.append(new_data[i])
            temp_data.append(mid_point)
        temp_data.append(new_data[-1])
        new_data = list(temp_data)
    return new_data


def judge_trajectory_output_state(tag_and_shape, PWL, divide_path_time_ratio=1):
    """[[colour, function, time], ...] for each waypoint of the trajectory."""
    state_time_list = []
    if len(PWL) > 3:
        for item in interpolate_list(PWL, divide_path_time_ratio):
            color, function, block_item = state_check(item, tag_and_shape)
            if block_item is not None:
                state_time_list.append([color, function, str(item[-1])])
            else:
                state_time_list.append(["white", "road", str(item[-1])])
    return state_time_list


def abbreviate_list_state_check(lst):
    """Collapse runs in the same place to their first and last entry."""
    result = []
    temp_group = []
    for i in range(len(lst)):
        if temp_group and (lst[i][:2] != temp_group[0][:2]):
            if temp_group[0] != temp_group[-1]:
                result.append(temp_group[0])
                result.append(temp_group[-1])
            else:
                result.append(temp_group[0])
            temp_group = []
        temp_group.append(lst[i])
    if temp_group:
        if temp_group[0] != temp_group[-1]:
            result.append(temp_group[0])
            result.append(temp_group[-1])
        else:
            result.append(temp_group[0])
    return result


def state_sequence_text(tag_and_shape, position_time_list) -> str:
    """The state sequence as the release writes it into the semantic check prompt."""
    states = abbreviate_list_state_check(judge_trajectory_output_state(tag_and_shape, position_time_list, 1))
    text = "["
    for index, item in enumerate(states):
        text += "in " + str(item[0]) + " " + str(item[1]) + " at time " + str(item[2])
        text += ", " if index != len(states) - 1 else "]."
    return text
