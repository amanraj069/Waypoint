"""Fly one query, or a prompt at a time, through the rebuilt PEACE agent.

    python -m compare.peace "Take off to 5 meters then move forward 3 meters"
    python -m compare.peace                      # a prompt, like peace-cli
    python -m compare.peace --model ollama:qwen2.5:14b "find the person"

The aircraft stands in the evaluation scene (five objects on open ground) and
keeps its state between queries, as the released CLI's does.
"""

from __future__ import annotations

import argparse

from compare.common.llm import default_client
from compare.peace.agent import PeaceAgent
from compare.peace.evaluation.run_dataset import evaluation_stack
from compare.peace.evaluation.scene import EVALUATION_SCENE


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="The rebuilt PEACE agent")
    parser.add_argument("query", nargs="*", help="a mission in plain English; none opens a prompt")
    parser.add_argument("--model", default=None)
    args = parser.parse_args(argv)

    stack = evaluation_stack()
    agent = PeaceAgent(stack, default_client(args.model), objects=EVALUATION_SCENE)

    def fly(query: str) -> None:
        result = agent.run(query)
        for plan_index, plan in enumerate(result.plans):
            print(f"\n{'plan' if plan_index == 0 else 'replan ' + str(plan_index)}: {plan.reasoning}")
            for step in plan.steps:
                arguments = ", ".join(f"{k}={v}" for k, v in step.args.items())
                print(f"    {step.tool}({arguments})")
        print()
        for step in result.steps:
            print(f"  [{step['t']:7.1f}s] {step['tool']} -> {step['result']}")
        x, y, z = stack.true_position
        print(f"\n{result.state.upper()}: {result.message}")
        print(f"  at ({x:.1f}, {y:.1f}, {z:.1f}), yaw {stack.yaw_deg % 360:.0f} deg, {stack.mode}, armed={stack.armed}; "
              f"{result.llm_calls} LLM call(s), {result.replan_count} replan(s)")

    try:
        if args.query:
            fly(" ".join(args.query))
            return
        print("PEACE (rebuilt). Type a mission; an empty line or Ctrl-D quits.")
        while True:
            try:
                query = input("> ").strip()
            except EOFError:
                break
            if not query:
                break
            fly(query)
    finally:
        agent.close()


if __name__ == "__main__":
    main()
