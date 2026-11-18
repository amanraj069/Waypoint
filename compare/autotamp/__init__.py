"""AutoTAMP: Autoregressive Task and Motion Planning with LLMs as Translators and Checkers, rebuilt.

Chen, Arkin, Dawson, Zhang, Roy, Fan (MIT), ICRA 2024, arXiv:2306.06531.
Rebuilt from the paper and the authors' release
(https://github.com/yongchao98/AutoTAMP, MIT, commit 71e5000): the prompts,
the syntax checker, the semantic check, the environments and the STL planner's
encoding are the released ones; Gurobi is replaced by HiGHS.
`compare/autotamp/README.md` lists every element and every difference.

    from compare.common.llm import default_client
    from compare.autotamp import nl_to_trajectory

    outcome = nl_to_trajectory(
        default_client(), "Go to the green bedroom, then the yellow restroom.",
        start_position=[1.3, -1.3], end_position=[0.0, 1.3], domain="HouseWorld",
    )
    outcome.stl, outcome.trajectory
"""

from compare.autotamp.pipeline import Outcome, build_spec, nl_to_trajectory, plan_stl

__all__ = ["Outcome", "build_spec", "nl_to_trajectory", "plan_stl"]
