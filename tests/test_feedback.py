from feedback import Feedback, observe
from sim import initial_state


def test_observe_returns_feedback():
    assert isinstance(observe(initial_state()), Feedback)


def test_stub_never_requests_a_replan():
    signal = observe(initial_state())
    assert signal.replan_required is False
    assert signal.reasons == []


def test_feedback_can_carry_replan_reasons():
    signal = Feedback(replan_required=True, reasons=["battery below reserve"])
    assert signal.replan_required is True
    assert signal.reasons == ["battery below reserve"]
