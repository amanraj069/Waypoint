from orchestrator import run_mission
from orchestrator.run import main


def test_run_mission_completes(capsys):
    report = run_mission("survey the north field")
    assert report.verification.passed
    assert report.states
    assert report.completed
    assert "landed" in capsys.readouterr().out


def test_run_mission_prints_every_tick(capsys):
    report = run_mission("survey the north field")
    out = capsys.readouterr().out
    for state in report.states:
        assert f"t={state.timestamp:6.1f}s" in out


def test_quiet_mode_prints_nothing(capsys):
    run_mission("survey the north field", verbose=False)
    assert capsys.readouterr().out == ""


def test_cli_exits_zero(capsys):
    assert main(["--mission", "survey the north field"]) == 0
    assert "Mission complete" in capsys.readouterr().out


def test_cli_requires_a_mission():
    import pytest

    with pytest.raises(SystemExit):
        main([])
