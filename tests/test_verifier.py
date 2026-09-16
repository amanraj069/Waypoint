from verifier import VerificationResult, verify


def test_verify_passes_for_a_valid_plan(sample_plan):
    result = verify(sample_plan)
    assert isinstance(result, VerificationResult)
    assert result.passed is True
    assert result.violations == []


def test_verification_result_defaults_to_no_violations():
    assert VerificationResult(passed=True).violations == []


def test_verification_result_carries_violations():
    result = VerificationResult(passed=False, violations=["altitude ceiling exceeded"])
    assert result.passed is False
    assert result.violations == ["altitude ceiling exceeded"]
