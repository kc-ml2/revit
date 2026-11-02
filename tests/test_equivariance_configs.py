"""Positive equivariance contracts and explicit geometry/stochastic diagnostics."""

import pytest

from scripts.check_equivariance import cases, run_case, validate_result


@pytest.mark.parametrize("case", cases(), ids=lambda case: case.id)
def test_equivariance_contract_or_documented_limitation(case):
    validate_result(run_case(case))
