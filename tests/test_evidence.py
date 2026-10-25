from __future__ import annotations

from rundiff.evidence import make_evidence, validate_evidence, validate_findings
from rundiff.loader import load_pair
from rundiff.models import ClaimType, EvidenceKind, Finding


def test_invalid_evidence_id_and_bad_source_are_rejected(run_pair_factory):
    pair = load_pair(*run_pair_factory())
    fake = make_evidence(
        kind=EvidenceKind.CONFIG,
        run_id="not-a-run",
        file="config.json",
        json_path="model.lr",
        value=0.1,
    )
    validation = validate_evidence([fake], pair)
    assert validation["valid"] is False
    assert any("outside this task" in error for error in validation["errors"])


def test_json_path_value_mismatch_is_detected(run_pair_factory):
    pair = load_pair(*run_pair_factory())
    fake = make_evidence(
        kind=EvidenceKind.CONFIG,
        run_id=pair.run_a.run_id,
        file="config.json",
        json_path="model.lr",
        value=999,
    )
    validation = validate_evidence([fake], pair)
    assert validation["valid"] is False
    assert any("does not match" in error for error in validation["errors"])


def test_forged_stable_id_is_rejected(run_pair_factory):
    pair = load_pair(*run_pair_factory())
    actual = make_evidence(
        kind=EvidenceKind.CONFIG,
        run_id=pair.run_a.run_id,
        file="config.json",
        json_path="model.lr",
        value=0.1,
    )
    forged = actual.model_copy(update={"evidence_id": "E_CONFIG_forged"})
    validation = validate_evidence([forged], pair)
    assert validation["valid"] is False
    assert any("stable source ID" in error for error in validation["errors"])


def test_wrong_log_excerpt_is_detected(run_pair_factory):
    pair = load_pair(*run_pair_factory(log_a="one\nactual line\n"))
    fake = make_evidence(
        kind=EvidenceKind.LOG,
        run_id=pair.run_a.run_id,
        file="train.log",
        line_number=2,
        excerpt="invented line",
    )
    assert validate_evidence([fake], pair)["valid"] is False


def test_claim_with_fictitious_evidence_becomes_unverified(run_pair_factory):
    pair = load_pair(*run_pair_factory())
    claim = Finding(
        claim_id="fake",
        classification=ClaimType.SUPPORTED_HYPOTHESIS,
        text="This claim has made-up evidence.",
        evidence_ids=["E_CONFIG_forged"],
    )
    validated, warnings = validate_findings([claim], [], pair)
    assert validated[0].classification == ClaimType.UNVERIFIED
    assert validated[0].evidence_ids == []
    assert warnings


def test_claim_with_unsupported_number_becomes_unverified(run_pair_factory):
    from rundiff.tools import compare_configs

    pair = load_pair(*run_pair_factory())
    evidence = compare_configs(pair)["evidence"]
    claim = Finding(
        claim_id="numeric",
        classification=ClaimType.SUPPORTED_HYPOTHESIS,
        text="The value was increased by 100.",
        evidence_ids=[item.evidence_id for item in evidence],
    )
    validated, warnings = validate_findings([claim], evidence, pair)
    assert validated[0].classification == ClaimType.UNVERIFIED
    assert any("unsupported numeric" in warning for warning in warnings)
