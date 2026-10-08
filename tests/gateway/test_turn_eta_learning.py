"""ETA must be evidence-based, cohort-specific, and learn from every turn."""

from agent.turn_receipt import build_receipt
from gateway import turn_eta
from gateway.turn_eta import (
    cohort_chain,
    correction_factor,
    estimate_turn_for_profile,
    record_turn_outcome,
    request_features,
)


def test_check_in_question_is_its_own_cohort():
    f = request_features("이제 들려?")
    assert f["kind"] == "question" and f["size"] == "s"
    chain = cohort_chain("telegram", "이제 들려?")
    # Never falls back to a platform-wide "everything" bucket.
    assert all(c.count(".") >= 2 for c in chain)


def test_no_number_without_evidence(tmp_path):
    assert estimate_turn_for_profile(tmp_path, "telegram", "랜딩 페이지 고쳐서 배포해줘") is None
    for _ in range(turn_eta.MIN_SAMPLES - 1):
        record_turn_outcome(tmp_path, "telegram", "랜딩 페이지 고쳐서 배포해줘", 300)
    assert estimate_turn_for_profile(tmp_path, "telegram", "랜딩 페이지 고쳐서 배포해줘") is None


def test_question_does_not_inherit_long_task_times(tmp_path):
    for _ in range(8):
        record_turn_outcome(tmp_path, "telegram", "랜딩 페이지 레이아웃 전부 다시 만들고 배포까지 해줘", 660)
        record_turn_outcome(tmp_path, "telegram", "이제 들려?", 6)
    est = estimate_turn_for_profile(tmp_path, "telegram", "지금 들려?")
    assert est is not None and est["p80_ms"] < 60_000
    assert est["cohort"].endswith(".question.s")


def test_calibration_learns_from_misses(tmp_path):
    msg = "리서치 정리해서 보고서 써줘"
    for _ in range(6):
        record_turn_outcome(tmp_path, "telegram", msg, 120)
    first = estimate_turn_for_profile(tmp_path, "telegram", msg)
    assert first and first["correction"] == 1.0
    # Reality keeps taking 2x the quoted upper bound -> next quote widens.
    for _ in range(turn_eta.MIN_CALIBRATION):
        record_turn_outcome(tmp_path, "telegram", msg, first["raw_p80_ms"] * 2 / 1000, shown=first)
    cal = correction_factor(tmp_path, first["cohort"])
    assert cal["samples"] >= turn_eta.MIN_CALIBRATION and cal["factor"] > 1.0
    second = estimate_turn_for_profile(tmp_path, "telegram", msg)
    assert second["p80_ms"] > second["raw_p80_ms"]


def test_receipt_states_its_basis():
    eta = {"available": True, "sample_count": 12, "p50_ms": 120_000, "p80_ms": 300_000}
    text = build_receipt("랜딩 페이지 고쳐서 배포해줘", eta=eta)
    assert "12건 기준" in text
    assert "하던 작업" not in build_receipt("다 고쳐", eta=eta)
