"""Evidence-based turn ETA backed by a MemKraft Delay Ledger (learns every turn).

Simon (2026-10-09): an ETA must never be a reflex number. It has to come from
minimal data + reasoning, arrive fast, and keep getting more accurate.

How it works
------------
1. **Features, not one global bucket.** A request is reduced (no text kept) to
   ``subject`` (development/research/…), ``kind`` (question / short order /
   request) and ``size`` (s/m/l). Cohort keys go from specific to broad::

       eta.<platform>.<subject>.<kind>.<size>
       eta.<platform>.<subject>.<kind>
       eta.<platform>.<subject>

   The most specific cohort with enough samples wins. The platform-wide
   "everything" bucket is never used (that is what produced "11분" for
   "이제 들려?").
2. **Evidence gate.** A number is shown only with >= MIN_SAMPLES completed
   turns in the chosen cohort and a bounded spread (p80/p50 <= MAX_SPREAD).
   Otherwise no number is shown at all.
3. **Learning loop.** Every finished turn is written back to the ETA ledger
   (MemKraft ``delay_run_start/finish``) under its most specific cohort, and the
   shown prediction vs. actual is appended to a calibration log. The shown
   upper bound is multiplied by a learned per-cohort correction (the p80 of
   actual/predicted over recent turns), so systematic under/over-estimates
   shrink over time.

Everything is bounded and fail-open: any error means "no ETA shown".
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MIN_SAMPLES = 5
MAX_SPREAD = 3.0          # p80/p50 above this = too unpredictable to quote
CALIBRATION_WINDOW = 30
MIN_CALIBRATION = 5
CORRECTION_BOUNDS = (0.5, 3.0)
_LEDGER_DIR = "eta-ledger"
_CALIBRATION_FILE = "calibration.jsonl"
_lock = threading.Lock()

_REPLY_WRAPPER = re.compile(r"^\s*\[Replying to:.*?\]\s*", re.S)
_QUESTION_TAIL = re.compile(r"(\?|？|까|니|나|냐|어때|맞아|됐어|돼)\s*$")


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _platform(platform: Any) -> str:
    p = str(platform or "other").strip().lower()
    return p if p in {"cli", "telegram", "discord", "slack", "matrix", "web", "desktop"} else "other"


def request_features(message_text: str) -> Dict[str, str]:
    """Closed, text-free features of a request (privacy-safe cohort input)."""
    try:
        from agent.conversation_loop import _timing_subject_for_message
        subject = _timing_subject_for_message(message_text)
    except Exception:
        subject = "general"
    body = _REPLY_WRAPPER.sub("", str(message_text or "")).strip()
    letters = re.sub(r"[\W_]+", "", body, flags=re.U)
    n = len(letters)
    if _QUESTION_TAIL.search(body) and n <= 40:
        kind = "question"
    elif n <= 8:
        kind = "short"
    else:
        kind = "request"
    size = "s" if n <= 30 else ("m" if n <= 120 else "l")
    return {"subject": subject, "kind": kind, "size": size}


def cohort_chain(platform: Any, message_text: str) -> List[str]:
    f = request_features(message_text)
    base = f"eta.{_platform(platform)}.{f['subject']}"
    return [f"{base}.{f['kind']}.{f['size']}", f"{base}.{f['kind']}", base]


def _store(hermes_home: Path):
    from memkraft import MemKraft
    return MemKraft(base_dir=str(Path(hermes_home) / "memkraft" / _LEDGER_DIR))


def _calibration_path(hermes_home: Path) -> Path:
    return Path(hermes_home) / "memkraft" / _LEDGER_DIR / _CALIBRATION_FILE


def _nearest(values: List[float], q: float) -> float:
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round(q * len(ordered) + 0.5)) - 1))
    return ordered[idx]


def correction_factor(hermes_home: Path, cohort: str) -> Dict[str, Any]:
    """Learned multiplier for the shown upper bound of one cohort."""
    path = _calibration_path(hermes_home)
    ratios: List[float] = []
    hits = 0
    if path.exists():
        try:
            lines = path.read_text(encoding="utf-8").splitlines()[-2000:]
        except OSError:
            lines = []
        for line in reversed(lines):
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("cohort") != cohort:
                continue
            pred, actual = float(rec.get("raw_p80_ms") or 0), float(rec.get("actual_ms") or 0)
            if pred > 0 and actual > 0:
                ratios.append(actual / pred)
                hits += 1 if actual <= float(rec.get("shown_hi_ms") or pred) else 0
            if len(ratios) >= CALIBRATION_WINDOW:
                break
    if len(ratios) < MIN_CALIBRATION:
        return {"factor": 1.0, "samples": len(ratios), "hit_rate": None}
    lo, hi = CORRECTION_BOUNDS
    return {
        "factor": min(hi, max(lo, _nearest(ratios, 0.8))),
        "samples": len(ratios),
        "hit_rate": hits / len(ratios),
    }


def estimate_turn_for_profile(hermes_home: Path, platform: str, message_text: str) -> Optional[Dict[str, Any]]:
    """Return a trustworthy estimate dict or None (=> show no number)."""
    try:
        store = _store(hermes_home)
    except Exception:
        return None
    if not (Path(hermes_home) / "memkraft" / _LEDGER_DIR / ".memkraft" / "delay" / "events.jsonl").exists():
        return None
    now = _now()
    for level, cohort in enumerate(cohort_chain(platform, message_text)):
        try:
            est = store.delay_estimate("task", cohort, now=now, window=50)
        except Exception as exc:  # noqa: BLE001
            logger.debug("turn ETA lookup failed (%s)", type(exc).__name__)
            return None
        if not (isinstance(est, dict) and est.get("available")):
            continue
        p50, p80 = int(est["p50_ms"]), int(est["p80_ms"])
        if int(est.get("sample_count") or 0) < MIN_SAMPLES or p50 <= 0:
            continue
        if p80 / p50 > MAX_SPREAD:
            # Too unpredictable here; a broader cohort will not be better.
            return None
        cal = correction_factor(hermes_home, cohort)
        out = dict(est)
        out.update(
            cohort=cohort,
            level=level,
            raw_p50_ms=p50,
            raw_p80_ms=p80,
            p50_ms=int(p50 * min(1.0, cal["factor"]) if cal["factor"] < 1 else p50),
            p80_ms=int(p80 * cal["factor"]),
            correction=cal["factor"],
            calibration_samples=cal["samples"],
            hit_rate=cal["hit_rate"],
        )
        out["p50_ms"] = min(out["p50_ms"], out["p80_ms"])
        return out
    return None


def record_turn_outcome(
    hermes_home: Path,
    platform: str,
    message_text: str,
    elapsed_s: float,
    *,
    shown: Optional[Dict[str, Any]] = None,
    outcome: str = "completed",
) -> bool:
    """Feed one finished turn back into the ledger + calibration log."""
    try:
        elapsed_ms = max(0, int(float(elapsed_s) * 1000))
        chain = cohort_chain(platform, message_text)
        store = _store(hermes_home)
        now = _now()
        run_base = f"t{time.time_ns()}"
        with _lock:
            # Record at every level so broader cohorts fill up as fallbacks.
            for i, cohort in enumerate(chain):
                rid = f"{run_base}-{i}"
                store.delay_run_start(rid, "task", cohort, now=now)
                store.delay_run_finish(rid, elapsed_ms, now=now, outcome=outcome)
            if shown and outcome == "completed" and shown.get("raw_p80_ms"):
                path = _calibration_path(hermes_home)
                path.parent.mkdir(parents=True, exist_ok=True)
                rec = {
                    "ts": now,
                    "cohort": shown.get("cohort"),
                    "raw_p80_ms": int(shown["raw_p80_ms"]),
                    "shown_hi_ms": int(shown.get("p80_ms") or shown["raw_p80_ms"]),
                    "actual_ms": elapsed_ms,
                }
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("turn ETA record failed (%s)", type(exc).__name__)
        return False
