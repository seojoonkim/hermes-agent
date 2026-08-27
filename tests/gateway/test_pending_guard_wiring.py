from __future__ import annotations

import ast
from pathlib import Path


def test_final_response_is_guarded_after_sanitization_before_session_update():
    path = Path(__file__).parents[2] / "gateway" / "run.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    imports_guard = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "gateway.pending_request_guard"
        and any(alias.name == "guard_false_closure" for alias in node.names)
        for node in ast.walk(tree)
    )
    assert imports_guard

    sanitize_at = source.index("response = _sanitize_gateway_final_response(source.platform, response)")
    guard_at = source.index("response, _pending_closure_blocked = guard_false_closure(", sanitize_at)
    session_update_at = source.index("# If the agent's session_id changed during compression", guard_at)

    assert sanitize_at < guard_at < session_update_at
    guarded_block = source[guard_at:session_update_at]
    assert "source=source" in guarded_block
    assert "hermes_home=_hermes_home" in guarded_block
    assert "if _pending_closure_blocked:" in guarded_block
    assert "작업 상태 guard 오류" in guarded_block
