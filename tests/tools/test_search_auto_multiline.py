"""Tests for search_files auto-multiline routing on \\n patterns."""

import json

import pytest

from tools.file_tools import search_tool


@pytest.fixture
def proj(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    d = tmp_path / "proj"
    d.mkdir()
    (d / "mod.py").write_text(
        "def setup():\n    init_db()\n    return True\n\n"
        "def teardown():\n    close_db()\n"
    )
    return d


class TestAutoMultiline:
    def test_newline_regex_matches_across_lines(self, proj):
        r = json.loads(search_tool(r"def setup\(\):\n    init_db\(\)", path=str(proj), task_id="t-ml"))
        assert "error" not in r
        assert r["total_count"] >= 1
        assert "multiline" in r.get("warning", "")

    def test_literal_newline_in_pattern_matches(self, proj):
        # A raw newline in the pattern (not the \n escape) also routes to
        # multiline mode. Keep the pattern free of regex metachars.
        r = json.loads(search_tool("return True\n\ndef teardown", path=str(proj), task_id="t-ml"))
        assert "error" not in r
        assert r["total_count"] >= 1

    def test_plain_pattern_unaffected(self, proj):
        r = json.loads(search_tool("init_db", path=str(proj), task_id="t-ml"))
        assert r["total_count"] == 1
        assert "multiline" not in r.get("warning", "")

    def test_escaped_backslash_n_stays_literal(self, proj):
        # \\n = literal backslash+n search, not a newline: no multiline mode.
        (proj / "strings.py").write_text('SEP = "a\\\\nb"\n')
        r = json.loads(search_tool(r"a\\nb", path=str(proj), task_id="t-ml"))
        assert "error" not in r
        assert "multiline" not in (r.get("warning") or "")

    def test_multiline_zero_match_is_clean(self, proj):
        r = json.loads(search_tool(r"def missing\(\):\n    nope\(\)", path=str(proj), task_id="t-ml"))
        assert "error" not in r
        assert r["total_count"] == 0

    @staticmethod
    def _disable_rg(monkeypatch):
        from tools import file_tools

        ops = file_tools._get_file_ops()
        real_has_command = ops._has_command
        monkeypatch.setattr(
            ops,
            "_has_command",
            lambda command: False if command == "rg" else real_has_command(command),
        )

    def test_python_fallback_searches_single_file(self, proj, monkeypatch):
        """The no-rg multiline fallback must not treat a file as a directory."""
        self._disable_rg(monkeypatch)
        target = proj / "mod.py"

        r = json.loads(
            search_tool(
                r"def setup\(\):\n    init_db\(\)",
                path=str(target),
                task_id="t-ml",
            )
        )

        assert "error" not in r
        assert r["total_count"] == 1
        assert r["matches"][0]["path"] == str(target)

    def test_python_fallback_handles_zero_width_match(self, proj, monkeypatch):
        """A valid empty match must not crash result serialization."""
        from tools import file_tools

        self._disable_rg(monkeypatch)
        result = file_tools._get_file_ops()._search_multiline_with_python(
            r"\n?", str(proj / "mod.py"), None, 3, 0, "content"
        )

        assert result.error is None
        assert result.total_count >= 1
        assert result.matches[0].content == ""
