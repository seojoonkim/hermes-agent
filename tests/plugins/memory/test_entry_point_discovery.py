"""Security and compatibility tests for packaged memory providers."""

from types import SimpleNamespace

import pytest

from agent.memory_provider import MemoryProvider
from plugins import memory


class Provider(MemoryProvider):
    def __init__(self, name="memkraft"):
        self._name = name

    @property
    def name(self):
        return self._name

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        pass

    def get_tool_schemas(self):
        return []


class EntryPoint:
    def __init__(self, name, loaded=None, error=None, summary=""):
        self.name = name
        self._loaded = loaded
        self._error = error
        self.load_calls = 0
        self.dist = SimpleNamespace(metadata={"Summary": summary})

    def load(self):
        self.load_calls += 1
        if self._error:
            raise self._error
        return self._loaded


class ModernEntryPoints(list):
    def select(self, **criteria):
        assert criteria == {"group": "hermes_agent.memory_providers"}
        return self


def install_metadata(monkeypatch, *entry_points):
    monkeypatch.setattr(
        memory.importlib.metadata, "entry_points", lambda: ModernEntryPoints(entry_points)
    )


@pytest.fixture
def empty_dirs(monkeypatch, tmp_path):
    bundled = tmp_path / "bundled"
    user = tmp_path / "user"
    bundled.mkdir()
    user.mkdir()
    monkeypatch.setattr(memory, "_MEMORY_PLUGINS_DIR", bundled)
    monkeypatch.setattr(memory, "_get_user_plugins_dir", lambda: user)
    return bundled, user


@pytest.mark.parametrize("location", ["bundled", "user"])
def test_directory_provider_takes_precedence_without_loading_entry_point(
    monkeypatch, empty_dirs, location
):
    bundled, user = empty_dirs
    packaged = EntryPoint("memkraft", Provider("wrong"))
    install_metadata(monkeypatch, packaged)
    provider_dir = (bundled if location == "bundled" else user) / "memkraft"
    provider_dir.mkdir()
    (provider_dir / "__init__.py").write_text("# MemoryProvider marker\n")
    expected = Provider()
    monkeypatch.setattr(memory, "_load_provider_from_dir", lambda path: expected)

    assert memory.load_memory_provider("memkraft") is expected
    assert packaged.load_calls == 0


@pytest.mark.parametrize("failure", [None, RuntimeError("directory failure")])
def test_directory_load_failure_does_not_fall_through(
    monkeypatch, empty_dirs, failure
):
    bundled, _ = empty_dirs
    packaged = EntryPoint("memkraft", Provider())
    install_metadata(monkeypatch, packaged)
    provider_dir = bundled / "memkraft"
    provider_dir.mkdir()
    (provider_dir / "__init__.py").write_text("# package\n")

    def fail_directory_load(path):
        if failure:
            raise failure
        return None

    monkeypatch.setattr(memory, "_load_provider_from_dir", fail_directory_load)

    assert memory.load_memory_provider("memkraft") is None
    assert packaged.load_calls == 0


def test_user_package_rejected_by_heuristic_still_reserves_name(monkeypatch, empty_dirs):
    _, user = empty_dirs
    packaged = EntryPoint("memkraft", Provider())
    install_metadata(monkeypatch, packaged)
    package = user / "memkraft"
    package.mkdir()
    (package / "__init__.py").write_text(
        "raise AssertionError('non-memory plugin must not execute')\n"
    )

    assert memory.list_memory_provider_names() == []
    assert memory.load_memory_provider("memkraft") is None
    assert packaged.load_calls == 0


def test_listing_uses_metadata_without_importing_provider_code(monkeypatch, empty_dirs):
    packaged = EntryPoint("memkraft", error=AssertionError("must not import"))
    install_metadata(monkeypatch, packaged)

    assert memory.list_memory_provider_names() == ["memkraft"]
    assert packaged.load_calls == 0


def test_memkraft_style_register_function_is_loaded(monkeypatch, empty_dirs):
    def register(ctx):
        ctx.register_memory_provider(Provider())

    packaged = EntryPoint("memkraft", register)
    install_metadata(monkeypatch, packaged)
    assert isinstance(memory.load_memory_provider("memkraft"), Provider)
    assert packaged.load_calls == 1


def test_subclass_and_instance_entry_points_are_supported(monkeypatch, empty_dirs):
    for loaded in (Provider, Provider()):
        install_metadata(monkeypatch, EntryPoint("memkraft", loaded))
        assert isinstance(memory.load_memory_provider("memkraft"), Provider)


def test_duplicate_entry_point_name_is_excluded_and_not_loaded(monkeypatch, empty_dirs):
    first = EntryPoint("memkraft", Provider())
    second = EntryPoint("memkraft", Provider())
    install_metadata(monkeypatch, first, second)

    assert memory.list_memory_provider_names() == []
    assert memory.load_memory_provider("memkraft") is None
    assert first.load_calls == second.load_calls == 0


def test_case_variant_duplicate_entry_points_are_excluded_from_discovery(
    monkeypatch, empty_dirs
):
    first = EntryPoint("MemKraft", Provider("MemKraft"))
    second = EntryPoint("memkraft", Provider())
    install_metadata(monkeypatch, first, second)

    assert memory.list_memory_provider_names() == []
    assert memory.discover_memory_providers() == []
    assert first.load_calls == second.load_calls == 0


def test_case_variant_directory_reserves_entry_point_without_loading_either(
    monkeypatch, empty_dirs
):
    _, user = empty_dirs
    package = user / "MemKraft"
    package.mkdir()
    (package / "__init__.py").write_text(
        "raise AssertionError('non-memory plugin must not execute')\n"
    )
    packaged = EntryPoint("memkraft", Provider())
    install_metadata(monkeypatch, packaged)

    assert memory.list_memory_provider_names() == []
    assert memory.find_provider_dir("memkraft") is None
    assert memory.load_memory_provider("memkraft") is None
    assert packaged.load_calls == 0


def test_case_variant_name_cannot_load_directory_provider(monkeypatch, empty_dirs):
    _, user = empty_dirs
    package = user / "MemKraft"
    package.mkdir()
    (package / "__init__.py").write_text("# MemoryProvider marker\n")
    expected = Provider("MemKraft")
    monkeypatch.setattr(memory, "_load_provider_from_dir", lambda path: expected)

    assert memory.find_provider_dir("memkraft") is None
    assert memory.load_memory_provider("memkraft") is None
    assert memory.load_memory_provider("MemKraft") is expected


@pytest.mark.parametrize("loaded", [object(), Provider("impostor")])
def test_wrong_type_or_provider_name_is_rejected(monkeypatch, empty_dirs, loaded):
    install_metadata(monkeypatch, EntryPoint("memkraft", loaded))
    assert memory.load_memory_provider("memkraft") is None


def test_entry_point_load_exception_returns_none_with_safe_logging(
    monkeypatch, empty_dirs, caplog
):
    install_metadata(monkeypatch, EntryPoint("memkraft", error=RuntimeError("secret-token")))
    assert memory.load_memory_provider("memkraft") is None
    assert "Failed to load memory provider entry point 'memkraft'" in caplog.text
    assert "secret-token" not in caplog.text


@pytest.mark.parametrize(
    "name",
    ["../escape", "bad/name", "bad.name", "bad\nforged", "_hidden", "", "a" * 65],
)
def test_invalid_input_name_fails_closed_without_logging_name(
    monkeypatch, empty_dirs, caplog, name
):
    install_metadata(monkeypatch, EntryPoint("memkraft", Provider()))
    assert memory.find_provider_dir(name) is None
    assert memory.load_memory_provider(name) is None
    if name:
        assert name not in caplog.text


def test_invalid_entry_point_name_is_ignored_without_logging_name(
    monkeypatch, empty_dirs, caplog
):
    invalid = EntryPoint("bad\nforged", Provider("bad\nforged"))
    install_metadata(monkeypatch, invalid)
    assert memory.list_memory_provider_names() == []
    assert invalid.load_calls == 0
    assert invalid.name not in caplog.text


def test_discover_merges_entry_points_with_summary_and_availability(
    monkeypatch, empty_dirs
):
    bundled, _ = empty_dirs
    package = bundled / "bundled"
    package.mkdir()
    (package / "__init__.py").write_text("# package\n")
    packaged = EntryPoint("memkraft", Provider(), summary="Packaged memory")
    shadowed = EntryPoint("bundled", Provider("bundled"))
    install_metadata(monkeypatch, packaged, shadowed)
    monkeypatch.setattr(memory, "_load_provider_from_dir", lambda path: Provider("bundled"))

    assert memory.discover_memory_providers() == [
        ("bundled", "", True),
        ("memkraft", "Packaged memory", True),
    ]
    assert packaged.load_calls == 1
    assert shadowed.load_calls == 0


def test_discover_marks_is_available_exception_false(monkeypatch, empty_dirs):
    provider = Provider()

    def fail_availability():
        raise RuntimeError("unavailable")

    provider.is_available = fail_availability
    install_metadata(monkeypatch, EntryPoint("memkraft", provider))

    assert memory.discover_memory_providers() == [("memkraft", "", False)]


def test_metadata_exception_degrades_to_empty(monkeypatch, empty_dirs):
    def fail_metadata():
        raise RuntimeError("broken metadata")

    monkeypatch.setattr(memory.importlib.metadata, "entry_points", fail_metadata)

    assert memory._memory_provider_entry_points() == []
    assert memory.list_memory_provider_names() == []


@pytest.mark.parametrize("summary", [
    "line\u0085break",
    "line\u2028break",
    "line\u2029break",
    "abc\u009b31m forged",
])
def test_discover_rejects_nonprintable_distribution_summary(
    monkeypatch, empty_dirs, summary
):
    packaged = EntryPoint("memkraft", Provider(), summary=summary)
    install_metadata(monkeypatch, packaged)

    assert memory.discover_memory_providers() == [("memkraft", "", True)]


def test_python39_legacy_entry_points_mapping_loads(monkeypatch, empty_dirs):
    packaged = EntryPoint("memkraft", Provider())
    monkeypatch.setattr(
        memory.importlib.metadata,
        "entry_points",
        lambda: {"hermes_agent.memory_providers": [packaged], "other": []},
    )

    assert memory.list_memory_provider_names() == ["memkraft"]
    assert packaged.load_calls == 0
    assert memory.load_memory_provider("memkraft") is not None
    assert packaged.load_calls == 1


def test_dashboard_options_include_entry_point_without_importing_it(
    monkeypatch, empty_dirs
):
    from hermes_cli.web_server import _memory_provider_options

    packaged = EntryPoint("memkraft", error=AssertionError("must not import"))
    install_metadata(monkeypatch, packaged)

    assert _memory_provider_options() == ["", "memkraft"]
    assert packaged.load_calls == 0


@pytest.mark.parametrize("name", ["bad.name", "a" * 65])
def test_dashboard_and_memory_loader_share_invalid_name_grammar(name):
    from fastapi import HTTPException
    from hermes_cli.web_server import _require_valid_memory_provider_name

    assert memory._is_valid_provider_name(name) is False
    with pytest.raises(HTTPException) as exc_info:
        _require_valid_memory_provider_name(name)
    assert exc_info.value.status_code == 404


@pytest.mark.parametrize("name", ["bad.name", "a" * 65])
def test_dashboard_config_api_rejects_invalid_provider_name(name):
    import asyncio

    from fastapi import HTTPException
    from hermes_cli.web_server import ConfigUpdate, update_config

    body = ConfigUpdate(config={"memory": {"provider": name}})
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(update_config(body))
    assert exc_info.value.status_code == 404
