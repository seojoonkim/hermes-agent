"""Completion claims and classification must use the producer's profile DB."""
from collections import OrderedDict
import threading
from unittest.mock import AsyncMock, patch

import pytest

from gateway.run import GatewayRunner, _SESSION_DB_UNPINNED
from gateway.config import GatewayConfig
from hermes_constants import get_hermes_home
from hermes_state import SessionDB


@pytest.mark.asyncio
async def test_named_profile_completion_uses_own_database(tmp_path, monkeypatch):
    import hermes_state
    root = tmp_path / 'hermes'
    home = root / 'profiles' / 'sano'
    home.mkdir(parents=True)
    (home / 'config.yaml').write_text('{}\n')
    monkeypatch.setenv('HERMES_HOME', str(root))
    monkeypatch.setattr(hermes_state, 'DEFAULT_DB_PATH', hermes_state._IMPORT_DEFAULT_DB_PATH)
    db = SessionDB(db_path=home / 'state.db')
    db.create_session(session_id='parent', source='telegram')
    db.close()
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._session_db_pinned = _SESSION_DB_UNPINNED
    runner._session_db_handles = {}
    runner._session_db_handles_lock = threading.Lock()
    runner._completion_delivery_lock = threading.Lock()
    runner._completion_deliveries_inflight = set()
    runner._completion_deliveries_delivered = OrderedDict()
    runner._completion_delivery_retention = 10
    seen = []
    async def inject(synth_text, evt):
        seen.append(get_hermes_home())
        return True
    runner._inject_watch_notification = inject
    evt = {'type': 'async_delegation', 'delegation_id': 'deleg_profile', 'parent_session_id': 'parent',
           'session_key': 'agent:sano:telegram:group:-5180401933:46291309'}
    claims = []
    acks = []
    with patch('tools.async_delegation.claim_completion_delivery', side_effect=lambda *a: claims.append(get_hermes_home()) or True), patch('tools.async_delegation.complete_completion_delivery', side_effect=lambda *a: acks.append(get_hermes_home())), patch('hermes_cli.profiles.get_profile_dir', return_value=home), patch('hermes_cli.profiles.profile_exists', return_value=True):
        assert await runner._deliver_completion_notification('done', evt) is True
    assert seen == [home]
    assert claims == acks == [home]
    assert get_hermes_home() == root
    runner.close_all_session_db_handles()


@pytest.mark.asyncio
async def test_unknown_parent_retains_pending_instead_of_terminal_drop():
    runner = object.__new__(GatewayRunner)
    runner._session_db = AsyncMock()
    runner._session_db.get_session.return_value = None
    assert await runner._classify_completion_target('missing') == 'retry'


@pytest.mark.asyncio
async def test_invalid_profile_does_not_inject():
    runner = object.__new__(GatewayRunner)
    runner._inject_watch_notification = AsyncMock(return_value=True)
    evt = {'type': 'async_delegation', 'session_key': 'agent:../other:telegram:dm:1'}
    assert await runner._deliver_completion_notification('done', evt) is False
    runner._inject_watch_notification.assert_not_called()


@pytest.mark.asyncio
async def test_main_completion_clears_inherited_named_scope(tmp_path):
    from gateway.run import _profile_runtime_scope
    root = tmp_path / 'root'
    named = tmp_path / 'named'
    root.mkdir()
    named.mkdir()
    runner = object.__new__(GatewayRunner)
    seen = []
    async def deliver(text, evt):
        seen.append(get_hermes_home())
        return True
    runner._deliver_completion_notification_scoped = deliver
    with patch('hermes_cli.profiles.get_profile_dir', return_value=root):
        with _profile_runtime_scope(named):
            assert await runner._deliver_completion_notification('done', {
                'session_key': 'agent:main:telegram:dm:1'
            }) is True
            assert get_hermes_home() == named
    assert seen == [root]


@pytest.mark.asyncio
async def test_persisted_replay_routes_and_acks_named_database(tmp_path, monkeypatch):
    import json
    import queue
    import time
    from types import SimpleNamespace
    import hermes_state
    from gateway.config import Platform
    from gateway.run import _profile_runtime_scope
    from tools import async_delegation as durable

    root = tmp_path / 'root'
    home = root / 'profiles' / 'sano'
    home.mkdir(parents=True)
    monkeypatch.setenv('HERMES_HOME', str(root))
    monkeypatch.setattr(hermes_state, 'DEFAULT_DB_PATH', hermes_state._IMPORT_DEFAULT_DB_PATH)
    db = SessionDB(db_path=home / 'state.db')
    db.create_session(session_id='parent', source='telegram')
    db.close()
    evt = {'type': 'async_delegation', 'delegation_id': 'persisted',
           'parent_session_id': 'parent', 'user_id': '46291309',
           'session_key': 'agent:sano:telegram:group:-5180401933:46291309'}
    with _profile_runtime_scope(home):
        with durable._transaction() as conn:
            conn.execute('INSERT INTO async_delegations '
                         '(delegation_id, origin_session, parent_session_id, state, '
                         'dispatched_at, completed_at, updated_at, event_json) '
                         'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                         ('persisted', evt['session_key'], 'parent', 'completed',
                          time.time(), time.time(), time.time(), json.dumps(evt)))
        restored = queue.Queue()
        assert durable.restore_undelivered_completions(restored) == 1
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._session_db_pinned = _SESSION_DB_UNPINNED
    runner._session_db_handles = {}
    runner._session_db_handles_lock = threading.Lock()
    runner._completion_delivery_lock = threading.Lock()
    runner._completion_deliveries_inflight = set()
    runner._completion_deliveries_delivered = OrderedDict()
    runner._completion_delivery_retention = 10
    runner.session_store = SimpleNamespace(_ensure_loaded=lambda: None, _entries={})
    runner._get_cached_session_source = lambda key: None
    adapter = SimpleNamespace(handle_message=AsyncMock(), supports_push=True)
    root_adapter = SimpleNamespace(handle_message=AsyncMock(), supports_push=True)
    runner.adapters = {Platform.TELEGRAM: root_adapter}
    runner._profile_adapters = {'sano': {Platform.TELEGRAM: adapter}}
    try:
        with patch('hermes_cli.profiles.get_profile_dir', return_value=home), patch('hermes_cli.profiles.profile_exists', return_value=True):
            assert await runner._deliver_completion_notification('done', restored.get_nowait()) is True
        adapter.handle_message.assert_awaited_once()
        root_adapter.handle_message.assert_not_awaited()
        runner._profile_adapters = {}
        assert await runner._inject_watch_notification('retry', evt) is False
        root_adapter.handle_message.assert_not_awaited()
        delivered = adapter.handle_message.call_args.args[0]
        assert delivered.source.chat_id == '-5180401933'
        assert delivered.source.profile == 'sano'
        assert delivered.source.thread_id is None
        assert delivered.metadata['gateway_session_id'] == 'parent'
        with _profile_runtime_scope(home):
            with durable._transaction() as conn:
                assert conn.execute('SELECT delivery_state FROM async_delegations WHERE delegation_id=?', ('persisted',)).fetchone()[0] == 'delivered'
            assert durable.restore_undelivered_completions(queue.Queue()) == 0
        assert get_hermes_home() == root
    finally:
        runner.close_all_session_db_handles()


def test_named_key_is_not_mistaken_for_raw_api_session():
    from gateway.run import _parse_session_key
    parsed = _parse_session_key('agent:sano:telegram:group:-5180401933:46291309')
    assert parsed is not None
    assert parsed['chat_id'] == '-5180401933'
    assert parsed['profile'] == 'sano'
    assert 'thread_id' not in parsed
