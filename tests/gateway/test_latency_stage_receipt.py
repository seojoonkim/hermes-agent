import inspect
from gateway.run import GatewayRunner
from gateway.latency import measure_agent_stage

def test_handler_uses_exception_safe_correlated_timer():
    source=inspect.getsource(GatewayRunner._handle_message_with_agent)
    assert 'with measure_agent_stage(_run_start_session_id, run_generation, _msg_started_monotonic)' in source
    assert 'gateway_latency: session=%s generation=%s prep_seconds=%.3f agent_seconds=%.3f' in inspect.getsource(measure_agent_stage)
