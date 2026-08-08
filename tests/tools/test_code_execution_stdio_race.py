"""Regression contract for process-global stdio ownership in execute_code RPC."""

import ast
import inspect
import textwrap

import pytest

import tools.code_execution_tool as code_execution


RPC_DISPATCHERS = ("_rpc_server_loop", "_rpc_poll_loop")


def _function_tree(name: str) -> tuple[str, ast.AST]:
    function = getattr(code_execution, name)
    source = textwrap.dedent(inspect.getsource(function))
    return source, ast.parse(source)


def _assigned_sys_stdio(tree: ast.AST) -> list[str]:
    assigned = []
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets.extend(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets.append(node.target)
        elif isinstance(node, ast.AugAssign):
            targets.append(node.target)
        for target in targets:
            flat_targets = target.elts if isinstance(target, (ast.Tuple, ast.List)) else [target]
            for item in flat_targets:
                if (
                    isinstance(item, ast.Attribute)
                    and isinstance(item.value, ast.Name)
                    and item.value.id == "sys"
                    and item.attr in {"stdout", "stderr"}
                ):
                    assigned.append(item.attr)
    return assigned


@pytest.mark.parametrize("function_name", RPC_DISPATCHERS)
def test_rpc_dispatch_does_not_take_process_stdio_ownership(function_name):
    """RPC workers must never replace or close process-global output streams.

    The old save/swap/restore pattern was racy across concurrent profile turns:
    one worker could restore another worker's already-closed devnull handle.
    """
    source, tree = _function_tree(function_name)

    assert _assigned_sys_stdio(tree) == []
    assert "redirect_stdout" not in source
    assert "redirect_stderr" not in source
    assert "_suppress_tool_stdio" not in source
    assert "devnull.close" not in source


@pytest.mark.parametrize("function_name", RPC_DISPATCHERS)
def test_rpc_dispatch_calls_tool_handler_directly(function_name):
    """Handler output stays on normal gateway stdio instead of being redirected."""
    source, tree = _function_tree(function_name)

    direct_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "handle_function_call"
    ]

    assert direct_calls, f"{function_name} must dispatch through handle_function_call"
    assert "sys.stdout" not in source
    assert "sys.stderr" not in source
