"""NixOS container supervision must honor the gateway's safe drain budget."""

import re
from pathlib import Path

from gateway.restart import DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
from gateway.shutdown_watchdog import DEFAULT_SHUTDOWN_WATCHDOG_GRACE_S


def test_nixos_container_stop_budget_outlives_gateway_watchdog():
    module = (
        Path(__file__).parents[2] / "nix" / "nixosModules.nix"
    ).read_text(encoding="utf-8")
    stop_timeout = int(
        re.search(r"stop -t (\d+) \$\{containerName\}", module).group(1)
    )
    service_timeout = int(
        re.search(r"TimeoutStopSec = (\d+);", module).group(1)
    )
    watchdog_deadline = (
        DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
        + DEFAULT_SHUTDOWN_WATCHDOG_GRACE_S
    )

    assert stop_timeout >= watchdog_deadline + 15
    assert service_timeout >= stop_timeout + 15