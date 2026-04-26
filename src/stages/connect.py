"""Connection stage for drone runtime pipeline."""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Any, Dict
from src.core.communication import UDPTelemetrySender


@dataclass(frozen=True)
class ConnectionConfig:
    """Settings used to establish outbound telemetry to ground station."""

    ground_station_ip: str
    telemetry_port: int = 9000
    source: str = "drone-pi"
    heartbeat_hz: float = 1.0


class DroneConnection:
    """Initialize and maintain telemetry connection to a PC ground station."""

    def __init__(self, config: ConnectionConfig) -> None:
        self._config = config
        self._sender = UDPTelemetrySender(
            host=config.ground_station_ip,
            port=config.telemetry_port,
            source=config.source,
        )
        self._last_heartbeat = 0.0

    def connect(self) -> int:
        """Send initial connect packet and return sequence number."""
        return self._sender.send(
            {
                "event": "connect",
                "status": "ok",
                "heartbeat_hz": self._config.heartbeat_hz,
            }
        )

    def send(self, payload: Dict[str, Any]) -> int:
        """Send telemetry payload through the established channel."""
        return self._sender.send(payload)

    def maybe_send_heartbeat(self) -> int | None:
        """Send heartbeat packet if heartbeat interval has elapsed."""
        now = time.time()
        interval = 1.0 / self._config.heartbeat_hz if self._config.heartbeat_hz > 0 else 1.0
        if now - self._last_heartbeat < interval:
            return None

        self._last_heartbeat = now
        return self._sender.send({"event": "heartbeat", "status": "alive"})

    def close(self) -> None:
        self._sender.close()
