"""Communication and telemetry helpers for Pi-to-PC data transfer.

This module uses UDP with JSON payloads for low-latency telemetry.
"""

from __future__ import annotations

import json
import socket
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class TelemetryPacket:
	"""Single telemetry message exchanged between drone and ground station."""

	seq: int
	timestamp: float
	source: str
	payload: Dict[str, Any]

	def to_bytes(self) -> bytes:
		"""Serialize packet to UTF-8 JSON bytes."""
		raw = {
			"seq": self.seq,
			"timestamp": self.timestamp,
			"source": self.source,
			"payload": self.payload,
		}
		return json.dumps(raw, separators=(",", ":")).encode("utf-8")

	@staticmethod
	def from_bytes(data: bytes) -> "TelemetryPacket":
		"""Deserialize UTF-8 JSON bytes to a TelemetryPacket."""
		decoded = json.loads(data.decode("utf-8"))
		required = {"seq", "timestamp", "source", "payload"}
		missing = required.difference(decoded)
		if missing:
			raise ValueError(f"Telemetry packet missing fields: {sorted(missing)}")

		payload = decoded["payload"]
		if not isinstance(payload, dict):
			raise ValueError("Telemetry packet payload must be an object")

		return TelemetryPacket(
			seq=int(decoded["seq"]),
			timestamp=float(decoded["timestamp"]),
			source=str(decoded["source"]),
			payload=payload,
		)


class UDPTelemetrySender:
	"""Sends telemetry packets to a known receiver host/port."""

	def __init__(self, host: str, port: int, source: str = "drone-pi") -> None:
		self._target = (host, port)
		self._source = source
		self._seq = 0
		self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

	def send(self, payload: Dict[str, Any]) -> int:
		"""Send a telemetry payload and return its sequence number."""
		packet = TelemetryPacket(
			seq=self._seq,
			timestamp=time.time(),
			source=self._source,
			payload=payload,
		)
		self._sock.sendto(packet.to_bytes(), self._target)
		sent_seq = self._seq
		self._seq += 1
		return sent_seq

	def close(self) -> None:
		self._sock.close()


class UDPTelemetryReceiver:
	"""Receives telemetry packets on a UDP socket."""

	def __init__(self, host: str = "0.0.0.0", port: int = 9000, timeout_s: Optional[float] = 1.0) -> None:
		self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		self._sock.bind((host, port))
		self._sock.settimeout(timeout_s)

	def receive(self, buffer_size: int = 65535) -> Optional[TelemetryPacket]:
		"""Return the next telemetry packet, or None on timeout."""
		try:
			data, _ = self._sock.recvfrom(buffer_size)
		except socket.timeout:
			return None

		return TelemetryPacket.from_bytes(data)

	def close(self) -> None:
		self._sock.close()
