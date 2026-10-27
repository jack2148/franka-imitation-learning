"""Non-blocking latest-state ZeroMQ subscriber for the MuJoCo process."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Any

from .protocol import OmyStateMessage, ProtocolError, decode_message


DEFAULT_ENDPOINT = "tcp://127.0.0.1:5557"
DEFAULT_STALE_TIMEOUT = 0.2


@dataclass(frozen=True)
class ClientState:
    message: OmyStateMessage | None
    fresh: bool
    age_seconds: float | None


class OmyStateClient:
    """Drain without blocking and expose only the newest valid sequence."""

    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        *,
        stale_timeout: float = DEFAULT_STALE_TIMEOUT,
        context: Any | None = None,
        socket: Any | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if stale_timeout <= 0:
            raise ValueError("stale_timeout must be positive")
        import zmq

        self._zmq = zmq
        self.endpoint = endpoint
        self.stale_timeout = float(stale_timeout)
        self._monotonic_ns = monotonic_ns
        self._owns_socket = socket is None
        if socket is None:
            self._context = context or zmq.Context.instance()
            self._socket = self._context.socket(zmq.SUB)
            self._socket.setsockopt(zmq.SUBSCRIBE, b"")
            self._socket.setsockopt(zmq.RCVHWM, 1)
            self._socket.setsockopt(zmq.CONFLATE, 1)
            self._socket.setsockopt(zmq.LINGER, 0)
            self._socket.connect(endpoint)
        else:
            self._context = context
            self._socket = socket
        self.latest: OmyStateMessage | None = None
        self.last_sequence: int | None = None
        self.last_receive_monotonic_ns: int | None = None
        self.rejected_messages = 0
        self.received_messages = 0
        self.dropped_sequences = 0
        self.stale_reads = 0

    def poll(self) -> OmyStateMessage | None:
        """Drain immediately available data; never wait for the publisher."""
        newest = None
        while True:
            try:
                payload = self._socket.recv(flags=self._zmq.NOBLOCK)
            except self._zmq.Again:
                break
            try:
                candidate = decode_message(payload, last_sequence=self.last_sequence)
            except ProtocolError:
                self.rejected_messages += 1
                continue
            if self.last_sequence is not None:
                self.dropped_sequences += max(
                    0, candidate.sequence - self.last_sequence - 1
                )
            self.latest = candidate
            self.last_sequence = candidate.sequence
            self.last_receive_monotonic_ns = self._monotonic_ns()
            self.received_messages += 1
            newest = candidate
        return newest

    def read(self) -> ClientState:
        self.poll()
        if self.latest is None or self.last_receive_monotonic_ns is None:
            return ClientState(message=None, fresh=False, age_seconds=None)
        age = max(
            0.0,
            (self._monotonic_ns() - self.last_receive_monotonic_ns) / 1e9,
        )
        if age > self.stale_timeout:
            self.stale_reads += 1
        return ClientState(
            message=self.latest,
            fresh=age <= self.stale_timeout,
            age_seconds=age,
        )

    def close(self) -> None:
        if self._owns_socket:
            self._socket.close(linger=0)
