"""SIP <-> Unix socket media relay. Runs inside the relay container.

The SIP/RTP stack it uses is compiled for Python 3.10, so it runs in its own
image rather than in EVA's environment. ``bridge.py`` starts it.

It dials a Guava ``guavasip-…`` code and bridges the call's RTP audio to a Unix
socket the host listens on:

    host --(8 kHz PCM16 LE)--> socket --> [µ-law encode] --> RTP --> Guava
    Guava --> RTP --> [µ-law decode] --> socket --(8 kHz PCM16 LE)--> host

Socket audio is raw (headerless) 8 kHz mono PCM16 little-endian in both
directions. Lifecycle events are printed to stdout, one per line:
``RELAY_CONNECTED remote=<ip>:<port> public=<ip>:<port>``, ``RELAY_HANGUP``,
``RELAY_ERROR <msg>``.
"""

import argparse
import socket
import sys
import threading
import time

import g711
import numpy as np
import rtp as rtp_module

# ---- audio / RTP geometry (PCMU, 8 kHz, 20 ms frames) ----------------------
_PCMU_PT = 0
_SAMPLES_PER_FRAME = 160  # 20 ms @ 8 kHz
_FRAME_BYTES = _SAMPLES_PER_FRAME * 2  # PCM16 => 2 bytes/sample => 320 bytes

FROM_NUMBER = "+15550000001"  # placeholder caller ID


def _log(msg: str) -> None:
    """Lifecycle line for the host to parse (stdout, line-buffered)."""
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def _recvall(conn: socket.socket, n: int) -> bytes | None:
    """Read exactly ``n`` bytes from a stream socket, or None at EOF."""
    buf = bytearray()
    while len(buf) < n:
        try:
            chunk = conn.recv(n - len(buf))
        except OSError:
            return None  # host closed (e.g. ECONNRESET during teardown)
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


class _RtpSendStream:
    """Encodes PCM16 frames to PCMU RTP and sends them on a shared UDP socket."""

    def __init__(self, sock: socket.socket, remote_addr: tuple[str, int]):
        self._sock = sock
        self._remote = remote_addr
        self._seq = 0
        self._ts = 0
        self._ssrc = 0x4EA00001  # fixed SSRC; value is arbitrary

    def send_pcm_frame(self, pcm16: bytes) -> None:
        samples = np.frombuffer(pcm16, dtype=np.int16)
        payload = g711.encode(samples).tobytes()
        packet = rtp_module.RTP(seq=self._seq, ts=self._ts, pt=_PCMU_PT, ssrc=self._ssrc, payload=payload).to_bytes()
        try:
            self._sock.sendto(packet, self._remote)
        except OSError:
            pass
        self._seq = (self._seq + 1) & 0xFFFF
        self._ts += _SAMPLES_PER_FRAME


def _host_to_rtp(conn: socket.socket, sender: _RtpSendStream, done: threading.Event) -> None:
    """Pump host PCM (unix socket) -> PCMU RTP -> Guava, one 20 ms frame at a time."""
    try:
        while not done.is_set():
            frame = _recvall(conn, _FRAME_BYTES)
            if frame is None:
                break  # host closed -> end the call
            sender.send_pcm_frame(frame)
    finally:
        done.set()


def _rtp_to_host(rtp_sock: socket.socket, conn: socket.socket, done: threading.Event) -> None:
    """Pump inbound PCMU RTP -> PCM16 -> host (unix socket)."""
    rtp_sock.settimeout(0.2)
    try:
        while not done.is_set():
            try:
                data, _ = rtp_sock.recvfrom(2048)
            except TimeoutError:
                continue
            except OSError:
                break
            if len(data) < 12:
                continue
            cc = data[0] & 0x0F
            header_size = 12 + 4 * cc
            if len(data) <= header_size:
                continue
            pt = data[1] & 0x7F
            payload = data[header_size:]
            if pt != _PCMU_PT:
                continue  # only PCMU is negotiated
            pcm16 = g711.decode(np.frombuffer(payload, dtype=np.uint8)).astype(np.int16).tobytes()
            try:
                conn.sendall(pcm16)
            except OSError:
                break
    finally:
        done.set()


def main() -> int:
    parser = argparse.ArgumentParser(description="SIP<->unix-socket media relay (container side).")
    parser.add_argument("--number", required=True, help="guavasip-… rendezvous code to dial")
    parser.add_argument("--sip-host", required=True)
    parser.add_argument("--sip-port", type=int, default=5060)
    parser.add_argument("--sock", required=True, help="Unix socket path the host listens on")
    parser.add_argument("--connect-timeout", type=float, default=15.0)
    args = parser.parse_args()

    sys.path.insert(0, "/opt/sip")  # SIP stack baked into the relay image
    from sip_client import CallSession  # noqa: E402
    from stun import discover_public_address  # noqa: E402

    # One UDP socket for both send and receive, so the STUN-discovered NAT
    # mapping matches the port RTP is sent from (symmetric RTP).
    rtp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rtp_sock.bind(("", 0))
    pub_ip, pub_port = discover_public_address(rtp_sock, "stun.l.google.com", 19302)

    session = CallSession(args.sip_host, args.sip_port)
    session.dial(
        to_number=args.number,
        from_number=FROM_NUMBER,
        local_rtp_ip=pub_ip,
        local_rtp_port=pub_port,
        local_sip_port=None,
    )
    if not session.wait_for_connected(timeout=args.connect_timeout):
        _log("RELAY_ERROR sip_not_answered")
        session.stop()
        return 1

    remote_ip, remote_port = session.remote_rtp_address
    _log(f"RELAY_CONNECTED remote={remote_ip}:{remote_port} public={pub_ip}:{pub_port}")

    # Connect to the host's listening unix socket.
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    deadline = time.monotonic() + 10.0
    while True:
        try:
            conn.connect(args.sock)
            break
        except (FileNotFoundError, ConnectionRefusedError):
            if time.monotonic() > deadline:
                _log("RELAY_ERROR unix_socket_connect_timeout")
                session.hangup()
                session.stop()
                return 1
            time.sleep(0.05)

    done = threading.Event()
    sender = _RtpSendStream(rtp_sock, (remote_ip, remote_port))
    t_out = threading.Thread(target=_host_to_rtp, args=(conn, sender, done), daemon=True)
    t_in = threading.Thread(target=_rtp_to_host, args=(rtp_sock, conn, done), daemon=True)
    t_out.start()
    t_in.start()

    # Also end if the far side hangs up.
    try:
        while not done.is_set():
            if session.hungup_event.is_set():
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        done.set()
        _log("RELAY_HANGUP")
        try:
            conn.close()
        except OSError:
            pass
        session.hangup()
        session.stop()
        try:
            rtp_sock.close()
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
