"""Record ComfyUI progress events over WebSocket using only the standard library."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--host', default='127.0.0.1')
p.add_argument('--port', type=int, default=8188)
p.add_argument('--timeout', type=float, default=3600)
p.add_argument('--output', default='artifacts/diagnostics/performance/comfy_progress.jsonl')
a = p.parse_args()
dest = Path(a.output)
dest.parent.mkdir(parents=True, exist_ok=True)
conn = socket.create_connection((a.host, a.port), timeout=20)
key = base64.b64encode(os.urandom(16)).decode()
request = f'GET /ws?clientId=b70-minimax-verification HTTP/1.1\r\nHost: {a.host}:{a.port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n'
conn.sendall(request.encode())
header = b''
while not header.endswith(b'\r\n\r\n'):
    header += conn.recv(1)
expected = base64.b64encode(hashlib.sha1((key + '258EAFA5-E914-47DA-95CA-C5AB0DC85B11').encode()).digest())
if b'101 Switching Protocols' not in header or expected not in header:
    raise RuntimeError(f'WebSocket handshake failed: {header!r}')

def read(size):
    result = b''
    while len(result) < size:
        chunk = conn.recv(size - len(result))
        if not chunk:
            raise EOFError('WebSocket closed')
        result += chunk
    return result

deadline = time.monotonic() + a.timeout
conn.settimeout(30)
with dest.open('a', encoding='utf-8', buffering=1) as out:
    while time.monotonic() < deadline:
        try:
            first, second = read(2)
        except socket.timeout:
            continue
        length = second & 127
        if length == 126:
            length = struct.unpack('!H', read(2))[0]
        elif length == 127:
            length = struct.unpack('!Q', read(8))[0]
        mask = read(4) if second & 128 else None
        payload = read(length)
        if mask:
            payload = bytes(x ^ mask[i % 4] for i, x in enumerate(payload))
        opcode = first & 15
        if opcode == 8:
            break
        if opcode == 9:
            masking = os.urandom(4)
            conn.sendall(bytes([0x8A, 0x80 | len(payload)]) + masking + bytes(x ^ masking[i % 4] for i, x in enumerate(payload)))
        if opcode == 1:
            event = json.loads(payload)
            out.write(json.dumps({'timestamp': time.time(), 'event': event}) + '\n')
            print(json.dumps(event), flush=True)
conn.close()
