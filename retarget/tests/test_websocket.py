"""Mehrere WebSocket-Clients duerfen sich die Frames nicht wegnehmen.

Regression zu dem Fehler, bei dem ein zweiter Client (oder ein alter
Browser-Tab) den gemeinsamen Puffer leerte und der Webviewer nur noch zwei
Bilder zu sehen bekam.
"""

import asyncio
import json
import sys
from pathlib import Path

import pytest

MAIN = Path(__file__).resolve().parents[2] / "avatar-pose" / "main"


class RecordingSocket:
    def __init__(self):
        self.messages = []

    async def send(self, data):
        self.messages.append(json.loads(data))


@pytest.fixture(scope="module")
def ws_module():
    if str(MAIN) not in sys.path:
        sys.path.insert(0, str(MAIN))
    import UnifiedWebSocket
    return UnifiedWebSocket


def _server(module):
    """Server ohne GLB laden; nur der Frame-Puffer ist relevant."""
    server = module.UnifiedWebSocket.__new__(module.UnifiedWebSocket)
    server.send_objects = [{"bone": {"rotation": [0.0, 0.0, 0.0, 1.0]}}]
    server._next_index = 1
    return server


def test_zwei_clients_bekommen_jeden_frame(ws_module, monkeypatch):
    real_sleep = asyncio.sleep

    async def fast_sleep(_delay, *args, **kwargs):
        await real_sleep(0)

    monkeypatch.setattr(ws_module.asyncio, "sleep", fast_sleep)
    server = _server(ws_module)
    frames = [{"index": i, "bone": {"rotation": [0.0, 0.0, 0.0, 1.0]}}
              for i in range(1, 6)]
    rest = {"bone": {"rotation": [0.0, 0.0, 0.0, 1.0]}}

    first, second = RecordingSocket(), RecordingSocket()

    async def run():
        tasks = [asyncio.create_task(server.send_data(sock))
                 for sock in (first, second)]
        # Beide Clients sind verbunden und haben den Ruheframe gesehen ...
        await real_sleep(0)
        # ... erst danach kommen die neuen Frames dazu.
        for frame in frames:
            server.add_send_object(frame)
        for _ in range(10):
            await real_sleep(0)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(run())

    expected = json.loads(json.dumps(frames))
    for sock in (first, second):
        got = [msg for msg in sock.messages if msg != rest]
        assert got[:len(expected)] == expected
        # danach wird der letzte Frame wiederholt
        assert sock.messages[-1] == expected[-1]


def test_puffer_bleibt_begrenzt(ws_module):
    server = _server(ws_module)
    for i in range(ws_module.HISTORY_FRAMES + 50):
        server.add_send_object({"bone": {"rotation": [0.0, 0.0, 0.0, 1.0]}})
    assert len(server.send_objects) == ws_module.HISTORY_FRAMES
    assert server._next_index == ws_module.HISTORY_FRAMES + 51
