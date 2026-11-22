"""
main.py

Authors: Carolin Gottschalk, Jonas D. Stephan
License: Apache License 2.0

Description:
This script initializes and starts the UnifiedWebSocket server.
It dynamically loads a specified language module and runs the WebSocket server
for real-time communication.
"""
import os
import asyncio
import importlib
from UnifiedWebSocket import UnifiedWebSocket

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == "__main__":
    """
    Main entry point of the WebSocket server.

    - Dynamically loads the language module.
    - Initializes the WebSocket server.
    - Starts the server asynchronously.
    """
    # Plugin per Umgebungsvariable waehlen, Standard bleibt "gsg".
    language = os.environ.get("AVATAR_PLUGIN", "gsg").strip() or "gsg"
    print(f"[main] Plugin: {language}", flush=True)
    language_module = importlib.import_module(f"plugins.{language}.{language}")
    language_class = getattr(language_module, language)

    # Initialisierung von UnifiedWebSocket
    unified_server = UnifiedWebSocket(language_class)

    # Start des WebSocket-Servers
    asyncio.run(unified_server.start_server())
