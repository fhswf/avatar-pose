"""
main.py

Authors: Carolin Gottschalk, Jonas D. Stephan
License: Apache License 2.0

Description:
This script initializes and starts the UnifiedWebSocket server.
It dynamically loads a specified language module and runs the WebSocket server
for real-time communication.
"""

import asyncio
import importlib
from UnifiedWebSocket import UnifiedWebSocket

if __name__ == "__main__":
    """
    Main entry point of the WebSocket server.

    - Dynamically loads the language module.
    - Initializes the WebSocket server.
    - Starts the server asynchronously.
    """
    # TODO: Change to environmental value
    language = "gsg"
    language_module = importlib.import_module(f"plugins.{language}.{language}")
    language_class = getattr(language_module, language)

    # Initialisierung von UnifiedWebSocket
    unified_server = UnifiedWebSocket(language_class)

    # Start des WebSocket-Servers
    asyncio.run(unified_server.start_server())
