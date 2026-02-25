"""
UnifiedWebSocket.py

Authors: Ann-Kristin Schulte, Jonas D. Stephan
License: Apache License 2.0

Description:
This module provides the `UnifiedWebSocket` class, which handles WebSocket communication
for real-time data transfer. It processes GLTF data, manages skeletal transformations,
and enables interaction between the server and external clients.
"""

import asyncio
import websockets
import json
import numpy as np
from pygltflib import GLTF2
import copy

class UnifiedWebSocket:
    """
    A WebSocket server for real-time communication, handling skeletal transformations and data exchange.
    """
    def __init__(self, language_class):
        """
        Initializes the WebSocket server and loads skeletal data.

        :param language_class: The language processing class for handling incoming messages.
        :type language_class: class
        """
        self.t_pose = {}
        self.load_gltf_data("../webviewer/media/avatar.glb")
        self.language_module = language_class(self)

        first_position = copy.deepcopy(self.t_pose)
        for key in first_position:
            if 'translation' in first_position[key]:
                del first_position[key]['translation']

        self.send_objects = [first_position]

    async def start_server(self):
        """
        Starts the WebSocket server and listens for connections.
        """
        async with websockets.serve(self.handler, "0.0.0.0", 8765):
            print("Unified WebSocket server is running on ws://0.0.0.0:8765")
            await asyncio.Future()

    async def handler(self, websocket):
        """
        Handles WebSocket connections and manages data transmission.

        :param websocket: The WebSocket connection.
        :type websocket: websockets.WebSocketServerProtocol
        """
        consumer_task = asyncio.create_task(self.receive_data(websocket))
        producer_task = asyncio.create_task(self.send_data(websocket))
        await asyncio.gather(consumer_task, producer_task)

    async def receive_data(self, websocket):
        """
        Listens for incoming messages and processes them.

        :param websocket: The WebSocket connection.
        :type websocket: websockets.WebSocketServerProtocol
        """
        while True:
            try:
                message = await websocket.recv()
                self.language_module.input(message)
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed (Input)")
                break

    async def send_data(self, websocket):
        """
        Sends outgoing data to the WebSocket client at a fixed frame rate.

        :param websocket: The WebSocket connection.
        :type websocket: websockets.WebSocketServerProtocol
        """
        while True:
            if len(self.send_objects) > 1:
                actual_frame = self.send_objects.pop(0)
            else:
                actual_frame = self.send_objects[0]

            await websocket.send(json.dumps(self.convert_numpy_to_list(actual_frame)))
            await asyncio.sleep(1 / 30)

    def add_send_object(self, send_object):
        """
        Adds an object to the list of data frames to be sent.

        :param send_object: The data object to send.
        :type send_object: dict
        """
        self.send_objects.append(send_object)

    def get_t_pose(self):
        """
        Retrieves the T-pose skeletal data.

        :return: The skeletal structure of the avatar.
        :rtype: dict
        """
        return self.t_pose

    def load_gltf_data(self, file_path):
        """
        Loads GLTF skeletal data and processes it into a structured format.

        :param file_path: The path to the GLTF file.
        :type file_path: str
        """
        gltf = GLTF2().load(file_path)
        tmp_t_pose = {}

        for node_index, node in enumerate(gltf.nodes):
            if node.name != "KIM_caucasian_male":
                if node.translation or node.rotation or node.scale:
                    node_object = {}
                    for parent_node in gltf.nodes:
                        if node.name != "Root" and node_index in (parent_node.children or []):
                            node_object["parent"] = parent_node.name
                            break
                    node_object["children"] = [gltf.nodes[child_index].name for child_index in (node.children or [])]
                    node_object["translation"] = np.array(node.translation or [0, 0, 0], dtype=np.float32)
                    node_object["rotation"] = np.array(node.rotation or [0, 0, 0, 1], dtype=np.float32)
                    tmp_t_pose[node.name] = node_object

        self.t_pose = tmp_t_pose
        #print(self.t_pose)

    def convert_numpy_to_list(self, data):
        """
        Converts NumPy arrays to lists for JSON serialization.

        :param data: The data structure containing NumPy arrays.
        :type data: dict
        :return: The converted structure with lists instead of NumPy arrays.
        :rtype: dict
        """
        for key, value in data.items():
            if isinstance(value, dict):
                self.convert_numpy_to_list(value)
            elif isinstance(value, np.ndarray):
                data[key] = value.tolist()
        return data
