"""
TestLang.py

Authors: Ann-Kristin Schulte, Carolin Gottschalk, Jonas D. Stephan
License: Apache License 2.0

Description:
This module defines the TestLang class, which extends LangPlugin to process pose estimation data.
It reads JSON data, computes final rotations using the Calculator class, and sends processed
frames to the output hook.
"""

from ..LangPlugin import LangPlugin
import json
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from Calculator import Calculator


class TestLang(LangPlugin):
    """
    A language processing plugin for handling skeleton-based pose estimation.
    It computes final rotations for pose data and sends processed frames to the output hook.
    """

    def __init__(self, unified_web_socket):
        """
        Initializes the TestLang plugin with an output hook and a Calculator instance.

        :param unified_web_socket: An object responsible for sending processed data.
        :type unified_web_socket: object
        """
        super().__init__(unified_web_socket)
        self.calculator = Calculator(unified_web_socket.get_t_pose())


    def input(self, string: str):
        """
        Processes an input command and computes pose rotations based on provided JSON data.

        :param string: Input command to determine the processing mode.
        :type string: str
        """
        if string == "test":
            with open('0001-0001-0001.pev') as file:
                data = json.load(file)

            for i in range(len(data["frames"])):
                self.unified_web_socket.add_send_object(
                    self.calculator.compute_final_rotations(data["frames"][i].get("skeletonpoints"))
                )
        else:
            with open('t_old.pei', 'r') as file:
                data = json.load(file)

            angles = self.calculator.compute_final_rotations(data.get("skeletonpoints"))
            self.unified_web_socket.add_send_object(angles)
