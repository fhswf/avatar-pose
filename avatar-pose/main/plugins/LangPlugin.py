"""
LangPlugin.py

Authors: Open Source Contributor
License: Apache License 2.0

Description:
This module defines an abstract base class `LangPlugin` for language processing plugins.
It provides a structure for handling input commands and interacting with an output hook.
"""

from abc import abstractmethod


class LangPlugin:
    """
    Abstract base class for language processing plugins.
    Each subclass must implement the `input` method to handle specific input commands.
    """

    def __init__(self, unified_web_socket):
        """
        Initializes the language plugin with an output hook and a base object.

        :param unified_web_socket: An object responsible for managing processed data.
        :type unified_web_socket: object
        """
        self.output_hook = unified_web_socket
        self.base_object = unified_web_socket.get_t_pose()

    @abstractmethod
    def input(self, string: str):
        """
        Abstract method to process input commands.
        Must be implemented by subclasses.

        :param string: Input command string.
        :type string: str
        """
        pass
