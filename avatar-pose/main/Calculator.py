"""
Calculator.py

Authors: Ann-Kristin Schulte, Jonas D. Stephan
License: Apache License 2.0

Description:
This module provides a Calculator class for various mathematical and
geometric computations, including quaternion transformations,
Three.js coordinate conversions, and skeleton position adjustments.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import json


class Calculator:
    """
    A class to handle transformations and calculations related to skeletal animations.
    """
    def __init__(self, t_pose):
        """
        Initializes the Calculator class with the given T-pose and loads the bone mapping.

        :param t_pose: Dictionary containing the T-pose data.
        :type t_pose: dict
        """
        self.t_pose = t_pose

        # Load mapping
        with open('data/bone_mapping.json', "r") as mapping_file:
            self.mapping = json.load(mapping_file)

    @staticmethod
    def quaternion_from_two_vectors(v_start, v_target):
        """
        Computes the rotation quaternion to rotate v_start to v_target.

        :param v_start: Initial vector.
        :type v_start: np.ndarray
        :param v_target: Target vector.
        :type v_target: np.ndarray
        :return: Quaternion representing the rotation.
        :rtype: np.ndarray
        """
        if np.linalg.norm(v_start) < 1e-6 or np.linalg.norm(v_target) < 1e-6:
            return np.array([0, 0, 0, 1])  # Identity quaternion fallback

        v_start /= np.linalg.norm(v_start)
        v_target /= np.linalg.norm(v_target)
        dot = np.dot(v_start, v_target)

        if dot < -0.999999:
            axis = np.cross(v_start, np.array([1, 0, 0]))
            if np.linalg.norm(axis) < 1e-6:
                axis = np.cross(v_start, np.array([0, 1, 0]))
            axis /= np.linalg.norm(axis)
            return R.from_rotvec(np.pi * axis).as_quat()

        axis = np.cross(v_start, v_target)
        if np.linalg.norm(axis) < 1e-6:
            return np.array([0, 0, 0, 1])

        angle = np.arccos(dot)
        return R.from_rotvec(angle * axis / np.linalg.norm(axis)).as_quat()

    @staticmethod
    def convert_to_threejs_coords(vector):
        """
        Converts coordinates to Three.js format.

        :param vector: Dictionary containing 'x', 'y', 'z' values.
        :type vector: dict
        :return: Converted NumPy array.
        :rtype: np.ndarray
        """
        return np.array([vector["x"], -vector["y"], -vector["z"]])

    def shift_origin_poe_data(self, poe_data):
        """
        Shifts POE data so that the origin is at the hip (after Three.js conversion).

        :param poe_data: List of dictionaries containing position data.
        :type poe_data: list
        :return: Adjusted POE data.
        :rtype: list
        """
        root_tail_position = self.get_target_position("Root", poe_data, point_type="tail")

        if np.linalg.norm(root_tail_position) < 1e-6:
            print("Warning: Invalid Root-Tail position. No shift applied.")
            return poe_data

        shifted_poe_data = []
        for entry in poe_data:
            shifted_entry = entry.copy()
            entry_position = np.array([entry["x"], entry["y"], entry["z"]])
            shifted_position = entry_position - root_tail_position
            shifted_position = self.convert_to_threejs_coords({
                "x": shifted_position[0],
                "y": shifted_position[1],
                "z": shifted_position[2]
            })
            shifted_entry["x"], shifted_entry["y"], shifted_entry["z"] = shifted_position
            shifted_poe_data.append(shifted_entry)

        return shifted_poe_data

    def get_target_position(self, bone_name, poe_data, point_type="head"):
        """
        Retrieves the target position for a bone from POE data based on mapping.

        :param bone_name: Name of the bone.
        :type bone_name: str
        :param poe_data: List of position data.
        :type poe_data: list
        :param point_type: Either 'head' or 'tail'.
        :type point_type: str
        :return: NumPy array of the target position.
        :rtype: np.ndarray
        """
        if bone_name not in self.mapping:
            return np.array([0.0, 0.0, 0.0])

        bone_map = self.mapping[bone_name]
        target_ids = bone_map.get(point_type)

        if target_ids is None:
            print(f"Warning: No {point_type} point found for {bone_name}.")
            return np.array([0.0, 0.0, 0.0])

        if isinstance(target_ids, list):
            positions = [self.get_point_position(target_id, poe_data) for target_id in target_ids]
            positions = [pos for pos in positions if np.linalg.norm(pos) > 1e-6]
            if positions:
                return np.mean(positions, axis=0)

        return self.get_point_position(target_ids, poe_data)

    @staticmethod
    def get_point_position(point_id, poe_data):
        """
        Retrieves the 3D coordinates of a point from POE data by ID.

        :param point_id: ID of the point.
        :type point_id: int
        :param poe_data: List of position data.
        :type poe_data: list
        :return: NumPy array with coordinates.
        :rtype: np.ndarray
        """
        for entry in poe_data:
            if entry["id"] == point_id:
                return np.array([entry["x"], entry["y"], entry["z"]])
        return np.array([0.0, 0.0, 0.0])

    def compute_final_rotations(self, poe_data):
        """
        Computes the final quaternions for all bones based on hierarchy.

        :param poe_data: List of position data.
        :type poe_data: list
        :return: Dictionary of final rotations.
        :rtype: dict
        """
        poe_data = self.shift_origin_poe_data(poe_data)
        final_rotations = {}

        for bone_name, bone in self.t_pose.items():
            if bone_name == "KIM_caucasian_male.body":
                continue

            parent_name = bone.get('parent')
            first_child_name = bone['children'][0] if bone.get('children') else None
            final_rotations.setdefault(bone_name, {})

            if self.mapping.get(bone_name, {}).get("ignore", False) or parent_name is None:
                final_rotations[bone_name] = {'rotation': np.array(bone['rotation'])}
                continue

            start_head_pos = np.array(bone['position'])
            start_tail_pos = np.array(self.t_pose[first_child_name]['position']) if first_child_name else start_head_pos
            target_head_pos = self.get_target_position(bone_name, poe_data, point_type="head")
            target_tail_pos = self.get_target_position(bone_name, poe_data, point_type="tail")

            start_vector = start_tail_pos - start_head_pos
            target_vector = target_tail_pos - target_head_pos

            if np.linalg.norm(target_vector) < 1e-7:
                final_rotations[bone_name]['rotation'] = self.t_pose[bone_name]['rotation']
                continue

            rotation_quat = self.quaternion_from_two_vectors(start_vector, target_vector)
            bone_quat = np.array(bone['rotation'])
            if np.linalg.norm(bone_quat) > 1e-6:
                bone_quat /= np.linalg.norm(bone_quat)
            else:
                bone_quat = np.array([0, 0, 0, 1])

            final_quat = R.from_quat(rotation_quat) * R.from_quat(bone_quat)
            final_rotations[bone_name]['rotation'] = final_quat.as_quat()

        return final_rotations

    @ staticmethod
    def slerp(q1, q2, t):
        """
        Performs Spherical Linear Interpolation (SLERP) between two quaternions.

        :param q1: The starting quaternion as a numpy array [x, y, z, w].
        :type q1: np.ndarray
        :param q2: The target quaternion as a numpy array [x, y, z, w].
        :type q2: np.ndarray
        :param t: The interpolation factor (0.0 = q1, 1.0 = q2).
        :type t: float
        :return: The interpolated quaternion as a numpy array.
        :rtype: np.ndarray
        """
        q1 = np.array(q1, dtype=np.float64)
        q2 = np.array(q2, dtype=np.float64)

        # Compute the dot product (cosine of the angle between q1 and q2)
        dot = np.dot(q1, q2)

        # If dot product is negative, negate q2 to ensure shortest path interpolation
        if dot < 0.0:
            q2 = -q2
            dot = -dot

        # Clamp the dot product to prevent numerical errors in arccos
        dot = np.clip(dot, -1.0, 1.0)

        # Compute the angle between the quaternions
        theta_0 = np.arccos(dot)  # Original angle
        theta = theta_0 * t  # Interpolated angle

        # Compute the sine values for interpolation
        sin_theta = np.sin(theta)
        sin_theta_0 = np.sin(theta_0)

        # If the angle is too small, use linear interpolation to avoid division by zero
        if sin_theta_0 < 1e-6:
            return (1 - t) * q1 + t * q2

        # Compute the interpolation coefficients
        s1 = np.sin((1 - t) * theta) / sin_theta_0
        s2 = sin_theta / sin_theta_0

        return s1 * q1 + s2 * q2

    @staticmethod
    def quaternion_to_rotation_matrix(q):
        """
        Konvertiert ein Quaternion in eine 3x3 Rotationsmatrix.
        """
        x, y, z, w = q
        return np.array([
            [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
        ])

    def calculate_bone_positions(self, t_pose):
        """
        Berechnet die absoluten Positionen aller Knochen in der Hierarchie.
        """
        for root_bone in t_pose:
            if "parent" not in t_pose[root_bone]:
                self.compute_absolute_position(root_bone, np.identity(4), t_pose)

    def compute_absolute_position(self, bone_name, parent_matrix, t_pose):
        """
        Rekursive Berechnung der absoluten Position eines Knochens basierend auf der Transformation des Elternknochens.
        """
        if bone_name not in t_pose:
            return np.array([0, 0, 0])

        bone = t_pose[bone_name]
        local_matrix = np.identity(4)
        local_matrix[:3, 3] = bone.get("translation", np.zeros(3))
        rotation_matrix = self.quaternion_to_rotation_matrix(bone.get("rotation", [0, 0, 0, 1]))
        local_matrix[:3, :3] = rotation_matrix

        absolute_matrix = np.dot(parent_matrix, local_matrix)
        bone["position"] = absolute_matrix[:3, 3]

        for child in bone.get("children", []):
            self.compute_absolute_position(child, absolute_matrix, t_pose)

    @staticmethod
    def sort_skeleton(tmp_t_pose):
        """Sortiert das Skelett, um eine korrekte Hierarchie sicherzustellen."""
        back = {}
        while len(back) < len(tmp_t_pose):
            for key in tmp_t_pose.keys():
                if key not in back:
                    if "parent" in tmp_t_pose[key]:
                        if tmp_t_pose[key]["parent"] in back:
                            back[key] = tmp_t_pose[key]
                    else:
                        back[key] = tmp_t_pose[key]
        return back