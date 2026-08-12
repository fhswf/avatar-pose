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

        # Sort skeleton to guarantee topological ordering (parents before children)
        self.sorted_bones = self.sort_skeleton(self.t_pose)

        # Compute rest global transformations for T-pose
        self.init_rest_transforms()

    def init_rest_transforms(self):
        """
        Pre-computes the global 4x4 matrices, world positions, and global rotations
        for all bones in rest state (T-pose). Also populates 'position' in self.t_pose entries.
        """
        self.global_rest_matrices = {}
        self.global_rest_positions = {}
        self.global_rest_rotations = {}

        for bone_name, bone in self.sorted_bones.items():
            local_trans = bone.get("translation", np.zeros(3))
            local_rot_quat = bone.get("rotation", np.array([0, 0, 0, 1]))

            # Ensure rotation quaternion is normalized
            if np.linalg.norm(local_rot_quat) > 1e-6:
                local_rot_quat = np.array(local_rot_quat, dtype=np.float64) / np.linalg.norm(local_rot_quat)
            else:
                local_rot_quat = np.array([0, 0, 0, 1], dtype=np.float64)

            local_mat = np.identity(4)
            local_mat[:3, :3] = R.from_quat(local_rot_quat).as_matrix()
            local_mat[:3, 3] = local_trans

            parent_name = bone.get("parent")
            if parent_name and parent_name in self.global_rest_matrices:
                global_mat = np.dot(self.global_rest_matrices[parent_name], local_mat)
            else:
                global_mat = local_mat

            self.global_rest_matrices[bone_name] = global_mat
            self.global_rest_positions[bone_name] = global_mat[:3, 3]
            self.global_rest_rotations[bone_name] = R.from_matrix(global_mat[:3, :3])

            # Populate position field in self.t_pose for backward compatibility
            bone["position"] = global_mat[:3, 3]

    @staticmethod
    def quaternion_from_two_vectors(v_start, v_target):
        """
        Computes the rotation quaternion to rotate v_start to v_target.

        :param v_start: Initial vector.
        :type v_start: np.ndarray
        :param v_target: Target vector.
        :type v_target: np.ndarray
        :return: Quaternion representing the rotation [x, y, z, w].
        :rtype: np.ndarray
        """
        norm_start = np.linalg.norm(v_start)
        norm_target = np.linalg.norm(v_target)
        if norm_start < 1e-6 or norm_target < 1e-6:
            return np.array([0, 0, 0, 1])  # Identity quaternion fallback

        v_start = v_start / norm_start
        v_target = v_target / norm_target
        dot = np.dot(v_start, v_target)

        if dot > 0.999999:
            return np.array([0, 0, 0, 1])

        if dot < -0.999999:
            axis = np.cross(v_start, np.array([1, 0, 0]))
            if np.linalg.norm(axis) < 1e-6:
                axis = np.cross(v_start, np.array([0, 1, 0]))
            axis /= np.linalg.norm(axis)
            return R.from_rotvec(np.pi * axis).as_quat()

        axis = np.cross(v_start, v_target)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-6:
            return np.array([0, 0, 0, 1])

        angle = np.arccos(np.clip(dot, -1.0, 1.0))
        return R.from_rotvec(angle * axis / axis_norm).as_quat()

    @staticmethod
    def convert_to_threejs_coords(vector):
        """
        Converts coordinates to Three.js format. Raw PEV coordinates have +Y pointing UP.
        """
        return np.array([vector["x"], vector["y"], vector["z"]])

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
        Retrieves the target position for a bone from POE data based on mapping,
        handling side-specific X-axis alignment (Left arm +X vs Right arm -X).

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
            return np.array([0.0, 0.0, 0.0])

        if isinstance(target_ids, list):
            positions = [self.get_point_position(target_id, poe_data) for target_id in target_ids]
            positions = [pos for pos in positions if np.linalg.norm(pos) > 1e-6]
            if positions:
                pos = np.mean(positions, axis=0)
            else:
                pos = np.array([0.0, 0.0, 0.0])
        else:
            pos = self.get_point_position(target_ids, poe_data)

        # X-axis alignment:
        # In the PEV coordinate system (mirrored camera), both arms extend in the same
        # raw X direction, but the Avatar's T-pose has _l at +X and _r at -X.
        # Therefore we invert X for _l (raw -X → avatar +X) and also invert X for _r
        # when the raw wrist position crosses over to the opposite side of the image
        # (e.g. hand raised towards chest). Invert X consistently for all lateral bones.
        if bone_name.endswith("_l"):
            return np.array([-pos[0], pos[1], pos[2]])
        elif bone_name.endswith("_r"):
            return np.array([-pos[0], pos[1], pos[2]])
        else:
            return pos

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
        Computes the final quaternions for all bones based on Forward Kinematics hierarchy.

        :param poe_data: List of position data.
        :type poe_data: list
        :return: Dictionary of final rotations per bone.
        :rtype: dict
        """
        shifted_poe = self.shift_origin_poe_data(poe_data)
        final_rotations = {}
        global_pose_rotations = {}

        for bone_name, bone in self.sorted_bones.items():
            if bone_name == "KIM_caucasian_male.body":
                continue

            parent_name = bone.get("parent")
            first_child_name = bone["children"][0] if bone.get("children") else None

            # Retrieve rest global and local rotations
            rest_global_rot = self.global_rest_rotations[bone_name]
            rest_local_quat = bone.get("rotation", np.array([0, 0, 0, 1]))
            if np.linalg.norm(rest_local_quat) > 1e-6:
                rest_local_quat = np.array(rest_local_quat, dtype=np.float64) / np.linalg.norm(rest_local_quat)
            else:
                rest_local_quat = np.array([0, 0, 0, 1], dtype=np.float64)
            rest_local_rot = R.from_quat(rest_local_quat)

            parent_global_pose_rot = global_pose_rotations.get(parent_name, R.identity())

            # Check if bone is ignored or unmapped
            is_ignored = self.mapping.get(bone_name, {}).get("ignore", False)

            if is_ignored:
                local_pose_rot = rest_local_rot
                global_pose_rot = parent_global_pose_rot * local_pose_rot
            else:
                head_pos = self.global_rest_positions[bone_name]
                if first_child_name:
                    tail_pos = self.global_rest_positions[first_child_name]
                else:
                    tail_pos = head_pos

                start_vector = tail_pos - head_pos
                target_head_pos = self.get_target_position(bone_name, shifted_poe, point_type="head")
                target_tail_pos = self.get_target_position(bone_name, shifted_poe, point_type="tail")
                target_vector = target_tail_pos - target_head_pos

                if np.linalg.norm(start_vector) < 1e-6 or np.linalg.norm(target_vector) < 1e-6:
                    local_pose_rot = rest_local_rot
                    global_pose_rot = parent_global_pose_rot * local_pose_rot
                else:
                    # World delta rotation from start_vector to target_vector
                    delta_quat = self.quaternion_from_two_vectors(start_vector, target_vector)
                    delta_rot = R.from_quat(delta_quat)

                    # World pose rotation
                    global_pose_rot = delta_rot * rest_global_rot

                    # Convert to local rotation relative to parent pose rotation
                    if parent_name and parent_name in global_pose_rotations:
                        local_pose_rot = parent_global_pose_rot.inv() * global_pose_rot
                    else:
                        local_pose_rot = global_pose_rot

            global_pose_rotations[bone_name] = global_pose_rot
            final_rotations[bone_name] = {"rotation": local_pose_rot.as_quat()}

        return final_rotations

    @staticmethod
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

        dot = np.dot(q1, q2)

        if dot < 0.0:
            q2 = -q2
            dot = -dot

        dot = np.clip(dot, -1.0, 1.0)
        theta_0 = np.arccos(dot)
        theta = theta_0 * t

        sin_theta = np.sin(theta)
        sin_theta_0 = np.sin(theta_0)

        if sin_theta_0 < 1e-6:
            return (1 - t) * q1 + t * q2

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