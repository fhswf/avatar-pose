# ============================================================
# AVATAR POSE TOOL (Sign Language)
# - Random pose (upper body) + Export JSON (Local+World)
# - NEW: Import JSON -> Apply pose back to rig (replay pose)
# Blender 4.x compatible
# ============================================================

import bpy
from mathutils import Quaternion, Vector, Euler, Matrix
import math
import random
import json
import os
from datetime import datetime

# ============================================================
# CONFIG
# ============================================================
DEFAULT_ARMATURE_NAME = "KIM_caucasian_male"

# ============================================================
# FINGER PRESETS
# ============================================================
FINGER_PRESETS = {
    "relaxed": {"thumb": (0.30, 0.20, 0.10), "index": (0.10, 0.30, 0.50), "middle": (0.10, 0.40, 0.60),
                "ring": (0.20, 0.50, 0.70), "pinky": (0.30, 0.60, 0.80)},
    "fist": {"thumb": (0.80, 0.90, 0.30), "index": (0.90, 0.90, 0.90), "middle": (0.90, 0.90, 0.90),
             "ring": (0.90, 0.90, 0.90), "pinky": (0.90, 0.80, 0.80)},
    "pointing": {"thumb": (0.20, 0.10, 0.10), "index": (0.00, 0.00, 0.00), "middle": (0.60, 0.70, 0.50),
                 "ring": (0.70, 0.80, 0.60), "pinky": (0.80, 0.90, 0.70)},
    "open_hand": {"thumb": (0.10, 0.10, 0.10), "index": (0.00, 0.00, 0.00), "middle": (0.00, 0.00, 0.00),
                  "ring": (0.00, 0.00, 0.00), "pinky": (0.00, 0.00, 0.00)},
    "grasping": {"thumb": (0.60, 0.70, 0.40), "index": (0.70, 0.80, 0.60), "middle": (0.80, 0.90, 0.70),
                 "ring": (0.90, 0.90, 0.80), "pinky": (0.90, 0.90, 0.80)},
}
FINGER_TYPES = ["thumb", "index", "middle", "ring", "pinky"]

# ============================================================
# LIMITS HELPERS (used for random)
# ============================================================
def _deg_limits(xmin, xmax, ymin, ymax, zmin, zmax):
    return (xmin, xmax, ymin, ymax, zmin, zmax)

def guess_limits_for_bone(bone_name: str):
    n = bone_name.lower()
    if n.endswith("_end") or n.endswith("_tip"):
        return _deg_limits(0, 0, 0, 0, 0, 0)
    if n in {"root", "hips", "pelvis"}:
        return _deg_limits(-8, 8, -8, 8, -10, 10)
    if "spine" in n or "chest" in n or "torso" in n:
        return _deg_limits(-15, 30, -15, 15, -20, 20)
    if "neck" in n:
        return _deg_limits(-40, 20, -30, 30, -40, 40)
    if "head" in n:
        return _deg_limits(-40, 20, -45, 45, -60, 60)
    if "clavicle" in n or "shoulder" in n:
        return _deg_limits(-20, 40, -25, 25, -20, 20)
    if "upperarm" in n or ("arm" in n and "lower" not in n and "forearm" not in n):
        return _deg_limits(-60, 180, -60, 180, -90, 90)
    if "lowerarm" in n or "forearm" in n:
        return _deg_limits(0, 145, -25, 25, -90, 90)
    if "hand" in n or "wrist" in n:
        return _deg_limits(-70, 70, -35, 35, -60, 60)
    if "thigh" in n or "upperleg" in n:
        return _deg_limits(-40, 130, -40, 60, -45, 45)
    if "calf" in n or "shin" in n or "lowerleg" in n:
        return _deg_limits(0, 145, -15, 15, -15, 15)
    if "foot" in n or "ankle" in n:
        return _deg_limits(-35, 35, -25, 25, -45, 45)
    if "ball" in n or "toe" in n:
        return _deg_limits(-45, 45, -10, 10, -10, 10)
    if "metacarpal" in n:
        return _deg_limits(-25, 25, -20, 20, -25, 25)
    if any(f in n for f in ["thumb", "index", "middle", "ring", "pinky"]):
        if "_01" in n:
            return _deg_limits(-20, 95, -20, 20, -20, 20)
        if "_02" in n:
            return _deg_limits(0, 110, -10, 10, -10, 10)
        if "_03" in n:
            return _deg_limits(0, 90, -8, 8, -8, 8)
        return _deg_limits(-15, 80, -10, 10, -10, 10)
    return _deg_limits(-10, 10, -10, 10, -10, 10)

def build_joint_limits_for_armature(arm_obj):
    return {pb.name: guess_limits_for_bone(pb.name) for pb in arm_obj.pose.bones}

# ============================================================
# UPPER BODY FILTER
# ============================================================
LOWER_BODY_KEYWORDS = (
    "thigh", "upperleg", "calf", "shin", "lowerleg",
    "foot", "ankle", "ball", "toe"
)

UPPER_BODY_KEYWORDS = (
    "root", "hips", "pelvis",
    "spine", "chest", "torso",
    "neck", "head",
    "clavicle", "shoulder",
    "upperarm", "lowerarm", "forearm",
    "hand", "wrist",
    "metacarpal", "thumb", "index", "middle", "ring", "pinky"
)

def is_lower_body_bone(pb: bpy.types.PoseBone) -> bool:
    n = pb.name.lower()
    if any(k in n for k in LOWER_BODY_KEYWORDS):
        return True
    p = pb.parent
    while p is not None:
        if any(k in p.name.lower() for k in LOWER_BODY_KEYWORDS):
            return True
        p = p.parent
    return False

def is_upper_body_bone(pb: bpy.types.PoseBone) -> bool:
    if is_lower_body_bone(pb):
        return False
    return any(k in pb.name.lower() for k in UPPER_BODY_KEYWORDS)

# ============================================================
# ROTATION HELPERS
# ============================================================
def ensure_pose_quaternion_mode(arm_obj):
    for pb in arm_obj.pose.bones:
        pb.rotation_mode = 'QUATERNION'

def random_rotation_from_limits(lim, randomness=1.0, finger_bias=False):
    (xmin, xmax, ymin, ymax, zmin, zmax) = lim

    def pick(minv, maxv, scale=1.0, beta_bias=False):
        if minv == maxv:
            return 0.0
        t = random.betavariate(2.0, 3.5) if beta_bias else random.random()
        v = minv + (maxv - minv) * t
        return math.radians(v * scale)

    if finger_bias:
        ax = pick(xmin, xmax, scale=randomness, beta_bias=True)
        ay = pick(ymin, ymax, scale=randomness * 0.35, beta_bias=False)
        az = pick(zmin, zmax, scale=randomness * 0.25, beta_bias=False)
    else:
        ax = pick(xmin, xmax, scale=randomness, beta_bias=False)
        ay = pick(ymin, ymax, scale=randomness, beta_bias=False)
        az = pick(zmin, zmax, scale=randomness, beta_bias=False)

    qx = Quaternion((1, 0, 0), ax)
    qy = Quaternion((0, 1, 0), ay)
    qz = Quaternion((0, 0, 1), az)
    return qx @ qy @ qz

def blend_quaternion(current: Quaternion, target: Quaternion, blend=0.5):
    return current.slerp(target, blend)

# ============================================================
# FINGER POSE
# ============================================================
def get_preset_value(preset_name, finger_type):
    if preset_name in FINGER_PRESETS and finger_type in FINGER_PRESETS[preset_name]:
        return FINGER_PRESETS[preset_name][finger_type]
    if finger_type == "thumb":
        return (0.25, 0.20, 0.10)
    return (0.10, 0.25, 0.40)

def apply_finger_pose(arm_obj, side_token: str, preset_name="relaxed",
                      intensity=0.6, limits=None, blend=0.7):
    if limits is None:
        limits = build_joint_limits_for_armature(arm_obj)

    side_lower = side_token.lower()
    side_upper = side_token.upper()
    possible_suffixes = [f"_{side_lower}", f".{side_upper}"]

    for finger in FINGER_TYPES:
        v1, v2, v3 = get_preset_value(preset_name, finger)

        for suf in possible_suffixes:
            meta_name = f"metacarpal_{finger}{suf}"
            if meta_name in arm_obj.pose.bones:
                pb = arm_obj.pose.bones[meta_name]
                lim = limits.get(meta_name, guess_limits_for_bone(meta_name))
                target = random_rotation_from_limits(lim, randomness=0.6 * intensity, finger_bias=True)
                pb.rotation_quaternion = blend_quaternion(pb.rotation_quaternion, target, blend=blend)

        for joint_i, v in enumerate([v1, v2, v3], start=1):
            for suf in possible_suffixes:
                bone_name = f"{finger}_{joint_i:02d}{suf}"
                if bone_name not in arm_obj.pose.bones:
                    continue

                pb = arm_obj.pose.bones[bone_name]
                lim = limits.get(bone_name, guess_limits_for_bone(bone_name))
                xmin, xmax, ymin, ymax, zmin, zmax = lim

                ax = math.radians(xmin + (xmax - xmin) * max(0.0, min(1.0, v)) * intensity)
                ay = math.radians(random.uniform(ymin, ymax) * 0.25 * intensity)
                az = math.radians(random.uniform(zmin, zmax) * 0.20 * intensity)

                qx = Quaternion((1, 0, 0), ax)
                qy = Quaternion((0, 1, 0), ay)
                qz = Quaternion((0, 0, 1), az)
                target = qx @ qy @ qz

                pb.rotation_quaternion = blend_quaternion(pb.rotation_quaternion, target, blend=blend)

# ============================================================
# RANDOM POSE
# ============================================================
def apply_random_pose(arm_obj,
                      randomness=1.0,
                      use_symmetry=True,
                      include_fingers=True,
                      finger_preset="random",
                      finger_intensity=0.6,
                      blend=0.5,
                      seed=None,
                      upper_body_only=True,
                      allow_pelvis_small=True):
    if seed is None:
        seed = random.randint(0, 10_000_000)
    random.seed(seed)

    ensure_pose_quaternion_mode(arm_obj)
    limits = build_joint_limits_for_armature(arm_obj)

    if finger_preset == "random":
        finger_preset = random.choice(list(FINGER_PRESETS.keys()))

    for pb in arm_obj.pose.bones:
        n = pb.name.lower()
        if n.endswith("_end") or n.endswith("_tip"):
            continue
        if upper_body_only and (not is_upper_body_bone(pb)):
            continue
        if is_lower_body_bone(pb):
            continue

        pb_lim = limits[pb.name]
        if allow_pelvis_small and pb.name.lower() in {"root", "hips", "pelvis"}:
            pb_lim = _deg_limits(-5, 5, -5, 5, -8, 8)

        is_fingerish = any(k in n for k in ["thumb", "index", "middle", "ring", "pinky", "metacarpal"])
        target = random_rotation_from_limits(pb_lim, randomness=randomness, finger_bias=is_fingerish)
        pb.rotation_quaternion = blend_quaternion(pb.rotation_quaternion, target, blend=blend)

    if include_fingers:
        apply_finger_pose(arm_obj, "l", preset_name=finger_preset, intensity=finger_intensity,
                          limits=limits, blend=0.75)
        apply_finger_pose(arm_obj, "r", preset_name=finger_preset, intensity=finger_intensity,
                          limits=limits, blend=0.75)

    bpy.context.view_layer.update()
    return seed

# ============================================================
# EXPORT (kept minimal here - assumes you already have exporter)
# ============================================================
def _vec3(v: Vector):
    return {"x": float(v.x), "y": float(v.y), "z": float(v.z)}

def _quat4(q: Quaternion):
    q = q.copy()
    q.normalize()
    return {"w": float(q.w), "x": float(q.x), "y": float(q.y), "z": float(q.z)}

def _mat4(m: Matrix):
    return [[float(m[r][c]) for c in range(4)] for r in range(4)]

def get_bone_data_local_and_world_evaluated(arm_obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    arm_eval = arm_obj.evaluated_get(depsgraph)
    mw = arm_eval.matrix_world.copy()

    data = {}
    for pb in arm_eval.pose.bones:
        head_a = pb.head.copy()
        tail_a = pb.tail.copy()
        head_w = mw @ head_a
        tail_w = mw @ tail_a

        loc_local = pb.location.copy()
        rot_local = pb.rotation_quaternion.copy()
        scl_local = pb.scale.copy()
        basis = pb.matrix_basis.copy()

        bone_world_matrix = mw @ pb.matrix
        rot_world = bone_world_matrix.to_quaternion()

        parent = pb.parent.name if pb.parent else None
        has_children = len(pb.children) > 0

        data[pb.name] = {
            "name": pb.name,
            "local": {
                "location": _vec3(loc_local),
                "rotation_quaternion": _quat4(rot_local),
                "scale": _vec3(scl_local),
                "matrix_basis": _mat4(basis),
                "rotation_mode": pb.rotation_mode,
            },
            "armature_space": {"head": _vec3(head_a), "tail": _vec3(tail_a)},
            "world_space": {
                "head": _vec3(head_w),
                "tail": _vec3(tail_w),
                "rotation_quaternion": _quat4(rot_world),
                "matrix": _mat4(bone_world_matrix),
            },
            "bone_length_armature_space": float((tail_a - head_a).length),
            "parent": parent,
            "has_children": has_children,
            "is_end_bone": (not has_children),
        }
    return data

def export_bones_to_json(armature_name: str, output_dir=None):
    if armature_name not in bpy.data.objects:
        print(f"❌ Armature '{armature_name}' not found")
        return None

    arm = bpy.data.objects[armature_name]
    bone_data = get_bone_data_local_and_world_evaluated(arm)

    export_data = {
        "metadata": {
            "armature_name": armature_name,
            "export_time": datetime.now().isoformat(),
            "blender_version": bpy.app.version_string,
            "bone_count": len(bone_data),
            "end_bone_count": sum(1 for b in bone_data.values() if b["is_end_bone"]),
            "spaces_included": ["LOCAL_CHANNELS", "ARMATURE_SPACE", "WORLD_SPACE"],
            "note": "LOCAL matches Transform panel; apply back via matrix_basis for best fidelity.",
        },
        "bones": bone_data,
    }

    if output_dir is None:
        output_dir = os.path.join(os.path.expanduser("~"), "Desktop", "Blender_Bone_Exports")
    os.makedirs(output_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_path = os.path.join(output_dir, f"bone_data_full_{ts}.json")

    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Exported bone data JSON: {full_path}")
    return export_data

# ============================================================
# IMPORT / APPLY POSE FROM JSON (NEW)
# ============================================================

def _read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _matrix_from_rows(rows):
    # rows: 4 lists of 4 floats
    m = Matrix.Identity(4)
    for r in range(4):
        for c in range(4):
            m[r][c] = float(rows[r][c])
    return m

def apply_pose_from_json(arm_obj, pose_json: dict,
                         use_matrix_basis=True,
                         upper_body_only=False,
                         ignore_missing=True):
    """
    Applies pose to arm_obj from JSON exported by this tool.
    Best fidelity: use_matrix_basis=True -> pb.matrix_basis = stored matrix
    Fallback: set pb.location, pb.rotation_quaternion, pb.scale
    """
    ensure_pose_quaternion_mode(arm_obj)

    bones_block = pose_json.get("bones", {})
    if not isinstance(bones_block, dict):
        raise ValueError("JSON does not contain a valid 'bones' dict")

    applied = 0
    skipped = 0
    missing = 0

    # IMPORTANT: set in a stable order (parents first) to reduce dependency surprises
    # We'll just iterate in armature pose bones order.
    for pb in arm_obj.pose.bones:
        name = pb.name
        if upper_body_only and (not is_upper_body_bone(pb)):
            continue

        rec = bones_block.get(name)
        if rec is None:
            if ignore_missing:
                missing += 1
                continue
            else:
                raise KeyError(f"Bone '{name}' not in JSON")

        loc = rec.get("local", {}).get("location")
        rot = rec.get("local", {}).get("rotation_quaternion")
        scl = rec.get("local", {}).get("scale")
        basis_rows = rec.get("local", {}).get("matrix_basis")

        try:
            if use_matrix_basis and basis_rows:
                pb.matrix_basis = _matrix_from_rows(basis_rows)
                # rotation_mode stays QUAT; matrix_basis overrides channels effectively
            else:
                # fallback to channels
                if loc:
                    pb.location = Vector((loc["x"], loc["y"], loc["z"]))
                if rot:
                    pb.rotation_mode = 'QUATERNION'
                    pb.rotation_quaternion = Quaternion((rot["w"], rot["x"], rot["y"], rot["z"]))
                if scl:
                    pb.scale = Vector((scl["x"], scl["y"], scl["z"]))
            applied += 1
        except Exception:
            skipped += 1
            if not ignore_missing:
                raise

    bpy.context.view_layer.update()
    return {"applied": applied, "skipped": skipped, "missing": missing}

# ============================================================
# UI HELPERS
# ============================================================

def get_armature_from_scene(context):
    name = context.scene.avatar_armature_name
    if name in bpy.data.objects and bpy.data.objects[name].type == "ARMATURE":
        return bpy.data.objects[name]
    if DEFAULT_ARMATURE_NAME in bpy.data.objects and bpy.data.objects[DEFAULT_ARMATURE_NAME].type == "ARMATURE":
        return bpy.data.objects[DEFAULT_ARMATURE_NAME]
    return None

def set_pose_mode(arm_obj):
    bpy.context.view_layer.objects.active = arm_obj
    if arm_obj.mode != "POSE":
        bpy.ops.object.mode_set(mode="POSE")

# ============================================================
# OPERATORS
# ============================================================

class AVATAR_OT_random_pose(bpy.types.Operator):
    bl_idname = "avatar.random_pose"
    bl_label = "Random Pose (Upper Body)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        arm = get_armature_from_scene(context)
        if not arm:
            self.report({'ERROR'}, "Armature not found")
            return {'CANCELLED'}

        set_pose_mode(arm)

        seed = apply_random_pose(
            arm,
            randomness=context.scene.avatar_randomness,
            include_fingers=context.scene.avatar_include_fingers,
            finger_preset=context.scene.avatar_finger_preset,
            finger_intensity=context.scene.avatar_finger_intensity,
            blend=context.scene.avatar_blend,
            upper_body_only=context.scene.avatar_upper_body_only,
            allow_pelvis_small=context.scene.avatar_allow_pelvis_small,
        )
        self.report({'INFO'}, f"Random pose applied (seed={seed})")
        return {'FINISHED'}

class AVATAR_OT_export_json(bpy.types.Operator):
    bl_idname = "avatar.export_json"
    bl_label = "Export Bone Data to JSON (Local + World)"
    bl_options = {'REGISTER'}

    def execute(self, context):
        arm = get_armature_from_scene(context)
        if not arm:
            self.report({'ERROR'}, "Armature not found")
            return {'CANCELLED'}

        export_bones_to_json(arm.name, output_dir=context.scene.avatar_export_dir.strip() or None)
        self.report({'INFO'}, "Bone JSON exported")
        return {'FINISHED'}

class AVATAR_OT_import_pose_json(bpy.types.Operator):
    bl_idname = "avatar.import_pose_json"
    bl_label = "Import Pose from JSON (Apply to Rig)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        arm = get_armature_from_scene(context)
        if not arm:
            self.report({'ERROR'}, "Armature not found")
            return {'CANCELLED'}

        path = context.scene.avatar_import_json_path.strip()
        if not path or not os.path.exists(path):
            self.report({'ERROR'}, "JSON path not found. Set Import JSON Path.")
            return {'CANCELLED'}

        set_pose_mode(arm)

        try:
            pose_json = _read_json(path)
            stats = apply_pose_from_json(
                arm,
                pose_json,
                use_matrix_basis=context.scene.avatar_import_use_matrix_basis,
                upper_body_only=context.scene.avatar_import_upper_body_only,
                ignore_missing=context.scene.avatar_import_ignore_missing,
            )
        except Exception as e:
            self.report({'ERROR'}, f"Import failed: {e}")
            return {'CANCELLED'}

        self.report({'INFO'}, f"Pose applied: {stats['applied']} bones (missing:{stats['missing']}, skipped:{stats['skipped']})")
        return {'FINISHED'}

class AVATAR_OT_reset_pose(bpy.types.Operator):
    bl_idname = "avatar.reset_pose"
    bl_label = "Reset Pose (Clear Transforms)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        arm = get_armature_from_scene(context)
        if not arm:
            self.report({'ERROR'}, "Armature not found")
            return {'CANCELLED'}
        set_pose_mode(arm)
        bpy.ops.pose.transforms_clear()
        bpy.context.view_layer.update()
        self.report({'INFO'}, "Pose reset")
        return {'FINISHED'}

# ============================================================
# PANEL
# ============================================================

class AVATAR_PT_panel(bpy.types.Panel):
    bl_label = "Avatar Pose & JSON (Replay)"
    bl_idname = "AVATAR_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Avatar"

    def draw(self, context):
        layout = self.layout

        box = layout.box()
        box.label(text="Armature")
        box.prop(context.scene, "avatar_armature_name", text="Name")

        box = layout.box()
        box.label(text="Random Pose")
        box.prop(context.scene, "avatar_upper_body_only", text="Upper Body Only")
        box.prop(context.scene, "avatar_allow_pelvis_small", text="Allow Small Pelvis Motion")
        box.prop(context.scene, "avatar_randomness", text="Randomness")
        box.prop(context.scene, "avatar_blend", text="Blend")
        box.prop(context.scene, "avatar_include_fingers", text="Include Fingers")
        box.prop(context.scene, "avatar_finger_preset", text="Finger Preset")
        box.prop(context.scene, "avatar_finger_intensity", text="Finger Intensity")

        row = box.row()
        row.scale_y = 1.25
        row.operator("avatar.random_pose", text="Apply Random Pose")

        box = layout.box()
        box.label(text="Export")
        box.prop(context.scene, "avatar_export_dir", text="Export Dir")
        row = box.row()
        row.scale_y = 1.2
        row.operator("avatar.export_json", text="Export JSON")

        box = layout.box()
        box.label(text="Import / Replay Pose from JSON")
        box.prop(context.scene, "avatar_import_json_path", text="Import JSON Path")
        box.prop(context.scene, "avatar_import_use_matrix_basis", text="Use Matrix Basis (best)")
        box.prop(context.scene, "avatar_import_upper_body_only", text="Apply Upper Body Only")
        box.prop(context.scene, "avatar_import_ignore_missing", text="Ignore Missing Bones")
        row = box.row(align=True)
        row.scale_y = 1.2
        row.operator("avatar.import_pose_json", text="Apply Pose from JSON")
        row.operator("avatar.reset_pose", text="Reset Pose")

# ============================================================
# REGISTER
# ============================================================

CLASSES = (
    AVATAR_OT_random_pose,
    AVATAR_OT_export_json,
    AVATAR_OT_import_pose_json,
    AVATAR_OT_reset_pose,
    AVATAR_PT_panel,
)

def register():
    for c in CLASSES:
        bpy.utils.register_class(c)

    bpy.types.Scene.avatar_armature_name = bpy.props.StringProperty(
        name="Avatar Armature Name",
        default=DEFAULT_ARMATURE_NAME
    )

    # random pose props
    bpy.types.Scene.avatar_upper_body_only = bpy.props.BoolProperty(name="Upper Body Only", default=True)
    bpy.types.Scene.avatar_allow_pelvis_small = bpy.props.BoolProperty(name="Allow Small Pelvis Motion", default=True)
    bpy.types.Scene.avatar_randomness = bpy.props.FloatProperty(name="Randomness", default=0.8, min=0.0, max=2.0)
    bpy.types.Scene.avatar_blend = bpy.props.FloatProperty(name="Blend", default=0.55, min=0.0, max=1.0)
    bpy.types.Scene.avatar_include_fingers = bpy.props.BoolProperty(name="Include Fingers", default=True)
    bpy.types.Scene.avatar_finger_preset = bpy.props.EnumProperty(
        name="Finger Preset",
        items=[(k, k.title(), "") for k in (["random"] + list(FINGER_PRESETS.keys()))],
        default="random"
    )
    bpy.types.Scene.avatar_finger_intensity = bpy.props.FloatProperty(name="Finger Intensity", default=0.65, min=0.0, max=1.5)

    # export
    bpy.types.Scene.avatar_export_dir = bpy.props.StringProperty(name="Export Directory", default="", subtype='DIR_PATH')

    # import pose
    bpy.types.Scene.avatar_import_json_path = bpy.props.StringProperty(
        name="Import JSON Path",
        default="",
        subtype='FILE_PATH'
    )
    bpy.types.Scene.avatar_import_use_matrix_basis = bpy.props.BoolProperty(
        name="Use Matrix Basis",
        default=True
    )
    bpy.types.Scene.avatar_import_upper_body_only = bpy.props.BoolProperty(
        name="Apply Upper Body Only",
        default=False
    )
    bpy.types.Scene.avatar_import_ignore_missing = bpy.props.BoolProperty(
        name="Ignore Missing Bones",
        default=True
    )

def unregister():
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)

    del bpy.types.Scene.avatar_armature_name
    del bpy.types.Scene.avatar_upper_body_only
    del bpy.types.Scene.avatar_allow_pelvis_small
    del bpy.types.Scene.avatar_randomness
    del bpy.types.Scene.avatar_blend
    del bpy.types.Scene.avatar_include_fingers
    del bpy.types.Scene.avatar_finger_preset
    del bpy.types.Scene.avatar_finger_intensity
    del bpy.types.Scene.avatar_export_dir

    del bpy.types.Scene.avatar_import_json_path
    del bpy.types.Scene.avatar_import_use_matrix_basis
    del bpy.types.Scene.avatar_import_upper_body_only
    del bpy.types.Scene.avatar_import_ignore_missing

if __name__ == "__main__":
    try:
        unregister()
    except Exception:
        pass
    register()
    print("✅ Avatar Pose & JSON Replay Tool registered.")
    print("Open: Viewport → N panel → Avatar tab")
