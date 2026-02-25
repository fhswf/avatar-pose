import json
import numpy as np
import torch
from pathlib import Path
from copy import deepcopy

from config import DEVICE, BONES, NUM_KEYPOINTS
from model import PoseMLP
from dataset import blender_to_synthetic_coco_keypoints
from rotations import axis_angle_to_quat
from infer import make_blender_pose_json

# Specific file selected for this test
SOURCE_FILE = Path("DatasetBlender/pose_0001_seed1000_open_hand_20260113_202427.json")

def strip_quaternions(json_data):
    """
    Removes rotation_quaternion from bones to simulate inference condition
    where we don't have the ground truth rotations.
    """
    data = deepcopy(json_data)
    bones = data.get("bones", {})
    
    for bone_name, bone_data in bones.items():
        # Strip from local channel (this is what counts for 'input' logic usuallly, 
        # though our input logic uses world_space positions which remain).
        if "local" in bone_data:
            if "rotation_quaternion" in bone_data["local"]:
                del bone_data["local"]["rotation_quaternion"]
        
        # Strip from world_space as well just to be clean, 
        # although dataset.py uses world_space HEAD/TAIL positions, not rotations.
        if "world_space" in bone_data:
            if "rotation_quaternion" in bone_data["world_space"]:
                del bone_data["world_space"]["rotation_quaternion"]
                
    return data

def main():
    if not SOURCE_FILE.exists():
        print(f"Error: File {SOURCE_FILE} not found.")
        return

    print(f"Loading {SOURCE_FILE}...")
    with open(SOURCE_FILE, "r", encoding="utf-8") as f:
        original_data = json.load(f)

    # 1. Strip Quaternions
    print("Stripping quaternions...")
    stripped_data = strip_quaternions(original_data)
    
    # Save stripped file for verification (optional)
    stripped_path = Path("test_input_stripped.json")
    with open(stripped_path, "w", encoding="utf-8") as f:
        json.dump(stripped_data, f, indent=2)
    print(f"Saved stripped input to {stripped_path}")

    # 2. Generate Model Input
    # dataset.py uses 'world_space' -> 'head'/'tail' positions which are still present.
    print("Generating model input...")
    kp_input = blender_to_synthetic_coco_keypoints(stripped_data) # (133, 3)
    x_input = kp_input.flatten().astype(np.float32) # (399,)
    
    # 3. Load Model
    model_path = Path("pose_mlp.pt")
    if not model_path.exists():
        print(f"Error: Model {model_path} not found.")
        return

    print(f"Loading model from {model_path}...")
    input_dim = NUM_KEYPOINTS * 3
    output_dim = len(BONES) * 3
    
    model = PoseMLP(input_dim=input_dim, output_dim=output_dim).to(DEVICE)
    sd = torch.load(str(model_path), map_location=DEVICE)
    model.load_state_dict(sd)
    model.eval()

    # 4. Run Inference
    print("Running inference...")
    with torch.no_grad():
        x_tensor = torch.tensor(x_input, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        aa_out = model(x_tensor).squeeze(0).cpu() # (len(BONES) * 3)
        aa_reshaped = aa_out.reshape(len(BONES), 3) # (len(BONES), 3)

        # Convert to Quaternions
        q_xyzw = axis_angle_to_quat(aa_reshaped).numpy() # (len(BONES), 4)

    # 5. Create Output JSON
    print("Creating output JSON...")
    # Frame index 0 because it's a single file/single frame test
    output_json = make_blender_pose_json(
        quats_xyzw=q_xyzw,
        frame_index=original_data.get("metadata", {}).get("frame_index", 0),
        source_file=SOURCE_FILE.name,
        armature_name=original_data.get("metadata", {}).get("armature_name", "Armature")
    )

    # 6. Save Result
    output_filename = f"testprediction_{SOURCE_FILE.name}"
    output_path = Path(output_filename)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2)
    
    print(f"Success! Output saved to: {output_path.resolve()}")

if __name__ == "__main__":
    main()
