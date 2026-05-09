import json

with open('/Users/jack/Documents/gitcode/serveless-allinone/workflows/wan2_2_animate_api.json', 'r') as f:
    base = json.load(f)

# 1. Create wan2_2_dwpose_extract_api.json
extract = {}
for node_id in ["14", "16", "33", "63", "72", "113", "115", "120"]:
    if node_id in base:
        extract[node_id] = base[node_id]

# Add two VHS_VideoCombine nodes for saving the outputs
extract["201"] = {
    "class_type": "VHS_VideoCombine",
    "inputs": {
        "frame_rate": ["115", 0],
        "loop_count": 0,
        "filename_prefix": "dwpose_stickman",
        "format": "video/h264-mp4",
        "pix_fmt": "yuv420p",
        "crf": 19,
        "save_metadata": False,
        "trim_to_audio": False,
        "pingpong": False,
        "save_output": True,
        "images": ["33", 0]
    }
}

extract["202"] = {
    "class_type": "VHS_VideoCombine",
    "inputs": {
        "frame_rate": ["115", 0],
        "loop_count": 0,
        "filename_prefix": "dwpose_face",
        "format": "video/h264-mp4",
        "pix_fmt": "yuv420p",
        "crf": 19,
        "save_metadata": False,
        "trim_to_audio": False,
        "pingpong": False,
        "save_output": True,
        "images": ["63", 0]
    }
}

with open('/Users/jack/Documents/gitcode/serveless-allinone/workflows/wan2_2_dwpose_extract_api.json', 'w') as f:
    json.dump(extract, f, indent=2)

# 2. Update wan2_2_animate_api.json for Generation Mode
animate = json.loads(json.dumps(base))

# Delete extraction nodes
for node_id in ["14", "16", "33", "63", "72", "113", "115", "120"]:
    if node_id in animate:
        del animate[node_id]

# Add VHS_LoadVideo for pose (stickman)
animate["301"] = {
    "class_type": "VHS_LoadVideo",
    "inputs": {
        "video": "stickman.mp4",
        "force_rate": 0,
        "force_size": "Disabled",
        "custom_width": 0,
        "custom_height": 0,
        "frame_load_cap": 0,
        "skip_first_frames": 0,
        "select_every_nth": 1,
        "format": "AnimateDiff"
    }
}

# Add VHS_LoadVideo for face crop
animate["302"] = {
    "class_type": "VHS_LoadVideo",
    "inputs": {
        "video": "face.mp4",
        "force_rate": 0,
        "force_size": "Disabled",
        "custom_width": 0,
        "custom_height": 0,
        "frame_load_cap": 0,
        "skip_first_frames": 0,
        "select_every_nth": 1,
        "format": "AnimateDiff"
    }
}

# Fix WanVideoAnimateEmbeds (node 84) to use these new inputs
animate["84"]["inputs"]["pose_images"] = ["301", 0]
animate["84"]["inputs"]["face_images"] = ["302", 0]

# Fix number of frames. We can use the frame count from the pose video.
animate["84"]["inputs"]["num_frames"] = ["301", 1]

# Fix VideoCombine audio input
if "audio" in animate["81"]["inputs"]:
    del animate["81"]["inputs"]["audio"]
animate["81"]["inputs"]["frame_rate"] = 16
animate["81"]["inputs"]["trim_to_audio"] = False

# Fix any PurgeVRAM that depended on node 33 (DWPose). Node 90 used 33.
if "90" in animate:
    del animate["90"]

with open('/Users/jack/Documents/gitcode/serveless-allinone/workflows/wan2_2_animate_api.json', 'w') as f:
    json.dump(animate, f, indent=2)

print("Split completed successfully!")
