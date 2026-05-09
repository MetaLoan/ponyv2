import re
import os

with open('/Users/jack/Documents/gitcode/serveless-allinone/app/handler.py', 'r') as f:
    content = f.read()

funcs = """
def _generate_wan_dwpose_extract(data: dict, request_id: str, event: dict = None) -> dict:
    from utils import load_json, resolve_media_to_comfy_filename, upload_to_s3
    from handler import COMFY_OUTPUT_DIR, queue_prompt, wait_history, _check_cancelled, _register_active_prompt, _unregister_active_prompt, _summarize_history
    import tempfile, subprocess
    
    prompt = load_json("/workspace/runpod-slim/ComfyUI/wan2_2_dwpose_extract_api.json")
    
    # 1. Resolve video
    video_url = data.get("video_url")
    if not video_url:
        raise ValueError("video_url is required for dwpose extract")
    video_filename = resolve_media_to_comfy_filename(video_url, "video")
    prompt["1"]["inputs"]["video"] = video_filename
    prompt["3"]["inputs"]["filename_prefix"] = f"dwpose_{request_id}"
    
    _check_cancelled(request_id)
    prompt_id = queue_prompt(prompt)
    _register_active_prompt(request_id, prompt_id)
    try:
        history_obj = wait_history(prompt_id, event=event, request_id=request_id)
    finally:
        _unregister_active_prompt(request_id)
        
    outputs = history_obj.get("outputs", {})
    output_mp4 = None
    for nid, nout in outputs.items():
        if "gifs" in nout:
            for vid_info in nout["gifs"]:
                fname = vid_info.get("filename", "")
                filepath = COMFY_OUTPUT_DIR / vid_info.get("subfolder", "") / fname
                if filepath.exists():
                    output_mp4 = filepath
                    break
    
    if not output_mp4:
        raise RuntimeError("No output mp4 generated for dwpose extract")
        
    s3_key = f"outputs/{request_id}/dwpose_{request_id}.mp4"
    final_url = upload_to_s3(output_mp4, s3_key)
    
    return {
        "ok": True,
        "video_url": final_url,
        "request_id": request_id,
        "prompt_id": prompt_id
    }


def _generate_wan_animate(data: dict, request_id: str, event: dict = None) -> dict:
    from utils import load_json, resolve_media_to_comfy_filename, upload_to_s3
    from handler import COMFY_OUTPUT_DIR, queue_prompt, wait_history, _check_cancelled, _register_active_prompt, _unregister_active_prompt, _summarize_history
    import tempfile, subprocess
    
    prompt = load_json("/workspace/runpod-slim/ComfyUI/wan2_2_animate_api.json")
    
    # Inputs
    video_url = data.get("action_video_url")
    if not video_url:
        raise ValueError("action_video_url is required")
    char_img = data.get("character_image_url")
    if not char_img:
        raise ValueError("character_image_url is required")
        
    width = int(data.get("width", 720))
    height = int(data.get("height", 1280))
    
    prompt["72"]["inputs"]["video"] = resolve_media_to_comfy_filename(video_url, "video")
    prompt["55"]["inputs"]["image"] = resolve_media_to_comfy_filename(char_img, "image")
    prompt["122"]["inputs"]["text"] = data.get("prompt", "这个角色在跳舞")
    prompt["81"]["inputs"]["filename_prefix"] = f"wan_animate_{request_id}"
    
    # Resolution
    prompt["14"]["inputs"]["image_gen_width"] = width
    prompt["14"]["inputs"]["image_gen_height"] = height
    prompt["39"]["inputs"]["width"] = width
    prompt["39"]["inputs"]["height"] = height
    prompt["84"]["inputs"]["width"] = width
    prompt["84"]["inputs"]["height"] = height
    
    _check_cancelled(request_id)
    prompt_id = queue_prompt(prompt)
    _register_active_prompt(request_id, prompt_id)
    try:
        history_obj = wait_history(prompt_id, event=event, request_id=request_id)
    finally:
        _unregister_active_prompt(request_id)
        
    outputs = history_obj.get("outputs", {})
    output_mp4 = None
    for nid, nout in outputs.items():
        if "gifs" in nout:
            for vid_info in nout["gifs"]:
                fname = vid_info.get("filename", "")
                filepath = COMFY_OUTPUT_DIR / vid_info.get("subfolder", "") / fname
                if filepath.exists():
                    output_mp4 = filepath
                    break
    
    if not output_mp4:
        raise RuntimeError("No output mp4 generated for animate")
        
    s3_key = f"outputs/{request_id}/wan_animate_{request_id}.mp4"
    final_url = upload_to_s3(output_mp4, s3_key)
    
    return {
        "ok": True,
        "video_url": final_url,
        "request_id": request_id,
        "prompt_id": prompt_id
    }

"""

content = content.replace("def _generate_wan_extend_any_frame_comfy", funcs + "\ndef _generate_wan_extend_any_frame_comfy")

routing = """    if data["mode"] == "wan2_2_dwpose_extract":
        return _generate_wan_dwpose_extract(data, request_id, event=event)
    if data["mode"] == "wan2_2_animate":
        return _generate_wan_animate(data, request_id, event=event)

    prompt = load_json(WORKFLOW_API_PATH)"""

content = content.replace("    prompt = load_json(WORKFLOW_API_PATH)", routing)

with open('/Users/jack/Documents/gitcode/serveless-allinone/app/handler.py', 'w') as f:
    f.write(content)

