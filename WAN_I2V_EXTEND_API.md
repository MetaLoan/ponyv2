# WAN2.2 Video Extension API Documentation

This document describes how to use the `wan2_2_i2v_extend_any_frame` mode to generate long, multi-segment videos with seamless transitions and AI-driven prompt decomposition.

## Endpoint
**POST** `https://api.runpod.ai/v2/<YOUR_ENDPOINT_ID>/run`

## Authentication
Include your RunPod API Key in the header:
`Authorization: Bearer <YOUR_API_KEY>`

---

## Input Parameters (Payload)

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `mode` | string | **Yes** | Must be `"wan2_2_i2v_extend_any_frame"` |
| `startimg` | string | **Yes** | URL or Base64 data of the starting frame image. |
| `prompt` | string | **Yes** | The main scene description or narrative. |
| `frames` | integer | No | Total frames for the final video. Default is 81. (Note: ~16 frames = 1 second). |
| `i2v_resolution` | string | No | Resolution and aspect ratio. Supports `"720P"`, `"480*832"`, `"1280*720"`, etc. Default is `"720P"`. |
| `auto_segment_prompts` | boolean | No | If `true`, the AI will automatically decompose the main prompt into logical sub-actions for each segment. |
| `seed` | integer | No | Random seed for the first segment. Subsequent segments use `seed + index`. |
| `negative_prompt` | string | No | Tokens to exclude from generation. |
| `endimg` | string | No | Optional target end frame (only applies to the final segment). |
| `wan_seeds` | array[int] | No | Optional array of specific seeds for each segment. |
| `wan_prompts` | array[str] | No | Optional array of specific prompts for each segment. |

---

## Request Example (cURL)

```bash
curl -X POST "https://api.runpod.ai/v2/<YOUR_ENDPOINT_ID>/run" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer <YOUR_API_KEY>" \
     -d '{
  "input": {
    "mode": "wan2_2_i2v_extend_any_frame",
    "prompt": "A student putting on shoes, tying laces, and standing up to walk.",
    "startimg": "https://example.com/start_frame.jpg",
    "frames": 241,
    "i2v_resolution": "720*1280",
    "auto_segment_prompts": true,
    "seed": 12345
  }
}'
```

## Response Format

```json
{
  "id": "job_id_abc123",
  "status": "COMPLETED",
  "output": {
    "ok": true,
    "mode": "wan2_2_i2v_extend_any_frame",
    "request_id": "req_uuid_xyz",
    "segment_count": 2,
    "final_video_url": "https://r2-cdn.com/outputs/final_video.mp4",
    "segment_video_urls": [
      "https://r2-cdn.com/outputs/segments/01_part.mp4",
      "https://r2-cdn.com/outputs/segments/02_part.mp4"
    ]
  }
}
```

---

## Advanced Logic Notes

### 1. Segmentation Strategy
The system automatically splits requests into segments based on a limit (default 161 frames / ~10 seconds).
*   Segment 1: Uses `startimg` as the first frame.
*   Segment N: Automatically extracts the last frame of Segment N-1 and uses it as the starting frame for Segment N.

### 2. AI Prompt Decomposition (`auto_segment_prompts`)
When enabled, the backend uses a Qwen-Plus model to analyze the main `prompt` and generate a sequence of sub-prompts.
*   **Input**: "A student putting on shoes"
*   **AI Decomposition**: 
    1. "Close up of hands reaching for sneakers."
    2. "The student pulling the laces tight."
    3. "The student standing up and taking a first step."

### 3. Merging
All segments are concatenated using `ffmpeg` on the worker before uploading, ensuring a single, continuous MP4 file is returned as `final_video_url`.
