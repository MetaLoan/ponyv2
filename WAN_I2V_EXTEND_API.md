# WAN2.2 视频扩展 API 文档

本文档描述如何使用 `wan2_2_i2v_extend_any_frame` 模式，基于首帧图片生成多段无缝衔接的长视频，并支持 AI 自动拆分提示词和预生成换脸功能。

## 接口地址
**POST** `https://api.runpod.ai/v2/<YOUR_ENDPOINT_ID>/run`

## 认证方式
在请求头中携带 RunPod API Key：
`Authorization: Bearer <YOUR_API_KEY>`

---

## 请求参数

### 基础参数

| 参数 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| `mode` | string | **是** | 必须为 `"wan2_2_i2v_extend_any_frame"` |
| `startimg` | string | 条件必填 | 首帧图片（支持公开 URL 或 Base64 编码）。如果提供了 `startvideo` 则自动从视频末帧提取 |
| `startvideo` | string | 否 | 前置视频（支持 URL 或 Base64 编码）。提供后会自动提取最后一帧作为生成起点，生成的视频会与原视频合并输出，分辨率以视频为准不可修改 |
| `prompt` | string | **是** | 主场景描述/叙事内容 |
| `frames` | integer | 否 | 总帧数，默认 81（约 16 帧 = 1 秒） |
| `i2v_resolution` | string | 否 | 分辨率，支持 `"720P"`、`"480*832"`、`"832*480"`、`"1280*720"` 等，默认 `"720P"`。**注意：提供 `startvideo` 时此参数自动锁定为视频分辨率** |
| `negative_prompt` | string | 否 | 负面提示词，排除不希望出现的内容 |
| `seed` | integer | 否 | 首段随机种子，后续段使用 `seed + index` |
| `endimg` | string | 否 | 可选的结束帧图片（仅应用于最后一段） |
| `async` | boolean | 否 | 是否异步执行，默认 `true` |

### 多段视频参数

| 参数 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| `segment_limit` | integer | 否 | 每段帧数上限，默认 161（约 10 秒） |
| `auto_segment_prompts` | boolean | 否 | 是否启用 AI 自动将主提示词拆分为多段子提示词 |
| `wan_seeds` | array[int] | 否 | 手动指定每段的随机种子 |
| `wan_prompts` | array[str] | 否 | 手动指定每段的提示词 |

### 模型与采样参数

| 参数 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| `wan_unet_high_name` | string | 否 | UNet High 模型名称，默认 `"WAN2.2-NSFW-FastMove-V2-H.safetensors"` |
| `wan_unet_low_name` | string | 否 | UNet Low 模型名称，默认 `"WAN2.2-NSFW-FastMove-V2-L.safetensors"` |
| `wan_vae_name` | string | 否 | VAE 模型名称，默认 `"wan_2.1_vae.safetensors"` |
| `wan_clip_vision_name` | string | 否 | CLIP Vision 模型名称，默认 `"clip_vision_h.safetensors"` |
| `wan_clip_name` | string | 否 | CLIP 文本模型名称，默认 `"umt5_xxl_fp8_e4m3fn_scaled.safetensors"` |
| `base_steps` | integer | 否 | 基础采样步数，默认 4 |
| `steps` | integer | 否 | 精细采样步数，默认 4 |
| `base_cfg` | float | 否 | 基础 CFG 引导比例，默认 2.0 |
| `cfg` | float | 否 | 精细 CFG 引导比例，默认 1.0 |

### 预生成换脸参数（Qwen Face Swap）

| 参数 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| `wan_face_swap` | boolean | 否 | 是否在生成视频前对首帧进行换脸处理 |
| `face_image` | string | 否 | 脸部参考图片（支持 URL 或 Base64），当 `wan_face_swap=true` 时必填 |
| `wan_face_swap_prompt` | string | 否 | 换脸提示词，描述如何融合两张图片的面部特征 |

> **换脸逻辑说明**：
> - **图1**（`startimg`）= 底图，保留人物姿势、构图、服装、光照、背景
> - **图2**（`face_image`）= 脸部来源，仅提取面部特征进行融合
> - 换脸成功后，替换后的首帧将用于后续视频生成
> - 换脸失败时（如 API 异常），系统会自动降级使用原始首帧继续生成视频
> - 换脸后的首帧图片会作为中间产物上传到 S3 并返回在 `intermediate_urls` 中

---

## 请求示例

### 基础用法（cURL）

```bash
curl -X POST "https://api.runpod.ai/v2/<YOUR_ENDPOINT_ID>/run" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer <YOUR_API_KEY>" \
     -d '{
  "input": {
    "mode": "wan2_2_i2v_extend_any_frame",
    "prompt": "一个学生穿上鞋子，系好鞋带，站起来走路。",
    "startimg": "https://example.com/start_frame.jpg",
    "frames": 241,
    "i2v_resolution": "480*832",
    "auto_segment_prompts": true,
    "seed": 12345
  }
}'
```

### 带换脸的用法

```bash
curl -X POST "https://api.runpod.ai/v2/<YOUR_ENDPOINT_ID>/run" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer <YOUR_API_KEY>" \
     -d '{
  "input": {
    "mode": "wan2_2_i2v_extend_any_frame",
    "prompt": "一个女孩在沙滩上行走",
    "startimg": "https://example.com/start_frame.jpg",
    "face_image": "https://example.com/face_reference.jpg",
    "wan_face_swap": true,
    "wan_face_swap_prompt": "以图1为最终画面底图，严格保留图1的人物姿势、构图、服装、光照、背景；仅将图2中的面部特征自然融合到图1人物脸上，保持真实自然、五官清晰、肤质统一。",
    "frames": 81,
    "i2v_resolution": "480*832",
    "base_steps": 4,
    "steps": 4,
    "base_cfg": 2,
    "cfg": 1,
    "seed": 42
  }
}'
```

---

## 响应格式

```json
{
  "id": "job_id_abc123",
  "status": "COMPLETED",
  "output": {
    "ok": true,
    "mode": "wan2_2_i2v_extend_any_frame",
    "request_id": "req_uuid_xyz",
    "final_video_url": "https://r2-cdn.com/outputs/final_video.mp4",
    "final_video_urls": ["https://r2-cdn.com/outputs/final_video.mp4"],
    "segment_video_urls": [
      "https://r2-cdn.com/outputs/segments/01_part.mp4",
      "https://r2-cdn.com/outputs/segments/02_part.mp4"
    ],
    "intermediate_urls": [
      "https://r2-cdn.com/intermediate/req_uuid/swapped_start.png"
    ]
  }
}
```

### 响应字段说明

| 字段 | 说明 |
| :--- | :--- |
| `final_video_url` | 合并后的完整视频 URL |
| `final_video_urls` | 最终视频 URL 数组（兼容多输出格式） |
| `segment_video_urls` | 各分段视频的 URL 列表 |
| `intermediate_urls` | 中间产物 URL（如换脸后的首帧图片） |

---

## 高级逻辑说明

### 1. 分段策略
系统根据 `segment_limit`（默认 161 帧 / 约 10 秒）自动拆分请求：
- **第 1 段**：使用 `startimg` 作为首帧（如开启换脸，使用换脸后的图片）
- **第 N 段**：自动提取第 N-1 段的最后一帧作为第 N 段的首帧
- 所有分段最终通过 `ffmpeg` 合并为一个连续的 MP4 文件

### 2. AI 提示词拆分（`auto_segment_prompts`）
启用后，后端使用 Qwen-Plus 模型分析主提示词，自动为每一段生成子提示词：
- **输入**：`"一个学生穿上鞋子"`
- **AI 拆分结果**：
  1. `"特写镜头：手伸向运动鞋"`
  2. `"学生系紧鞋带"`
  3. `"学生站起来迈出第一步"`

### 3. 预生成换脸（`wan_face_swap`）
启用后，在视频生成之前先对首帧进行面部替换：
1. 将首帧图片（图1）和脸部图片（图2）上传到可访问的 URL
2. 调用 DashScope Qwen 图像编辑 API 进行换脸
3. 换脸成功后，用新首帧替代原始首帧进行视频生成
4. 换脸后的图片同时上传到 S3 作为中间产物返回
5. 如果换脸失败，自动降级使用原始首帧，不影响视频生成

### 4. 视频续写（`startvideo`）
提供一个已有视频后，系统会自动：
1. 使用 `ffprobe` 检测视频的分辨率（宽 × 高）
2. 使用 `ffmpeg` 提取视频最后一帧作为生成起始图
3. 强制将生成分辨率设置为视频分辨率（忽略 `i2v_resolution` 参数）
4. 生成新的视频段后，将**原始视频 + 新生成段**合并为一个连续 MP4
5. 如果同时启用了换脸，换脸操作在提取的末帧上进行

> **注意**：提供 `startvideo` 时，可以不提供 `startimg`（系统自动从视频提取）。如果同时提供了 `startimg` 和 `startvideo`，以 `startvideo` 的末帧为准。

### 5. 视频合并
所有分段在 Worker 上通过 `ffmpeg` 拼接后上传，确保返回的 `final_video_url` 是一个完整、连续的 MP4 文件。
