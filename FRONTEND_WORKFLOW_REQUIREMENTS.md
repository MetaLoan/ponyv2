# V16 Frontend + Workflow 改造需求文档

## 目标

新增一个前端 Web 测试页，并同步改造 V16 workflow / handler，使系统支持五种出图模式、节点级开关和完整参数调节能力。

本文件定义需求边界、交互方式、接口契约和工作流改造范围，不包含具体实现代码。

## 当前现状

当前能力已经验证可运行，但能力边界较窄：

- 已支持：
  - 传 `reference_image` + `prompt`
  - 可选传 `pose_image`
  - Dual-pass 自动姿态模式
  - PuLID 换脸
  - LoRA 参数覆盖
  - `use_upscale` / `enable_upscale` 布尔开关
- 当前限制：
  - `pose_image` 一旦传入，就直接跳过第一阶段
  - 不支持“先按 pose 出图，再换脸”作为独立显式模式
  - 不支持“只传 pose 图，不传 reference_image”
  - 不支持“什么参考图都没有，只传 prompt”
  - LoRA 需要升级为显式多条串联能力
  - 参数控制分散在 handler 注入里，前端没有成体系配置面板

## 本次改造目标

### 功能目标

需要支持以下 8 类能力：

1. 传人像图 + 提示词，完整走完 2 阶段生图
2. 传人像图 + Pose 图解析 + 提示词生图后再换脸
3. 只传 Pose 图解析，根据提示词生图
4. 不传任何参考图，只传提示词直接生图
5. 通过 DashScope Qwen 做换脸后处理
6. 上传 face 图和 pose 图，直接用 Qwen 做融合
7. 串联开关 LoRA 节点、开关 4x 放大节点
8. 调整所有核心节点参数，包括种子、分辨率、批量、权重、步数、采样器、ControlNet 强度等

### 交付目标

本轮后续开发应至少产出：

- 一个前端 Web 测试页
- 一套扩展后的 API 输入契约
- 一份改造后的 workflow JSON
- 一版可解析新参数并改写 workflow 的 handler

## 术语定义

### 模式 A：Dual-Pass Auto Pose

输入：
- `reference_image`
- `prompt`

行为：
- 第一阶段低步数生成中间图
- 从中间图提取 depth + pose
- 第二阶段再注入 PuLID 和 ControlNet，输出最终图

用途：
- 当前默认模式

### 模式 B：Pose-First Then Face-Swap

输入：
- `reference_image`
- `pose_image`
- `prompt`

行为：
- 先以 `pose_image` 为姿态条件进行生图
- 再将 `reference_image` 作为换脸参考注入 PuLID
- 输出最终图

用途：
- 用户希望显式控制姿态，同时保留人脸一致性

### 模式 C：Pose-Only Generation

输入：
- `pose_image`
- `prompt`

行为：
- 使用 `pose_image` 解析 depth + pose
- 不做 PuLID 换脸
- 直接根据提示词生成最终图

用途：
- 只控姿态，不指定人物身份

### 模式 D：Text-Only Generation

输入：
- `prompt`

可选：
- `negative_prompt`

行为：
- 不依赖 `reference_image`
- 不依赖 `pose_image`
- 不做 PuLID
- 直接根据提示词执行最终出图阶段

用途：
- 纯文生图测试
- 校验基础 checkpoint / LoRA / sampler / upscale 配置

### 模式 E：Qwen Swap Face

输入：
- `reference_image`
- `prompt`

可选：
- `qwen_swap_prompt`
- `qwen_model`
- `qwen_size`
- `qwen_extra_image`

行为：
- 先按 `prompt` 完成基础生图
- 再调用 DashScope 千问图像编辑模型做人脸替换
- 输出最终成品图

用途：
- 需要把参考人脸融合进生成图
- 将换脸能力从 ComfyUI 工作流中独立出来

## 前端页面需求

### 页面目标

该页面首先是“测试台”，不是最终商用页面。

要求：
- 低开发复杂度
- 直接服务于参数联调
- 能清晰展示输入、模式、运行状态、输出图和返回 JSON

### 页面结构

建议分成 4 个区域：

1. 输入区
- `reference_image` 上传
- `pose_image` 上传
- `prompt`
- `negative_prompt`

2. 模式区
- 模式选择：
  - `A: 双阶段自动姿态`
  - `B: Pose 生图后换脸`
  - `C: 仅 Pose 生图`
  - `D: 仅提示词生图`

3. 参数区
- 基础参数
- 第一阶段参数
- 最终输出阶段参数
- ControlNet 参数
- PuLID 参数
- LoRA 参数
- Upscale 参数

4. 结果区
- 请求 payload 预览
- 任务状态
- 最终图
- 中间图
- 原始 JSON 响应

### 模式与字段联动

- 选择模式 A：
  - `reference_image` 必填
  - `pose_image` 隐藏或忽略
  - `PuLID` 默认开启
  - `base_*` 参数显示
- 选择模式 B：
  - `reference_image` 必填
  - `pose_image` 必填
  - `PuLID` 默认开启
  - `base_*` 参数隐藏或忽略
- 选择模式 C：
  - `pose_image` 必填
  - `reference_image` 隐藏或不必填
  - `PuLID` 默认关闭且不可开启
  - `base_*` 参数隐藏或忽略
- 选择模式 D：
  - `reference_image` 隐藏或不必填
  - `pose_image` 隐藏或不必填
  - `PuLID` 默认关闭且不可开启
  - `base_*` 参数隐藏或忽略
- 选择模式 E：
  - `reference_image` 必填
  - `pose_image` 隐藏或不必填
  - `PuLID` 默认关闭且不可开启
  - `base_*` 参数隐藏或忽略

### 运行体验

- 点击“生成”后禁用重复提交
- 显示 request id / job id
- 轮询状态并显示：
  - `IN_QUEUE`
  - `IN_PROGRESS`
  - `COMPLETED`
  - `FAILED`
- 失败时展示完整错误 JSON

### 调试体验

- 支持“复制 payload”
- 支持“复制最终 URL”
- 支持切换是否展示中间图

## 参数面板需求

### 一、最终输出阶段参数

- `width`
- `height`
- `batch_size`
- `seed`
- `steps`
- `cfg`
- `denoise`
- `sampler_name`
- `scheduler`

说明：
- `width` / `height` / `batch_size` 默认指向最终输出阶段
- 若开启 4x 放大，最终返回图尺寸会在该基础上继续放大
- 需要在前端清楚区分“第一阶段 seed”和“最终阶段 seed”

### 二、第一阶段参数

仅模式 A 使用：

- `base_seed`
- `base_steps`
- `base_cfg`
- `base_sampler_name`
- `base_scheduler`
- `base_denoise`

说明：
- 模式 A 是两阶段链路，因此允许第一阶段和最终阶段使用不同 seed
- 模式 B / C / D 默认只执行最终阶段，因此 `base_*` 参数应隐藏、置灰或忽略

### 三、PuLID 参数

- `enable_pulid`
- `pulid_weight`
- `pulid_start_at`
- `pulid_end_at`
- `pulid_method`

说明：
- 模式 C / D 中 `enable_pulid` 必须强制为 `false`

### 四、ControlNet 参数

- `cn_depth_strength`
- `cn_depth_start_percent`
- `cn_depth_end_percent`
- `cn_pose_strength`
- `cn_pose_start_percent`
- `cn_pose_end_percent`

说明：
- 模式 A / B / C 需要支持完整 ControlNet 起止百分比调节
- 模式 D 不依赖 pose/depth 条件图，因此相关参数应隐藏或忽略

### 五、LoRA 参数

- `enable_lora`
- `loras[]`
  - `name`
  - `strength_model`
  - `strength_clip`

说明：
- 必须支持多 LoRA 串联
- 前端应支持增删 LoRA 行
- 后端必须保持 LoRA 顺序有意义

### 六、Qwen 换脸参数

- `qwen_swap_prompt`
- `qwen_model`
- `qwen_size`

说明：
- `qwen_swap_prompt` 用于描述换脸融合要求，默认会强调只融合脸部、不改变图1的沙滩环境
- `qwen_pose_fusion` 直接复用 `prompt` 作为融合模板，默认会强调保留 pose 图并只融合脸部
- `qwen_model` 默认 `qwen-image-2.0-pro`
- `qwen_size` 可选，格式为 `宽*高`
- `qwen_extra_image` 是可选第三张图，用于扩展融合参考
- 模式 E 下前端应展示这些参数

### 七、4x 放大参数

- `enable_upscale`
- `upscale_model_name`
- `output_format`
- `jpg_quality`

说明：
- `enable_upscale` 必须是真正节点链开关，不是仅改变返回格式

## 模型列表动态获取需求

### 目标

前端不再写死：

- 大模型列表
- LoRA 列表
- 4x 放大模型列表

而是由后端动态读取对象存储中的模型目录并返回给前端。

## 已定位的对象存储目录

基于本地凭据文件 [s3-credentials.txt](/Users/leo/Desktop/sdxl2img/s3-credentials.txt) 对 RunPod S3 兼容桶的实际检查，当前模型路径在：

- 桶：`hqvl05xyne`
- endpoint：`https://s3api-eu-ro-1.runpod.io`
- 根前缀：`runpod-slim/ComfyUI/models/`

已确认存在的关键目录：

- `runpod-slim/ComfyUI/models/checkpoints/`
- `runpod-slim/ComfyUI/models/loras/`
- `runpod-slim/ComfyUI/models/upscale_models/`

已确认存在的样例文件：

- checkpoints:
  - `SDXL_Photorealistic_Mix_nsfw.safetensors`
  - `babesByStableYogiPony_v60FP16.safetensors`
  - `ponyRealism_V23ULTRA.safetensors`
- loras:
  - `NSFW_POV_AllInOne.safetensors`
- upscale models:
  - `4x-UltraSharp.pth`

### 安全要求

前端页面不能直接持有或使用对象存储 AK/SK。

因此必须采用：

- 前端调用后端接口
- 后端使用服务端环境变量或本地凭据去列目录
- 后端只返回筛选后的模型元数据

### 建议新增后端接口

`GET /models/catalog`

返回前端可选模型清单：

```json
{
  "checkpoints": [
    {
      "name": "SDXL_Photorealistic_Mix_nsfw.safetensors",
      "path": "runpod-slim/ComfyUI/models/checkpoints/SDXL_Photorealistic_Mix_nsfw.safetensors",
      "type": "checkpoint"
    }
  ],
  "loras": [
    {
      "name": "NSFW_POV_AllInOne.safetensors",
      "path": "runpod-slim/ComfyUI/models/loras/NSFW_POV_AllInOne.safetensors",
      "type": "lora"
    }
  ],
  "upscale_models": [
    {
      "name": "4x-UltraSharp.pth",
      "path": "runpod-slim/ComfyUI/models/upscale_models/4x-UltraSharp.pth",
      "type": "upscale_model"
    }
  ]
}
```

## 目录扫描规则

后端扫描时只取白名单目录：

- checkpoints: `runpod-slim/ComfyUI/models/checkpoints/`
- loras: `runpod-slim/ComfyUI/models/loras/`
- upscale models: `runpod-slim/ComfyUI/models/upscale_models/`

忽略：

- `put_*_here`
- 非模型占位文件
- 子目录中的临时缓存

建议白名单扩展名：

- checkpoints:
  - `.safetensors`
  - `.ckpt`
- loras:
  - `.safetensors`
  - `.ckpt`
- upscale:
  - `.pth`
  - `.pt`

## 参数契约补充

建议统一输入契约：

```json
{
  "input": {
    "mode": "dual_pass_auto_pose | pose_then_face_swap | pose_only | text_only | qwen_swap_face | qwen_pose_fusion | wan2_2_i2v_extend_any_frame",
    "reference_image": "base64_or_url",
    "pose_image": "base64_or_url",
    "startimg": "base64_or_url",
    "endimg": "base64_or_url",
    "prompt": "text",
    "frames": 81,
    "negative_prompt": "text",
    "qwen_swap_prompt": "text",
    "qwen_model": "qwen-image-2.0-pro",
    "qwen_size": "1024*1536"
  }
}
```

### 各模式字段约束

`dual_pass_auto_pose`
- 必填：
  - `reference_image`
  - `prompt`
- 可选：
  - `negative_prompt`
  - `base_*`

`pose_then_face_swap`
- 必填：
  - `reference_image`
  - `pose_image`
  - `prompt`

`pose_only`
- 必填：
  - `pose_image`
  - `prompt`
- 不应要求：
  - `reference_image`

`text_only`
- 必填：
  - `prompt`
- 不应要求：
  - `reference_image`
  - `pose_image`
- 默认：
  - `enable_pulid=false`

`qwen_swap_face`
- 必填：
  - `reference_image`
  - `prompt`
- 可选：
  - `negative_prompt`
  - `qwen_swap_prompt`
  - `qwen_model`
  - `qwen_size`
  - `qwen_extra_image`
- 不应要求：
  - `pose_image`
- 默认：
  - `enable_pulid=false`

`qwen_pose_fusion`
- 必填：
  - `reference_image`
  - `pose_image`
  - `prompt`
- 可选：
  - `negative_prompt`
  - `qwen_model`
  - `qwen_size`
- 不应要求：
  - `qwen_extra_image`
- 默认：
  - `enable_pulid=false`

`wan2_2_i2v_extend_any_frame`
- 必填：
  - `startimg`
  - `prompt`
  - `frames`
- 可选：
  - `endimg`
  - `negative_prompt`
  - `i2v_resolution`
  - `i2v_prompt_extend`
  - `i2v_watermark`
- 仍然保留：
  - `enable_lora`
  - `loras[]`
- 不应要求：
  - `reference_image`
  - `pose_image`
- 默认：
  - `enable_pulid=false`
  - `enable_upscale=false`
  - `enable_i2v=false`

## handler / workflow 改造影响

当前 workflow 中相关节点是固定写死的：

- checkpoint 节点：
  - `CheckpointLoaderSimple(1)`
- LoRA 节点：
  - `LoraLoader(17)`
- 4x 模型节点：
  - `UpscaleModelLoader(18)`

因此后续实现时必须支持：

1. handler 从请求中注入 `ckpt_name`
2. handler 从请求中注入多条 `loras[]`
3. handler 从请求中注入 `upscale_model_name`
4. 若对应开关关闭，则相关节点要被真正绕开
5. 模式 D 时跳过 pose/depth/PuLID 相关链路

## workflow JSON 改造需求

### 总体原则

不要继续依赖“传没传 pose_image 就隐式决定模式”。

改造后 workflow 应具备：

- 可显式切换五种模式
- 可显式跳过 PuLID
- 可显式跳过 LoRA
- 可显式跳过 4x 放大
- 可显式启用 Qwen 换脸后处理
- 输出节点保持稳定，不因模式变化导致 handler 取错节点

### workflow 需要具备的分支

#### 分支 1：第一阶段自动姿态分支

用于模式 A：
- `KSampler(22)` 生成中间图
- `VAEDecode(23)`
- `DepthAnything(10)`
- `DWPreprocessor(24)`

#### 分支 2：外部 Pose 解析分支

用于模式 B / C：
- `LoadImage(9)`
- `DepthAnything(10)` 改接 `9`
- `DWPreprocessor(24)` 改接 `9`

#### 分支 3：PuLID 换脸分支

用于模式 A / B：
- `PulidModelLoader(4)`
- `PulidEvaClipLoader(5)`
- `PulidInsightFaceLoader(6)`
- `ApplyPulid(8)`

模式 C / D 必须可绕过此分支。

#### 分支 4：LoRA 分支

当前是固定 `LoraLoader(17)`。

需要支持：
- 开启 LoRA：沿用 `17 -> model/clip`，并允许后续追加多个 LoRA 节点
- 关闭 LoRA：直接使用 `CheckpointLoaderSimple(1)` 的 `model/clip`

#### 分支 5：4x 放大分支

需要明确：
- 原图直接输出
- 4x 后输出

#### 分支 6：Text-Only 直接出图分支

用于模式 D：
- 不接 `LoadImage(9)`
- 不接 `KSampler(22) -> VAEDecode(23)` 自动姿态预处理
- 不接 `DepthAnything(10)` 与 `DWPreprocessor(24)` 条件图链路
- 最终采样器直接使用文本正负提示和基础模型链路出图

#### 分支 7：Qwen 换脸后处理分支

用于模式 E：
- 基础图先完成 Comfy 生成
- 再把最终图和参考脸送入 DashScope Qwen 图像编辑接口
- 输出以 Qwen 返回图为最终图

## 输出节点要求

无论哪种模式，都要保持统一输出约定：

- 最终图输出节点固定
- 中间图输出节点固定
- handler 不需要根据模式去猜最终图在哪个节点

## 接口返回要求

统一返回：

- `ok`
- `mode`
- `prompt_id`
- `request_id`
- `final_url`
- `intermediate_urls`
- `meta`
  - `enable_pulid`
  - `enable_lora`
  - `enable_upscale`
  - `final_format`
