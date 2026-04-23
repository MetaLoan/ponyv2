# 前端工作模式速查

这份文档只讲一件事：前端测试页里每种模式是干什么的，什么时候用，最少要填什么。

如果只想快速上手，可以先记住两点：

- `Render` 是“先看工作流预览”
- `Generate` 是“真正提交出图”

页面里的“模式”才是决定流程的关键。

## 0. 建议配置

如果你只是先想把流程跑通，建议先用下面这组默认值：

- `mode`: 按实际场景选择
- `reference_image`: 有参考脸就传，没有就不传
- `pose_image`: 只有需要姿态控制时再传
- `prompt`: 必填，先写清楚主体、风格、动作
- `enable_lora`: `true`，先保留 LoRA 链，方便测试效果
- `enable_upscale`: `true`，先打开 4x 放大，方便看最终图清晰度

推荐的最小可调思路：

- 先只调 `mode`
- 再补 `reference_image` / `pose_image`
- 再改 `prompt`
- 最后再调 `enable_lora`、`enable_upscale`

## 1. 模式总览

| 模式 | 需要的输入 | 主要用途 |
| --- | --- | --- |
| `dual_pass_auto_pose` | `reference_image + prompt` | 先自动分析姿态，再正式出图 |
| `pose_then_face_swap` | `reference_image + pose_image + prompt` | 用外部姿态图控制动作，再做换脸 |
| `pose_only` | `pose_image + prompt` | 只按姿态出图，不换脸 |
| `text_only` | `prompt` | 纯文生图，不用参考图 |
| `qwen_swap_face` | `reference_image + prompt` | 先出图，再用 Qwen 做换脸 |
| `qwen_pose_fusion` | `reference_image + pose_image + prompt` | 直接用 Qwen 融合人脸和姿态图 |
| `qwen_edit_face` | `prompt` | 先出图，再用 Qwen 按文字改脸 |
| `wan2_2_i2v_extend_any_frame` | `startimg + prompt + frames` | 按 81 帧分段续写视频，再合并成最终长视频，同时保留 LoRA 串联 |

## 2. 每种模式怎么理解

### A. `dual_pass_auto_pose`

最常用的默认模式。

它会先用参考图和提示词跑一遍，自动提取姿态和深度，再进行第二次正式生成。

适合：

- 想保留参考人物的大致风格
- 又不想自己单独准备 pose 图

最小示例：

```json
{
  "mode": "dual_pass_auto_pose",
  "reference_image": "人物参考图",
  "prompt": "写实摄影风格，站立姿势，柔和灯光",
  "enable_lora": true,
  "enable_upscale": true
}
```

### B. `pose_then_face_swap`

这个模式是“先按姿势画，再换脸”。

你需要同时给：

- 参考脸图 `reference_image`
- 姿态图 `pose_image`
- 提示词 `prompt`

适合：

- 已经有一张想要的动作图
- 还想把另一张人物脸换进去

最小示例：

```json
{
  "mode": "pose_then_face_swap",
  "reference_image": "要换进去的人脸参考图",
  "pose_image": "动作参考图",
  "prompt": "站姿全身照，电影光感",
  "enable_lora": true,
  "enable_upscale": true
}
```

### C. `pose_only`

这个模式只看姿态，不做换脸。

它会用 `pose_image` 约束动作，但不会强制保留参考人物身份。

适合：

- 想测试姿态控制是否稳定
- 只想锁定动作，不想换脸

最小示例：

```json
{
  "mode": "pose_only",
  "pose_image": "动作参考图",
  "prompt": "穿红色连衣裙，坐姿，室内光",
  "enable_lora": true,
  "enable_upscale": true
}
```

### D. `text_only`

纯提示词模式。

不需要参考图，也不需要姿态图，适合测试基础模型本身的出图能力。

适合：

- 测 checkpoint 是否正常
- 测 LoRA、采样器、分辨率这些基础参数

最小示例：

```json
{
  "mode": "text_only",
  "prompt": "赛博朋克街景，雨夜，霓虹灯，电影感",
  "enable_lora": true,
  "enable_upscale": true
}
```

### E. `qwen_swap_face`

这个模式是“先出图，再用 Qwen 换脸”。

它和普通 PuLID 换脸不同，换脸是在生成完成后做后处理。

适合：

- 想把某张参考脸融合进最终图
- 参考脸和生成图的风格差异比较大

最小示例：

```json
{
  "mode": "qwen_swap_face",
  "reference_image": "人脸参考图",
  "prompt": "写实人像，半身，柔光",
  "enable_lora": true,
  "enable_upscale": true
}
```

如果需要更强的融合控制，可以再加：

- `qwen_swap_prompt`
- `qwen_extra_image`

### F. `qwen_pose_fusion`

这个模式是不先出图，直接拿 `pose_image` 和 `reference_image` 交给 Qwen 做融合。

它适合你已经有一张姿态图和一张脸图，想直接得到融合结果的场景。

最小示例：

```json
{
  "mode": "qwen_pose_fusion",
  "reference_image": "脸图",
  "pose_image": "姿态图",
  "prompt": "保留 pose 图姿势和构图，只融合 face 图的人脸特征",
  "enable_lora": true,
  "enable_upscale": true
}
```

### G. `qwen_edit_face`

这个模式也是先出图，但不是拿参考脸去换，而是直接按文字去改脸。

适合：

- 不想上传参考脸
- 只想用文字描述脸部特征做后处理

最小示例：

```json
{
  "mode": "qwen_edit_face",
  "prompt": "写实人像，清晰五官，干净背景",
  "enable_lora": true,
  "enable_upscale": true
}
```

### H. `wan2_2_i2v_extend_any_frame`

这个模式是“先定起始帧，再按帧数连续续写视频”。

你需要至少给：

- `startimg`
- `prompt`
- `frames`

可选再给：

- `endimg`

它会按 81 帧一段循环：

- `1-81` 为第一段
- `82-162` 用上一段的尾帧继续写
- 后面继续循环

`endimg` 只作为整个序列最后一帧，不会拿来当每段的终止帧。

最小示例：

```json
{
  "mode": "wan2_2_i2v_extend_any_frame",
  "startimg": "start.jpg",
  "prompt": "沙滩海边的写实视频，人物自然行走，镜头稳定",
  "frames": 309
}
```

## 3. 选模式时怎么判断

- 有参考脸，但没有姿态图: 用 `dual_pass_auto_pose`
- 有参考脸，也有姿态图: 用 `pose_then_face_swap`
- 只有姿态图: 用 `pose_only`
- 什么图都没有: 用 `text_only`
- 想在出图后再换脸: 用 `qwen_swap_face`
- 想直接拿 pose 图和 face 图做融合: 用 `qwen_pose_fusion`
- 想在出图后直接文字改脸: 用 `qwen_edit_face`
- 想按帧数扩展视频并合并: 用 `wan2_2_i2v_extend_any_frame`

## 4. 最常见的几个例子

### 例子 1: 参考图 + 自动姿态

你只有一张人物图，想让系统自动接着画。

```json
{
  "mode": "dual_pass_auto_pose",
  "reference_image": "portrait.jpg",
  "prompt": "全身照，站姿，室外自然光",
  "enable_lora": true,
  "enable_upscale": true
}
```

### 例子 2: 姿态图 + 换脸

你已经有一张动作图，想换成某个人脸。

```json
{
  "mode": "pose_then_face_swap",
  "reference_image": "face.jpg",
  "pose_image": "pose.jpg",
  "prompt": "时尚写真，黑色背景",
  "enable_lora": true,
  "enable_upscale": true
}
```

### 例子 3: 纯文生图

只想验证模型本身效果。

```json
{
  "mode": "text_only",
  "prompt": "一只戴墨镜的橘猫，电影海报风格",
  "enable_lora": true,
  "enable_upscale": true
}
```

### 例子 4: 先出图再 Qwen 换脸

```json
{
  "mode": "qwen_swap_face",
  "reference_image": "face.jpg",
  "prompt": "高清人像，白衬衫，浅景深",
  "enable_lora": true,
  "enable_upscale": true
}
```

### 例子 5: 文字改脸

```json
{
  "mode": "qwen_edit_face",
  "prompt": "年轻女性肖像，五官清晰，皮肤自然，柔光，背景干净",
  "enable_lora": true,
  "enable_upscale": true
}
```

## 5. 这几个字段一般不用纠结

如果只是想先把流程跑通，优先关注这几个字段就够了：

- `mode`
- `reference_image`
- `pose_image`
- `prompt`
- `enable_lora`
- `enable_upscale`

其他参数像 `seed`、`steps`、`cfg`、`sampler_name` 可以后面再慢慢调。

## 6. 一句话总结

- `dual_pass_auto_pose` = 参考图自动接着画
- `pose_then_face_swap` = 先按动作画，再换脸
- `pose_only` = 只控动作
- `text_only` = 纯提示词出图
- `qwen_swap_face` = 出图后用 Qwen 换脸
- `qwen_edit_face` = 出图后用 Qwen 按文字改脸
- `wan2_2_i2v_extend_any_frame` = 按帧数分段续写视频，再合并输出
