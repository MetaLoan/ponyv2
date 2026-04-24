import { ChangeEvent, useEffect, useMemo, useState } from "react";
import { useTasks, TaskCenter } from "./TaskCenter";

type Mode = "dual_pass_auto_pose" | "pose_then_face_swap" | "pose_only" | "text_only" | "qwen_swap_face" | "qwen_pose_fusion" | "qwen_edit_face" | "wan2_2_i2v_extend_any_frame";
type WanPreset = "manual" | "realistic_video" | "anime_video";

type CatalogItem = {
  name: string;
  path: string;
  type: string;
};

type CatalogResponse = {
  checkpoints: CatalogItem[];
  unets: CatalogItem[];
  loras: CatalogItem[];
  upscale_models: CatalogItem[];
  vaes: CatalogItem[];
  clip_visions: CatalogItem[];
  text_encoders: CatalogItem[];
};

type HealthResponse = {
  ok: boolean;
  engine: string;
  comfy_api_url?: string;
  runpod_ready?: boolean;
  s3_ready?: boolean;
};

type LoraRow = {
  id: string;
  name: string;
  strength_model: number;
  strength_clip: number;
};

type GenerateResult = Record<string, unknown>;

type MediaState = {
  kind: "file" | "url";
  file: File | null;
  url: string;
  preview: string;
};

const defaultLora = (): LoraRow => ({
  id: crypto.randomUUID(),
  name: "",
  strength_model: 0.6,
  strength_clip: 0.9,
});

const DEFAULT_QWEN_SWAP_PROMPT =
  "以图1为最终画面底图，严格保留图1的人物姿势、构图、服装、光照、背景和沙滩环境；仅将图2中的面部特征自然融合到图1人物脸上，保持真实自然、五官清晰、肤质统一；图3如存在，仅作为辅助参考，不要改变其他区域。";
const DEFAULT_QWEN_POSE_FUSION_PROMPT =
  "以图1的pose图作为最终构图底图，严格保留人物姿势、肢体角度、镜头、服装、场景和光照；将图2的face图中的面部身份特征自然融合到图1人物脸上；保持五官清晰、肤质统一、真实摄影感，不要改动背景、衣服、身体姿态或镜头结构，融合结果要自然连贯。";
const DEFAULT_QWEN_EDIT_PROMPT =
  "将图中的角色脸部特征形象进行调整，使其符合如下描述中关于脸部的特征描述:{{生图提示词的主提示词变量}}";
const DEFAULT_WAN_EXTEND_PROMPT =
  "沙滩，海边，晴天，自然光，蓝天白云，海浪，金色细沙，轻微海风，真实摄影感，画面通透，动作自然连贯，镜头稳定，细节清晰，电影感成片";
const DEFAULT_WAN_REALISTIC_PROMPT =
  "真实摄影感，电影感，稳定镜头，自然动作，干净光线，细节清晰，人物动作连贯，画面通透， realistic, photorealistic, high detail";
const DEFAULT_WAN_ANIME_PROMPT =
  "anime style, clean lineart, vibrant colors, smooth motion, expressive character design, cinematic framing, detailed background, dynamic composition";
const DEFAULT_WAN_UNET_HIGH_NAME = "wan22I2V8StepsNSFWFP8_fp8Highnoise10.safetensors";
const DEFAULT_WAN_UNET_LOW_NAME = "wan22I2V8StepsNSFWFP8_fp8Lownoise10.safetensors";
const DEFAULT_WAN_VAE_NAME = "wan_2.1_vae.safetensors";
const DEFAULT_WAN_CLIP_VISION_NAME = "clip_vision_h.safetensors";
const DEFAULT_WAN_CLIP_NAME = "umt5_xxl_fp8_e4m3fn_scaled.safetensors";
const WAN_ANIME_LORA_HINTS = ["live-wallpaper-style", "wan22-2d-animation-effects-2d", "wan-22-live2d-background", "2309690"];
const WAN_MODEL_HINTS = ["wan", "wan2", "kf2v", "i2v", "svi"];

function matchesHints(item: CatalogItem, hints: string[]): boolean {
  const haystack = `${item.name} ${item.path}`.toLowerCase();
  return hints.some((hint) => haystack.includes(hint.toLowerCase()));
}

function pickCatalogName(items: CatalogItem[], preferred: string, fallback: string): string {
  const names = items.map((item) => item.name).filter((name) => name.trim() !== "");
  if (preferred.trim() && names.includes(preferred.trim())) {
    return preferred.trim();
  }
  if (fallback.trim() && names.includes(fallback.trim())) {
    return fallback.trim();
  }
  return names[0] || fallback;
}

function pickCatalogDefault(items: CatalogItem[], fallback: string, preferred = ""): string {
  const names = items.map((item) => item.name.trim()).filter((name) => name !== "");
  if (fallback.trim() && names.includes(fallback.trim())) {
    return fallback.trim();
  }
  if (preferred.trim() && names.includes(preferred.trim())) {
    return preferred.trim();
  }
  return names[0] || fallback;
}

function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [catalog, setCatalog] = useState<CatalogResponse>({
    checkpoints: [],
    unets: [],
    loras: [],
    upscale_models: [],
    vaes: [],
    clip_visions: [],
    text_encoders: [],
  });
  const [catalogError, setCatalogError] = useState<string>("");
  const [mode, setMode] = useState<Mode>("dual_pass_auto_pose");
  const [referenceMedia, setReferenceMedia] = useState<MediaState>({ kind: "file", file: null, url: "", preview: "" });
  const [poseMedia, setPoseMedia] = useState<MediaState>({ kind: "file", file: null, url: "", preview: "" });
  const [qwenExtraMedia, setQwenExtraMedia] = useState<MediaState>({ kind: "file", file: null, url: "", preview: "" });
  const [wanStartMedia, setWanStartMedia] = useState<MediaState>({ kind: "file", file: null, url: "", preview: "" });
  const [wanEndMedia, setWanEndMedia] = useState<MediaState>({ kind: "file", file: null, url: "", preview: "" });
  const [prompt, setPrompt] = useState(
    "沙滩，海边，晴天，自然光，蓝天白云，海浪，金色细沙，轻微海风，真实摄影感，画面通透，细节清晰，人物自然融入环境，photorealistic, best quality, ultra detailed"
  );
  const [qwenPoseFusionPrompt, setQwenPoseFusionPrompt] = useState(DEFAULT_QWEN_POSE_FUSION_PROMPT);
  const [wanExtendPrompt, setWanExtendPrompt] = useState(DEFAULT_WAN_EXTEND_PROMPT);
  const [wanUnetHighName, setWanUnetHighName] = useState(DEFAULT_WAN_UNET_HIGH_NAME);
  const [wanUnetLowName, setWanUnetLowName] = useState(DEFAULT_WAN_UNET_LOW_NAME);
  const [wanVaeName, setWanVaeName] = useState(DEFAULT_WAN_VAE_NAME);
  const [wanClipVisionName, setWanClipVisionName] = useState(DEFAULT_WAN_CLIP_VISION_NAME);
  const [wanClipName, setWanClipName] = useState(DEFAULT_WAN_CLIP_NAME);
  const [wanPreset, setWanPreset] = useState<WanPreset>("manual");
  const [wanAdvancedOpen, setWanAdvancedOpen] = useState(false);
  const [frames, setFrames] = useState(81);
  const [negativePrompt, setNegativePrompt] = useState(
    "bad anatomy, poorly drawn hands, deformed hands, mutated hands, extra fingers, fused fingers, bad hands, blurry, low quality, worst quality, lowres, text, watermark, censored, ugly, deformed, extra limbs, bad proportions, open mouth, tongue out, tongue visible, saliva, oral sex, blowjob, fellatio, penis, any male genital, ahegao, rolling eyes"
  );
  const [qwenSwapPrompt, setQwenSwapPrompt] = useState(DEFAULT_QWEN_SWAP_PROMPT);
  const [qwenEditPrompt, setQwenEditPrompt] = useState(DEFAULT_QWEN_EDIT_PROMPT);
  const [qwenModel, setQwenModel] = useState("qwen-image-2.0-pro");
  const [qwenSize, setQwenSize] = useState("");
  const [ckptName, setCkptName] = useState("SDXL_Photorealistic_Mix_nsfw.safetensors");
  const [width, setWidth] = useState(832);
  const [height, setHeight] = useState(1216);
  const [batchSize, setBatchSize] = useState(1);
  const [baseSteps, setBaseSteps] = useState(8);
  const [steps, setSteps] = useState(40);
  const [cfg, setCfg] = useState(4);
  const [baseCfg, setBaseCfg] = useState(4);
  const [baseDenoise, setBaseDenoise] = useState(1);
  const [denoise, setDenoise] = useState(1);
  const [baseSeed, setBaseSeed] = useState<number>(967549018325766);
  const [seed, setSeed] = useState<number>(642164445859951);
  const [baseSamplerName, setBaseSamplerName] = useState("dpmpp_2m_sde");
  const [baseScheduler, setBaseScheduler] = useState("karras");
  const [samplerName, setSamplerName] = useState("dpmpp_2m_sde");
  const [scheduler, setScheduler] = useState("karras");
  const [enablePulid, setEnablePulid] = useState(true);
  const [pulidWeight, setPulidWeight] = useState(0.7);
  const [pulidStartAt, setPulidStartAt] = useState(0.5);
  const [pulidEndAt, setPulidEndAt] = useState(1);
  const [pulidMethod, setPulidMethod] = useState("fidelity");
  const [cnDepthStrength, setCnDepthStrength] = useState(0.6);
  const [cnDepthStartPercent, setCnDepthStartPercent] = useState(0);
  const [cnDepthEndPercent, setCnDepthEndPercent] = useState(1);
  const [cnPoseStrength, setCnPoseStrength] = useState(0.6);
  const [cnPoseStartPercent, setCnPoseStartPercent] = useState(0);
  const [cnPoseEndPercent, setCnPoseEndPercent] = useState(1);
  const [enableLora, setEnableLora] = useState(true);
  const [loras, setLoras] = useState<LoraRow[]>([defaultLora()]);
  const [enableUpscale, setEnableUpscale] = useState(true);
  const [upscaleModelName, setUpscaleModelName] = useState("4x-UltraSharp.pth");
  const [keepIntermediate, setKeepIntermediate] = useState(true);
  const [enableI2V, setEnableI2V] = useState(false);
  const [i2vPrompt, setI2VPrompt] = useState(
    "保持主体一致，生成自然流畅、镜头稳定、画面连贯的动态视频，动作真实，细节清晰，电影感自然。"
  );
  const [i2vModel, setI2VModel] = useState("wan2.7-i2v");
  const [i2vResolution, setI2VResolution] = useState("720P");
  const [i2vDuration, setI2VDuration] = useState(5);
  const [i2vSeed, setI2VSeed] = useState<number>(12345);
  const [i2vNegativePrompt, setI2VNegativePrompt] = useState(
    "low quality, blurry, flicker, jitter, motion artifacts, deformed, extra limbs, bad proportions"
  );
  const [wanSeeds, setWanSeeds] = useState<number[]>([]);
  const [wanPrompts, setWanPrompts] = useState<string[]>([]);
  const [wanSegmentFrames, setWanSegmentFrames] = useState<number>(161);
  const [autoSegmentPrompts, setAutoSegmentPrompts] = useState(false);
  const [isAiSplitting, setIsAiSplitting] = useState(false);
  const [i2vAudioURL, setI2VAudioURL] = useState("");
  const [i2vPromptExtend, setI2VPromptExtend] = useState(true);
  const [i2vWatermark, setI2VWatermark] = useState(false);
  const [outputFormat, setOutputFormat] = useState<"jpg" | "png">("jpg");
  const [jpgQuality, setJpgQuality] = useState(85);
  const [renderResult, setRenderResult] = useState<GenerateResult | null>(null);
  const [generateResult, setGenerateResult] = useState<GenerateResult | null>(null);
  const [busy, setBusy] = useState<"" | "render" | "generate">("");
  const [error, setError] = useState("");
  const { tasks, addTask, clearTasks } = useTasks();
  const [payloadJsonText, setPayloadJsonText] = useState("");
  const [payloadImportError, setPayloadImportError] = useState("");
  const isWanMode = mode === "wan2_2_i2v_extend_any_frame";
  const wanModelOptions = useMemo(() => {
    const source = (catalog.unets || []).filter((item) => matchesHints(item, WAN_MODEL_HINTS));
    const pool = source.length > 0 ? source : (catalog.unets || []);
    const names = new Set<string>();
    for (const item of pool) {
      if (item.name.trim()) {
        names.add(item.name.trim());
      }
    }
    if (wanUnetHighName.trim()) {
      names.add(wanUnetHighName.trim());
    }
    if (wanUnetLowName.trim()) {
      names.add(wanUnetLowName.trim());
    }
    if (names.size === 0) {
      names.add(DEFAULT_WAN_UNET_HIGH_NAME);
      names.add(DEFAULT_WAN_UNET_LOW_NAME);
    }
    return Array.from(names);
  }, [catalog.unets, wanUnetHighName, wanUnetLowName]);
  const wanVaeOptions = useMemo(() => {
    const names = new Set<string>([DEFAULT_WAN_VAE_NAME]);
    for (const item of catalog.vaes || []) {
      if (item.name.trim()) {
        names.add(item.name.trim());
      }
    }
    if (wanVaeName.trim()) {
      names.add(wanVaeName.trim());
    }
    return Array.from(names);
  }, [catalog.vaes, wanVaeName]);
  const wanClipVisionOptions = useMemo(() => {
    const names = new Set<string>([DEFAULT_WAN_CLIP_VISION_NAME]);
    for (const item of catalog.clip_visions || []) {
      if (item.name.trim()) {
        names.add(item.name.trim());
      }
    }
    if (wanClipVisionName.trim()) {
      names.add(wanClipVisionName.trim());
    }
    return Array.from(names);
  }, [catalog.clip_visions, wanClipVisionName]);
  const wanTextEncoderOptions = useMemo(() => {
    const names = new Set<string>([DEFAULT_WAN_CLIP_NAME]);
    for (const item of catalog.text_encoders || []) {
      if (item.name.trim()) {
        names.add(item.name.trim());
      }
    }
    if (wanClipName.trim()) {
      names.add(wanClipName.trim());
    }
    return Array.from(names);
  }, [catalog.text_encoders, wanClipName]);
  const wanAnimeLoraOptions = useMemo(() => {
    const matches = (catalog.loras || []).filter((item) =>
      WAN_ANIME_LORA_HINTS.some((hint) => item.name.toLowerCase().includes(hint.toLowerCase()) || item.path.toLowerCase().includes(hint.toLowerCase()))
    );
    return matches.length > 0 ? matches : catalog.loras || [];
  }, [catalog.loras]);

  useEffect(() => {
    void (async () => {
      const healthResp = await fetch("/api/health");
      const healthText = await healthResp.text();
      if (!healthText.trim()) {
        throw new Error("Empty /api/health response");
      }
      const healthJson = JSON.parse(healthText) as HealthResponse;
      setHealth(healthJson);

      const catalogResp = await fetch("/api/models/catalog");
      if (!catalogResp.ok) {
        const text = await catalogResp.text();
        setCatalogError(text);
        return;
      }
      const catalogText = await catalogResp.text();
      if (!catalogText.trim()) {
        setCatalogError("Empty /api/models/catalog response");
        return;
      }
      const catalogJson = JSON.parse(catalogText) as CatalogResponse;
      setCatalog(catalogJson);
      if (catalogJson?.checkpoints?.[0] && !ckptName) {
        setCkptName(catalogJson.checkpoints[0].name);
      }
      if (catalogJson?.loras?.[0] && !loras[0]?.name) {
        setLoras((rows) =>
          rows.map((row, idx) => (idx === 0 ? { ...row, name: catalogJson.loras[0].name } : row))
        );
      }
      if (catalogJson?.upscale_models?.[0] && !upscaleModelName) {
        setUpscaleModelName(catalogJson.upscale_models[0].name);
      }
    })().catch((err: unknown) => setCatalogError(String(err)));
  }, []);

  useEffect(() => {
    if (!isWanMode) {
      return;
    }
    const wanUnets = (catalog.unets || []).filter((item) => matchesHints(item, WAN_MODEL_HINTS));
    const unetSource = wanUnets.length > 0 ? wanUnets : catalog.unets || [];
    setWanUnetHighName((current) => pickCatalogDefault(unetSource, DEFAULT_WAN_UNET_HIGH_NAME, current));
    setWanUnetLowName((current) => pickCatalogDefault(unetSource, DEFAULT_WAN_UNET_LOW_NAME, current));
    setWanVaeName((current) => pickCatalogDefault(catalog.vaes || [], DEFAULT_WAN_VAE_NAME, current));
    setWanClipVisionName((current) => pickCatalogDefault(catalog.clip_visions || [], DEFAULT_WAN_CLIP_VISION_NAME, current));
    setWanClipName((current) => pickCatalogDefault(catalog.text_encoders || [], DEFAULT_WAN_CLIP_NAME, current));
  }, [catalog.unets, catalog.vaes, catalog.clip_visions, catalog.text_encoders, isWanMode]);

  useEffect(() => {
    if (
      mode === "pose_only" ||
      mode === "text_only" ||
      mode === "qwen_swap_face" ||
      mode === "qwen_pose_fusion" ||
      mode === "qwen_edit_face" ||
      mode === "wan2_2_i2v_extend_any_frame"
    ) {
      setEnablePulid(false);
    } else if (mode === "dual_pass_auto_pose" || mode === "pose_then_face_swap") {
      setEnablePulid(true);
    }
  }, [mode]);

  const payload = useMemo(() => {
    const cleanLoras = enableLora
      ? loras
          .filter((item) => item.name.trim() !== "")
          .map(({ name, strength_model, strength_clip }) => ({
            name,
            strength_model,
            strength_clip,
          }))
      : [];
    const activePrompt =
      mode === "qwen_pose_fusion" ? qwenPoseFusionPrompt : isWanMode ? wanExtendPrompt : prompt;
    const body: Record<string, unknown> = {
      mode,
      prompt: activePrompt,
      negative_prompt: negativePrompt,
      enable_lora: enableLora,
      loras: cleanLoras,
      async: true
    };
    if (isWanMode) {
      body.startimg = mediaToPayloadValue(wanStartMedia);
      const wanEndValue = mediaToPayloadValue(wanEndMedia);
      if (wanEndValue) {
        body.endimg = wanEndValue;
      }
      body.frames = frames;
      body.wan_unet_high_name = wanUnetHighName;
      body.wan_unet_low_name = wanUnetLowName;
      body.wan_vae_name = wanVaeName;
      body.wan_clip_vision_name = wanClipVisionName;
      body.wan_clip_name = wanClipName;
      body.i2v_resolution = i2vResolution;
      body.base_steps = baseSteps;
      body.steps = steps;
      body.base_cfg = baseCfg;
      body.cfg = cfg;
      body.segment_limit = wanSegmentFrames;
      body.auto_segment_prompts = autoSegmentPrompts;
      const computedSegmentCount = Math.max(1, Math.ceil((frames - 1) / Math.max(1, wanSegmentFrames - 1)));
      body.wan_seeds = Array.from({ length: computedSegmentCount }).map((_, idx) => wanSeeds[idx] ?? seed + idx);
      body.wan_prompts = Array.from({ length: computedSegmentCount }).map((_, idx) => wanPrompts[idx] || "");
    } else {
      body.width = width;
      body.height = height;
      body.batch_size = batchSize;
      body.ckpt_name = ckptName;
      body.base_steps = baseSteps;
      body.base_seed = baseSeed;
      body.base_cfg = baseCfg;
      body.base_sampler_name = baseSamplerName;
      body.base_scheduler = baseScheduler;
      body.base_denoise = baseDenoise;
      body.steps = steps;
      body.seed = seed;
      body.cfg = cfg;
      body.sampler_name = samplerName;
      body.scheduler = scheduler;
      body.denoise = denoise;
      body.enable_pulid = enablePulid;
      body.pulid_weight = pulidWeight;
      body.pulid_start_at = pulidStartAt;
      body.pulid_end_at = pulidEndAt;
      body.pulid_method = pulidMethod;
      body.cn_depth_strength = cnDepthStrength;
      body.cn_depth_start_percent = cnDepthStartPercent;
      body.cn_depth_end_percent = cnDepthEndPercent;
      body.cn_pose_strength = cnPoseStrength;
      body.cn_pose_start_percent = cnPoseStartPercent;
      body.cn_pose_end_percent = cnPoseEndPercent;
      body.enable_upscale = enableUpscale;
      body.upscale_model_name = upscaleModelName;
      body.keep_intermediate = keepIntermediate;
      body.enable_i2v = enableI2V;
      body.i2v_prompt = i2vPrompt;
      body.i2v_model = i2vModel;
      body.i2v_resolution = i2vResolution;
      body.i2v_duration = i2vDuration;
      body.i2v_seed = i2vSeed;
      body.i2v_negative_prompt = i2vNegativePrompt;
      body.i2v_audio_url = i2vAudioURL;
      body.i2v_prompt_extend = i2vPromptExtend;
      body.i2v_watermark = i2vWatermark;
      body.output_format = outputFormat;
      body.jpg_quality = jpgQuality;
    }
    const referenceImageValue = mediaToPayloadValue(referenceMedia);
    const poseImageValue = mediaToPayloadValue(poseMedia);
    const qwenExtraImageValue = mediaToPayloadValue(qwenExtraMedia);
    if (referenceImageValue && (mode === "dual_pass_auto_pose" || mode === "pose_then_face_swap" || mode === "qwen_swap_face" || mode === "qwen_pose_fusion")) {
      body.reference_image = referenceImageValue;
    }
    if (poseImageValue && (mode === "pose_then_face_swap" || mode === "pose_only" || mode === "qwen_pose_fusion")) {
      body.pose_image = poseImageValue;
    }
    if (qwenExtraImageValue && mode === "qwen_swap_face") {
      body.qwen_extra_image = qwenExtraImageValue;
    }
    if (mode === "qwen_pose_fusion") {
      body.reference_image = referenceImageValue;
      body.pose_image = poseImageValue;
    }
    if (mode === "qwen_swap_face") {
      body.qwen_swap_prompt = qwenSwapPrompt;
      body.qwen_model = qwenModel;
      body.qwen_size = qwenSize;
    }
    if (mode === "qwen_edit_face") {
      body.qwen_edit_prompt = qwenEditPrompt;
      body.qwen_model = qwenModel;
      body.qwen_size = qwenSize;
    }
    return body;
  }, [
    mode,
    isWanMode,
    prompt,
    qwenPoseFusionPrompt,
    qwenSwapPrompt,
    qwenEditPrompt,
    wanExtendPrompt,
    qwenModel,
    qwenSize,
    negativePrompt,
    wanStartMedia,
    wanEndMedia,
    wanSeeds,
    wanPrompts,
    wanSegmentFrames,
    frames,
    wanUnetHighName,
    wanUnetLowName,
    qwenExtraMedia,
    i2vResolution,
    i2vAudioURL,
    i2vPromptExtend,
    i2vWatermark,
    width,
    height,
    batchSize,
    ckptName,
    baseSteps,
    baseSeed,
    baseCfg,
    baseDenoise,
    baseSamplerName,
    baseScheduler,
    steps,
    seed,
    cfg,
    samplerName,
    scheduler,
    denoise,
    enablePulid,
    pulidWeight,
    pulidStartAt,
    pulidEndAt,
    pulidMethod,
    cnDepthStrength,
    cnDepthStartPercent,
    cnDepthEndPercent,
    cnPoseStrength,
    cnPoseStartPercent,
    cnPoseEndPercent,
    enableUpscale,
    upscaleModelName,
    keepIntermediate,
    enableI2V,
    i2vPrompt,
    i2vModel,
    i2vDuration,
    i2vSeed,
    i2vNegativePrompt,
    outputFormat,
    jpgQuality,
    referenceMedia,
    poseMedia,
    qwenExtraMedia,
    enableLora,
    loras,
  ]);

  const modeSummary = {
    dual_pass_auto_pose: "Reference image + prompt. Stage 1 auto derives pose and depth, stage 2 renders final result.",
    pose_then_face_swap: "Reference image + pose image + prompt. Render with external pose, then apply face identity.",
    pose_only: "Pose image + prompt. Pose-guided generation without PuLID face swap.",
    text_only: "Prompt only. No reference image, no pose image, no PuLID.",
    qwen_swap_face: "Base image is generated first, then Qwen uses reference face image and optional image 3 for face swap.",
    qwen_pose_fusion: "Pose image + face image + prompt. Qwen fuses the face directly onto the pose image.",
    qwen_edit_face: "Base image is generated first, then Qwen edits the face directly from the prompt without a reference face image.",
    wan2_2_i2v_extend_any_frame: "Start image + prompt + frame count. Generates video frames natively in a single pass and merges them.",
  }[mode];

  async function onRenderWorkflow() {
    await submit("render");
  }

  async function onGenerate() {
    await submit("generate");
  }

  async function submit(target: "render" | "generate") {
    setBusy(target);
    setError("");
    try {
      const body = { ...payload };
      if (mode === "dual_pass_auto_pose" || mode === "pose_then_face_swap") {
        body.reference_image = await resolveMedia(referenceMedia);
      }
      if (mode === "qwen_swap_face") {
        body.reference_image = await resolveMedia(referenceMedia);
        if (qwenExtraMedia.file || qwenExtraMedia.url.trim()) {
          body.qwen_extra_image = await resolveMedia(qwenExtraMedia);
        }
      }
      if (mode === "qwen_pose_fusion") {
        body.reference_image = await resolveMedia(referenceMedia);
        body.pose_image = await resolveMedia(poseMedia);
      }
      if (mode === "wan2_2_i2v_extend_any_frame") {
        body.startimg = await resolveMedia(wanStartMedia);
        const wanEndValue = await resolveOptionalMedia(wanEndMedia);
        if (wanEndValue) {
          body.endimg = wanEndValue;
        }
        body.frames = frames;
      }
      if (mode === "pose_then_face_swap" || mode === "pose_only") {
        body.pose_image = await resolveMedia(poseMedia);
      }

      const endpoint = target === "render" ? "/api/workflow/render" : "/api/generate";
      const resp = await fetch(endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(body),
      });
      const json = (await resp.json()) as GenerateResult;
      if (!resp.ok || (json as any).error || (json.ok === false)) {
        throw new Error(String((json as { error?: string }).error ?? "Request failed"));
      }
      if (target === "render") {
        setRenderResult(json);
      } else {
        if (json.status === "IN_QUEUE" && json.job_id) {
          addTask({
            id: String(json.job_id),
            job_id: String(json.job_id),
            mode: mode,
            prompt: String(payload.prompt || ""),
            status: "IN_QUEUE",
            timestamp: Date.now()
          });
          setGenerateResult(null);
        } else {
          setGenerateResult(json);
        }
      }
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy("");
    }
  }

  async function handleAiSplit() {
    if (!wanExtendPrompt) {
      setError("Please enter a main prompt first.");
      return;
    }
    const segmentCount = Math.max(1, Math.ceil((frames - 1) / Math.max(1, wanSegmentFrames - 1)));
    if (segmentCount <= 1) {
      setError("Total frames must exceed segment frames to use AI split.");
      return;
    }
    setIsAiSplitting(true);
    setError("");
    try {
      const resp = await fetch("/api/ai/split_prompt", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: wanExtendPrompt, segments: segmentCount }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.error || "Failed to call AI split");
      }
      setWanPrompts(data.prompts || []);
    } catch (err) {
      setError(String(err));
    } finally {
      setIsAiSplitting(false);
    }
  }

  function updateMedia(which: "reference" | "pose" | "qwenExtra" | "wanStart" | "wanEnd", patch: Partial<MediaState>) {
    const setter =
      which === "reference"
        ? setReferenceMedia
        : which === "pose"
          ? setPoseMedia
          : which === "qwenExtra"
            ? setQwenExtraMedia
            : which === "wanStart"
              ? setWanStartMedia
              : setWanEndMedia;
    setter((prev) => ({ ...prev, ...patch }));
  }

  function onFileChange(which: "reference" | "pose" | "qwenExtra" | "wanStart" | "wanEnd", e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0] ?? null;
    if (!file) {
      updateMedia(which, { file: null, preview: "" });
      return;
    }
    const preview = URL.createObjectURL(file);
    updateMedia(which, { kind: "file", file, preview, url: "" });
  }

  function onURLChange(which: "reference" | "pose" | "qwenExtra" | "wanStart" | "wanEnd", value: string) {
    updateMedia(which, { kind: "url", url: value, file: null, preview: value });
  }

  function addLora() {
    setLoras((rows) => [...rows, defaultLora()]);
  }

  function updateLora(id: string, patch: Partial<LoraRow>) {
    setLoras((rows) => rows.map((row) => (row.id === id ? { ...row, ...patch } : row)));
  }

  function removeLora(id: string) {
    setLoras((rows) => rows.filter((row) => row.id !== id));
  }

  function applyWanPreset(preset: WanPreset) {
    setWanPreset(preset);
    if (preset === "manual") {
      return;
    }
    setMode("wan2_2_i2v_extend_any_frame");
    setEnableLora(preset === "anime_video");
    setWanExtendPrompt(preset === "anime_video" ? DEFAULT_WAN_ANIME_PROMPT : DEFAULT_WAN_REALISTIC_PROMPT);
    setNegativePrompt(
      preset === "anime_video"
        ? "bad anatomy, blurry, low quality, worst quality, text, watermark, extra fingers, extra limbs, deformed, distorted, messy background"
        : "bad anatomy, poorly drawn hands, deformed hands, mutated hands, extra fingers, fused fingers, blurry, low quality, worst quality, text, watermark, extra limbs, bad proportions"
    );
    setFrames(81);
    setWanStartMedia({ kind: "file", file: null, url: "", preview: "" });
    setWanEndMedia({ kind: "file", file: null, url: "", preview: "" });
    setWanSeeds([]);
    const wanUnets = (catalog.unets || []).filter((item) => matchesHints(item, WAN_MODEL_HINTS));
    const unetSource = wanUnets.length > 0 ? wanUnets : catalog.unets || [];
    setWanUnetHighName(pickCatalogDefault(unetSource, DEFAULT_WAN_UNET_HIGH_NAME));
    setWanUnetLowName(pickCatalogDefault(unetSource, DEFAULT_WAN_UNET_LOW_NAME));
    setBaseSteps(4);
    setSteps(4);
    setBaseCfg(2.0);
    setCfg(1.0);
    setI2VResolution("720P");
    if (preset === "realistic_video") {
      setLoras([]);
      setEnableLora(false);
      return;
    }
    const animeRows = wanAnimeLoraOptions.slice(0, 2).map((item, idx) => ({
      id: crypto.randomUUID(),
      name: item.name,
      strength_model: idx === 0 ? 0.3 : 0.2,
      strength_clip: idx === 0 ? 0.3 : 0.2,
    }));
    setLoras(animeRows.length > 0 ? animeRows : [defaultLora()]);
  }

  function saveDefaultPayload() {
    localStorage.setItem(`default_payload_${mode}`, JSON.stringify(payload, null, 2));
    alert("Saved as default payload for mode: " + mode);
  }

  function loadPayloadFromJson() {
    setPayloadImportError("");
    try {
      const parsed = JSON.parse(payloadJsonText) as unknown;
      const source = extractImportSource(parsed);

      const importedMode = asMode(source.mode);
      if (importedMode) {
        setMode(importedMode);
      }

      if (typeof source.prompt === "string") {
        if (source.mode === "qwen_pose_fusion") {
          setQwenPoseFusionPrompt(source.prompt);
        } else if (source.mode === "wan2_2_i2v_extend_any_frame") {
          setWanExtendPrompt(source.prompt);
        } else {
          setPrompt(source.prompt);
        }
      }
      if (typeof source.negative_prompt === "string") {
        setNegativePrompt(source.negative_prompt);
      }
      if (typeof source.qwen_swap_prompt === "string") {
        setQwenSwapPrompt(source.qwen_swap_prompt);
      }
      if (typeof source.qwen_edit_prompt === "string") {
        setQwenEditPrompt(source.qwen_edit_prompt);
      }
      if (typeof source.frames === "number") {
        setFrames(source.frames);
      }
      if (typeof source.wan_unet_high_name === "string") {
        setWanUnetHighName(source.wan_unet_high_name);
      }
      if (typeof source.wan_unet_low_name === "string") {
        setWanUnetLowName(source.wan_unet_low_name);
      }
      if (typeof source.wan_vae_name === "string") {
        setWanVaeName(source.wan_vae_name);
      }
      if (typeof source.wan_clip_vision_name === "string") {
        setWanClipVisionName(source.wan_clip_vision_name);
      }
      if (typeof source.wan_clip_name === "string") {
        setWanClipName(source.wan_clip_name);
      }
      if (typeof source.startimg === "string" || typeof source.startimg === "object") {
        setWanStartMedia(mediaFromImportedValue(source.startimg));
      }
      if (typeof source.endimg === "string" || typeof source.endimg === "object") {
        setWanEndMedia(mediaFromImportedValue(source.endimg));
      }
      if (typeof source.qwen_model === "string") {
        setQwenModel(source.qwen_model);
      }
      if (typeof source.qwen_size === "string") {
        setQwenSize(source.qwen_size);
      }
      if (typeof source.ckpt_name === "string") {
        setCkptName(source.ckpt_name);
      }
      if (typeof source.width === "number") {
        setWidth(source.width);
      }
      if (typeof source.height === "number") {
        setHeight(source.height);
      }
      if (typeof source.batch_size === "number") {
        setBatchSize(source.batch_size);
      }
      if (typeof source.base_steps === "number") {
        setBaseSteps(source.base_steps);
      }
      if (typeof source.base_seed === "number") {
        setBaseSeed(source.base_seed);
      }
      if (typeof source.base_cfg === "number") {
        setBaseCfg(source.base_cfg);
      }
      if (typeof source.base_denoise === "number") {
        setBaseDenoise(source.base_denoise);
      }
      if (typeof source.base_sampler_name === "string") {
        setBaseSamplerName(source.base_sampler_name);
      }
      if (typeof source.base_scheduler === "string") {
        setBaseScheduler(source.base_scheduler);
      }
      if (typeof source.steps === "number") {
        setSteps(source.steps);
      }
      if (typeof source.seed === "number") {
        setSeed(source.seed);
      }
      if (typeof source.cfg === "number") {
        setCfg(source.cfg);
      }
      if (typeof source.sampler_name === "string") {
        setSamplerName(source.sampler_name);
      }
      if (typeof source.scheduler === "string") {
        setScheduler(source.scheduler);
      }
      if (typeof source.denoise === "number") {
        setDenoise(source.denoise);
      }
      if (typeof source.enable_pulid === "boolean") {
        setEnablePulid(source.enable_pulid);
      }
      if (typeof source.pulid_weight === "number") {
        setPulidWeight(source.pulid_weight);
      }
      if (typeof source.pulid_start_at === "number") {
        setPulidStartAt(source.pulid_start_at);
      }
      if (typeof source.pulid_end_at === "number") {
        setPulidEndAt(source.pulid_end_at);
      }
      if (typeof source.pulid_method === "string") {
        setPulidMethod(source.pulid_method);
      }
      if (typeof source.cn_depth_strength === "number") {
        setCnDepthStrength(source.cn_depth_strength);
      }
      if (typeof source.cn_depth_start_percent === "number") {
        setCnDepthStartPercent(source.cn_depth_start_percent);
      }
      if (typeof source.cn_depth_end_percent === "number") {
        setCnDepthEndPercent(source.cn_depth_end_percent);
      }
      if (typeof source.cn_pose_strength === "number") {
        setCnPoseStrength(source.cn_pose_strength);
      }
      if (typeof source.cn_pose_start_percent === "number") {
        setCnPoseStartPercent(source.cn_pose_start_percent);
      }
      if (typeof source.cn_pose_end_percent === "number") {
        setCnPoseEndPercent(source.cn_pose_end_percent);
      }
      if (typeof source.enable_lora === "boolean") {
        setEnableLora(source.enable_lora);
      }
      if (Array.isArray(source.loras)) {
        const rows = source.loras
          .filter((item) => item && typeof item === "object")
          .map((item) => {
            const row = item as Record<string, unknown>;
            return {
              id: crypto.randomUUID(),
              name: typeof row.name === "string" ? row.name : "",
              strength_model: typeof row.strength_model === "number" ? row.strength_model : 0.6,
              strength_clip: typeof row.strength_clip === "number" ? row.strength_clip : 0.9,
            } satisfies LoraRow;
          });
        setLoras(rows.length > 0 ? rows : [defaultLora()]);
      }
      if (typeof source.enable_upscale === "boolean") {
        setEnableUpscale(source.enable_upscale);
      }
      if (typeof source.upscale_model_name === "string") {
        setUpscaleModelName(source.upscale_model_name);
      }
      if (typeof source.keep_intermediate === "boolean") {
        setKeepIntermediate(source.keep_intermediate);
      }
      if (typeof source.enable_i2v === "boolean") {
        setEnableI2V(source.enable_i2v);
      }
      if (typeof source.i2v_prompt === "string") {
        setI2VPrompt(source.i2v_prompt);
      }
      if (typeof source.i2v_model === "string") {
        setI2VModel(source.i2v_model);
      }
      if (Array.isArray(source.wan_seeds)) {
        setWanSeeds(source.wan_seeds.map(Number));
      }
      if (typeof source.i2v_resolution === "string") {
        setI2VResolution(source.i2v_resolution);
      }
      if (typeof source.i2v_duration === "number") {
        setI2VDuration(source.i2v_duration);
      }
      if (typeof source.i2v_seed === "number") {
        setI2VSeed(source.i2v_seed);
      }
      if (typeof source.i2v_negative_prompt === "string") {
        setI2VNegativePrompt(source.i2v_negative_prompt);
      }
      if (typeof source.i2v_audio_url === "string") {
        setI2VAudioURL(source.i2v_audio_url);
      }
      if (typeof source.i2v_prompt_extend === "boolean") {
        setI2VPromptExtend(source.i2v_prompt_extend);
      }
      if (typeof source.i2v_watermark === "boolean") {
        setI2VWatermark(source.i2v_watermark);
      }
      if (typeof source.output_format === "string" && (source.output_format === "jpg" || source.output_format === "png")) {
        setOutputFormat(source.output_format);
      }
      if (typeof source.jpg_quality === "number") {
        setJpgQuality(source.jpg_quality);
      }

      setReferenceMedia(mediaFromImportedValue(source.reference_image));
      setPoseMedia(mediaFromImportedValue(source.pose_image));
      setQwenExtraMedia(mediaFromImportedValue(source.qwen_extra_image));
    } catch (err) {
      setPayloadImportError(String(err));
    }
  }

  function handleFileRead(key: "reference" | "pose" | "qwenExtra" | "wanStart" | "wanEnd", file: File) {
    if (!file) {
      return;
    }
    const reader = new FileReader();
    reader.onload = (ev) => {
      onURLChange(key, String(ev.target?.result || ""));
    };
    reader.readAsDataURL(file);
  }

  useEffect(() => {
    const saved = localStorage.getItem(`default_payload_${mode}`);
    if (saved) {
      setPayloadJsonText(saved);
    } else {
      setPayloadJsonText("");
    }
  }, [mode]);

  return (
    <div className="page">
      <aside className="sidebar">
        <h1>V16 Workflow Tester</h1>
        <p className="muted">
          React front-end + Go API for workflow preview, RunPod execution, and dynamic model catalog.
        </p>

        <section className="card">
          <h2>Runtime</h2>
          <div className="kv">
            <span>Engine</span>
            <strong>{health?.engine ?? "..."}</strong>
          </div>
          <div className="kv">
            <span>Catalog</span>
            <strong>{catalogError ? "error" : "ready"}</strong>
          </div>
          {catalogError && <pre className="errorBox">{catalogError}</pre>}
        </section>

        <section className="card">
          <h2>Mode</h2>
          <select value={mode} onChange={(e) => setMode(e.target.value as Mode)}>
            <option value="dual_pass_auto_pose">dual_pass_auto_pose: two-stage auto pose</option>
            <option value="pose_then_face_swap">pose_then_face_swap: pose first, then face swap</option>
            <option value="pose_only">pose_only: pose-guided only</option>
            <option value="text_only">text_only: prompt only</option>
            <option value="qwen_swap_face">qwen_swap_face: prompt then Qwen face swap</option>
            <option value="qwen_pose_fusion">qwen_pose_fusion: pose + face Qwen fusion</option>
            <option value="qwen_edit_face">qwen_edit_face: prompt then Qwen face edit</option>
            <option value="wan2_2_i2v_extend_any_frame">wan2.2 i2v-extend-any-frame</option>
          </select>
          <p className="muted compact">{modeSummary}</p>
        </section>

        {(mode === "dual_pass_auto_pose" || mode === "pose_then_face_swap" || mode === "qwen_swap_face") && (
          <MediaCard
            title={mode === "qwen_swap_face" ? "Qwen Reference Face Image" : "Reference Image"}
            media={referenceMedia}
            onKindChange={(kind) => updateMedia("reference", { kind })}
            onFileChange={(e) => onFileChange("reference", e)}
            onURLChange={(value) => onURLChange("reference", value)}
          />
        )}

        {mode === "qwen_pose_fusion" && (
          <MediaCard
            title="Qwen Face Image"
            media={referenceMedia}
            onKindChange={(kind) => updateMedia("reference", { kind })}
            onFileChange={(e) => onFileChange("reference", e)}
            onURLChange={(value) => onURLChange("reference", value)}
          />
        )}

        {mode === "qwen_swap_face" && (
          <MediaCard
            title="Qwen Optional Image 3"
            media={qwenExtraMedia}
            onKindChange={(kind) => updateMedia("qwenExtra", { kind })}
            onFileChange={(e) => onFileChange("qwenExtra", e)}
            onURLChange={(value) => onURLChange("qwenExtra", value)}
          />
        )}

        {mode === "qwen_edit_face" && (
          <section className="card">
            <h2>Qwen Edit</h2>
            <div className="stack">
              <label>
                Qwen Edit Prompt
                <textarea rows={4} value={qwenEditPrompt} onChange={(e) => setQwenEditPrompt(e.target.value)} />
              </label>
              <label>
                Qwen Model
                <input value={qwenModel} onChange={(e) => setQwenModel(e.target.value)} />
              </label>
              <label>
                Qwen Size
                <input placeholder="1024*1536" value={qwenSize} onChange={(e) => setQwenSize(e.target.value)} />
              </label>
            </div>
          </section>
        )}

        {(mode === "pose_then_face_swap" || mode === "pose_only") && (
          <MediaCard
            title="Pose Image"
            media={poseMedia}
            onKindChange={(kind) => updateMedia("pose", { kind })}
            onFileChange={(e) => onFileChange("pose", e)}
            onURLChange={(value) => onURLChange("pose", value)}
          />
        )}

        {mode === "qwen_pose_fusion" && (
          <MediaCard
            title="Qwen Pose Image"
            media={poseMedia}
            onKindChange={(kind) => updateMedia("pose", { kind })}
            onFileChange={(e) => onFileChange("pose", e)}
            onURLChange={(value) => onURLChange("pose", value)}
          />
        )}

        {mode === "wan2_2_i2v_extend_any_frame" && (
          <MediaCard
            title="WAN Start Image"
            media={wanStartMedia}
            onKindChange={(kind) => updateMedia("wanStart", { kind })}
            onFileChange={(e) => onFileChange("wanStart", e)}
            onURLChange={(value) => onURLChange("wanStart", value)}
          />
        )}

        {mode === "wan2_2_i2v_extend_any_frame" && (
          <MediaCard
            title="WAN End Image (Optional)"
            media={wanEndMedia}
            onKindChange={(kind) => updateMedia("wanEnd", { kind })}
            onFileChange={(e) => onFileChange("wanEnd", e)}
            onURLChange={(value) => onURLChange("wanEnd", value)}
          />
        )}
      </aside>

      <main className="main">
        <section className="card">
          <h2>{mode === "qwen_pose_fusion" ? "Qwen Pose Fusion Prompt" : mode === "wan2_2_i2v_extend_any_frame" ? "Wan Video Prompt" : "Prompt"}</h2>
          {mode === "qwen_pose_fusion" ? (
            <textarea rows={4} value={qwenPoseFusionPrompt} onChange={(e) => setQwenPoseFusionPrompt(e.target.value)} />
          ) : mode === "wan2_2_i2v_extend_any_frame" ? (
            <div className="stack">
              <textarea rows={4} value={wanExtendPrompt} onChange={(e) => setWanExtendPrompt(e.target.value)} />
              <div className="inline">
                <NumberField label="Total Duration (Seconds)" value={Math.floor((frames - 1) / 16)} onChange={(v) => setFrames(Math.max(1, v) * 16 + 1)} min={1} max={60} step={1} />
                <NumberField label="Total Frames" value={frames} onChange={setFrames} min={1} max={999999} step={1} />
              </div>
              <div className="inline" style={{ marginTop: "0.5rem" }}>
                <NumberField label="Segment Duration (Sec)" value={Math.floor((wanSegmentFrames - 1) / 16)} onChange={(v) => setWanSegmentFrames(Math.max(1, v) * 16 + 1)} min={1} max={30} step={1} />
                <NumberField label="Segment Frames" value={wanSegmentFrames} onChange={setWanSegmentFrames} min={2} max={1000} step={1} />
              </div>
              {frames > wanSegmentFrames && (
                <div className="stack" style={{ marginTop: "1rem" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <h4 style={{ margin: "0", fontSize: "0.9rem", opacity: 0.8 }}>Segment Configuration</h4>
                    <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                      <label style={{ display: "flex", alignItems: "center", gap: "4px", fontSize: "0.8rem", cursor: "pointer" }}>
                        <input type="checkbox" checked={autoSegmentPrompts} onChange={(e) => setAutoSegmentPrompts(e.target.checked)} />
                        Auto-Generate Segment Prompts
                      </label>
                      <button 
                        onClick={handleAiSplit} 
                        disabled={isAiSplitting}
                        style={{ padding: "4px 12px", fontSize: "0.8rem", cursor: "pointer", background: "linear-gradient(45deg, #8a2be2, #4b0082)", border: "none", color: "white", borderRadius: "12px", display: "flex", alignItems: "center", gap: "6px" }}
                      >
                        {isAiSplitting ? "✨ Orchestrating..." : "✨ AI Auto-Script (Manual)"}
                      </button>
                    </div>
                  </div>
                  <div className="stack" style={{ gap: '1rem' }}>
                    {Array.from({ length: Math.ceil((frames - 1) / Math.max(1, wanSegmentFrames - 1)) }).map((_, idx) => (
                      <div key={idx} style={{ padding: "0.5rem", border: "1px solid #444", borderRadius: "4px" }}>
                        <div style={{ marginBottom: "0.5rem", fontWeight: "bold" }}>Segment {idx + 1}</div>
                        <NumberField 
                          label="Seed" 
                          value={wanSeeds[idx] ?? seed + idx} 
                          onChange={(v) => {
                            const newSeeds = [...wanSeeds];
                            newSeeds[idx] = v;
                            setWanSeeds(newSeeds);
                          }} 
                          min={0} max={9999999999999999} step={1} 
                        />
                        <div style={{ marginTop: "0.5rem" }}>
                          <label style={{ display: "block", fontSize: "0.8rem", marginBottom: "0.2rem" }}>Segment Prompt (Leave blank to use main)</label>
                          <textarea 
                            rows={2} 
                            value={wanPrompts[idx] || ""} 
                            onChange={(e) => {
                              const newPrompts = [...wanPrompts];
                              newPrompts[idx] = e.target.value;
                              setWanPrompts(newPrompts);
                            }} 
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <textarea rows={4} value={prompt} onChange={(e) => setPrompt(e.target.value)} />
          )}
          <textarea rows={3} value={negativePrompt} onChange={(e) => setNegativePrompt(e.target.value)} />
          {mode === "qwen_swap_face" && (
            <div className="stack">
              <label>
                Qwen Swap Prompt
                <textarea rows={4} value={qwenSwapPrompt} onChange={(e) => setQwenSwapPrompt(e.target.value)} />
              </label>
              <label>
                Qwen Model
                <input value={qwenModel} onChange={(e) => setQwenModel(e.target.value)} />
              </label>
              <label>
                Qwen Size
                <input placeholder="1024*1536" value={qwenSize} onChange={(e) => setQwenSize(e.target.value)} />
              </label>
            </div>
          )}
        </section>

        <section className="grid two">
          {isWanMode ? (
            <section className="card">
              <h2>WAN Preset</h2>
              <label>
                Preset
                <select value={wanPreset} onChange={(e) => applyWanPreset(e.target.value as WanPreset)}>
                  <option value="manual">Manual</option>
                  <option value="realistic_video">Realistic Video</option>
                  <option value="anime_video">Anime Video</option>
                </select>
              </label>
              <p className="muted compact">
                Realistic keeps the stack minimal. Anime tries to prefill installed style LoRAs and a stylized prompt.
              </p>
              <div className="subcard">
                <h3 style={{ marginTop: 0 }}>WAN Main Models</h3>
                <p className="muted compact">
                  Choose the high and low UNet pair used by WAN before you generate video.
                </p>
                <div className="stack">
                  <label>
                    WAN High UNet
                    <select value={wanUnetHighName} onChange={(e) => setWanUnetHighName(e.target.value)}>
                      {wanModelOptions.map((name) => (
                        <option key={`wan-high-${name}`} value={name}>
                          {name}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label>
                    WAN Low UNet
                    <select value={wanUnetLowName} onChange={(e) => setWanUnetLowName(e.target.value)}>
                      {wanModelOptions.map((name) => (
                        <option key={`wan-low-${name}`} value={name}>
                          {name}
                        </option>
                      ))}
                    </select>
                  </label>
                </div>
                <p className="muted compact">
                  Active pair: {wanUnetHighName || "(unset)"} / {wanUnetLowName || "(unset)"}
                </p>
              </div>
              <h2>WAN LoRA Chain</h2>
              <label className="toggle">
                <input type="checkbox" checked={enableLora} onChange={(e) => setEnableLora(e.target.checked)} />
                Enable LoRA chain
              </label>
              {enableLora && (
                <div className="stack">
                  {loras.map((row) => (
                    <div className="subcard" key={row.id}>
                      <select value={row.name} onChange={(e) => updateLora(row.id, { name: e.target.value })}>
                        <option value="">Select LoRA</option>
                        {(catalog.loras || []).map((item) => (
                          <option key={item.path} value={item.name}>
                            {item.name}
                          </option>
                        ))}
                      </select>
                      <div className="inline">
                        <NumberField label="Model" value={row.strength_model} onChange={(v) => updateLora(row.id, { strength_model: v })} min={0} max={2} step={0.05} />
                        <NumberField label="Clip" value={row.strength_clip} onChange={(v) => updateLora(row.id, { strength_clip: v })} min={0} max={2} step={0.05} />
                      </div>
                      <button type="button" className="ghost" onClick={() => removeLora(row.id)}>
                        Remove
                      </button>
                    </div>
                  ))}
                  <button type="button" className="ghost" onClick={addLora}>
                    Add LoRA
                  </button>
                </div>
              )}
              <details className="subcard" open={wanAdvancedOpen} onToggle={(e) => setWanAdvancedOpen(e.currentTarget.open)}>
                <summary style={{ cursor: "pointer", fontWeight: 600 }}>WAN Advanced</summary>
                <div className="stack" style={{ marginTop: 12 }}>
                  <p className="muted compact">
                    Low-level video controls and runtime model names. Leave these alone unless you know the WAN workflow expects a different setting.
                  </p>
                  <div className="inline">
                    <label>
                      WAN VAE
                      <select value={wanVaeName} onChange={(e) => setWanVaeName(e.target.value)}>
                        {wanVaeOptions.map((name) => (
                          <option key={`wan-vae-${name}`} value={name}>
                            {name}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label>
                      WAN CLIP Vision
                      <select value={wanClipVisionName} onChange={(e) => setWanClipVisionName(e.target.value)}>
                        {wanClipVisionOptions.map((name) => (
                          <option key={`wan-clip-vision-${name}`} value={name}>
                            {name}
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>
                  <label>
                    WAN Text Encoder
                    <select value={wanClipName} onChange={(e) => setWanClipName(e.target.value)}>
                      {wanTextEncoderOptions.map((name) => (
                        <option key={`wan-text-encoder-${name}`} value={name}>
                          {name}
                        </option>
                      ))}
                    </select>
                  </label>
                  <div className="inline">
                    <NumberField label="High Noise Steps" value={baseSteps} onChange={setBaseSteps} min={1} max={50} step={1} />
                    <NumberField label="Low Noise Steps" value={steps} onChange={setSteps} min={1} max={50} step={1} />
                  </div>
                  <div className="inline">
                    <NumberField label="High Noise CFG" value={baseCfg} onChange={setBaseCfg} min={1} max={20} step={0.5} />
                    <NumberField label="Low Noise CFG" value={cfg} onChange={setCfg} min={1} max={20} step={0.5} />
                  </div>
                  <div className="inline">
                    <label>
                      WAN Resolution
                      <select value={i2vResolution} onChange={(e) => setI2VResolution(e.target.value)}>
                        <option value="480P">480P</option>
                        <option value="720P">720P</option>
                        <option value="1080P">1080P</option>
                      </select>
                    </label>
                  </div>
                </div>
              </details>
            </section>
          ) : (
            <section className="card">
              <h2>Model Selection</h2>
              <label>
                Checkpoint
                <select value={ckptName} onChange={(e) => setCkptName(e.target.value)}>
                  {(catalog.checkpoints || []).map((item) => (
                    <option key={item.path} value={item.name}>
                      {item.name}
                    </option>
                  ))}
                </select>
              </label>
              <label className="toggle">
                <input type="checkbox" checked={enableLora} onChange={(e) => setEnableLora(e.target.checked)} />
                Enable LoRA chain
              </label>
              {enableLora && (
                <div className="stack">
                  {loras.map((row) => (
                    <div className="subcard" key={row.id}>
                      <select value={row.name} onChange={(e) => updateLora(row.id, { name: e.target.value })}>
                        <option value="">Select LoRA</option>
                        {(catalog.loras || []).map((item) => (
                          <option key={item.path} value={item.name}>
                            {item.name}
                          </option>
                        ))}
                      </select>
                      <div className="inline">
                        <NumberField label="Model" value={row.strength_model} onChange={(v) => updateLora(row.id, { strength_model: v })} min={0} max={2} step={0.05} />
                        <NumberField label="Clip" value={row.strength_clip} onChange={(v) => updateLora(row.id, { strength_clip: v })} min={0} max={2} step={0.05} />
                      </div>
                      <button type="button" className="ghost" onClick={() => removeLora(row.id)}>
                        Remove
                      </button>
                    </div>
                  ))}
                  <button type="button" className="ghost" onClick={addLora}>
                    Add LoRA
                  </button>
                </div>
              )}
              <label className="toggle">
                <input type="checkbox" checked={enableUpscale} onChange={(e) => setEnableUpscale(e.target.checked)} />
                Enable 4x upscale
              </label>
              {enableUpscale && (
                <label>
                  Upscale Model
                  <select value={upscaleModelName} onChange={(e) => setUpscaleModelName(e.target.value)}>
                    {(catalog.upscale_models || []).map((item) => (
                      <option key={item.path} value={item.name}>
                        {item.name}
                      </option>
                    ))}
                  </select>
                </label>
              )}
            </section>
          )}

          {!isWanMode && (
            <section className="card">
              <h2>Stage Params</h2>
              <div className="inline">
                <NumberField label="Width" value={width} onChange={setWidth} min={256} max={2048} step={64} />
                <NumberField label="Height" value={height} onChange={setHeight} min={256} max={2048} step={64} />
                <NumberField label="Final Batch" value={batchSize} onChange={setBatchSize} min={1} max={8} step={1} />
              </div>
              <div className="inline">
                <label>
                  Final Sampler
                  <input value={samplerName} onChange={(e) => setSamplerName(e.target.value)} />
                </label>
                <label>
                  Final Scheduler
                  <input value={scheduler} onChange={(e) => setScheduler(e.target.value)} />
                </label>
              </div>
              <div className="inline">
                <NumberField label="Final Seed" value={seed} onChange={setSeed} min={0} max={9999999999999999} step={1} />
                <NumberField label="Final Denoise" value={denoise} onChange={setDenoise} min={0} max={1} step={0.05} />
              </div>
              <div className="inline">
                <NumberField label="Final Steps" value={steps} onChange={setSteps} min={1} max={120} step={1} />
                <NumberField label="Final CFG" value={cfg} onChange={setCfg} min={1} max={20} step={0.5} />
              </div>
              {mode === "dual_pass_auto_pose" && (
                <>
                  <div className="inline">
                    <NumberField label="Base Seed" value={baseSeed} onChange={setBaseSeed} min={0} max={9999999999999999} step={1} />
                    <NumberField label="Base Denoise" value={baseDenoise} onChange={setBaseDenoise} min={0} max={1} step={0.05} />
                  </div>
                  <div className="inline">
                    <NumberField label="Base Steps" value={baseSteps} onChange={setBaseSteps} min={1} max={100} step={1} />
                    <NumberField label="Base CFG" value={baseCfg} onChange={setBaseCfg} min={1} max={20} step={0.5} />
                  </div>
                  <div className="inline">
                    <label>
                      Base Sampler
                      <input value={baseSamplerName} onChange={(e) => setBaseSamplerName(e.target.value)} />
                    </label>
                    <label>
                      Base Scheduler
                      <input value={baseScheduler} onChange={(e) => setBaseScheduler(e.target.value)} />
                    </label>
                  </div>
                </>
              )}
            </section>
          )}
        </section>

        <section className="grid two">
          {!isWanMode && (
            <section className="card">
              <h2>Output</h2>
              <div className="inline">
                <label>
                  Output Format
                  <select value={outputFormat} onChange={(e) => setOutputFormat(e.target.value as "jpg" | "png")}>
                    <option value="jpg">jpg</option>
                    <option value="png">png</option>
                  </select>
                </label>
                <NumberField label="JPEG Quality" value={jpgQuality} onChange={setJpgQuality} min={1} max={100} step={1} />
              </div>
              <label className="toggle">
                <input type="checkbox" checked={keepIntermediate} onChange={(e) => setKeepIntermediate(e.target.checked)} />
                Keep intermediate outputs
              </label>
            </section>
          )}

          {isWanMode ? (
            <section className="card">
              <h2>WAN Workflow</h2>
              <p className="muted compact">
                Core inputs stay above. Advanced video controls are hidden in the WAN Advanced panel.
              </p>
            </section>
          ) : (
            <section className="card">
              <h2>I2V Postprocess</h2>
              <label className="toggle">
                <input type="checkbox" checked={enableI2V} onChange={(e) => setEnableI2V(e.target.checked)} />
                Enable i2v for final outputs
              </label>
              <div className="stack">
                <label>
                  I2V Prompt
                  <textarea rows={4} value={i2vPrompt} onChange={(e) => setI2VPrompt(e.target.value)} />
                </label>
                <div className="inline">
                  <label>
                    I2V Model
                    <input value={i2vModel} onChange={(e) => setI2VModel(e.target.value)} />
                  </label>
                  <label>
                    I2V Resolution
                    <select value={i2vResolution} onChange={(e) => setI2VResolution(e.target.value)}>
                      <option value="480P">480P</option>
                      <option value="720P">720P</option>
                      <option value="1080P">1080P</option>
                    </select>
                  </label>
                </div>
                <div className="inline">
                  <NumberField label="I2V Duration" value={i2vDuration} onChange={setI2VDuration} min={2} max={15} step={1} />
                  <NumberField label="I2V Seed" value={i2vSeed} onChange={setI2VSeed} min={0} max={2147483647} step={1} />
                </div>
                <label>
                  I2V Negative Prompt
                  <textarea rows={3} value={i2vNegativePrompt} onChange={(e) => setI2VNegativePrompt(e.target.value)} />
                </label>
                <label>
                  I2V Audio URL
                  <input placeholder="https://..." value={i2vAudioURL} onChange={(e) => setI2VAudioURL(e.target.value)} />
                </label>
                <div className="inline">
                  <label className="toggle">
                    <input type="checkbox" checked={i2vPromptExtend} onChange={(e) => setI2VPromptExtend(e.target.checked)} />
                    Prompt extend
                  </label>
                  <label className="toggle">
                    <input type="checkbox" checked={i2vWatermark} onChange={(e) => setI2VWatermark(e.target.checked)} />
                    Watermark
                  </label>
                </div>
              </div>
            </section>
          )}

          {!isWanMode && (
            <section className="card">
              <h2>PuLID</h2>
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={enablePulid}
                  onChange={(e) => setEnablePulid(e.target.checked)}
                  disabled={mode === "pose_only" || mode === "text_only" || mode === "qwen_swap_face" || mode === "qwen_pose_fusion" || mode === "qwen_edit_face"}
                />
                Enable PuLID
              </label>
              <label>
                Method
                <input value={pulidMethod} onChange={(e) => setPulidMethod(e.target.value)} />
              </label>
              <div className="inline">
                <NumberField label="Weight" value={pulidWeight} onChange={setPulidWeight} min={0} max={2} step={0.05} />
                <NumberField label="Start" value={pulidStartAt} onChange={setPulidStartAt} min={0} max={1} step={0.05} />
                <NumberField label="End" value={pulidEndAt} onChange={setPulidEndAt} min={0} max={1} step={0.05} />
              </div>
            </section>
          )}

          {!isWanMode &&
            mode !== "text_only" &&
            mode !== "qwen_swap_face" &&
            mode !== "qwen_pose_fusion" &&
            mode !== "qwen_edit_face" && (
              <section className="card">
                <h2>ControlNet</h2>
                <div className="inline">
                  <NumberField label="Depth Strength" value={cnDepthStrength} onChange={setCnDepthStrength} min={0} max={2} step={0.05} />
                  <NumberField label="Pose Strength" value={cnPoseStrength} onChange={setCnPoseStrength} min={0} max={2} step={0.05} />
                </div>
                <div className="inline">
                  <NumberField label="Depth Start" value={cnDepthStartPercent} onChange={setCnDepthStartPercent} min={0} max={1} step={0.05} />
                  <NumberField label="Depth End" value={cnDepthEndPercent} onChange={setCnDepthEndPercent} min={0} max={1} step={0.05} />
                </div>
                <div className="inline">
                  <NumberField label="Pose Start" value={cnPoseStartPercent} onChange={setCnPoseStartPercent} min={0} max={1} step={0.05} />
                  <NumberField label="Pose End" value={cnPoseEndPercent} onChange={setCnPoseEndPercent} min={0} max={1} step={0.05} />
                </div>
              </section>
            )}
        </section>

        <section className="card">
          <h2>Actions</h2>
          <div className="inline actions">
            <button type="button" disabled={busy !== ""} onClick={onRenderWorkflow}>
              {busy === "render" ? "Rendering..." : "Preview Workflow"}
            </button>
            <button type="button" disabled={busy !== ""} onClick={onGenerate}>
              {busy === "generate" ? "Generating..." : "Generate"}
            </button>
          </div>
          {error && <pre className="errorBox">{error}</pre>}
        </section>

        <section className="grid two">
          <section className="card">
            <h2>Import Payload</h2>
            <textarea
              rows={10}
              value={payloadJsonText}
              onChange={(e) => setPayloadJsonText(e.target.value)}
              placeholder='Paste request JSON here, then click "Load JSON into form".'
            />
            <div className="inline actions">
              <button type="button" disabled={!payloadJsonText.trim()} onClick={loadPayloadFromJson}>
                Load JSON into form
              </button>
              <button type="button" className="ghost" onClick={() => setPayloadJsonText("")}>
                Clear
              </button>
            </div>
            {payloadImportError && <pre className="errorBox">{payloadImportError}</pre>}
          </section>
          <section className="card">
            <h2>Payload Preview</h2>
            <div className="toolbar">
              <button type="button" className="ghost" onClick={() => void copyText(JSON.stringify(payload, null, 2))}>
                Copy Payload
              </button>
              <button type="button" className="ghost" onClick={saveDefaultPayload}>
                Save to default payload
              </button>
            </div>
            <pre>{JSON.stringify(payload, null, 2)}</pre>
          </section>
          <section className="card">
            <h2>Workflow Preview</h2>
            <pre>{renderResult ? JSON.stringify(renderResult, null, 2) : "No workflow preview yet."}</pre>
          </section>
        </section>

        <section className="card">
          <h2>Generation Result</h2>
          {generateResult ? (
            <>
              <ResultGallery result={generateResult} />
              <pre>{JSON.stringify(generateResult, null, 2)}</pre>
            </>
          ) : (
            <p className="muted">No generation result yet.</p>
          )}
        </section>
      </main>
      <TaskCenter tasks={tasks} clearTasks={clearTasks} />
    </div>
  );
}

function MediaCard({
  title,
  media,
  onKindChange,
  onFileChange,
  onURLChange,
}: {
  title: string;
  media: MediaState;
  onKindChange: (kind: "file" | "url") => void;
  onFileChange: (e: ChangeEvent<HTMLInputElement>) => void;
  onURLChange: (value: string) => void;
}) {
  return (
    <section className="card">
      <h2>{title}</h2>
      <div className="inline">
        <label className="toggle">
          <input type="radio" checked={media.kind === "file"} onChange={() => onKindChange("file")} />
          File
        </label>
        <label className="toggle">
          <input type="radio" checked={media.kind === "url"} onChange={() => onKindChange("url")} />
          URL
        </label>
      </div>
      <input type="file" accept="image/*" onChange={onFileChange} />
      <input
        placeholder="https://..."
        value={media.url}
        onChange={(e) => onURLChange(e.target.value)}
      />
      {media.preview && <img className="preview" src={media.preview} alt={title} />}
    </section>
  );
}

function ResultGallery({ result }: { result: GenerateResult }) {
  const finalURLs =
    extractStringArray(result, ["final_urls"]) ??
    extractStringArray(result, ["meta", "output", "final_urls"]) ??
    extractSingleStringArray(result, ["final_url"]) ??
    extractSingleStringArray(result, ["meta", "output", "final_url"]) ??
    [];
  const finalVideoURLs =
    extractStringArray(result, ["final_video_urls"]) ??
    extractStringArray(result, ["meta", "output", "final_video_urls"]) ??
    extractSingleStringArray(result, ["final_video_url"]) ??
    extractSingleStringArray(result, ["meta", "output", "final_video_url"]) ??
    [];
  const segmentVideoURLs =
    extractStringArray(result, ["segment_video_urls"]) ??
    extractStringArray(result, ["meta", "output", "segment_video_urls"]) ??
    [];
  const intermediates =
    extractStringArray(result, ["intermediate_urls"]) ??
    extractStringArray(result, ["meta", "output", "intermediate_urls"]) ??
    [];

  return (
    <div className="gallery">
      {finalURLs.map((url, idx) => (
        <div key={url} className="imageCard">
          <div className="imageCardHeader">
            <span>{finalURLs.length > 1 ? `Final ${idx + 1}` : "Final"}</span>
            <button type="button" className="ghost small" onClick={() => void copyText(url)}>
              Copy URL
            </button>
          </div>
          <a href={url} target="_blank" rel="noreferrer">
            <img src={url} alt={`final result ${idx + 1}`} />
          </a>
        </div>
      ))}
      {segmentVideoURLs.map((url, idx) => (
        <div key={url} className="imageCard">
          <div className="imageCardHeader">
            <span>{`Segment Video ${idx + 1}`}</span>
            <button type="button" className="ghost small" onClick={() => void copyText(url)}>
              Copy URL
            </button>
          </div>
          <a href={url} target="_blank" rel="noreferrer">
            <video className="mediaPreview" src={url} controls playsInline />
          </a>
        </div>
      ))}
      {finalVideoURLs.map((url, idx) => (
        <div key={url} className="imageCard">
          <div className="imageCardHeader">
            <span>{finalVideoURLs.length > 1 ? `Final Video ${idx + 1}` : "Final Video"}</span>
            <button type="button" className="ghost small" onClick={() => void copyText(url)}>
              Copy URL
            </button>
          </div>
          <a href={url} target="_blank" rel="noreferrer">
            <video className="mediaPreview" src={url} controls playsInline />
          </a>
        </div>
      ))}
      {intermediates.map((url) => (
        <div key={url} className="imageCard">
          <div className="imageCardHeader">
            <span>Intermediate</span>
            <button type="button" className="ghost small" onClick={() => void copyText(url)}>
              Copy URL
            </button>
          </div>
          <a href={url} target="_blank" rel="noreferrer">
            <img src={url} alt="intermediate result" />
          </a>
        </div>
      ))}
    </div>
  );
}

function NumberField({
  label,
  value,
  onChange,
  min,
  max,
  step,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step: number;
}) {
  return (
    <label>
      {label}
      <input
        type="number"
        value={Number.isFinite(value) ? value : ""}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}

async function resolveMedia(media: MediaState): Promise<string> {
  if (media.kind === "url" && media.url.trim()) {
    return media.url.trim();
  }
  if (!media.file) {
    throw new Error("Missing required image input.");
  }
  const arrayBuffer = await media.file.arrayBuffer();
  let binary = "";
  const bytes = new Uint8Array(arrayBuffer);
  for (const byte of bytes) {
    binary += String.fromCharCode(byte);
  }
  return btoa(binary);
}

async function resolveOptionalMedia(media: MediaState): Promise<string | undefined> {
  const value = media.kind === "url" ? media.url.trim() : media.file ? media.preview.trim() : "";
  if (!value) {
    return undefined;
  }
  return resolveMedia(media);
}

async function copyText(value: string): Promise<void> {
  await navigator.clipboard.writeText(value);
}

function asPlainObject(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  return value as Record<string, unknown>;
}

function extractImportSource(parsed: unknown): Record<string, unknown> {
  const root = asPlainObject(parsed);
  if (!root) {
    throw new Error("JSON payload must be an object.");
  }
  const input = asPlainObject(root.input);
  if (input) {
    return input;
  }
  const metaOutput = resolvePath(root, ["meta", "output"]);
  const metaOutputObj = asPlainObject(metaOutput);
  if (metaOutputObj) {
    const nestedMeta = asPlainObject(metaOutputObj.meta);
    return nestedMeta ? { ...metaOutputObj, ...nestedMeta } : metaOutputObj;
  }
  return root;
}

function asMode(value: unknown): Mode | undefined {
  if (
    value === "dual_pass_auto_pose" ||
    value === "pose_then_face_swap" ||
    value === "pose_only" ||
    value === "text_only" ||
    value === "qwen_swap_face" ||
    value === "qwen_pose_fusion" ||
    value === "qwen_edit_face" ||
    value === "wan2_2_i2v_extend_any_frame"
  ) {
    return value;
  }
  return undefined;
}

function mediaFromImportedValue(value: unknown): MediaState {
  if (typeof value === "string" && value.trim() !== "") {
    const trimmed = value.trim();
    return {
      kind: "url",
      file: null,
      url: trimmed,
      preview: trimmed,
    };
  }
  const obj = asPlainObject(value);
  if (obj) {
    const candidate = [obj.url, obj.image, obj.img_url, obj.reference_image, obj.pose_image]
      .find((item) => typeof item === "string" && item.trim() !== "");
    if (typeof candidate === "string") {
      const trimmed = candidate.trim();
      return {
        kind: "url",
        file: null,
        url: trimmed,
        preview: trimmed,
      };
    }
  }
  if (typeof value !== "string" || value.trim() === "") {
    return { kind: "file", file: null, url: "", preview: "" };
  }
  return { kind: "file", file: null, url: "", preview: "" };
}

function mediaToPayloadValue(media: MediaState): string | undefined {
  if (media.kind === "url") {
    const trimmed = media.url.trim();
    return trimmed || undefined;
  }
  const preview = media.preview.trim();
  return preview || undefined;
}

function extractSingleString(value: unknown, path: string[]): string | undefined {
  const resolved = resolvePath(value, path);
  return typeof resolved === "string" && resolved ? resolved : undefined;
}

function extractSingleStringArray(value: unknown, path: string[]): string[] | undefined {
  const resolved = extractSingleString(value, path);
  return resolved ? [resolved] : undefined;
}

function extractStringArray(value: unknown, path: string[]): string[] | undefined {
  const resolved = resolvePath(value, path);
  if (!Array.isArray(resolved)) {
    return undefined;
  }
  const items = resolved.filter((item): item is string => typeof item === "string" && item.trim() !== "");
  return items.length > 0 ? items : undefined;
}

function resolvePath(value: unknown, path: string[]): unknown {
  let current: unknown = value;
  for (const key of path) {
    if (!current || typeof current !== "object" || Array.isArray(current) || !(key in current)) {
      return undefined;
    }
    current = (current as Record<string, unknown>)[key];
  }
  return current;
}

export default App;
