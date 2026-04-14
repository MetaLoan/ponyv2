import { ChangeEvent, useEffect, useMemo, useState } from "react";

type Mode = "dual_pass_auto_pose" | "pose_then_face_swap" | "pose_only" | "text_only" | "qwen_swap_face";

type CatalogItem = {
  name: string;
  path: string;
  type: string;
};

type CatalogResponse = {
  checkpoints: CatalogItem[];
  loras: CatalogItem[];
  upscale_models: CatalogItem[];
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

function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [catalog, setCatalog] = useState<CatalogResponse>({
    checkpoints: [],
    loras: [],
    upscale_models: [],
  });
  const [catalogError, setCatalogError] = useState<string>("");
  const [mode, setMode] = useState<Mode>("dual_pass_auto_pose");
  const [referenceMedia, setReferenceMedia] = useState<MediaState>({ kind: "file", file: null, url: "", preview: "" });
  const [poseMedia, setPoseMedia] = useState<MediaState>({ kind: "file", file: null, url: "", preview: "" });
  const [qwenExtraMedia, setQwenExtraMedia] = useState<MediaState>({ kind: "file", file: null, url: "", preview: "" });
  const [prompt, setPrompt] = useState(
    "masterpiece, best quality, ultra detailed, 8k raw photo, photorealistic, sharp focus, intricate details, rich skin texture, subsurface scattering, glossy skin, realistic anatomy, depth of field, volumetric lighting,"
  );
  const [negativePrompt, setNegativePrompt] = useState(
    "bad anatomy, poorly drawn hands, deformed hands, mutated hands, extra fingers, fused fingers, bad hands, blurry, low quality, worst quality, lowres, text, watermark, censored, ugly, deformed, extra limbs, bad proportions, open mouth, tongue out, tongue visible, saliva, oral sex, blowjob, fellatio, penis, any male genital, ahegao, rolling eyes"
  );
  const [qwenSwapPrompt, setQwenSwapPrompt] = useState(
    "将参考图中的人脸自然融合到生成图人物上，保持姿势、构图、光照、背景和服装不变，保证真实自然，五官清晰，肤质真实。"
  );
  const [qwenModel, setQwenModel] = useState("qwen-image-edit-max");
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
  const [i2vResolution, setI2VResolution] = useState("1080P");
  const [i2vDuration, setI2VDuration] = useState(5);
  const [i2vSeed, setI2VSeed] = useState<number>(12345);
  const [i2vNegativePrompt, setI2VNegativePrompt] = useState(
    "low quality, blurry, flicker, jitter, motion artifacts, deformed, extra limbs, bad proportions"
  );
  const [i2vAudioURL, setI2VAudioURL] = useState("");
  const [i2vPromptExtend, setI2VPromptExtend] = useState(true);
  const [i2vWatermark, setI2VWatermark] = useState(false);
  const [outputFormat, setOutputFormat] = useState<"jpg" | "png">("jpg");
  const [jpgQuality, setJpgQuality] = useState(85);
  const [renderResult, setRenderResult] = useState<GenerateResult | null>(null);
  const [generateResult, setGenerateResult] = useState<GenerateResult | null>(null);
  const [busy, setBusy] = useState<"" | "render" | "generate">("");
  const [error, setError] = useState("");

  useEffect(() => {
    void (async () => {
      const healthResp = await fetch("/api/health");
      const healthJson = (await healthResp.json()) as HealthResponse;
      setHealth(healthJson);

      const catalogResp = await fetch("/api/models/catalog");
      if (!catalogResp.ok) {
        const text = await catalogResp.text();
        setCatalogError(text);
        return;
      }
      const catalogJson = (await catalogResp.json()) as CatalogResponse;
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
    if (mode === "pose_only" || mode === "text_only" || mode === "qwen_swap_face") {
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
    const body: Record<string, unknown> = {
      mode,
      reference_image: undefined as string | undefined,
      qwen_extra_image: undefined as string | undefined,
      pose_image: undefined as string | undefined,
      prompt,
      negative_prompt: negativePrompt,
      width,
      height,
      batch_size: batchSize,
      ckpt_name: ckptName,
      base_steps: baseSteps,
      base_seed: baseSeed,
      base_cfg: baseCfg,
      base_sampler_name: baseSamplerName,
      base_scheduler: baseScheduler,
      base_denoise: baseDenoise,
      steps,
      seed,
      cfg,
      sampler_name: samplerName,
      scheduler,
      denoise,
      enable_pulid: enablePulid,
      pulid_weight: pulidWeight,
      pulid_start_at: pulidStartAt,
      pulid_end_at: pulidEndAt,
      pulid_method: pulidMethod,
      cn_depth_strength: cnDepthStrength,
      cn_depth_start_percent: cnDepthStartPercent,
      cn_depth_end_percent: cnDepthEndPercent,
      cn_pose_strength: cnPoseStrength,
      cn_pose_start_percent: cnPoseStartPercent,
      cn_pose_end_percent: cnPoseEndPercent,
      enable_lora: enableLora,
      loras: cleanLoras,
      enable_upscale: enableUpscale,
      upscale_model_name: upscaleModelName,
      keep_intermediate: keepIntermediate,
      enable_i2v: enableI2V,
      i2v_prompt: i2vPrompt,
      i2v_model: i2vModel,
      i2v_resolution: i2vResolution,
      i2v_duration: i2vDuration,
      i2v_seed: i2vSeed,
      i2v_negative_prompt: i2vNegativePrompt,
      i2v_audio_url: i2vAudioURL,
      i2v_prompt_extend: i2vPromptExtend,
      i2v_watermark: i2vWatermark,
      output_format: outputFormat,
      jpg_quality: jpgQuality,
    };
    if (mode === "qwen_swap_face") {
      body.qwen_swap_prompt = qwenSwapPrompt;
      body.qwen_model = qwenModel;
      body.qwen_size = qwenSize;
    }
    return body;
  }, [
    mode,
    prompt,
    qwenSwapPrompt,
    qwenModel,
    qwenSize,
    qwenExtraMedia,
    negativePrompt,
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
    enableLora,
    loras,
    enableUpscale,
    upscaleModelName,
    keepIntermediate,
    enableI2V,
    i2vPrompt,
    i2vModel,
    i2vResolution,
    i2vDuration,
    i2vSeed,
    i2vNegativePrompt,
    i2vAudioURL,
    i2vPromptExtend,
    i2vWatermark,
    outputFormat,
    jpgQuality,
  ]);

  const modeSummary = {
    dual_pass_auto_pose: "Reference image + prompt. Stage 1 auto derives pose and depth, stage 2 renders final result.",
    pose_then_face_swap: "Reference image + pose image + prompt. Render with external pose, then apply face identity.",
    pose_only: "Pose image + prompt. Pose-guided generation without PuLID face swap.",
    text_only: "Prompt only. No reference image, no pose image, no PuLID.",
    qwen_swap_face: "Base image is generated first, then Qwen uses reference face image and optional image 3 for face swap.",
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
      if (!resp.ok) {
        throw new Error(String((json as { error?: string }).error ?? "Request failed"));
      }
      if (target === "render") {
        setRenderResult(json);
      } else {
        setGenerateResult(json);
      }
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy("");
    }
  }

  function updateMedia(which: "reference" | "pose" | "qwenExtra", patch: Partial<MediaState>) {
    const setter = which === "reference" ? setReferenceMedia : which === "pose" ? setPoseMedia : setQwenExtraMedia;
    setter((prev) => ({ ...prev, ...patch }));
  }

  function onFileChange(which: "reference" | "pose" | "qwenExtra", e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0] ?? null;
    if (!file) {
      updateMedia(which, { file: null, preview: "" });
      return;
    }
    const preview = URL.createObjectURL(file);
    updateMedia(which, { kind: "file", file, preview, url: "" });
  }

  function onURLChange(which: "reference" | "pose" | "qwenExtra", value: string) {
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

        {mode === "qwen_swap_face" && (
          <MediaCard
            title="Qwen Optional Image 3"
            media={qwenExtraMedia}
            onKindChange={(kind) => updateMedia("qwenExtra", { kind })}
            onFileChange={(e) => onFileChange("qwenExtra", e)}
            onURLChange={(value) => onURLChange("qwenExtra", value)}
          />
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
      </aside>

      <main className="main">
        <section className="card">
          <h2>Prompt</h2>
          <textarea rows={4} value={prompt} onChange={(e) => setPrompt(e.target.value)} />
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
        </section>

        <section className="grid two">
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

          <section className="card">
            <h2>PuLID</h2>
            <label className="toggle">
              <input
                type="checkbox"
                checked={enablePulid}
                onChange={(e) => setEnablePulid(e.target.checked)}
                disabled={mode === "pose_only" || mode === "text_only" || mode === "qwen_swap_face"}
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

          {mode !== "text_only" && mode !== "qwen_swap_face" && (
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
            <h2>Payload Preview</h2>
            <div className="toolbar">
              <button type="button" className="ghost" onClick={() => void copyText(JSON.stringify(payload, null, 2))}>
                Copy Payload
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

async function copyText(value: string): Promise<void> {
  await navigator.clipboard.writeText(value);
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
