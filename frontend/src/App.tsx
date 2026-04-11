import { ChangeEvent, useEffect, useMemo, useState } from "react";

type Mode = "dual_pass_auto_pose" | "pose_then_face_swap" | "pose_only" | "text_only";

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
  const [prompt, setPrompt] = useState("masterpiece, best quality, photorealistic, 1girl");
  const [negativePrompt, setNegativePrompt] = useState("blurry, low quality, watermark");
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
      if (catalogJson.checkpoints[0] && !ckptName) {
        setCkptName(catalogJson.checkpoints[0].name);
      }
      if (catalogJson.loras[0] && !loras[0]?.name) {
        setLoras((rows) =>
          rows.map((row, idx) => (idx === 0 ? { ...row, name: catalogJson.loras[0].name } : row))
        );
      }
      if (catalogJson.upscale_models[0] && !upscaleModelName) {
        setUpscaleModelName(catalogJson.upscale_models[0].name);
      }
    })().catch((err: unknown) => setCatalogError(String(err)));
  }, []);

  useEffect(() => {
    if (mode === "pose_only" || mode === "text_only") {
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
    return {
      mode,
      reference_image: undefined as string | undefined,
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
      output_format: outputFormat,
      jpg_quality: jpgQuality,
    };
  }, [
    mode,
    prompt,
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
    outputFormat,
    jpgQuality,
  ]);

  const modeSummary = {
    dual_pass_auto_pose: "Reference image + prompt. Stage 1 auto derives pose and depth, stage 2 renders final result.",
    pose_then_face_swap: "Reference image + pose image + prompt. Render with external pose, then apply face identity.",
    pose_only: "Pose image + prompt. Pose-guided generation without PuLID face swap.",
    text_only: "Prompt only. No reference image, no pose image, no PuLID.",
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

  function updateMedia(which: "reference" | "pose", patch: Partial<MediaState>) {
    const setter = which === "reference" ? setReferenceMedia : setPoseMedia;
    setter((prev) => ({ ...prev, ...patch }));
  }

  function onFileChange(which: "reference" | "pose", e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0] ?? null;
    if (!file) {
      updateMedia(which, { file: null, preview: "" });
      return;
    }
    const preview = URL.createObjectURL(file);
    updateMedia(which, { kind: "file", file, preview, url: "" });
  }

  function onURLChange(which: "reference" | "pose", value: string) {
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
          </select>
          <p className="muted compact">{modeSummary}</p>
        </section>

        {(mode === "dual_pass_auto_pose" || mode === "pose_then_face_swap") && (
          <MediaCard
            title="Reference Image"
            media={referenceMedia}
            onKindChange={(kind) => updateMedia("reference", { kind })}
            onFileChange={(e) => onFileChange("reference", e)}
            onURLChange={(value) => onURLChange("reference", value)}
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
        </section>

        <section className="grid two">
          <section className="card">
            <h2>Model Selection</h2>
            <label>
              Checkpoint
              <select value={ckptName} onChange={(e) => setCkptName(e.target.value)}>
                {catalog.checkpoints.map((item) => (
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
                      {catalog.loras.map((item) => (
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
                  {catalog.upscale_models.map((item) => (
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
            <h2>PuLID</h2>
            <label className="toggle">
              <input type="checkbox" checked={enablePulid} onChange={(e) => setEnablePulid(e.target.checked)} disabled={mode === "pose_only" || mode === "text_only"} />
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

          {mode !== "text_only" && (
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
