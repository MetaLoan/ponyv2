package v16web

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"encoding/xml"
	"errors"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"log"
	"mime"
	"mime/multipart"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
)

type App struct {
	Config     Config
	httpClient *http.Client
	router     *http.ServeMux
	s3Client   *minio.Client
}

type Config struct {
	Addr                string
	RepoRoot            string
	FrontendDist        string
	WorkflowTemplate    string
	WanWorkflowTemplate string
	KeyEnvFile          string
	S3CredsFile         string
	ComfyAPIURL         string
	RunPodAPIKey        string
	RunPodEndpointID    string
	S3Endpoint          string
	S3Region            string
	S3Bucket            string
	S3AccessKey         string
	S3SecretKey         string
	S3RootPrefix        string
	S3CatalogPython     string
	S3CatalogScript     string
}

type GenerateRequest struct {
	Mode                string       `json:"mode"`
	ReferenceImage      string       `json:"reference_image"`
	StartImg            string       `json:"startimg"`
	EndImg              string       `json:"endimg"`
	WanVaeName          string       `json:"wan_vae_name"`
	WanClipVisionName   string       `json:"wan_clip_vision_name"`
	WanClipName         string       `json:"wan_clip_name"`
	WanUnetHighName     string       `json:"wan_unet_high_name"`
	WanUnetLowName      string       `json:"wan_unet_low_name"`
	QwenExtraImage      string       `json:"qwen_extra_image"`
	PoseImage           string       `json:"pose_image"`
	Prompt              string       `json:"prompt"`
	Frames              int          `json:"frames"`
	QwenSwapPrompt      string       `json:"qwen_swap_prompt"`
	QwenEditPrompt      string       `json:"qwen_edit_prompt"`
	QwenModel           string       `json:"qwen_model"`
	QwenSize            string       `json:"qwen_size"`
	NegativePrompt      string       `json:"negative_prompt"`
	Width               int          `json:"width"`
	Height              int          `json:"height"`
	BatchSize           int          `json:"batch_size"`
	CKPTName            string       `json:"ckpt_name"`
	BaseSteps           int          `json:"base_steps"`
	BaseSeed            int64        `json:"base_seed"`
	BaseCFG             float64      `json:"base_cfg"`
	BaseSamplerName     string       `json:"base_sampler_name"`
	BaseScheduler       string       `json:"base_scheduler"`
	BaseDenoise         float64      `json:"base_denoise"`
	Steps               int          `json:"steps"`
	Seed                int64        `json:"seed"`
	CFG                 float64      `json:"cfg"`
	SamplerName         string       `json:"sampler_name"`
	Scheduler           string       `json:"scheduler"`
	Denoise             float64      `json:"denoise"`
	EnablePulid         *bool        `json:"enable_pulid"`
	PulidMethod         string       `json:"pulid_method"`
	PulidWeight         float64      `json:"pulid_weight"`
	PulidStartAt        float64      `json:"pulid_start_at"`
	PulidEndAt          float64      `json:"pulid_end_at"`
	CNDepthStrength     float64      `json:"cn_depth_strength"`
	CNDepthStartPercent float64      `json:"cn_depth_start_percent"`
	CNDepthEndPercent   float64      `json:"cn_depth_end_percent"`
	CNPoseStrength      float64      `json:"cn_pose_strength"`
	CNPoseStartPercent  float64      `json:"cn_pose_start_percent"`
	CNPoseEndPercent    float64      `json:"cn_pose_end_percent"`
	EnableLora          *bool        `json:"enable_lora"`
	Loras               []LoraConfig `json:"loras"`
	EnableUpscale       *bool        `json:"enable_upscale"`
	UpscaleModelName    string       `json:"upscale_model_name"`
	OutputFormat        string       `json:"output_format"`
	JPGQuality          int          `json:"jpg_quality"`
	KeepIntermediate    *bool        `json:"keep_intermediate"`
	EnableI2V           *bool        `json:"enable_i2v"`
	I2VPrompt           string       `json:"i2v_prompt"`
	I2VModel            string       `json:"i2v_model"`
	I2VResolution       string       `json:"i2v_resolution"`
	I2VDuration         int          `json:"i2v_duration"`
	I2VSeed             int64        `json:"i2v_seed"`
	I2VNegativePrompt   string       `json:"i2v_negative_prompt"`
	I2VAudioURL         string       `json:"i2v_audio_url"`
	I2VPromptExtend     *bool        `json:"i2v_prompt_extend"`
	I2VWatermark        *bool        `json:"i2v_watermark"`
	RequestID           string       `json:"request_id"`
	WanSeeds            []int64      `json:"wan_seeds"`
	WanPrompts          []string     `json:"wan_prompts"`
	SegmentLimit        int          `json:"segment_limit"`
	AutoSegmentPrompts  bool         `json:"auto_segment_prompts"`
	Async               bool         `json:"async"`
}

type LoraConfig struct {
	Name          string  `json:"name"`
	Strength      float64 `json:"strength"`
	StrengthModel float64 `json:"strength_model"`
	StrengthClip  float64 `json:"strength_clip"`
}

type RenderResponse struct {
	Mode             string                 `json:"mode"`
	Engine           string                 `json:"engine"`
	Warnings         []string               `json:"warnings,omitempty"`
	NormalizedInput  GenerateRequest        `json:"normalized_input"`
	RenderedWorkflow map[string]interface{} `json:"rendered_workflow"`
}

type GenerateResponse struct {
	OK               bool                   `json:"ok"`
	Mode             string                 `json:"mode"`
	Engine           string                 `json:"engine"`
	Warnings         []string               `json:"warnings,omitempty"`
	PromptID         string                 `json:"prompt_id,omitempty"`
	JobID            string                 `json:"job_id,omitempty"`
	Status           string                 `json:"status,omitempty"`
	RequestID        string                 `json:"request_id,omitempty"`
	FinalURL         string                 `json:"final_url,omitempty"`
	FinalURLs        []string               `json:"final_urls,omitempty"`
	FinalVideoURL    string                 `json:"final_video_url,omitempty"`
	FinalVideoURLs   []string               `json:"final_video_urls,omitempty"`
	SegmentVideoURLs []string               `json:"segment_video_urls,omitempty"`
	IntermediateURLs []string               `json:"intermediate_urls,omitempty"`
	Error            string                 `json:"error,omitempty"`
	Meta             map[string]interface{} `json:"meta,omitempty"`
	Raw              map[string]interface{} `json:"raw,omitempty"`
}

type CatalogResponse struct {
	Checkpoints   []CatalogItem `json:"checkpoints"`
	Unets         []CatalogItem `json:"unets"`
	Loras         []CatalogItem `json:"loras"`
	UpscaleModels []CatalogItem `json:"upscale_models"`
	Vaes          []CatalogItem `json:"vaes"`
	ClipVisions   []CatalogItem `json:"clip_visions"`
	TextEncoders  []CatalogItem `json:"text_encoders"`
}

type CatalogItem struct {
	Name string `json:"name"`
	Path string `json:"path"`
	Type string `json:"type"`
}

type comfyImage struct {
	Filename  string `json:"filename"`
	Subfolder string `json:"subfolder"`
	Type      string `json:"type"`
}

const (
	qwenAPIURL                     = "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
	qwenModelDefault               = "qwen-image-2.0-pro"
	qwenDataInspectionHeader       = "{\"input\":\"disable\", \"output\":\"disable\"}"
	qwenDefaultSwapPrompt          = "以图1为最终画面底图，严格保留图1的人物姿势、构图、服装、光照、背景和沙滩环境；仅将图2中的面部特征自然融合到图1人物脸上，保持真实自然、五官清晰、肤质统一；图3如存在，仅作为辅助参考，不要改变其他区域。"
	qwenDefaultPoseFusionPrompt    = "以图1的pose图作为最终构图底图，严格保留人物姿势、肢体角度、镜头、服装、场景和光照；将图2的face图中的面部身份特征自然融合到图1人物脸上；保持五官清晰、肤质统一、真实摄影感，不要改动背景、衣服、身体姿态或镜头结构，融合结果要自然连贯。"
	qwenDefaultEditPrompt          = "将图中的角色脸部特征形象进行调整，使其符合如下描述中关于脸部的特征描述:{{生图提示词的主提示词变量}}"
	qwenDirectPoseFusionMode       = "qwen_pose_fusion"
	wanExtendAnyFrameMode          = "wan2_2_i2v_extend_any_frame"
	wanExtendAnyFrameModel         = "wan2.2-kf2v-flash"
	wanExtendAnyFrameSegmentLimit  = 81
	wanExtendAnyFrameDefaultPrompt = "沙滩，海边，晴天，自然光，蓝天白云，海浪，金色细沙，轻微海风，真实摄影感，画面通透，动作自然连贯，镜头稳定，细节清晰，电影感成片"
)

func NewApp() (*App, error) {
	repoRoot := mustGetwd()
	cfg := Config{
		Addr:                envOrDefault("V16WEB_ADDR", ":8080"),
		RepoRoot:            repoRoot,
		FrontendDist:        filepath.Join(repoRoot, "frontend", "dist"),
		WorkflowTemplate:    filepath.Join(repoRoot, "workflows", "pulid_sdxl_workflow_web_api.json"),
		WanWorkflowTemplate: filepath.Join(repoRoot, "workflows", "wan2_2_i2v_extend_any_frame_api.json"),
		KeyEnvFile:          envOrDefault("KEY_ENV_FILE", filepath.Clean(filepath.Join(repoRoot, "..", "key.env"))),
		S3CredsFile:         envOrDefault("S3_CREDENTIALS_FILE", filepath.Clean(filepath.Join(repoRoot, "..", "s3-credentials.txt"))),
		S3RootPrefix:        envOrDefault("S3_MODEL_ROOT_PREFIX", "runpod-slim/ComfyUI/models"),
		S3Region:            envOrDefault("S3_REGION", "eu-ro-1"),
		S3CatalogScript:     envOrDefault("S3_CATALOG_SCRIPT", filepath.Join(repoRoot, "scripts", "list_model_catalog.py")),
	}
	cfg.S3CatalogPython = detectCatalogPython()
	loadKeyEnvFile(cfg.KeyEnvFile)
	cfg.ComfyAPIURL = strings.TrimRight(os.Getenv("COMFY_API_URL"), "/")
	cfg.RunPodAPIKey = firstNonEmpty(os.Getenv("RUNPOD_API_KEY"), os.Getenv("runpod"))
	cfg.RunPodEndpointID = firstNonEmpty(os.Getenv("RUNPOD_ENDPOINT_ID"), os.Getenv("Endpoint"))
	loadS3CredsFile(&cfg)

	app := &App{
		Config: cfg,
		httpClient: &http.Client{
			Timeout: 120 * time.Second,
		},
		router: http.NewServeMux(),
	}
	if cfg.S3Endpoint != "" && cfg.S3AccessKey != "" && cfg.S3SecretKey != "" && cfg.S3Bucket != "" {
		endpoint := strings.TrimPrefix(strings.TrimPrefix(cfg.S3Endpoint, "https://"), "http://")
		client, err := minio.New(endpoint, &minio.Options{
			Creds:        credentials.NewStaticV4(cfg.S3AccessKey, cfg.S3SecretKey, ""),
			Secure:       strings.HasPrefix(cfg.S3Endpoint, "https://"),
			Region:       cfg.S3Region,
			BucketLookup: minio.BucketLookupPath,
		})
		if err != nil {
			return nil, err
		}
		app.s3Client = client
	}
	app.routes()
	return app, nil
}

func (a *App) ListenAndServe() error {
	return http.ListenAndServe(a.Config.Addr, a.router)
}

func (a *App) routes() {
	a.router.HandleFunc("/api/health", a.handleHealth)
	a.router.HandleFunc("/api/models/catalog", a.handleCatalog)
	a.router.HandleFunc("/api/workflow/render", a.handleRenderWorkflow)
	a.router.HandleFunc("/api/generate", a.handleGenerate)
	a.router.HandleFunc("/api/status", a.handleStatus)
	a.router.HandleFunc("/api/comfy/view", a.handleComfyView)
	a.router.HandleFunc("/api/ai/split_prompt", a.handleAiSplitPrompt)

	fs := http.FileServer(http.Dir(a.Config.FrontendDist))
	a.router.Handle("/", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/api/") {
			http.NotFound(w, r)
			return
		}
		p := filepath.Join(a.Config.FrontendDist, strings.TrimPrefix(path.Clean(r.URL.Path), "/"))
		if info, err := os.Stat(p); err == nil && !info.IsDir() {
			fs.ServeHTTP(w, r)
			return
		}
		http.ServeFile(w, r, filepath.Join(a.Config.FrontendDist, "index.html"))
	}))
}

func (a *App) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"ok":            true,
		"engine":        a.engine(),
		"comfy_api_url": a.Config.ComfyAPIURL,
		"runpod_ready":  a.Config.RunPodAPIKey != "" && a.Config.RunPodEndpointID != "",
		"s3_ready":      a.s3Client != nil,
	})
}

func (a *App) handleCatalog(w http.ResponseWriter, r *http.Request) {
	if resp, err := a.catalogViaScript(r.Context()); err == nil {
		writeJSON(w, http.StatusOK, resp)
		return
	}

	if a.s3Client == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]interface{}{
			"error": "S3 catalog is not configured",
		})
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()

	resp := CatalogResponse{
		Checkpoints:   a.listCatalog(ctx, path.Join(a.Config.S3RootPrefix, "checkpoints")+"/", "checkpoint"),
		Unets:         a.listCatalog(ctx, path.Join(a.Config.S3RootPrefix, "unet")+"/", "unet"),
		Loras:         a.listCatalog(ctx, path.Join(a.Config.S3RootPrefix, "loras")+"/", "lora"),
		UpscaleModels: a.listCatalog(ctx, path.Join(a.Config.S3RootPrefix, "upscale_models")+"/", "upscale_model"),
		Vaes:          a.listCatalog(ctx, path.Join(a.Config.S3RootPrefix, "vae")+"/", "vae"),
		ClipVisions:   a.listCatalog(ctx, path.Join(a.Config.S3RootPrefix, "clip_vision")+"/", "clip_vision"),
		TextEncoders:  a.listCatalog(ctx, path.Join(a.Config.S3RootPrefix, "text_encoders")+"/", "text_encoder"),
	}
	writeJSON(w, http.StatusOK, resp)
}

func (a *App) catalogViaScript(ctx context.Context) (*CatalogResponse, error) {
	if a.Config.S3CatalogPython == "" || a.Config.S3CatalogScript == "" {
		return nil, errors.New("catalog script not configured")
	}
	if _, err := os.Stat(a.Config.S3CatalogScript); err != nil {
		return nil, err
	}
	cmd := exec.CommandContext(
		ctx,
		a.Config.S3CatalogPython,
		a.Config.S3CatalogScript,
		a.Config.S3CredsFile,
		a.Config.S3RootPrefix,
	)
	cmd.Env = append(os.Environ(), "S3_REGION="+a.Config.S3Region)
	out, err := cmd.Output()
	if err != nil {
		return nil, err
	}
	var resp CatalogResponse
	if err := json.Unmarshal(out, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

func (a *App) handleRenderWorkflow(w http.ResponseWriter, r *http.Request) {
	req, err := decodeGenerateRequest(r)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}
	if req.Mode == qwenDirectPoseFusionMode {
		writeJSON(w, http.StatusOK, RenderResponse{
			Mode:   req.Mode,
			Engine: "qwen",
			Warnings: []string{
				"qwen_pose_fusion uses direct DashScope fusion and has no Comfy workflow",
			},
			NormalizedInput: req,
			RenderedWorkflow: map[string]interface{}{
				"mode": req.Mode,
				"note": "Direct Qwen fusion preview. Face image is image 2 and pose image is image 1.",
			},
		})
		return
	}
	if req.Mode == wanExtendAnyFrameMode {
		segmentCount := 1
		if req.Frames > 0 {
			segmentCount = (req.Frames + wanExtendAnyFrameSegmentLimit - 1) / wanExtendAnyFrameSegmentLimit
		}
		writeJSON(w, http.StatusOK, RenderResponse{
			Mode:   req.Mode,
			Engine: "wan",
			Warnings: []string{
				"wan2.2 i2v extend mode is orchestrated by the RunPod worker and has no Comfy workflow",
			},
			NormalizedInput: req,
			RenderedWorkflow: map[string]interface{}{
				"mode":               req.Mode,
				"segment_limit":      wanExtendAnyFrameSegmentLimit,
				"segment_count":      segmentCount,
				"frames":             req.Frames,
				"wan_unet_high_name": req.WanUnetHighName,
				"wan_unet_low_name":  req.WanUnetLowName,
				"workflow_template":  a.Config.WanWorkflowTemplate,
				"startimg_required":  true,
				"endimg_optional":    true,
				"prompt_field":       "prompt",
				"final_output":       "merged_video + per-segment_videos",
			},
		})
		return
	}
	rendered, warnings, err := a.renderWorkflow(req, false)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, RenderResponse{
		Mode:             req.Mode,
		Engine:           a.engine(),
		Warnings:         warnings,
		NormalizedInput:  req,
		RenderedWorkflow: rendered,
	})
}

func (a *App) handleGenerate(w http.ResponseWriter, r *http.Request) {
	req, err := decodeGenerateRequest(r)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}
	if req.Mode == qwenDirectPoseFusionMode {
		resp, err := a.generateWithQwenPoseFusion(r.Context(), req)
		if err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
			return
		}
		writeJSON(w, http.StatusOK, resp)
		return
	}
	if req.Mode == wanExtendAnyFrameMode && a.engine() != "runpod" {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{
			"error": "wan2.2 i2v extend mode currently requires the RunPod worker path.",
		})
		return
	}

	switch a.engine() {
	case "comfy":
		resp, err := a.generateWithComfy(r.Context(), req)
		if err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
			return
		}
		writeJSON(w, http.StatusOK, resp)
	case "runpod":
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		flusher, ok := w.(http.Flusher)
		if ok {
			flusher.Flush()
		}

		done := make(chan struct{})
		go func() {
			ticker := time.NewTicker(30 * time.Second)
			defer ticker.Stop()
			for {
				select {
				case <-done:
					return
				case <-ticker.C:
					w.Write([]byte(" "))
					if ok {
						flusher.Flush()
					}
				}
			}
		}()

		res, err := a.generateWithRunPod(r.Context(), req)
		close(done)

		if err != nil {
			res = &GenerateResponse{
				OK:    false,
				Error: err.Error(),
			}
		}
		json.NewEncoder(w).Encode(res)
	default:
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{
			"error": "No execution engine configured. Set COMFY_API_URL or RUNPOD_API_KEY/RUNPOD_ENDPOINT_ID.",
		})
	}
}

func (a *App) handleComfyView(w http.ResponseWriter, r *http.Request) {
	if a.Config.ComfyAPIURL == "" {
		http.Error(w, "COMFY_API_URL not configured", http.StatusBadRequest)
		return
	}
	q := r.URL.Query()
	viewURL := a.Config.ComfyAPIURL + "/view?" + q.Encode()
	req, err := http.NewRequestWithContext(r.Context(), http.MethodGet, viewURL, nil)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	resp, err := a.httpClient.Do(req)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()
	for k, vals := range resp.Header {
		for _, v := range vals {
			w.Header().Add(k, v)
		}
	}
	w.WriteHeader(resp.StatusCode)
	_, _ = io.Copy(w, resp.Body)
}

func (a *App) handleAiSplitPrompt(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Prompt   string `json:"prompt"`
		Segments int    `json:"segments"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}

	apiKey := os.Getenv("DASHSCOPE_API_KEY")
	if apiKey == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "DASHSCOPE_API_KEY not set in web environment"})
		return
	}

	systemMsg := "You are a video storyboard assistant. The user is generating a long video by splitting a main action into multiple segments. Decompose the main prompt into a sequence of detailed sub-actions, one for each segment. Output ONLY the prompts, one per line."
	userMsg := fmt.Sprintf("Main prompt: '%s'. Please provide %d segment prompts.", req.Prompt, req.Segments)

	payload := map[string]interface{}{
		"model": "qwen-plus",
		"input": map[string]interface{}{
			"messages": []map[string]string{
				{"role": "system", "content": systemMsg},
				{"role": "user", "content": userMsg},
			},
		},
		"parameters": map[string]interface{}{
			"result_format": "message",
		},
	}

	jsonPayload, _ := json.Marshal(payload)
	httpReq, _ := http.NewRequestWithContext(r.Context(), "POST", "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/text-generation/generation", bytes.NewBuffer(jsonPayload))
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+apiKey)

	resp, err := a.httpClient.Do(httpReq)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	defer resp.Body.Close()

	var dashscopeResp struct {
		Output struct {
			Choices []struct {
				Message struct {
					Content string `json:"content"`
				} `json:"message"`
			} `json:"choices"`
		} `json:"output"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&dashscopeResp); err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to parse dashscope response"})
		return
	}

	if len(dashscopeResp.Output.Choices) == 0 {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "empty response from AI"})
		return
	}

	content := dashscopeResp.Output.Choices[0].Message.Content
	lines := strings.Split(strings.TrimSpace(content), "\n")
	var prompts []string
	for _, l := range lines {
		if t := strings.TrimSpace(l); t != "" {
			prompts = append(prompts, t)
		}
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{"prompts": prompts})
}


func (a *App) engine() string {
	if a.Config.ComfyAPIURL != "" {
		return "comfy"
	}
	if a.Config.RunPodAPIKey != "" && a.Config.RunPodEndpointID != "" {
		return "runpod"
	}
	return "none"
}

func (a *App) listCatalog(ctx context.Context, prefix, kind string) []CatalogItem {
	items := []CatalogItem{}
	if a.Config.S3Endpoint == "" || a.Config.S3AccessKey == "" || a.Config.S3SecretKey == "" || a.Config.S3Bucket == "" {
		return items
	}

	result, err := a.s3ListObjectsV2(ctx, prefix)
	if err != nil {
		log.Printf("catalog list failed for %s: %v", prefix, err)
		return items
	}

	for _, obj := range result.Contents {
		key := obj.Key
		// Decode URL-encoded keys (encoding-type=url)
		if decoded, err := url.QueryUnescape(key); err == nil {
			key = decoded
		}
		if strings.HasSuffix(key, "/") {
			continue
		}
		name := path.Base(key)
		if strings.HasPrefix(name, "put_") {
			continue
		}
		if !isAllowedModelFile(name, kind) {
			continue
		}
		items = append(items, CatalogItem{Name: name, Path: key, Type: kind})
	}
	sort.Slice(items, func(i, j int) bool { return items[i].Name < items[j].Name })
	return items
}

// s3ListObjectsV2Result represents the XML response from S3 ListObjectsV2.
type s3ListObjectsV2Result struct {
	XMLName     xml.Name   `xml:"ListBucketResult"`
	Contents    []s3Object `xml:"Contents"`
	IsTruncated bool       `xml:"IsTruncated"`
}

type s3Object struct {
	Key  string `xml:"Key"`
	Size int64  `xml:"Size"`
}

// s3ListObjectsV2 calls S3 ListObjectsV2 via HTTP with AWS Signature V4,
// bypassing minio-go which adds encoding-type=url that RunPod S3 doesn't support.
func (a *App) s3ListObjectsV2(ctx context.Context, prefix string) (*s3ListObjectsV2Result, error) {
	endpointURL := strings.TrimRight(a.Config.S3Endpoint, "/")
	region := a.Config.S3Region
	bucket := a.Config.S3Bucket

	// Build the request URL: <endpoint>/<bucket>?list-type=2&prefix=...
	query := url.Values{}
	query.Set("list-type", "2")
	query.Set("prefix", prefix)
	query.Set("delimiter", "/")
	query.Set("max-keys", "1000")
	query.Set("encoding-type", "url")

	reqURL := fmt.Sprintf("%s/%s?%s", endpointURL, bucket, query.Encode())

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, reqURL, nil)
	if err != nil {
		return nil, err
	}

	// Sign with AWS Signature V4
	now := time.Now().UTC()
	dateStamp := now.Format("20060102")
	amzDate := now.Format("20060102T150405Z")

	req.Header.Set("x-amz-date", amzDate)
	req.Header.Set("x-amz-content-sha256", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855") // empty body hash
	req.Header.Set("Host", req.URL.Host)

	// Canonical request
	canonicalURI := "/" + bucket
	canonicalQuerystring := query.Encode()
	canonicalHeaders := fmt.Sprintf("host:%s\nx-amz-content-sha256:%s\nx-amz-date:%s\n",
		req.URL.Host, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", amzDate)
	signedHeaders := "host;x-amz-content-sha256;x-amz-date"
	payloadHash := "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

	canonicalRequest := fmt.Sprintf("%s\n%s\n%s\n%s\n%s\n%s",
		"GET", canonicalURI, canonicalQuerystring, canonicalHeaders, signedHeaders, payloadHash)

	// String to sign
	scope := fmt.Sprintf("%s/%s/s3/aws4_request", dateStamp, region)
	canonicalRequestHash := sha256Hex([]byte(canonicalRequest))
	stringToSign := fmt.Sprintf("AWS4-HMAC-SHA256\n%s\n%s\n%s", amzDate, scope, canonicalRequestHash)

	// Signing key
	kDate := hmacSHA256([]byte("AWS4"+a.Config.S3SecretKey), []byte(dateStamp))
	kRegion := hmacSHA256(kDate, []byte(region))
	kService := hmacSHA256(kRegion, []byte("s3"))
	kSigning := hmacSHA256(kService, []byte("aws4_request"))

	signature := hex.EncodeToString(hmacSHA256(kSigning, []byte(stringToSign)))

	authHeader := fmt.Sprintf("AWS4-HMAC-SHA256 Credential=%s/%s, SignedHeaders=%s, Signature=%s",
		a.Config.S3AccessKey, scope, signedHeaders, signature)
	req.Header.Set("Authorization", authHeader)

	resp, err := a.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode >= 300 {
		return nil, fmt.Errorf("S3 ListObjectsV2 failed (status %d): %s", resp.StatusCode, string(body))
	}

	var result s3ListObjectsV2Result
	if err := xml.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("S3 ListObjectsV2 XML parse failed: %v (body: %s)", err, string(body))
	}
	return &result, nil
}

func hmacSHA256(key, data []byte) []byte {
	h := hmac.New(sha256.New, key)
	h.Write(data)
	return h.Sum(nil)
}

func sha256Hex(data []byte) string {
	h := sha256.Sum256(data)
	return hex.EncodeToString(h[:])
}

func (a *App) renderWorkflow(req GenerateRequest, requireMedia bool) (map[string]interface{}, []string, error) {
	req = normalizeRequest(req)
	if err := validateRequest(req, requireMedia); err != nil {
		return nil, nil, err
	}

	templatePath := a.Config.WorkflowTemplate
	if req.Mode == wanExtendAnyFrameMode && a.Config.WanWorkflowTemplate != "" {
		templatePath = a.Config.WanWorkflowTemplate
	}
	raw, err := os.ReadFile(templatePath)
	if err != nil {
		return nil, nil, err
	}
	var workflow map[string]interface{}
	if err := json.Unmarshal(raw, &workflow); err != nil {
		return nil, nil, err
	}
	nodes, err := toNodeMap(workflow)
	if err != nil {
		return nil, nil, err
	}

	warnings := []string{}
	if req.Mode == "pose_only" {
		f := false
		req.EnablePulid = &f
	}
	if req.Mode == "text_only" {
		f := false
		req.EnablePulid = &f
	}
	if req.Mode == "qwen_swap_face" || req.Mode == "qwen_edit_face" || req.Mode == qwenDirectPoseFusionMode || req.Mode == wanExtendAnyFrameMode {
		f := false
		req.EnablePulid = &f
	}

	modelSourceModel := []interface{}{"1", 0}
	modelSourceClip := []interface{}{"1", 1}

	if req.CKPTName != "" {
		nodes["1"]["inputs"].(map[string]interface{})["ckpt_name"] = req.CKPTName
	}
	if req.ReferenceImage != "" {
		nodes["7"]["inputs"].(map[string]interface{})["image"] = req.ReferenceImage
	}
	if req.PoseImage != "" {
		nodes["9"]["inputs"].(map[string]interface{})["image"] = req.PoseImage
	}

	if req.EnableLora != nil && *req.EnableLora && len(req.Loras) > 0 {
		modelSourceModel, modelSourceClip = applyLoraChain(nodes, req.Loras)
	} else {
		nodes["2"]["inputs"].(map[string]interface{})["clip"] = []interface{}{"1", 1}
		nodes["3"]["inputs"].(map[string]interface{})["clip"] = []interface{}{"1", 1}
	}

	nodes["2"]["inputs"].(map[string]interface{})["clip"] = modelSourceClip
	nodes["3"]["inputs"].(map[string]interface{})["clip"] = modelSourceClip

	if req.Mode == "dual_pass_auto_pose" {
		source := []interface{}{"23", 0}
		nodes["10"]["inputs"].(map[string]interface{})["image"] = source
		nodes["24"]["inputs"].(map[string]interface{})["image"] = source
		nodes["27"]["inputs"].(map[string]interface{})["images"] = source
		delete(nodes, "9")
	} else if req.Mode == "pose_then_face_swap" || req.Mode == "pose_only" {
		source := []interface{}{"9", 0}
		nodes["10"]["inputs"].(map[string]interface{})["image"] = source
		nodes["24"]["inputs"].(map[string]interface{})["image"] = source
		nodes["27"]["inputs"].(map[string]interface{})["images"] = source
		delete(nodes, "22")
		delete(nodes, "23")
		delete(nodes, "27")
	} else if req.Mode == "text_only" || req.Mode == "qwen_swap_face" || req.Mode == "qwen_edit_face" || req.Mode == qwenDirectPoseFusionMode {
		nodes["14"]["inputs"].(map[string]interface{})["positive"] = []interface{}{"2", 0}
		nodes["14"]["inputs"].(map[string]interface{})["negative"] = []interface{}{"3", 0}
		delete(nodes, "4")
		delete(nodes, "5")
		delete(nodes, "6")
		delete(nodes, "7")
		delete(nodes, "8")
		delete(nodes, "9")
		delete(nodes, "10")
		delete(nodes, "11")
		delete(nodes, "12")
		delete(nodes, "22")
		delete(nodes, "23")
		delete(nodes, "24")
		delete(nodes, "25")
		delete(nodes, "26")
		delete(nodes, "27")
		delete(nodes, "28")
		delete(nodes, "29")
	}
	if req.Mode == "qwen_swap_face" || req.Mode == "qwen_edit_face" {
		warnings = append(warnings, req.Mode+" is a post-process DashScope face edit after base generation")
	}

	if req.Width > 0 {
		nodes["13"]["inputs"].(map[string]interface{})["width"] = req.Width
	}
	if req.Height > 0 {
		nodes["13"]["inputs"].(map[string]interface{})["height"] = req.Height
	}
	if req.BatchSize > 0 {
		nodes["13"]["inputs"].(map[string]interface{})["batch_size"] = req.BatchSize
	}
	if req.Prompt != "" {
		nodes["2"]["inputs"].(map[string]interface{})["text"] = req.Prompt
	}
	if req.NegativePrompt != "" {
		nodes["3"]["inputs"].(map[string]interface{})["text"] = req.NegativePrompt
	}

	if node22, ok := nodes["22"]; ok {
		overrideSampler(node22["inputs"].(map[string]interface{}), req.BaseSeed, req.BaseSteps, req.BaseCFG, req.BaseSamplerName, req.BaseScheduler, req.BaseDenoise)
	}
	overrideSampler(nodes["14"]["inputs"].(map[string]interface{}), req.Seed, req.Steps, req.CFG, req.SamplerName, req.Scheduler, req.Denoise)

	if node12, ok := nodes["12"]; ok {
		overrideControlNet(node12["inputs"].(map[string]interface{}), req.CNDepthStrength, req.CNDepthStartPercent, req.CNDepthEndPercent)
	}
	if node26, ok := nodes["26"]; ok {
		overrideControlNet(node26["inputs"].(map[string]interface{}), req.CNPoseStrength, req.CNPoseStartPercent, req.CNPoseEndPercent)
	}

	if req.EnablePulid != nil && *req.EnablePulid {
		nodes["8"]["inputs"].(map[string]interface{})["model"] = modelSourceModel
		if req.PulidMethod != "" {
			nodes["8"]["inputs"].(map[string]interface{})["method"] = req.PulidMethod
		}
		if req.PulidWeight > 0 {
			nodes["8"]["inputs"].(map[string]interface{})["weight"] = req.PulidWeight
		}
		if req.PulidStartAt >= 0 {
			nodes["8"]["inputs"].(map[string]interface{})["start_at"] = req.PulidStartAt
		}
		if req.PulidEndAt > 0 {
			nodes["8"]["inputs"].(map[string]interface{})["end_at"] = req.PulidEndAt
		}
		nodes["14"]["inputs"].(map[string]interface{})["model"] = []interface{}{"8", 0}
	} else {
		nodes["14"]["inputs"].(map[string]interface{})["model"] = modelSourceModel
		delete(nodes, "4")
		delete(nodes, "5")
		delete(nodes, "6")
		delete(nodes, "7")
		delete(nodes, "8")
	}
	if node22, ok := nodes["22"]; ok {
		node22["inputs"].(map[string]interface{})["model"] = modelSourceModel
	}
	if !(req.EnableLora != nil && *req.EnableLora && len(req.Loras) > 0) {
		delete(nodes, "17")
	}

	if req.EnableUpscale != nil && *req.EnableUpscale {
		if req.UpscaleModelName != "" {
			nodes["18"]["inputs"].(map[string]interface{})["model_name"] = req.UpscaleModelName
		}
		nodes["16"]["inputs"].(map[string]interface{})["images"] = []interface{}{"19", 0}
	} else {
		nodes["16"]["inputs"].(map[string]interface{})["images"] = []interface{}{"15", 0}
		delete(nodes, "18")
		delete(nodes, "19")
	}
	if req.KeepIntermediate != nil && !*req.KeepIntermediate {
		delete(nodes, "27")
		delete(nodes, "28")
		delete(nodes, "29")
	}

	return workflow, warnings, nil
}

func (a *App) generateWithComfy(ctx context.Context, req GenerateRequest) (*GenerateResponse, error) {
	req = normalizeRequest(req)
	if req.Mode == "qwen_swap_face" || req.Mode == "qwen_edit_face" {
		return nil, errors.New(req.Mode + " is supported via the RunPod engine path only")
	}
	rendered, warnings, err := a.renderWorkflow(req, true)
	if err != nil {
		return nil, err
	}

	if req.ReferenceImage != "" {
		name, err := a.uploadMediaToComfy(ctx, req.ReferenceImage, "reference")
		if err != nil {
			return nil, err
		}
		rendered["7"].(map[string]interface{})["inputs"].(map[string]interface{})["image"] = name
	}
	if req.QwenExtraImage != "" {
		warnings = append(warnings, "qwen_extra_image is ignored by the Comfy workflow preview and only used by the DashScope post-process")
	}
	if req.PoseImage != "" {
		name, err := a.uploadMediaToComfy(ctx, req.PoseImage, "pose")
		if err != nil {
			return nil, err
		}
		rendered["9"].(map[string]interface{})["inputs"].(map[string]interface{})["image"] = name
	}

	payload := map[string]interface{}{
		"prompt":    rendered,
		"client_id": fmt.Sprintf("v16web-%d", time.Now().UnixNano()),
	}
	body, _ := json.Marshal(payload)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, a.Config.ComfyAPIURL+"/prompt", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := a.httpClient.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("comfy /prompt failed: %s", string(b))
	}
	var promptResp map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&promptResp); err != nil {
		return nil, err
	}
	promptID := toString(promptResp["prompt_id"])
	if promptID == "" {
		return nil, errors.New("missing prompt_id from comfy")
	}

	history, err := a.waitComfyHistory(ctx, promptID)
	if err != nil {
		return nil, err
	}
	final, intermediate := collectHistoryImages(history)
	cacheBust := promptID
	out := &GenerateResponse{
		OK:       true,
		Mode:     req.Mode,
		Engine:   "comfy",
		Warnings: warnings,
		PromptID: promptID,
		Status:   "COMPLETED",
		Meta: map[string]interface{}{
			"enable_pulid":   req.EnablePulid != nil && *req.EnablePulid,
			"enable_lora":    req.EnableLora != nil && *req.EnableLora,
			"enable_upscale": req.EnableUpscale != nil && *req.EnableUpscale,
		},
	}
	for _, img := range final {
		url := a.makeLocalComfyViewURL(img, cacheBust)
		out.FinalURLs = append(out.FinalURLs, url)
	}
	if len(out.FinalURLs) > 0 {
		out.FinalURL = out.FinalURLs[0]
	}
	if req.KeepIntermediate != nil && *req.KeepIntermediate {
		for _, img := range intermediate {
			out.IntermediateURLs = append(out.IntermediateURLs, a.makeLocalComfyViewURL(img, cacheBust))
		}
	}
	return out, nil
}

func (a *App) generateWithRunPod(ctx context.Context, req GenerateRequest) (*GenerateResponse, error) {
	req = normalizeRequest(req)
	warnings := []string{}

	if req.AutoSegmentPrompts && req.Mode == wanExtendAnyFrameMode && req.Prompt != "" {
		segmentLimit := req.SegmentLimit
		if segmentLimit <= 0 {
			segmentLimit = wanExtendAnyFrameSegmentLimit
		}
		frames := req.Frames
		if frames <= 0 {
			frames = segmentLimit
		}
		frames = maxInt(frames, 2)
		segmentCount := maxInt(1, (frames-1+segmentLimit-2)/(segmentLimit-1))
		if segmentCount > 1 {
			prompts, err := a.callDashScopeSplitPrompt(ctx, req.Prompt, segmentCount)
			if err == nil && len(prompts) == segmentCount {
				req.WanPrompts = prompts
			} else {
				warnings = append(warnings, fmt.Sprintf("AI Auto-Segment failed: %v", err))
			}
		}
	}

	input := map[string]interface{}{
		"mode":                   req.Mode,
		"reference_image":        req.ReferenceImage,
		"startimg":               req.StartImg,
		"endimg":                 req.EndImg,
		"frames":                 req.Frames,
		"wan_vae_name":           req.WanVaeName,
		"wan_clip_vision_name":    req.WanClipVisionName,
		"wan_clip_name":          req.WanClipName,
		"wan_unet_high_name":     req.WanUnetHighName,
		"wan_unet_low_name":      req.WanUnetLowName,
		"wan_seeds":              req.WanSeeds,
		"wan_prompts":            req.WanPrompts,
		"segment_limit":          req.SegmentLimit,
		"prompt":                 req.Prompt,
		"qwen_swap_prompt":       req.QwenSwapPrompt,
		"qwen_edit_prompt":       req.QwenEditPrompt,
		"qwen_model":             req.QwenModel,
		"qwen_size":              req.QwenSize,
		"qwen_extra_image":       req.QwenExtraImage,
		"negative_prompt":        req.NegativePrompt,
		"width":                  req.Width,
		"height":                 req.Height,
		"batch_size":             req.BatchSize,
		"ckpt_name":              req.CKPTName,
		"base_steps":             req.BaseSteps,
		"base_seed":              req.BaseSeed,
		"base_cfg":               req.BaseCFG,
		"base_sampler_name":      req.BaseSamplerName,
		"base_scheduler":         req.BaseScheduler,
		"base_denoise":           req.BaseDenoise,
		"steps":                  req.Steps,
		"cfg":                    req.CFG,
		"seed":                   req.Seed,
		"sampler_name":           req.SamplerName,
		"scheduler":              req.Scheduler,
		"denoise":                req.Denoise,
		"enable_pulid":           req.EnablePulid != nil && *req.EnablePulid,
		"pulid_weight":           req.PulidWeight,
		"pulid_start_at":         req.PulidStartAt,
		"pulid_method":           req.PulidMethod,
		"pulid_end_at":           req.PulidEndAt,
		"cn_depth_strength":      req.CNDepthStrength,
		"cn_depth_start_percent": req.CNDepthStartPercent,
		"cn_depth_end_percent":   req.CNDepthEndPercent,
		"cn_pose_strength":       req.CNPoseStrength,
		"cn_pose_start_percent":  req.CNPoseStartPercent,
		"cn_pose_end_percent":    req.CNPoseEndPercent,
		"enable_lora":            req.EnableLora != nil && *req.EnableLora,
		"enable_upscale":         req.EnableUpscale != nil && *req.EnableUpscale,
		"use_upscale":            req.EnableUpscale != nil && *req.EnableUpscale,
		"upscale_model_name":     req.UpscaleModelName,
		"output_format":          req.OutputFormat,
		"jpg_quality":            req.JPGQuality,
		"keep_intermediate":      req.KeepIntermediate != nil && *req.KeepIntermediate,
		"enable_i2v":             req.EnableI2V != nil && *req.EnableI2V,
		"i2v_prompt":             req.I2VPrompt,
		"i2v_model":              req.I2VModel,
		"i2v_resolution":         req.I2VResolution,
		"i2v_duration":           req.I2VDuration,
		"i2v_seed":               req.I2VSeed,
		"i2v_negative_prompt":    req.I2VNegativePrompt,
		"i2v_audio_url":          req.I2VAudioURL,
		"i2v_prompt_extend":      req.I2VPromptExtend != nil && *req.I2VPromptExtend,
		"i2v_watermark":          req.I2VWatermark != nil && *req.I2VWatermark,
		"request_id":             req.RequestID,
	}
	if req.PoseImage != "" {
		input["pose_image"] = req.PoseImage
	}
	if req.EnableLora != nil && *req.EnableLora && len(req.Loras) > 0 {
		loras := make([]map[string]interface{}, 0, len(req.Loras))
		for _, l := range req.Loras {
			entry := map[string]interface{}{"name": l.Name}
			if l.Strength != 0 {
				entry["strength"] = l.Strength
			}
			if l.StrengthModel != 0 {
				entry["strength_model"] = l.StrengthModel
			}
			if l.StrengthClip != 0 {
				entry["strength_clip"] = l.StrengthClip
			}
			loras = append(loras, entry)
		}
		input["loras"] = loras
	}

	runURL := fmt.Sprintf("https://api.runpod.ai/v2/%s/run", a.Config.RunPodEndpointID)
	body, _ := json.Marshal(map[string]interface{}{"input": input})
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, runURL, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+a.Config.RunPodAPIKey)
	resp, err := a.httpClient.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("runpod /run failed: %s", string(b))
	}
	var runResp map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&runResp); err != nil {
		return nil, err
	}
	jobID := toString(runResp["id"])
	if jobID == "" {
		return nil, errors.New("missing RunPod job id")
	}

	if req.Async {
		return &GenerateResponse{
			OK:     true,
			Mode:   req.Mode,
			Engine: "runpod",
			JobID:  jobID,
			Status: "IN_QUEUE",
		}, nil
	}

	statusURL := fmt.Sprintf("https://api.runpod.ai/v2/%s/status/%s", a.Config.RunPodEndpointID, jobID)
	for {
		time.Sleep(5 * time.Second)
		reqStatus, _ := http.NewRequestWithContext(ctx, http.MethodGet, statusURL, nil)
		reqStatus.Header.Set("Authorization", "Bearer "+a.Config.RunPodAPIKey)
		statusResp, err := a.httpClient.Do(reqStatus)
		if err != nil {
			return nil, err
		}
		var data map[string]interface{}
		_ = json.NewDecoder(statusResp.Body).Decode(&data)
		statusResp.Body.Close()
		status := toString(data["status"])
		switch status {
		case "COMPLETED":
			out := &GenerateResponse{
				OK:       true,
				Mode:     req.Mode,
				Engine:   "runpod",
				Warnings: warnings,
				JobID:    jobID,
				Status:   status,
				Raw:      data,
			}
			if output, ok := data["output"].(map[string]interface{}); ok {
				out.RequestID = toString(output["request_id"])
				out.PromptID = toString(output["prompt_id"])
				cacheBust := firstNonEmpty(out.RequestID, out.JobID, out.PromptID)
				out.FinalURL = addCacheBust(toString(output["final_url"]), cacheBust)
				if urls, ok := output["final_urls"].([]interface{}); ok {
					for _, u := range urls {
						out.FinalURLs = append(out.FinalURLs, addCacheBust(toString(u), cacheBust))
					}
				}
				if videoURL := addCacheBust(toString(output["final_video_url"]), cacheBust); videoURL != "" {
					out.FinalVideoURL = videoURL
				}
				if urls, ok := output["final_video_urls"].([]interface{}); ok {
					for _, u := range urls {
						out.FinalVideoURLs = append(out.FinalVideoURLs, addCacheBust(toString(u), cacheBust))
					}
				}
				if urls, ok := output["segment_video_urls"].([]interface{}); ok {
					for _, u := range urls {
						out.SegmentVideoURLs = append(out.SegmentVideoURLs, addCacheBust(toString(u), cacheBust))
					}
				}
				if urls, ok := output["intermediate_urls"].([]interface{}); ok {
					for _, u := range urls {
						out.IntermediateURLs = append(out.IntermediateURLs, addCacheBust(toString(u), cacheBust))
					}
				}
				if out.FinalVideoURL == "" && len(out.FinalVideoURLs) > 0 {
					out.FinalVideoURL = out.FinalVideoURLs[0]
				}
				out.Meta = map[string]interface{}{
					"output": output,
				}
			}
			return out, nil
		case "FAILED", "CANCELLED", "TIMED_OUT":
			return nil, fmt.Errorf("runpod job failed: %s", mustJSON(data))
		}
	}
}

func (a *App) generateWithQwenPoseFusion(ctx context.Context, req GenerateRequest) (*GenerateResponse, error) {
	req = normalizeRequest(req)
	if err := validateRequest(req, true); err != nil {
		return nil, err
	}
	req.RequestID = firstNonEmpty(req.RequestID, fmt.Sprintf("qwen-%d", time.Now().UnixNano()))
	qwenPrompt := strings.TrimSpace(req.Prompt)
	if qwenPrompt == "" {
		qwenPrompt = qwenDefaultPoseFusionPrompt
	}
	qwenModel := strings.TrimSpace(req.QwenModel)
	if qwenModel == "" {
		qwenModel = qwenModelDefault
	}
	qwenRaw, err := a.callDashScopeQwenPoseFusion(ctx, req.ReferenceImage, req.PoseImage, qwenPrompt, strings.TrimSpace(req.NegativePrompt), qwenModel, strings.TrimSpace(req.QwenSize))
	if err != nil {
		return nil, err
	}
	dataURL := "data:image/png;base64," + base64.StdEncoding.EncodeToString(qwenRaw)
	return &GenerateResponse{
		OK:        true,
		Mode:      req.Mode,
		Engine:    "qwen",
		Warnings:  []string{"qwen_pose_fusion uses direct DashScope fusion and does not run the Comfy workflow"},
		RequestID: req.RequestID,
		FinalURL:  dataURL,
		FinalURLs: []string{dataURL},
		Meta: map[string]interface{}{
			"mode":         req.Mode,
			"pose_mode":    "direct_qwen_fusion",
			"qwen_model":   qwenModel,
			"final_format": "png",
		},
	}, nil
}

func (a *App) callDashScopeQwenPoseFusion(ctx context.Context, faceMedia, poseMedia, prompt, negativePrompt, model, sizeOverride string) ([]byte, error) {
	apiKey := strings.TrimSpace(os.Getenv("DASHSCOPE_API_KEY"))
	if apiKey == "" {
		return nil, errors.New("DASHSCOPE_API_KEY is required for qwen pose fusion mode")
	}
	faceBytes, faceName, err := readMedia(faceMedia, "face")
	if err != nil {
		return nil, err
	}
	poseBytes, poseName, err := readMedia(poseMedia, "pose")
	if err != nil {
		return nil, err
	}
	faceInput, _, err := qwenImageBytesToDataURL(faceBytes, faceName)
	if err != nil {
		return nil, err
	}
	poseInput, sizeText, err := qwenImageBytesToDataURL(poseBytes, poseName)
	if err != nil {
		return nil, err
	}
	if strings.TrimSpace(sizeOverride) != "" {
		sizeText = strings.TrimSpace(sizeOverride)
	}
	params := map[string]interface{}{
		"n":               1,
		"negative_prompt": strings.TrimSpace(negativePrompt),
		"prompt_extend":   false,
		"watermark":       false,
		"size":            sizeText,
	}
	if strings.TrimSpace(negativePrompt) == "" {
		params["negative_prompt"] = " "
	}
	payload := map[string]interface{}{
		"model": model,
		"input": map[string]interface{}{
			"messages": []interface{}{
				map[string]interface{}{
					"role": "user",
					"content": []interface{}{
						map[string]interface{}{"image": poseInput},
						map[string]interface{}{"image": faceInput},
						map[string]interface{}{"text": prompt},
					},
				},
			},
		},
		"parameters": params,
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, qwenAPIURL, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+apiKey)
	httpReq.Header.Set("X-DashScope-DataInspection", qwenDataInspectionHeader)
	resp, err := a.httpClient.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("dashscope qwen pose fusion failed: %s", string(b))
	}
	var respJSON map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&respJSON); err != nil {
		return nil, err
	}
	imageURL, err := extractDashScopeImageURL(respJSON)
	if err != nil {
		return nil, err
	}
	imgReq, err := http.NewRequestWithContext(ctx, http.MethodGet, imageURL, nil)
	if err != nil {
		return nil, err
	}
	imgResp, err := a.httpClient.Do(imgReq)
	if err != nil {
		return nil, err
	}
	defer imgResp.Body.Close()
	if imgResp.StatusCode >= 300 {
		b, _ := io.ReadAll(imgResp.Body)
		return nil, fmt.Errorf("dashscope qwen pose fusion image download failed: %s", string(b))
	}
	return io.ReadAll(imgResp.Body)
}

func qwenImageBytesToDataURL(blob []byte, filename string) (string, string, error) {
	cfg, _, err := image.DecodeConfig(bytes.NewReader(blob))
	if err != nil {
		return "", "", err
	}
	ext := strings.ToLower(filepath.Ext(filename))
	mimeType := mime.TypeByExtension(ext)
	if mimeType == "" {
		mimeType = "image/jpeg"
	} else if idx := strings.Index(mimeType, ";"); idx >= 0 {
		mimeType = mimeType[:idx]
	}
	return fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(blob)), fmt.Sprintf("%d*%d", cfg.Width, cfg.Height), nil
}

func extractDashScopeImageURL(respJSON map[string]interface{}) (string, error) {
	output, _ := respJSON["output"].(map[string]interface{})
	if output != nil {
		if choices, ok := output["choices"].([]interface{}); ok {
			for _, choice := range choices {
				choiceMap, _ := choice.(map[string]interface{})
				if choiceMap == nil {
					continue
				}
				message, _ := choiceMap["message"].(map[string]interface{})
				if message == nil {
					continue
				}
				content, _ := message["content"].([]interface{})
				for _, item := range content {
					itemMap, _ := item.(map[string]interface{})
					if itemMap == nil {
						continue
					}
					if imageURL := strings.TrimSpace(toString(itemMap["image"])); imageURL != "" {
						return imageURL, nil
					}
				}
			}
		}
	}
	return "", fmt.Errorf("dashscope response missing image url: %s", mustJSON(respJSON))
}

func (a *App) waitComfyHistory(ctx context.Context, promptID string) (map[string]interface{}, error) {
	deadline := time.Now().Add(20 * time.Minute)
	for time.Now().Before(deadline) {
		req, _ := http.NewRequestWithContext(ctx, http.MethodGet, a.Config.ComfyAPIURL+"/history/"+promptID, nil)
		resp, err := a.httpClient.Do(req)
		if err != nil {
			return nil, err
		}
		var history map[string]interface{}
		if err := json.NewDecoder(resp.Body).Decode(&history); err != nil {
			resp.Body.Close()
			return nil, err
		}
		resp.Body.Close()
		if item, ok := history[promptID].(map[string]interface{}); ok {
			if outputs, ok := item["outputs"].(map[string]interface{}); ok && len(outputs) > 0 {
				return item, nil
			}
		}
		time.Sleep(1500 * time.Millisecond)
	}
	return nil, errors.New("timed out waiting for Comfy history")
}

func collectHistoryImages(history map[string]interface{}) ([]comfyImage, []comfyImage) {
	outs, _ := history["outputs"].(map[string]interface{})
	final := collectNodeImages(outs, "16")
	intermediate := []comfyImage{}
	for _, nid := range []string{"27", "28", "29"} {
		intermediate = append(intermediate, collectNodeImages(outs, nid)...)
	}
	return final, intermediate
}

func collectNodeImages(outputs map[string]interface{}, nodeID string) []comfyImage {
	node, ok := outputs[nodeID].(map[string]interface{})
	if !ok {
		return nil
	}
	rawImgs, ok := node["images"].([]interface{})
	if !ok {
		return nil
	}
	imgs := make([]comfyImage, 0, len(rawImgs))
	for _, item := range rawImgs {
		b, _ := json.Marshal(item)
		var img comfyImage
		if json.Unmarshal(b, &img) == nil {
			imgs = append(imgs, img)
		}
	}
	return imgs
}

func (a *App) makeLocalComfyViewURL(img comfyImage, cacheBust string) string {
	values := url.Values{}
	values.Set("filename", img.Filename)
	if img.Subfolder != "" {
		values.Set("subfolder", img.Subfolder)
	}
	if img.Type != "" {
		values.Set("type", img.Type)
	} else {
		values.Set("type", "output")
	}
	return addCacheBust("/api/comfy/view?"+values.Encode(), cacheBust)
}

func addCacheBust(rawURL, cacheBust string) string {
	if rawURL == "" || cacheBust == "" {
		return rawURL
	}
	parsed, err := url.Parse(rawURL)
	if err != nil {
		sep := "?"
		if strings.Contains(rawURL, "?") {
			sep = "&"
		}
		return rawURL + sep + "v=" + url.QueryEscape(cacheBust)
	}
	q := parsed.Query()
	q.Set("v", cacheBust)
	parsed.RawQuery = q.Encode()
	return parsed.String()
}

func (a *App) uploadMediaToComfy(ctx context.Context, media, prefix string) (string, error) {
	data, name, err := readMedia(media, prefix)
	if err != nil {
		return "", err
	}
	buf := &bytes.Buffer{}
	writer := multipart.NewWriter(buf)
	part, err := writer.CreateFormFile("image", name)
	if err != nil {
		return "", err
	}
	if _, err := part.Write(data); err != nil {
		return "", err
	}
	_ = writer.WriteField("type", "input")
	_ = writer.WriteField("overwrite", "true")
	if err := writer.Close(); err != nil {
		return "", err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, a.Config.ComfyAPIURL+"/upload/image", buf)
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())
	resp, err := a.httpClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		b, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("comfy upload failed: %s", string(b))
	}
	var uploadResp map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&uploadResp); err != nil {
		return "", err
	}
	filename := toString(uploadResp["name"])
	if filename == "" {
		filename = name
	}
	return filename, nil
}

func decodeGenerateRequest(r *http.Request) (GenerateRequest, error) {
	var req GenerateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		return req, err
	}
	return normalizeRequest(req), nil
}

func normalizeRequest(req GenerateRequest) GenerateRequest {
	req.Mode = normalizeMode(req.Mode, req.ReferenceImage, req.PoseImage, req.StartImg)
	if req.Mode == wanExtendAnyFrameMode {
		req.BatchSize = firstPositive(req.BatchSize, 1)
		req.Width = firstPositive(req.Width, 480)
		req.Height = firstPositive(req.Height, 832)
		req.Steps = firstPositive(req.Steps, 4)
		req.CFG = firstPositiveFloat(req.CFG, 1)
		req.SamplerName = firstNonEmpty(req.SamplerName, "euler")
		req.Scheduler = firstNonEmpty(req.Scheduler, "simple")
	} else {
		req.BatchSize = firstPositive(req.BatchSize, 1)
		req.Width = firstPositive(req.Width, 832)
		req.Height = firstPositive(req.Height, 1216)
	}
	req.BaseSteps = firstPositive(req.BaseSteps, 8)
	if req.Mode != wanExtendAnyFrameMode {
		req.Steps = firstPositive(req.Steps, 40)
	}
	req.BaseCFG = firstPositiveFloat(req.BaseCFG, 4)
	if req.Mode != wanExtendAnyFrameMode {
		req.CFG = firstPositiveFloat(req.CFG, 4)
	}
	req.BaseDenoise = firstPositiveFloat(req.BaseDenoise, 1)
	if req.Mode != wanExtendAnyFrameMode {
		req.Denoise = firstPositiveFloat(req.Denoise, 1)
	}
	req.BaseSamplerName = firstNonEmpty(req.BaseSamplerName, "dpmpp_2m_sde")
	req.BaseScheduler = firstNonEmpty(req.BaseScheduler, "karras")
	if req.Mode != wanExtendAnyFrameMode {
		req.SamplerName = firstNonEmpty(req.SamplerName, "dpmpp_2m_sde")
		req.Scheduler = firstNonEmpty(req.Scheduler, "karras")
	}
	req.CNDepthStrength = firstPositiveFloat(req.CNDepthStrength, 0.6)
	req.CNPoseStrength = firstPositiveFloat(req.CNPoseStrength, 0.6)
	req.CNDepthStartPercent = clampDefault(req.CNDepthStartPercent, 0)
	req.CNDepthEndPercent = clampDefault(req.CNDepthEndPercent, 1)
	req.CNPoseStartPercent = clampDefault(req.CNPoseStartPercent, 0)
	req.CNPoseEndPercent = clampDefault(req.CNPoseEndPercent, 1)
	req.PulidWeight = firstPositiveFloat(req.PulidWeight, 0.7)
	req.PulidStartAt = clampDefault(req.PulidStartAt, 0.5)
	req.PulidEndAt = clampDefault(req.PulidEndAt, 1)
	req.PulidMethod = firstNonEmpty(req.PulidMethod, "fidelity")
	req.QwenSwapPrompt = firstNonEmpty(req.QwenSwapPrompt, qwenDefaultSwapPrompt)
	req.QwenEditPrompt = firstNonEmpty(req.QwenEditPrompt, qwenDefaultEditPrompt)
	if req.Mode == qwenDirectPoseFusionMode {
		req.Prompt = firstNonEmpty(req.Prompt, qwenDefaultPoseFusionPrompt)
	}
	if req.Mode == wanExtendAnyFrameMode {
		req.Prompt = firstNonEmpty(req.Prompt, wanExtendAnyFrameDefaultPrompt)
	}
	req.QwenModel = firstNonEmpty(req.QwenModel, qwenModelDefault)
	req.QwenSize = strings.TrimSpace(req.QwenSize)
	req.QwenExtraImage = strings.TrimSpace(req.QwenExtraImage)
	req.StartImg = strings.TrimSpace(req.StartImg)
	req.EndImg = strings.TrimSpace(req.EndImg)
	req.WanUnetHighName = firstNonEmpty(req.WanUnetHighName, "wan22I2V8StepsNSFWFP8_fp8Highnoise10.safetensors")
	req.WanUnetLowName = firstNonEmpty(req.WanUnetLowName, "wan22I2V8StepsNSFWFP8_fp8Lownoise10.safetensors")
	req.Frames = firstPositive(req.Frames, wanExtendAnyFrameSegmentLimit)
	req.CKPTName = firstNonEmpty(req.CKPTName, "SDXL_Photorealistic_Mix_nsfw.safetensors")
	req.UpscaleModelName = firstNonEmpty(req.UpscaleModelName, "4x-UltraSharp.pth")
	req.OutputFormat = firstNonEmpty(req.OutputFormat, "jpg")
	req.JPGQuality = firstPositive(req.JPGQuality, 85)
	req.I2VPrompt = firstNonEmpty(req.I2VPrompt, req.Prompt, "保持主体一致，生成自然流畅、画面连贯的动态视频，镜头稳定，动作真实，细节清晰。")
	req.I2VModel = firstNonEmpty(req.I2VModel, "wan2.7-i2v")
	req.I2VResolution = firstNonEmpty(req.I2VResolution, "1080P")
	req.I2VDuration = firstPositive(req.I2VDuration, 5)
	req.I2VNegativePrompt = strings.TrimSpace(req.I2VNegativePrompt)
	req.I2VAudioURL = strings.TrimSpace(req.I2VAudioURL)
	if req.EnablePulid == nil {
		b := req.Mode != "pose_only" && req.Mode != "text_only"
		if req.Mode == "qwen_swap_face" || req.Mode == "qwen_edit_face" || req.Mode == qwenDirectPoseFusionMode || req.Mode == wanExtendAnyFrameMode {
			b = false
		}
		req.EnablePulid = &b
	}
	if req.Mode == "qwen_swap_face" || req.Mode == "qwen_edit_face" || req.Mode == qwenDirectPoseFusionMode || req.Mode == wanExtendAnyFrameMode {
		f := false
		req.EnablePulid = &f
	}
	if req.Mode == wanExtendAnyFrameMode {
		f := false
		req.EnableUpscale = &f
		req.EnableI2V = &f
	}
	if req.EnableLora == nil {
		b := len(req.Loras) > 0
		req.EnableLora = &b
	}
	if req.EnableUpscale == nil {
		b := false
		req.EnableUpscale = &b
	}
	if req.EnableI2V == nil {
		b := false
		req.EnableI2V = &b
	}
	if req.I2VPromptExtend == nil {
		b := true
		req.I2VPromptExtend = &b
	}
	if req.I2VWatermark == nil {
		b := false
		req.I2VWatermark = &b
	}
	if req.KeepIntermediate == nil {
		b := true
		req.KeepIntermediate = &b
	}
	return req
}

func normalizeMode(mode, referenceImage, poseImage, startImg string) string {
	switch strings.TrimSpace(mode) {
	case "dual_pass_auto_pose", "pose_then_face_swap", "pose_only", "text_only", "qwen_swap_face", qwenDirectPoseFusionMode, "qwen_edit_face", wanExtendAnyFrameMode:
		return strings.TrimSpace(mode)
	}
	if strings.TrimSpace(startImg) != "" {
		return wanExtendAnyFrameMode
	}
	if strings.TrimSpace(referenceImage) == "" && strings.TrimSpace(poseImage) == "" {
		return "text_only"
	}
	if strings.TrimSpace(poseImage) != "" {
		if strings.TrimSpace(referenceImage) != "" {
			return "pose_then_face_swap"
		}
		return "pose_only"
	}
	return "dual_pass_auto_pose"
}

func validateRequest(req GenerateRequest, requireMedia bool) error {
	switch req.Mode {
	case "dual_pass_auto_pose":
		if req.ReferenceImage == "" && requireMedia {
			return errors.New("reference_image is required for dual_pass_auto_pose")
		}
	case "pose_then_face_swap":
		if req.ReferenceImage == "" && requireMedia {
			return errors.New("reference_image is required for pose_then_face_swap")
		}
		if req.PoseImage == "" && requireMedia {
			return errors.New("pose_image is required for pose_then_face_swap")
		}
	case "pose_only":
		if req.PoseImage == "" && requireMedia {
			return errors.New("pose_image is required for pose_only")
		}
	case "qwen_swap_face":
		if req.ReferenceImage == "" && requireMedia {
			return errors.New("reference_image is required for qwen_swap_face")
		}
	case "qwen_pose_fusion":
		if req.ReferenceImage == "" && requireMedia {
			return errors.New("reference_image is required for qwen_pose_fusion")
		}
		if req.PoseImage == "" && requireMedia {
			return errors.New("pose_image is required for qwen_pose_fusion")
		}
	case wanExtendAnyFrameMode:
		if req.StartImg == "" && requireMedia {
			return errors.New("startimg is required for wan2_2_i2v_extend_any_frame")
		}
		if req.Frames <= 0 {
			return errors.New("frames must be greater than 0 for wan2_2_i2v_extend_any_frame")
		}
	case "qwen_edit_face":
		// prompt-guided Qwen edit uses the generated image as the base input.
	case "text_only":
		// prompt-only mode
	default:
		return fmt.Errorf("unsupported mode: %s", req.Mode)
	}
	if strings.TrimSpace(req.Prompt) == "" {
		return errors.New("prompt is required")
	}
	return nil
}

func toNodeMap(workflow map[string]interface{}) (map[string]map[string]interface{}, error) {
	nodes := map[string]map[string]interface{}{}
	for k, v := range workflow {
		node, ok := v.(map[string]interface{})
		if !ok {
			return nil, fmt.Errorf("workflow node %s is not an object", k)
		}
		nodes[k] = node
	}
	return nodes, nil
}

func applyLoraChain(nodes map[string]map[string]interface{}, loras []LoraConfig) ([]interface{}, []interface{}) {
	modelSource := []interface{}{"1", 0}
	clipSource := []interface{}{"1", 1}
	if len(loras) == 0 {
		return modelSource, clipSource
	}

	maxID := 0
	for k := range nodes {
		if id, err := strconv.Atoi(k); err == nil && id > maxID {
			maxID = id
		}
	}

	for i, l := range loras {
		nodeID := "17"
		if i > 0 {
			maxID++
			nodeID = strconv.Itoa(maxID)
			nodes[nodeID] = map[string]interface{}{
				"class_type": "LoraLoader",
				"_meta": map[string]interface{}{
					"title": fmt.Sprintf("LoraLoader %d", i+1),
				},
				"inputs": map[string]interface{}{},
			}
		}
		strengthModel := l.StrengthModel
		strengthClip := l.StrengthClip
		if l.Strength != 0 {
			strengthModel = l.Strength
			strengthClip = l.Strength
		}
		inputs := nodes[nodeID]["inputs"].(map[string]interface{})
		inputs["lora_name"] = l.Name
		inputs["strength_model"] = strengthModel
		inputs["strength_clip"] = strengthClip
		inputs["model"] = modelSource
		inputs["clip"] = clipSource
		modelSource = []interface{}{nodeID, 0}
		clipSource = []interface{}{nodeID, 1}
	}
	return modelSource, clipSource
}

func overrideSampler(inputs map[string]interface{}, seed int64, steps int, cfg float64, sampler, scheduler string, denoise float64) {
	if seed != 0 {
		inputs["seed"] = seed
	}
	if steps > 0 {
		inputs["steps"] = steps
	}
	if cfg > 0 {
		inputs["cfg"] = cfg
	}
	if sampler != "" {
		inputs["sampler_name"] = sampler
	}
	if scheduler != "" {
		inputs["scheduler"] = scheduler
	}
	if denoise > 0 {
		inputs["denoise"] = denoise
	}
}

func overrideControlNet(inputs map[string]interface{}, strength, start, end float64) {
	if strength > 0 {
		inputs["strength"] = strength
	}
	inputs["start_percent"] = start
	inputs["end_percent"] = end
}

func loadKeyEnvFile(filename string) {
	raw, err := os.ReadFile(filename)
	if err != nil {
		return
	}
	lines := strings.Split(string(raw), "\n")
	for _, line := range lines {
		if !strings.Contains(line, "=") {
			continue
		}
		parts := strings.SplitN(line, "=", 2)
		key := strings.TrimSpace(parts[0])
		val := strings.TrimSpace(parts[1])
		if key == "" || strings.HasPrefix(key, "#") {
			continue
		}
		if os.Getenv(key) == "" {
			_ = os.Setenv(key, val)
		}
	}
}

func loadS3CredsFile(cfg *Config) {
	cfg.S3AccessKey = firstNonEmpty(os.Getenv("S3_ACCESS_KEY_ID"), os.Getenv("AWS_ACCESS_KEY_ID"))
	cfg.S3SecretKey = firstNonEmpty(os.Getenv("S3_SECRET_ACCESS_KEY"), os.Getenv("AWS_SECRET_ACCESS_KEY"))
	cfg.S3Bucket = firstNonEmpty(os.Getenv("S3_BUCKET"))
	cfg.S3Endpoint = firstNonEmpty(os.Getenv("S3_ENDPOINT_URL"))
	if cfg.S3AccessKey != "" && cfg.S3SecretKey != "" && cfg.S3Bucket != "" && cfg.S3Endpoint != "" {
		return
	}
	raw, err := os.ReadFile(cfg.S3CredsFile)
	if err != nil {
		return
	}
	for _, line := range strings.Split(string(raw), "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "aws_access_key_id") {
			cfg.S3AccessKey = strings.TrimSpace(strings.SplitN(line, "=", 2)[1])
		}
		if strings.HasPrefix(line, "aws_secret_access_key") {
			cfg.S3SecretKey = strings.TrimSpace(strings.SplitN(line, "=", 2)[1])
		}
		if strings.HasPrefix(line, "Bucket name") {
			cfg.S3Bucket = strings.TrimSpace(strings.SplitN(line, "=", 2)[1])
		}
		if strings.HasPrefix(line, "Endpoint URL") {
			cfg.S3Endpoint = strings.TrimSpace(strings.SplitN(line, "=", 2)[1])
		}
	}
}

func readMedia(media, prefix string) ([]byte, string, error) {
	if strings.HasPrefix(media, "http://") || strings.HasPrefix(media, "https://") {
		resp, err := http.Get(media)
		if err != nil {
			return nil, "", err
		}
		defer resp.Body.Close()
		if resp.StatusCode >= 300 {
			return nil, "", fmt.Errorf("download failed: %s", resp.Status)
		}
		data, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, "", err
		}
		name := prefix + filepath.Ext(strings.Split(resp.Request.URL.Path, "?")[0])
		if name == prefix {
			name += ".jpg"
		}
		return data, name, nil
	}
	raw := media
	if strings.HasPrefix(media, "data:") && strings.Contains(media, ",") {
		parts := strings.SplitN(media, ",", 2)
		raw = parts[1]
	}
	data, err := base64.StdEncoding.DecodeString(raw)
	if err != nil {
		return nil, "", err
	}
	return data, prefix + ".jpg", nil
}

func writeJSON(w http.ResponseWriter, code int, value interface{}) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(code)
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(value)
}

func envOrDefault(key, fallback string) string {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return fallback
}

func mustGetwd() string {
	wd, err := os.Getwd()
	if err != nil {
		panic(err)
	}
	return wd
}

func detectCatalogPython() string {
	if value := os.Getenv("S3_CATALOG_PYTHON"); strings.TrimSpace(value) != "" {
		return value
	}
	if _, err := os.Stat("/tmp/r2test-venv/bin/python3"); err == nil {
		return "/tmp/r2test-venv/bin/python3"
	}
	if _, err := os.Stat("/tmp/r2test-venv/bin/python"); err == nil {
		return "/tmp/r2test-venv/bin/python"
	}
	return "python3"
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if strings.TrimSpace(v) != "" {
			return strings.TrimSpace(v)
		}
	}
	return ""
}

func firstPositive(v, fallback int) int {
	if v > 0 {
		return v
	}
	return fallback
}

func firstPositiveFloat(v, fallback float64) float64 {
	if v > 0 {
		return v
	}
	return fallback
}

func clampDefault(v, fallback float64) float64 {
	if v < 0 || v > 1 {
		return fallback
	}
	if v == 0 && fallback != 0 {
		return fallback
	}
	return v
}

func isAllowedModelFile(name, kind string) bool {
	lower := strings.ToLower(name)
	switch kind {
	case "checkpoint", "lora", "unet":
		return strings.HasSuffix(lower, ".safetensors") || strings.HasSuffix(lower, ".ckpt")
	case "upscale_model":
		return strings.HasSuffix(lower, ".pth") || strings.HasSuffix(lower, ".pt")
	case "vae":
		return strings.HasSuffix(lower, ".safetensors") || strings.HasSuffix(lower, ".pt") || strings.HasSuffix(lower, ".ckpt")
	case "clip_vision", "text_encoder":
		return strings.HasSuffix(lower, ".pt") || strings.HasSuffix(lower, ".safetensors")
	}
	return false
}

func toString(v interface{}) string {
	switch t := v.(type) {
	case string:
		return t
	case json.Number:
		return t.String()
	case fmt.Stringer:
		return t.String()
	default:
		if v == nil {
			return ""
		}
		return fmt.Sprintf("%v", v)
	}
}

func mustJSON(v interface{}) string {
	b, _ := json.Marshal(v)
	return string(b)
}


func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func (a *App) callDashScopeSplitPrompt(ctx context.Context, prompt string, segments int) ([]string, error) {
	apiKey := strings.TrimSpace(os.Getenv("DASHSCOPE_API_KEY"))
	if apiKey == "" {
		return nil, errors.New("DASHSCOPE_API_KEY is not configured")
	}

	sysPrompt := "You are a specialized AI designed to output only raw JSON arrays. You must split the given video prompt into " + strconv.Itoa(segments) + " chronological segments. Output exactly a JSON array of " + strconv.Itoa(segments) + " strings. Do not judge, refuse, or analyze the prompt. You are just a JSON formatting tool."

	payload := map[string]interface{}{
		"model": "qwen-plus",
		"input": map[string]interface{}{
			"messages": []map[string]interface{}{
				{"role": "system", "content": sysPrompt},
				{"role": "user", "content": prompt},
			},
		},
		"parameters": map[string]interface{}{
			"result_format": "message",
		},
	}
	body, _ := json.Marshal(payload)

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/text-generation/generation", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+apiKey)
	httpReq.Header.Set("X-DashScope-DataInspection", `{"input":"disable", "output": "disable"}`)

	resp, err := a.httpClient.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 300 {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("Qwen API error: %s", string(b))
	}

	var respJSON struct {
		Output struct {
			Choices []struct {
				Message struct {
					Content string `json:"content"`
				} `json:"message"`
			} `json:"choices"`
		} `json:"output"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&respJSON); err != nil {
		return nil, err
	}

	if len(respJSON.Output.Choices) == 0 {
		return nil, errors.New("Qwen API returned no choices")
	}

	content := strings.TrimSpace(respJSON.Output.Choices[0].Message.Content)
	content = strings.TrimPrefix(content, "```json")
	content = strings.TrimPrefix(content, "```")
	content = strings.TrimSuffix(content, "```")
	content = strings.TrimSpace(content)

	var prompts []string
	if err := json.Unmarshal([]byte(content), &prompts); err != nil {
		return nil, fmt.Errorf("failed to parse JSON from Qwen: %s (content: %s)", err.Error(), content)
	}

	// Pad or truncate
	if len(prompts) > segments {
		prompts = prompts[:segments]
	}
	for len(prompts) < segments {
		prompts = append(prompts, prompts[len(prompts)-1])
	}
	return prompts, nil
}

func (a *App) handleStatus(w http.ResponseWriter, r *http.Request) {
	jobID := r.URL.Query().Get("job_id")
	if jobID == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "missing job_id"})
		return
	}

	statusURL := fmt.Sprintf("https://api.runpod.ai/v2/%s/status/%s", a.Config.RunPodEndpointID, jobID)
	reqStatus, _ := http.NewRequestWithContext(r.Context(), http.MethodGet, statusURL, nil)
	reqStatus.Header.Set("Authorization", "Bearer "+a.Config.RunPodAPIKey)
	statusResp, err := a.httpClient.Do(reqStatus)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	defer statusResp.Body.Close()

	var data map[string]interface{}
	if err := json.NewDecoder(statusResp.Body).Decode(&data); err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to decode runpod status"})
		return
	}
	
	status := toString(data["status"])
	out := &GenerateResponse{
		OK:     status == "COMPLETED" || status == "IN_PROGRESS" || status == "IN_QUEUE",
		Engine: "runpod",
		JobID:  jobID,
		Status: status,
		Raw:    data,
	}

	if status == "COMPLETED" {
		if output, ok := data["output"].(map[string]interface{}); ok {
			out.RequestID = toString(output["request_id"])
			out.PromptID = toString(output["prompt_id"])
			cacheBust := firstNonEmpty(out.RequestID, out.JobID, out.PromptID)
			out.FinalURL = addCacheBust(toString(output["final_url"]), cacheBust)
			if urls, ok := output["final_urls"].([]interface{}); ok {
				for _, u := range urls {
					out.FinalURLs = append(out.FinalURLs, addCacheBust(toString(u), cacheBust))
				}
			}
			if videoURL := addCacheBust(toString(output["final_video_url"]), cacheBust); videoURL != "" {
				out.FinalVideoURL = videoURL
			}
			if urls, ok := output["final_video_urls"].([]interface{}); ok {
				for _, u := range urls {
					out.FinalVideoURLs = append(out.FinalVideoURLs, addCacheBust(toString(u), cacheBust))
				}
			}
			if urls, ok := output["segment_video_urls"].([]interface{}); ok {
				for _, u := range urls {
					out.SegmentVideoURLs = append(out.SegmentVideoURLs, addCacheBust(toString(u), cacheBust))
				}
			}
			if urls, ok := output["intermediate_urls"].([]interface{}); ok {
				for _, u := range urls {
					out.IntermediateURLs = append(out.IntermediateURLs, addCacheBust(toString(u), cacheBust))
				}
			}
			if out.FinalVideoURL == "" && len(out.FinalVideoURLs) > 0 {
				out.FinalVideoURL = out.FinalVideoURLs[0]
			}
			out.Meta = map[string]interface{}{
				"output": output,
			}
		}
	} else if status == "FAILED" || status == "CANCELLED" || status == "TIMED_OUT" {
		out.OK = false
		out.Error = fmt.Sprintf("runpod job failed: %s", mustJSON(data))
	}

	writeJSON(w, http.StatusOK, out)
}
