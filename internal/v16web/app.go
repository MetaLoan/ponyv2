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
	"io"
	"log"
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
	Addr             string
	RepoRoot         string
	FrontendDist     string
	WorkflowTemplate string
	KeyEnvFile       string
	S3CredsFile      string
	ComfyAPIURL      string
	RunPodAPIKey     string
	RunPodEndpointID string
	S3Endpoint       string
	S3Region         string
	S3Bucket         string
	S3AccessKey      string
	S3SecretKey      string
	S3RootPrefix     string
	S3CatalogPython  string
	S3CatalogScript  string
}

type GenerateRequest struct {
	Mode                string       `json:"mode"`
	ReferenceImage      string       `json:"reference_image"`
	QwenExtraImage      string       `json:"qwen_extra_image"`
	PoseImage           string       `json:"pose_image"`
	Prompt              string       `json:"prompt"`
	QwenSwapPrompt      string       `json:"qwen_swap_prompt"`
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
	RequestID           string       `json:"request_id"`
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
	IntermediateURLs []string               `json:"intermediate_urls,omitempty"`
	Meta             map[string]interface{} `json:"meta,omitempty"`
	Raw              map[string]interface{} `json:"raw,omitempty"`
}

type CatalogResponse struct {
	Checkpoints   []CatalogItem `json:"checkpoints"`
	Loras         []CatalogItem `json:"loras"`
	UpscaleModels []CatalogItem `json:"upscale_models"`
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

func NewApp() (*App, error) {
	repoRoot := mustGetwd()
	cfg := Config{
		Addr:             envOrDefault("V16WEB_ADDR", ":8080"),
		RepoRoot:         repoRoot,
		FrontendDist:     filepath.Join(repoRoot, "frontend", "dist"),
		WorkflowTemplate: filepath.Join(repoRoot, "workflows", "pulid_sdxl_workflow_web_api.json"),
		KeyEnvFile:       envOrDefault("KEY_ENV_FILE", filepath.Clean(filepath.Join(repoRoot, "..", "key.env"))),
		S3CredsFile:      envOrDefault("S3_CREDENTIALS_FILE", filepath.Clean(filepath.Join(repoRoot, "..", "s3-credentials.txt"))),
		S3RootPrefix:     envOrDefault("S3_MODEL_ROOT_PREFIX", "runpod-slim/ComfyUI/models"),
		S3Region:         envOrDefault("S3_REGION", "eu-ro-1"),
		S3CatalogScript:  envOrDefault("S3_CATALOG_SCRIPT", filepath.Join(repoRoot, "scripts", "list_model_catalog.py")),
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
	a.router.HandleFunc("/api/comfy/view", a.handleComfyView)

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
		Loras:         a.listCatalog(ctx, path.Join(a.Config.S3RootPrefix, "loras")+"/", "lora"),
		UpscaleModels: a.listCatalog(ctx, path.Join(a.Config.S3RootPrefix, "upscale_models")+"/", "upscale_model"),
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

	switch a.engine() {
	case "comfy":
		resp, err := a.generateWithComfy(r.Context(), req)
		if err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
			return
		}
		writeJSON(w, http.StatusOK, resp)
	case "runpod":
		resp, err := a.generateWithRunPod(r.Context(), req)
		if err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
			return
		}
		writeJSON(w, http.StatusOK, resp)
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
	XMLName     xml.Name    `xml:"ListBucketResult"`
	Contents    []s3Object  `xml:"Contents"`
	IsTruncated bool        `xml:"IsTruncated"`
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

	raw, err := os.ReadFile(a.Config.WorkflowTemplate)
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
	if req.Mode == "qwen_swap_face" {
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
	} else if req.Mode == "text_only" || req.Mode == "qwen_swap_face" {
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
	if req.Mode == "qwen_swap_face" {
		warnings = append(warnings, "qwen_swap_face is a post-process DashScope face swap after base text-only generation")
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
	if req.Mode == "qwen_swap_face" {
		return nil, errors.New("qwen_swap_face is supported via the RunPod engine path only")
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
	if len(final) > 0 {
		out.FinalURL = a.makeLocalComfyViewURL(final[0])
	}
	if req.KeepIntermediate != nil && *req.KeepIntermediate {
		for _, img := range intermediate {
			out.IntermediateURLs = append(out.IntermediateURLs, a.makeLocalComfyViewURL(img))
		}
	}
	return out, nil
}

func (a *App) generateWithRunPod(ctx context.Context, req GenerateRequest) (*GenerateResponse, error) {
	req = normalizeRequest(req)
	warnings := []string{}

	input := map[string]interface{}{
		"mode":                   req.Mode,
		"reference_image":        req.ReferenceImage,
		"prompt":                 req.Prompt,
	"qwen_swap_prompt":       req.QwenSwapPrompt,
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
				out.FinalURL = toString(output["final_url"])
				if urls, ok := output["final_urls"].([]interface{}); ok {
					for _, u := range urls {
						out.FinalURLs = append(out.FinalURLs, toString(u))
					}
				}
				if urls, ok := output["intermediate_urls"].([]interface{}); ok {
					for _, u := range urls {
						out.IntermediateURLs = append(out.IntermediateURLs, toString(u))
					}
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

func (a *App) makeLocalComfyViewURL(img comfyImage) string {
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
	return "/api/comfy/view?" + values.Encode()
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
	req.Mode = normalizeMode(req.Mode, req.ReferenceImage, req.PoseImage)
	req.BatchSize = firstPositive(req.BatchSize, 1)
	req.Width = firstPositive(req.Width, 832)
	req.Height = firstPositive(req.Height, 1216)
	req.BaseSteps = firstPositive(req.BaseSteps, 8)
	req.Steps = firstPositive(req.Steps, 40)
	req.BaseCFG = firstPositiveFloat(req.BaseCFG, 4)
	req.CFG = firstPositiveFloat(req.CFG, 4)
	req.BaseDenoise = firstPositiveFloat(req.BaseDenoise, 1)
	req.Denoise = firstPositiveFloat(req.Denoise, 1)
	req.BaseSamplerName = firstNonEmpty(req.BaseSamplerName, "dpmpp_2m_sde")
	req.BaseScheduler = firstNonEmpty(req.BaseScheduler, "karras")
	req.SamplerName = firstNonEmpty(req.SamplerName, "dpmpp_2m_sde")
	req.Scheduler = firstNonEmpty(req.Scheduler, "karras")
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
	req.QwenSwapPrompt = firstNonEmpty(req.QwenSwapPrompt, "将参考图中的人脸自然融合到生成图人物上，保持姿势、构图、光照、背景和服装不变，保证真实自然，五官清晰，肤质真实。")
	req.QwenModel = firstNonEmpty(req.QwenModel, "qwen-image-edit-max")
	req.QwenSize = strings.TrimSpace(req.QwenSize)
	req.QwenExtraImage = strings.TrimSpace(req.QwenExtraImage)
	req.CKPTName = firstNonEmpty(req.CKPTName, "SDXL_Photorealistic_Mix_nsfw.safetensors")
	req.UpscaleModelName = firstNonEmpty(req.UpscaleModelName, "4x-UltraSharp.pth")
	req.OutputFormat = firstNonEmpty(req.OutputFormat, "jpg")
	req.JPGQuality = firstPositive(req.JPGQuality, 85)
	if req.EnablePulid == nil {
		b := req.Mode != "pose_only" && req.Mode != "text_only"
		if req.Mode == "qwen_swap_face" {
			b = false
		}
		req.EnablePulid = &b
	}
	if req.Mode == "qwen_swap_face" {
		f := false
		req.EnablePulid = &f
	}
	if req.EnableLora == nil {
		b := len(req.Loras) > 0
		req.EnableLora = &b
	}
	if req.EnableUpscale == nil {
		b := false
		req.EnableUpscale = &b
	}
	if req.KeepIntermediate == nil {
		b := true
		req.KeepIntermediate = &b
	}
	return req
}

func normalizeMode(mode, referenceImage, poseImage string) string {
	switch strings.TrimSpace(mode) {
	case "dual_pass_auto_pose", "pose_then_face_swap", "pose_only", "text_only", "qwen_swap_face":
		return strings.TrimSpace(mode)
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
	case "checkpoint", "lora":
		return strings.HasSuffix(lower, ".safetensors") || strings.HasSuffix(lower, ".ckpt")
	case "upscale_model":
		return strings.HasSuffix(lower, ".pth") || strings.HasSuffix(lower, ".pt")
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
