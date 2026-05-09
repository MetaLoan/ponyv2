import re

with open('/Users/jack/Documents/gitcode/serveless-allinone/frontend/src/App.tsx', 'r') as f:
    content = f.read()

# 1. Define flags
content = content.replace(
    'const isWanMode = mode === "wan2_2_i2v_extend_any_frame";',
    'const isWanMode = mode === "wan2_2_i2v_extend_any_frame";\n  const isWanAnimateMode = mode === "wan2_2_animate";\n  const isWanExtractMode = mode === "wan2_2_dwpose_extract";'
)

# 2. Add custom Animation Generation Params section
animate_params = """
        {isWanAnimateMode && (
          <section className="card">
            <h2>Animation Generation Params</h2>
            <div className="inline">
              <NumberField label="Seed" value={seed} onChange={setSeed} min={0} max={9999999999999999} step={1} />
              <NumberField label="Steps" value={steps} onChange={setSteps} min={1} max={100} step={1} />
            </div>
            <div className="inline">
              <label>
                Resolution
                <select value={i2vResolution} onChange={(e) => setI2VResolution(e.target.value)}>
                   <option value="480P">480P (Standard)</option>
                   <option value="720P">720P (High Quality)</option>
                </select>
              </label>
            </div>
          </section>
        )}
"""
content = content.replace(
    '<section className="grid two">',
    '<section className="grid two">\n' + animate_params
)

# 3. Hide sections for Animate/Extract mode
# Hide Prompt section for Extract
content = content.replace(
    '<h2>{mode === "qwen_pose_fusion" ? "Qwen Pose Fusion Prompt" : mode === "wan2_2_i2v_extend_any_frame" ? "Wan Video Prompt" : "Prompt"}</h2>',
    '{!isWanExtractMode && <h2>{mode === "qwen_pose_fusion" ? "Qwen Pose Fusion Prompt" : mode === "wan2_2_i2v_extend_any_frame" ? "Wan Video Prompt" : "Prompt"}</h2>}'
)
content = content.replace(
    '<textarea rows={4} value={prompt} onChange={(e) => setPrompt(e.target.value)} />',
    '{!isWanExtractMode && <textarea rows={4} value={prompt} onChange={(e) => setPrompt(e.target.value)} />}'
)
content = content.replace(
    '<textarea rows={3} value={negativePrompt} onChange={(e) => setNegativePrompt(e.target.value)} />',
    '{!isWanExtractMode && <textarea rows={3} value={negativePrompt} onChange={(e) => setNegativePrompt(e.target.value)} />}'
)

# Hide WAN Preset / Model selection for Animate/Extract
content = content.replace(
    '{isWanMode ? (',
    '{(isWanMode && !isWanAnimateMode && !isWanExtractMode) ? ('
)
# The else block for Model Selection
content = content.replace(
    ': (',
    ': (!isWanAnimateMode && !isWanExtractMode) ? ('
)
# Close it
content = content.replace(
    '</section>\n          )}',
    '</section>\n          ) : null}'
)

# Hide Video Generation Params for Animate
content = content.replace(
    '{isWanMode && (',
    '{isWanMode && !isWanAnimateMode && !isWanExtractMode && ('
)

# Hide Stage Params for Animate/Extract
content = content.replace(
    '{!isWanMode && (',
    '{!isWanMode && !isWanAnimateMode && !isWanExtractMode && ('
)

with open('/Users/jack/Documents/gitcode/serveless-allinone/frontend/src/App.tsx', 'w') as f:
    f.write(content)
