import re

with open('/Users/jack/Documents/gitcode/serveless-allinone/frontend/src/App.tsx', 'r') as f:
    content = f.read()

# Update Mode type
content = content.replace(
    '| "wan2_2_i2v_extend_any_frame";',
    '| "wan2_2_i2v_extend_any_frame" | "wan2_2_dwpose_extract" | "wan2_2_animate";'
)

# Add actionVideoMedia state
content = content.replace(
    'const [wanStartMedia, setWanStartMedia] = useState<MediaState>({ kind: "file", file: null, url: "", preview: "" });',
    'const [wanStartMedia, setWanStartMedia] = useState<MediaState>({ kind: "file", file: null, url: "", preview: "" });\n  const [actionVideoMedia, setActionVideoMedia] = useState<MediaState>({ kind: "url", file: null, url: "", preview: "" });'
)

# Add modeSummary descriptions
mode_summary_patch = """
    wan2_2_i2v_extend_any_frame: "Generate extended video frames from an initial image or video segment.",
    wan2_2_dwpose_extract: "Extract a skeleton (DWPose) MP4 from an uploaded source dance video.",
    wan2_2_animate: "Animate a static character image using a source action video while precisely keeping facial expressions.",
"""
content = re.sub(
    r'wan2_2_i2v_extend_any_frame: ".*?",',
    mode_summary_patch.strip(),
    content
)

# Update payload builder
payload_builder = """
    if (mode === "wan2_2_dwpose_extract") {
      body.video_url = await resolveMedia(actionVideoMedia);
    }
    if (mode === "wan2_2_animate") {
      body.action_video_url = await resolveMedia(actionVideoMedia);
      body.character_image_url = await resolveMedia(referenceMedia);
      body.prompt = prompt;
      body.width = parseInt(i2vResolution.split("*")[0]);
      body.height = parseInt(i2vResolution.split("*")[1]);
    }
"""
content = content.replace(
    'if (mode === "wan2_2_i2v_extend_any_frame") {',
    payload_builder + '\n      if (mode === "wan2_2_i2v_extend_any_frame") {'
)

# Update select options
select_options = """
            <option value="wan2_2_i2v_extend_any_frame">wan2.2 i2v-extend-any-frame</option>
            <option value="wan2_2_dwpose_extract">wan2.2 Generate DWPose</option>
            <option value="wan2_2_animate">wan2.2 Animate Generate</option>
"""
content = content.replace(
    '<option value="wan2_2_i2v_extend_any_frame">wan2.2 i2v-extend-any-frame</option>',
    select_options.strip()
)

# Render MediaCard for referenceMedia in wan2_2_animate
ref_card_cond = 'mode === "qwen_swap_face" || mode === "wan2_2_animate"'
content = content.replace('mode === "qwen_swap_face") && (', ref_card_cond + ') && (')

# Render MediaCard for actionVideoMedia
action_video_card = """
        {(mode === "wan2_2_dwpose_extract" || mode === "wan2_2_animate") && (
          <MediaCard
            title="Action Video (MP4)"
            media={actionVideoMedia}
            onKindChange={(kind) => setActionVideoMedia(prev => ({ ...prev, kind }))}
            onFileChange={(e) => {
              if (e.target.files && e.target.files[0]) {
                const file = e.target.files[0];
                setActionVideoMedia({ kind: "file", file, url: "", preview: URL.createObjectURL(file) });
              }
            }}
            onURLChange={(value) => setActionVideoMedia(prev => ({ ...prev, url: value }))}
            accept="video/*"
          />
        )}
"""
content = content.replace(
    '{mode === "qwen_pose_fusion" && (',
    action_video_card + '\n        {mode === "qwen_pose_fusion" && ('
)

# update dependencies in App.tsx `const updateMedia` logic? Actually the `onFileChange` I hardcoded above is easier.

# Fix the accept prop in MediaCard! Wait, `MediaCard` component doesn't take `accept`. Let's check `MediaCard` definition.
# If it doesn't take `accept`, I'll leave it as default.
content = content.replace('accept="video/*"', '')

with open('/Users/jack/Documents/gitcode/serveless-allinone/frontend/src/App.tsx', 'w') as f:
    f.write(content)
