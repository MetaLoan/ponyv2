import re

with open('/Users/jack/Documents/gitcode/serveless-allinone/frontend/src/App.tsx', 'r') as f:
    content = f.read()

# Add faceMedia state next to actionVideoMedia
content = content.replace(
    'const [actionVideoMedia, setActionVideoMedia] = useState<MediaState>({ kind: "url", file: null, url: "", preview: "" });',
    'const [actionVideoMedia, setActionVideoMedia] = useState<MediaState>({ kind: "url", file: null, url: "", preview: "" });\n  const [faceMedia, setFaceMedia] = useState<MediaState>({ kind: "url", file: null, url: "", preview: "" });'
)

# Fix payload builder for dwpose_extract
# No change needed for actionVideoMedia -> video_url for dwpose_extract

# Fix payload builder for wan2_2_animate
payload_animate_old = """
    if (mode === "wan2_2_animate") {
      body.action_video_url = await resolveMedia(actionVideoMedia);
      body.character_image_url = await resolveMedia(referenceMedia);
      body.prompt = prompt;
      body.width = parseInt(i2vResolution.split("*")[0]);
      body.height = parseInt(i2vResolution.split("*")[1]);
    }
"""

payload_animate_new = """
    if (mode === "wan2_2_animate") {
      body.pose_video_url = await resolveMedia(poseMedia);
      body.face_video_url = await resolveMedia(faceMedia);
      body.character_image_url = await resolveMedia(referenceMedia);
      body.prompt = prompt;
      body.width = parseInt(i2vResolution.split("*")[0]);
      body.height = parseInt(i2vResolution.split("*")[1]);
    }
"""
content = content.replace(payload_animate_old.strip(), payload_animate_new.strip())

# Fix MediaCard rendering
# We have a MediaCard for actionVideoMedia for mode === "wan2_2_dwpose_extract" || mode === "wan2_2_animate"
# Change it to only dwpose_extract
content = content.replace(
    '{(mode === "wan2_2_dwpose_extract" || mode === "wan2_2_animate") && (',
    '{(mode === "wan2_2_dwpose_extract") && ('
)

# Add MediaCards for poseMedia and faceMedia for wan2_2_animate
new_cards = """
        {mode === "wan2_2_animate" && (
          <>
            <MediaCard
              title="Pose Stickman Video (MP4)"
              media={poseMedia}
              onKindChange={(kind) => setPoseMedia(prev => ({ ...prev, kind }))}
              onFileChange={(e) => {
                if (e.target.files && e.target.files[0]) {
                  const file = e.target.files[0];
                  setPoseMedia({ kind: "file", file, url: "", preview: URL.createObjectURL(file) });
                }
              }}
              onURLChange={(value) => setPoseMedia(prev => ({ ...prev, url: value }))}
            />
            <MediaCard
              title="Face Crop Video (MP4)"
              media={faceMedia}
              onKindChange={(kind) => setFaceMedia(prev => ({ ...prev, kind }))}
              onFileChange={(e) => {
                if (e.target.files && e.target.files[0]) {
                  const file = e.target.files[0];
                  setFaceMedia({ kind: "file", file, url: "", preview: URL.createObjectURL(file) });
                }
              }}
              onURLChange={(value) => setFaceMedia(prev => ({ ...prev, url: value }))}
            />
          </>
        )}
"""
content = content.replace(
    '{mode === "qwen_pose_fusion" && (',
    new_cards.strip() + '\n        {mode === "qwen_pose_fusion" && ('
)

with open('/Users/jack/Documents/gitcode/serveless-allinone/frontend/src/App.tsx', 'w') as f:
    f.write(content)
