import json
import subprocess
import base64
from pathlib import Path

def create_scene(quad_x):
    return {
        "version": 5,
        "objs": [
            {
                "type": "SingleRay",
                "p1": {"x": 50, "y": 150},
                "p2": {"x": 100, "y": 150}
            },
            {
                "type": "Glass",
                "path": [
                    {"x": quad_x - 25, "y": 125, "arc": False},
                    {"x": quad_x + 25, "y": 125, "arc": False},
                    {"x": quad_x + 25, "y": 175, "arc": False},
                    {"x": quad_x - 25, "y": 175, "arc": False}
                ],
                "refIndex": 1.5,
                "partialReflect": True
            },
            {
                "type": "Detector",
                "p1": {"x": 350, "y": 140},
                "p2": {"x": 350, "y": 160}
            },
            {
                "type": "CropBox",
                "p1": {"x": 0, "y": 100},
                "p4": {"x": 400, "y": 200},
                "width": 800
            }
        ]
    }

def render_scene(quad_x, output_path):
    scene = create_scene(quad_x)
    scene_json = json.dumps(scene)
    
    result = subprocess.run(
        ["node", "ray-optics/integrations/runner.js"],
        input=scene_json,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Error rendering scene: {result.stderr}")
        return
    
    output = json.loads(result.stdout)
    
    if output.get("images"):
        data_url = output["images"][0]["dataUrl"]
        png_data = data_url.split(",")[1]
        png_bytes = base64.b64decode(png_data)
        
        with open(output_path, "wb") as f:
            f.write(png_bytes)
        print(f"Saved: {output_path}")

Path("results").mkdir(exist_ok=True)

render_scene(100, "results/quad_x100.png")
render_scene(300, "results/quad_x300.png")
