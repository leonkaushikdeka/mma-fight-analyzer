from PIL import Image, ImageDraw, ImageFont
import os

os.chdir("D:/active projects/mma-fight-analyzer")

lines = [
    "$ pip install -r requirements.txt",
    "Requirement already satisfied: mediapipe>=0.10.0",
    "Requirement already satisfied: opencv-python>=4.9.0",
    "Requirement already satisfied: numpy>=1.24.0",
    "",
    "$ python -c \"import mediapipe, cv2, numpy\"",
    "mediapipe: 0.10.35",
    "opencv: 4.13.0",
    "numpy: 2.4.2",
    "All dependencies OK",
    "",
    "$ python -c \"from src.referee_ai import RefereeAI\"",
    "MMA Fight Analyzer core modules loaded successfully",
    "",
    "$ python --version",
    "Python 3.13.11",
]

img_w, img_h = 900, len(lines) * 24 + 80
img = Image.new("RGB", (img_w, img_h), "#1e1e2e")
draw = ImageDraw.Draw(img)

draw.rectangle([0, 0, img_w, 32], fill="#313244")
draw.text((15, 8), "MMA Fight Analyzer - D:\\active projects\\mma-fight-analyzer", fill="#cdd6f4", font=ImageFont.load_default())

try:
    font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 14)
except:
    font = ImageFont.load_default()

y = 42
for line in lines:
    if line.startswith("$"):
        draw.text((15, y), line, fill="#89b4fa", font=font)
    elif "successfully" in line.lower() or "OK" in line:
        draw.text((15, y), line, fill="#a6e3a1", font=font)
    else:
        draw.text((15, y), line, fill="#cdd6f4", font=font)
    y += 24

os.makedirs("screenshots", exist_ok=True)
img.save("screenshots/demo.png")
print("Screenshot saved")
