from flask import Flask, request, jsonify
from flask_cors import CORS
import io
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# DeeperCNN
class DeeperCNN(nn.Module):
    def __init__(self, num_classes):
        super(DeeperCNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 160 -> 80
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 80 -> 40
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 40 -> 20
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)           # 20 -> 10
        )
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 10 * 10, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x


# Device & model loading
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DeeperCNN(num_classes=5).to(device)
model.load_state_dict(torch.load("best_model1.pth", map_location=device))
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


class_labels = ['Covid-19', 'Emphysema', 'Normal', 'Pneumonia', 'Tuberculosis']

# Lightweight grayscale check
def is_possibly_xray(img: Image.Image) -> bool:
    rgb = img.convert("RGB").resize((64, 64))  # downsample for speed
    arr = np.asarray(rgb).astype(np.float32)   # (64,64,3)
    diffs = np.stack([
        np.abs(arr[...,0] - arr[...,1]),
        np.abs(arr[...,1] - arr[...,2]),
        np.abs(arr[...,2] - arr[...,0]),
    ], axis=-1)
    avg_diff = diffs.mean()
    return avg_diff < 10.0


# Flask app
app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Optional safety filter
        if not is_possibly_xray(image):
            return jsonify({"error": "The uploaded image does not appear to be a chest X-ray."}), 400

        # Preprocess
        x = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
            conf = conf.item()
            pred_idx = pred_idx.item()

        # optional confidence gate
        if conf < 0.75:
            return jsonify({
                "error": "Low confidence prediction.",
                "confidence": f"{conf*100:.2f}%"
            }), 400

        label = class_labels[pred_idx]
        return jsonify({
            "prediction": label,
            "confidence": f"{conf*100:.2f}%"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Use threaded=False if you run into PyTorch + Flask thread issues
    app.run(debug=True, port=3562, host="0.0.0.0", threaded=True)