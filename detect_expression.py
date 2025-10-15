import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os

# Model cnn
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 14 * 14, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# Select device
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load dataset to get class names
from torchvision import datasets
dataset = datasets.ImageFolder("images", transform=transforms.ToTensor())
expressions = dataset.classes
num_classes = len(expressions)

# Load trained model
model = SimpleCNN(num_classes).to(device)
model.load_state_dict(torch.load("expression_model.pth", map_location=device))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Load trigger images
trigger_images = {}
for expr in expressions:
    img_path = f"triggers/{expr}.jpg"
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        if img is not None:
            trigger_images[expr] = img
        else:
            print(f"Warning: {img_path} could not be decoded")
    else:
        print(f"Warning: Missing {img_path}")

# Resize all trigger images to match webcam window
display_height = 480
display_width = 640
for key in trigger_images:
    trigger_images[key] = cv2.resize(trigger_images[key], (display_width, display_height))

# Fallback neutral display if needed
neutral_display = trigger_images.get("neutral", np.zeros((display_height, display_width, 3), dtype=np.uint8))

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(3, display_width)
cap.set(4, display_height)

# Make sure window is large enough to show both sides
cv2.namedWindow("Expression Split View", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Expression Split View", 1280, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert webcam frame to tensor
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_tensor = transform(pil_img).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    confidence_value = confidence.item()
    label = expressions[pred.item()] if confidence_value > 0.6 else "neutral"
    trigger_path = f"triggers/{label}.jpg"

    # Print debug info to terminal
    print(f"Detected: {label} | Confidence: {confidence_value:.2f} | Trigger: {trigger_path}")

    # Add text overlay to webcam feed
    cv2.putText(frame, f"{label} ({confidence_value:.2f})", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Get matching trigger image
    display_img = trigger_images.get(label, neutral_display)

    # Combine webcam and trigger image horizontally
    if display_img is not None and display_img.shape == frame.shape:
        combined = np.hstack((frame, display_img))
    else:
        combined = frame

    # Show combined view
    cv2.imshow("Expression Split View", combined)

    # Exit on ESC key
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
