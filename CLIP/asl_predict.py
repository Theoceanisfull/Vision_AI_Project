import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import CLIPModel

import pyttsx3


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "openai/clip-vit-base-patch32"
MODEL_PATH = "clip_asl_ft.pth"

class_names = ['0','1','2','3','4','5','6','7','8','9',
               'a','b','c','d','e','f','g','h','i','j',
               'k','l','m','n','o','p','q','r','s','t',
               'u','v','w','x','y','z']


# Text-to-speech engine
# tts_engine = pyttsx3.init()
# tts_engine.setProperty("rate", 150)


def speak_prediction(prediction):
    if prediction.isdigit():
        text = f"Predicted sign is number {prediction}"
    else:
        text = f"Predicted sign is letter {prediction.upper()}"

    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.say(text)
    engine.runAndWait()
    engine.stop()


class CLIPFineTuneClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()

        self.clip = CLIPModel.from_pretrained(model_name)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.clip.config.projection_dim, num_classes)
        )

    def forward(self, pixel_values):
        vision_outputs = self.clip.vision_model(pixel_values=pixel_values)
        pooled_output = vision_outputs.pooler_output
        image_features = self.clip.visual_projection(pooled_output)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        logits = self.classifier(image_features)
        return logits


model = CLIPFineTuneClassifier(MODEL_NAME, len(class_names))
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

if isinstance(checkpoint, dict) and "model_state" in checkpoint:
    model.load_state_dict(checkpoint["model_state"])
    class_names = checkpoint["class_names"]
else:
    model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
])


def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1)

    pred_idx = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_idx].item()

    return class_names[pred_idx], confidence


def choose_image():
    file_path = filedialog.askopenfilename(
        title="Choose ASL Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
    )

    if not file_path:
        return

    img = Image.open(file_path).convert("RGB")
    img.thumbnail((300, 300))

    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk

    prediction, confidence = predict_image(file_path)

    result_label.config(
        text=f"Prediction: {prediction.upper()}\nConfidence: {confidence:.4f}"
    )

    speak_prediction(prediction)


root = tk.Tk()
root.title("ASL CLIP Prediction")
root.geometry("420x520")

title_label = tk.Label(
    root,
    text="ASL Image Prediction",
    font=("Arial", 18, "bold")
)
title_label.pack(pady=15)

btn = tk.Button(
    root,
    text="Choose Image",
    command=choose_image,
    font=("Arial", 14),
    width=20
)
btn.pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=15)

result_label = tk.Label(
    root,
    text="Prediction will appear here",
    font=("Arial", 16),
    justify="center"
)
result_label.pack(pady=20)

root.mainloop()