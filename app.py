from pathlib import Path
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ✅ Model yolu (Space içindeki bert_tiny klasörüne bakar)
MODEL_DIR = Path(__file__).parent / "bert_tiny"

# Tokenizer ve model yükle
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
model.eval()

# Etiketler
LABELS = ["human", "ai"]

# Tahmin fonksiyonu
def predict(text):
    if not text.strip():
        return "Empty input", {lbl: 0.0 for lbl in LABELS}

    enc = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    )
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    pred_id = int(probs.argmax())
    return LABELS[pred_id], {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}

# Gradio arayüzü
demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        lines=4,
        label="Enter Text",
        placeholder="Write a sentence...",
    ),
    outputs=[
        gr.Label(label="Prediction"),
        gr.JSON(label="Class Probabilities"),
    ],
    title="AI vs Human Text Classifier",
    description="Enter a text, and the model will predict whether it was written by a human or by AI.",
)

if __name__ == "__main__":
    demo.launch()

