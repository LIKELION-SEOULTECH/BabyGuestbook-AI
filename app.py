from flask import Flask, request, jsonify
import onnxruntime as ort
import torch
from kobert_transformers import get_tokenizer
import numpy as np

app = Flask(__name__)

# ONNX 모델 로드
onnx_model_path = "kobert_emotion.onnx"
session = ort.InferenceSession(onnx_model_path)

# KoBERT tokenizer 로드
tokenizer = get_tokenizer()
max_len = 64

# 라벨 맵 (필요 시 리턴용)
label_map = {
    0: "fear", 1: "surprise", 2: "angry", 3: "sadness",
    4: "neutral", 5: "happiness", 6: "disgust"
}

def preprocess(text):
    encoded = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors='pt'
    )

    input_ids = encoded['input_ids'].numpy()
    attention_mask = encoded['attention_mask'].numpy()
    token_type_ids = np.zeros_like(input_ids)  # KoBERT는 token_type_ids = 0

    return input_ids, attention_mask, token_type_ids

@app.route("/")
def home():
    return "서버가 잘 작동 중입니다!"

def analyze_emotion():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "텍스트가 필요합니다."}), 400

    text = data["text"]
    input_ids, attention_mask, token_type_ids = preprocess(text)

    ort_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids
    }

    outputs = session.run(None, ort_inputs)
    logits = outputs[0]
    predicted_class = int(np.argmax(logits))

    return jsonify({
        "input": text,
        "predicted_class": predicted_class,
        "label": label_map[predicted_class]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
