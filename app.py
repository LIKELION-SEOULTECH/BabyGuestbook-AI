from flask import Flask, request, jsonify
import onnxruntime as ort
import torch
from kobert_transformers import get_tokenizer
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 모든 도메인 허용

# ONNX 모델 로드
onnx_model_path = "kobert_emotion.onnx"
session = ort.InferenceSession(onnx_model_path)

# KoBERT tokenizer 로드
tokenizer = get_tokenizer()
max_len = 64

# 라벨 매핑
label_map = {
    0: "FEAR", 1: "SURPRISE", 2: "ANGRY", 3: "SADNESS",
    4: "NEUTRAL", 5: "HAPPINESS", 6: "DISGUST"
}

# 전처리 함수
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
    return "🥳 서버 작동 중! POST 요청은 /emotion/analysis 로 보내주세요."

@app.route('/emotion/analysis', methods=['POST'])
def analyze_emotion():
    # JSON 요청만 허용
    data = request.get_json(force=True, silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "⚠️ 'text' 필드가 필요합니다. 예: {'text': '오늘은 행복한 하루야!'}"}), 400

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
    app.run(host="0.0.0.0", port=8000)
