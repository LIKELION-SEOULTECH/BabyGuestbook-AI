from flask import Flask, request, jsonify
import onnxruntime as ort
import torch
from kobert_transformers import get_tokenizer
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 모든 도메인 허용 (원하면 특정 도메인만 허용 가능)


# ONNX 모델 로드
onnx_model_path = "kobert_emotion.onnx"
session = ort.InferenceSession(onnx_model_path)

# KoBERT tokenizer 로드
tokenizer = get_tokenizer()
max_len = 64

# 라벨 맵 (필요 시 리턴용)
label_map = {
    0: "FEAR", 1: "SURPRISE", 2: "ANGRY", 3: "SADNESS",
    4: "NEUTRAL", 5: "HAPPINESS", 6: "DISGUST"
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
@app.route('/emotion/analysis', methods=['GET', 'POST'])
def analyze_emotion():
    if request.method == 'GET':
        return '''
            <form action="/emotion/analysis" method="post">
                <input name="text" placeholder="텍스트 입력">
                <input type="submit">
            </form>
        '''
    elif request.method == 'POST':
        data = request.get_json() or request.form
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
