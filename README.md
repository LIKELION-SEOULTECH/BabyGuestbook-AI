# 🍼 babyguestbook-AI - 감정 분석 챗봇 API `♡´-

KoBERT 기반 감정 분석 모델을 Flask API로 제공합니다.  
사용자의 발화를 분석해 7가지 감정 중 하나를 예측합니다!

🎈 **"오늘은 정말 행복한 하루야!" → 😄 Happiness**

---

## 🚀 프로젝트 구성

| 파일 | 설명 |
|------|------|
| `app.py` | Flask 서버 실행 파일 (API 제공) |
| `ddkobert.py` | KoBERT 학습 및 ONNX 변환 스크립트 |
| `kobert_emotion.onnx` | 학습 완료된 ONNX 모델 파일 |
| `README.md` | 이 문서! |

---

## 🧠 감정 분류 클래스 (7개)

| 클래스 번호 | 감정 (영문) | 감정 (한글) |
|-------------|-------------|-------------|
| 0 | Fear | 공포 |
| 1 | Surprise | 놀람 |
| 2 | Angry | 분노 |
| 3 | Sadness | 슬픔 |
| 4 | Neutral | 중립 |
| 5 | Happiness | 기쁨 |
| 6 | Disgust | 혐오 |

---

## 📊 모델 성능

- ✅ **Train Accuracy:** 97.90%  
- ✅ **Test Accuracy:** 92.60%  

---

## 🔗 REST API 정보

- **URL:** `http://192.168.0.36:5000/emotion/analysis`  
- **요청 방식:** `POST`  
- **헤더:**  
- **요청 예시:**

```json
{
"text": "오늘은 정말 행복한 하루야!"
}
