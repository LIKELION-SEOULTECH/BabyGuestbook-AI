# 🍼 babyguestbook-AI  -`♡´-

KoBERT 기반 감정 분석 모델을 Flask API로 제공합니다.  
텍스트 데이터를 입력받아 감정을 분류합니다.  
(예: 행복, 슬픔, 분노 등)

---

## 🚀 프로젝트 구성

| 파일명 | 설명 |
|--------|------|
| `app.py` | Flask 서버 실행 파일 (API 제공) |
| `ddkobert.py` | KoBERT 학습 및 ONNX 변환 스크립트 |
| `kobert_emotion.onnx` | 학습 완료된 ONNX 모델 파일 |
| `requirements.txt` | 프로젝트 의존성 목록 |
| `Dockerfile` | Docker 컨테이너 환경 설정 |
| `README.md` | 이 문서! 프로젝트 설명서 ✨ |

---

## 📊 모델 성능

- ✅ **Train Accuracy:** 97.90%  
- ✅ **Test Accuracy:** 92.60%  

---

## 📡 감정 분석 API 사용법

- **URL**: `http://192.168.0.36:5000/emotion/analysis`
- **요청 방식**: `POST`
- **헤더**:  
  `Content-Type: application/json`

- **요청 예시**
```json
{
  "text": "오늘은 정말 행복한 하루야!"
}
