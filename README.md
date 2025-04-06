안녕하세요 babyguestbook-AI 입니다-`♡´-

- app.py: 모델 flask api 생성 
  http://192.168.0.36:5000/emotion/analysis
  <테스트할 때는 아래 참고>
                POST /emotion/analysis
        Content-Type: application/json
        
        {
          "text": "오늘은 정말 행복한 하루야!"
        }

- ddkobert.py: 모델 학습 및 onnx 파일로 변환 코드
             모델 정확도
             Train Acc: 0.9790
             Test Accuracy: 0.9260
             
- kobert_emotion.onnx: 모델 onnx 파일

