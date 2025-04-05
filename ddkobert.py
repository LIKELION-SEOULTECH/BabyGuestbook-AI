import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm

from torch.optim import AdamW
from kobert_transformers import get_tokenizer, get_kobert_model


from transformers.optimization import get_cosine_schedule_with_warmup
from sklearn.model_selection import train_test_split



# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA 사용 가능 여부:", torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

print("사용 중인 디바이스:", device)

# 1. 데이터 로드 및 전처리
import pandas as pd

file_path = r"C:\Users\DTLAB1\Downloads\5차년도_2차.csv"

# 파일 한 번만 읽기
data = pd.read_csv(file_path, encoding='cp949', sep=None, engine='python')

# 데이터 확인
print("불러오기 성공!")
print(data.head())


# 클래스 레이블 인코딩
label_map = {
    "fear": 0, "surprise": 1, "angry": 2, "sadness": 3,
    "neutral": 4, "happiness": 5, "disgust": 6
}
print(data.head()) 
# 7개의 감정 class → 숫자
data.loc[(data['상황'] == "fear"), '상황'] = 0  # fear → 0
data.loc[(data['상황'] == "surprise"), '상황'] = 1  # surprise → 1
data.loc[(data['상황'] == "angry"), '상황'] = 2  # angry → 2
data.loc[(data['상황'] == "sadness"), '상황'] = 3  # sadness → 3
data.loc[(data['상황'] == "neutral"), '상황'] = 4  # neutral → 4
data.loc[(data['상황'] == "happiness"), '상황'] = 5  # happiness → 5
data.loc[(data['상황'] == "disgust"), '상황'] = 6  # disgust → 6

# %%
print(data['상황'].unique())
# [발화문, 상황] data_list 생성
data_list = []
for ques, label in zip (data['발화문'], data['상황']):
  data = []
  data.append(ques)
  data.append(str(label))

  data_list.append(data)
print(data)
print(data_list[:10])

# 훈련/테스트 셋 분리
dataset_train, dataset_test = train_test_split(data_list, test_size=0.2, shuffle=True, random_state=42)

# 2. Tokenizer 및 모델 로딩
tokenizer = get_tokenizer()  # KoBERT 전용 토크나이저
bertmodel = get_kobert_model()  # KoBERT 모델


# 3. 데이터셋 클래스 정의
class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len):
        self.sentences = [dataset[i][sent_idx] for i in range(len(dataset))]
        self.labels = [int(dataset[i][label_idx]) for i in range(len(dataset))]
        self.tokenizer = bert_tokenizer
        self.max_len = max_len

    def __getitem__(self, i):
        encoded = self.tokenizer(
            self.sentences[i],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        # 안전하게 token_type_ids 강제 0 처리
        token_type_ids = torch.zeros_like(input_ids)
        label = self.labels[i]
        return input_ids, attention_mask, token_type_ids, torch.tensor(label)


    def __len__(self):
        return len(self.labels)

# 4. 모델 정의
class BERTClassifier(nn.Module):
    def __init__(self, bert, hidden_size=768, num_classes=7, dr_rate=0.5):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(p=dr_rate)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs.pooler_output
        out = self.dropout(pooled_output)
        return self.classifier(out)


# 5. 하이퍼파라미터
max_len = 64
batch_size = 64
num_epochs = 5
learning_rate = 5e-5
warmup_ratio = 0.1
max_grad_norm = 1
log_interval = 200

# 6. 데이터로더 준비
train_data = BERTDataset(dataset_train, 0, 1, tokenizer, max_len)
test_data = BERTDataset(dataset_test, 0, 1, tokenizer, max_len)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# 7. 학습 준비
model = BERTClassifier(bertmodel).to(device)
loss_fn = nn.CrossEntropyLoss()

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

t_total = len(train_loader) * num_epochs
warmup_step = int(t_total * warmup_ratio)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

def calc_accuracy(preds, labels):
    _, pred_max = torch.max(preds, 1)
    correct = (pred_max == labels).sum().item()
    return correct / len(labels)

# 8. 학습 루프
for epoch in range(num_epochs):
    model.train()
    total_acc = 0
    for batch_id, (input_ids, attention_mask, token_type_ids, labels) in enumerate(tqdm(train_loader)):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, token_type_ids)
        loss = loss_fn(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        acc = calc_accuracy(outputs, labels)
        total_acc += acc

        if batch_id % log_interval == 0:
            print(f"Epoch {epoch+1}, Step {batch_id+1}, Loss {loss.item():.4f}, Acc {acc:.4f}")

    print(f"Epoch {epoch+1} finished. Train Acc: {total_acc / len(train_loader):.4f}")

# 9. 테스트
model.eval()
total_acc = 0
with torch.no_grad():
    for input_ids, attention_mask, token_type_ids, labels in test_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        labels = labels.to(device)

        outputs = model(input_ids, attention_mask, token_type_ids)
        acc = calc_accuracy(outputs, labels)
        total_acc += acc
print(f"Test Accuracy: {total_acc / len(test_loader):.4f}")

# 10. ONNX 저장
import torch

# 모델과 더미 입력 정의
# dummy_input은 input_ids, attention_mask, token_type_ids 3개를 포함해야 돼!
dummy_input = (
    torch.randint(0, 2000, (1, 64)).to(device),  # input_ids
    torch.ones(1, 64, dtype=torch.long).to(device),  # attention_mask
    torch.zeros(1, 64, dtype=torch.long).to(device)  # token_type_ids
)

torch.onnx.export(
    model, 
    dummy_input, 
    "kobert_emotion.onnx",
    export_params=True,
    opset_version=17,
    input_names=['input_ids', 'attention_mask', 'token_type_ids'],
    output_names=['output'],
    dynamic_axes={
        'input_ids': {0: 'batch_size'},
        'attention_mask': {0: 'batch_size'},
        'token_type_ids': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
