# FinBERT-KR 모델 테스트 
from transformers import AutoTokenizer, AutoModel
import torch

# FinBERT-KR 모델 로드
model_name = "snunlp/KR-FinBert-SC"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 임베딩 테스트
text = "IRP 계좌 입금 시간이 궁금해요."
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

with torch.no_grad():
    outputs = model(**inputs)

# CLS 토큰의 임베딩 벡터 출력
embedding_vector = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
print("Embedding vector shape:", embedding_vector.shape)  # (768,)