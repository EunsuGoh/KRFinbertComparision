from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np
import pandas as pd
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ✅ 모델 설정
models = {
    "FinBERT-KR": "snunlp/KR-FinBert-SC",
    "Kakao DEBERTa": "kakaobank/kf-deberta-base"
}

# ✅ 토크나이저 & 모델 로드
tokenizers = {name: AutoTokenizer.from_pretrained(model) for name, model in models.items()}
models = {name: AutoModel.from_pretrained(model) for name, model in models.items()}

# ✅ FAQ 데이터 로드
faq_file = "./faq.xlsx"  # FAQ 파일 경로
df = pd.read_excel(faq_file)
questions = df["Question"].tolist()

# ✅ 임베딩 함수 (Mean Pooling 적용)
def get_embedding(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding / np.linalg.norm(embedding)  # 정규화하여 코사인 유사도 기반으로 변경

# ✅ 모든 질문 벡터화
embeddings = {name: [] for name in models.keys()}

for model_name, model in models.items():
    tokenizer = tokenizers[model_name]
    print(f"🔹 {model_name} 임베딩 생성 중...")
    
    for question in questions:
        emb = get_embedding(model, tokenizer, question)
        embeddings[model_name].append(emb)
    
    embeddings[model_name] = np.array(embeddings[model_name])

# ✅ FAISS 벡터 DB 생성 (코사인 유사도 기반으로 변경)
index_faiss = {name: faiss.IndexFlatIP(768) for name in models.keys()}  # Inner Product 기반 인덱스 생성

for model_name in models.keys():
    index_faiss[model_name].add(embeddings[model_name])

print("✅ FAISS 벡터 DB 저장 완료!")

# ✅ 유사 질문 검색 함수 (코사인 유사도 적용)
def search_similar(model_name, query, top_k=5):
    tokenizer = tokenizers[model_name]
    model = models[model_name]
    
    query_vec = get_embedding(model, tokenizer, query).reshape(1, -1)  # 정규화 포함
    D, I = index_faiss[model_name].search(query_vec, top_k)  # 거리(D), 인덱스(I) 반환
    
    print(f"\n🔍 {model_name} 검색 결과:")
    for i, idx in enumerate(I[0]):
        print(f"{i+1}. {questions[idx]} (유사도 점수: {D[0][i]:.4f})")

# ✅ 여러 개의 쿼리에 대해 검색 수행
def batch_search(queries, model_name, top_k=3):
    print(f"\n🚀 [{model_name}] 모델 검색 결과 비교\n" + "="*50)
    
    for i, query in enumerate(queries):
        print(f"\n🔍 테스트 쿼리 {i+1}: {query}")
        search_similar(model_name, query, top_k=top_k)
        print("-"*50)

# # ✅ 테스트 질문
# test_queries = [
#     "인터넷 뱅킹 비밀번호를 변경하려면 어떻게 하나요?",
#     "퇴직연금 계좌에서 출금할 수 있나요?",
#     "자동이체 설정을 변경하고 싶은데 어디서 하나요?",
#     "주식 계좌를 개설하려면 어떤 서류가 필요한가요?",
#     "신용카드 한도를 상향 조정하는 방법이 궁금합니다.",
#     "정기예금 금리를 확인하는 방법을 알려주세요.",
#     "모바일 앱에서 대출 신청이 가능한가요?",
#     "IRP 계좌에서 다른 계좌로 이체할 수 있나요?",
#     "해외 결제 시 적용되는 환율은 어떻게 결정되나요?",
#     "스마트폰에서 공인인증서를 등록하는 방법을 알고 싶어요."
# ]

# # ✅ 두 모델에 대해 비교 실행
# batch_search(test_queries, "FinBERT-KR")
# batch_search(test_queries, "Kakao DEBERTa")

# ✅ 테스트: 단일일 질문 검색
test_query = "IRP가 뭐에요?"
print(f"🔍 테스트 질문: {test_query}")
search_similar("FinBERT-KR", test_query)
search_similar("Kakao DEBERTa", test_query)
