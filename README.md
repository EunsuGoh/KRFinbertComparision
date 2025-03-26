# KRFinbertComparision

한국어 금융 데이터셋으로 사전훈련된 두가지 BERT모델(1. KR-Finvbert, 2. Kakaobank DeBERTa)의 임베딩성능을 비교합니다.

finbert_test.py : 단순 작동 테스트
finbert_vs_deverta.py : 유클리디안 유사도(L2)에 의거한 유사도 및 답변 비교
finbert_vs_deverta_cosine.py : 코사인 유사도에 의거한 유사도 및 답변 비교
