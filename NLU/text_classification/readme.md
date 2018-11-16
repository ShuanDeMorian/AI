# text_classification

모델 10개 찾고 장단점 적기

1. RNN
2. bi-LSTM, LSTM-GRNN
3. CNN-char, CNN-word
4. GRU
5. LEAM(Label Embedding Attentive Model)
6. PTE(Predictive Text Embeddings , Tang et al 2015)
7. SWEM
8. Bi-BloSAN
9. SVM(가장 기초적 classification, linear) TextFeatures
10. HAN(Hierachical Attention Network) HN-ATT
11. BoW TFIDF
12. Decision Tree

## 1. 논문 : Hierarchical Attention Networks for Document Classification
https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf

## 2. 논문 : 윤킴 Convolutional Neural Networks for Sentence Classification
https://arxiv.org/abs/1408.5882

## 선형성 모델
선형 분류 알고리즘은 클래스를 직선(또는 고차원의 아날로그)으로 구분할 수 있다고 가정, linear regression, SVM
데이터가 직선을 따르는 경향이 있다고 가정함. 이러한 가정은 일부 문제에 대해서는 그다지 나쁘지 않지만 어떤 면에서는 정확도가 떨어질 수 있음(선형으로는 절대 분류를 못 하는 데이터가 있다), 너무 단순하게 표현함
장점으로는 가장 먼저 시도해보기 좋을만큼 간단하고 학습 시간이 빠른 경향이 있다.
선형성 모델은 다중 클래스 분류에 매우 취약, 2클래스 분류에 적절, text classification에는 적합하지 못 함

## 모델이 얼마나 좋으냐 평가할 때 사용할 것
* 매개 변수 수
알고리즘의 동작에 영향을 주는 숫자, 알고리즘의 학습 시간 및 정확도에 크게 영향을 줌, 일반적으로 매개 변수가 많은 알고리즘의 경우 적절한 조합을 찾기 위해 대부분 시행착오와 오류를 겪어야 함, 
매개변수가 많을 때의 장점으로는 알고리즘의 유연성이 향상된다는 것, 정확도가 높아질 수 있다. 적절한 수의 매개변수를 선택하는 것이 중요

임베딩 추가
1. fast text
2. 엘리스 오 subword


