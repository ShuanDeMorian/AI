# text_classification

모델 10개 찾고 장단점 적기

1. RNN
* 장점 : 반복적이고 순차적인 데이터(Sequential data)학습에 특화, 현재의 학습과 과거의 학습이 연결, 과거의 정보를 통해 미래를 예측, 시간상의 순서가 있는 task에 적절, 이전의 정보를 현재의 문제 해결에 활용할 수 있음, 이벤트의 연속, 리스트에 관련된 문제를 해결하기 적절 
* 단점 : 시간이 오래 걸림, 장기 의존성 문제 해결 불가(처음 시작한 Weight의 값이 점차 학습이 될수록 상쇄됨)
2. bi-LSTM, LSTM-GRNN
* 장점 : cell gate를 통한 RNN의 장기 의존성 문제 해결(Weight를 계속 기억할지 결정하여 Gradient Vanishing 문제를 해결), 과거의 data를 계속해서 update 하므로, RNN보다 지속적, Cell State는 정보를 추가하거나 삭제하는 기능을 담당, 각각의 메모리 컨트롤 가능, 결과값 컨트롤 가능 
* 단점 : 메모리가 덮어씌워질 가능성(더 자세히?), 연산속도가 느리다
3. CNN-char, CNN-word
* 장점 : 데이터에서 feature 추출하여 패턴 파악에 용이, layer size 감소로 Parameter 갯수 효과적으로 축소, 노이즈 상쇄, 미세한 부분에서 일관적인 특징을 제공
* 단점 :
4. GRU(Gated Recurrent Unit)
* 장점 : LSTM 개선하여 더 단순화됨(Update gate(과거의 상태를 반영하는 Gate)와 Reset gate(현시점 정보와 과거시점 정보의 반영 여부를 결정)를 추가하여 과거의 정보를 어떻게 반영할 것인지 결정), 연산속도가 빠르며 메모리가 LSTM처럼 덮여 씌여질 가능성이 없음
* 단점 : 메모리와 결과값의 컨트롤이 불가능
5. LEAM(Label Embedding Attentive Model)
6. PTE(Predictive Text Embeddings , Tang et al 2015)
7. SWEM
8. Bi-BloSAN
9. SVM(가장 기초적 classification, linear) TextFeatures
* 장점 :
* 단점 : 주로 단어의
빈도수를 feature로 사용하였기 때문에 해당 단어가 그 문장, 혹은 문단에서 어떤 의미로 쓰였는지 알기 힘들었다. 
10. HAN(Hierachical Attention Network) HN-ATT
11. BoW TFIDF, BoW 방법을 이용한 Naive Bayesian Classifier
* 장점 : 
* 단점 : 주로 단어의 빈도수를 feature로 사용하였기 때문에 해당 단어가 그 문장, 혹은 문단에서 어떤 의미로 쓰였는지 알기 힘들었다. 
12. Decision Tree
13. N-gram
* 장점 : 단어의 의미적 문맥적 정보를 파악, 모든 문서 속 단어들의 의미적, 문맥적인 정보를 완벽하게 파악하지 않아도 적절한 성능이 나옴
* 단점 : 단순히 여러 단어를 보는 것만으로는 텍스트의 모든 의미 파악하는데 한계 존재, N 값이 커질수록 계산량이 급격히 늘어나는 단점

## 1. 논문 : Hierarchical Attention Networks for Document Classification
https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf

## 2. 논문 : 윤킴 Convolutional Neural Networks for Sentence Classification
https://arxiv.org/abs/1408.5882

## 3. 논문 : 순환 신경망 기반 대용량 텍스트 데이터 분류 기술
https://pdfs.semanticscholar.org/d0e4/aebbe0dcb6a014ecbd70562878e22bb888b5.pdf

문서 분류 문제는 오랜 기간 동안 자연어 처리 분야에서 연구되어 왔다. 우리는 기존 컨볼루션 신경망을
이용했던 연구에서 더 나아가, 순환 신경망에 기반을 둔 문서 분류를 수행하였다. 순환 신경망에서는 가장
성능이 좋다고 알려져 있는 장기-단기 기억 (Long-Short Term Memory; LSTM) 신경망과 회로형 순환 유
닛(Gated Recurrent Units; GRU)을 활용하였다. 실험 결과, 분류 정확도는 Multinomial Naive Bayesian
Classifier, SVM, LSTM, CNN, GRU의 순서로 나타났다. 따라서 텍스트 문서 분류 문제는 시퀀스를 고려하
는 것 보다는 문서의 feature를 뽑아 분류하는 문제에 가깝다는 것을 추측할 수 있었다. 그리고 GRU가
LSTM보다 문서의 feature 추출에 더 적합하다는 것을 알 수 있었으며 적절한 feature와 시퀀스 정보를 함
께 활용할 때 가장 성능이 잘 나온다는 것을 확인할 수 있었다.

과거의 연구 방향이자, 지금도 뛰어난 성능으로 많이 사용되고 있는 Bag-of-Words[3] (BoW) 방법을 이용한 Naive Bayesian Classifier[4], 서포트 벡터 머신(Support Vector Machine; SVM)[5] 등은 주로 단어의 빈도수를 feature로 사용하였기 때문에 해당 단어가 그 문장, 혹은 문단에서 어떤 의미로 쓰였는지 알기 힘들었다. 이를 극복하려는 시도로 한 번에 여러 단어를 보는 N-gram 방법을 통해 단어의 의미적, 문맥적 정보를 파악하려 했으나, 단순히 여러 단어를 보는 것만으로는 텍스트의 모든 의미를 파악하는데 한계가 있었다. 또한, N 값이 커질수록 계산 량이 급격히 늘어나는 단점도 있었다. 그러나 모든 문서 속 단어들의 의미적, 문맥적인 정보를 완벽하게 파악하지 않아도 적절한 성능이 나왔기 때문에 여전히 BoW 방법은 널리 사용되고 있다. 
이를 극복하려는 시도로 한 번에 여러 단어를 보는 N-gram 방법을 통해 단어의 의미적, 문맥적 정보를 파악하려 했으나, 단순히 여러 단어를 보는 것만으로는 텍스트의 모든 의미를 파악하는데 한계가 있었다. 또한, N 값이 커질수록 계산 량이 급격히 늘어나는 단점도 있었다. 그러나 모든 문서 속 단어들의 의미적, 문맥적인 정보를 완벽하게 파악하지 않아도 적절한 성능이 나왔기 때문에 여전히 BoW 방법은 널리 사용되고 있다.


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



