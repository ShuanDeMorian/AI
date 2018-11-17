# text_classification

모델 10개 찾고 장단점 적기

1. RNN
RNN은 신경망 속 셀의 현재 출력 결과가 이전의 계산 결과에 영향을 받는 인공신경망 모델이다. 다시 말해, 이전 계산 결과에 대한 메모리 정보를 가지고 있어 순차적인 데이터를 학습하는데 장점을 가지고 있다. 기본적인 RNN은 일반적으로 학습이 어려워 다양한 변형이 발생했는데, 그 중 가장 성공적인 모델은 장기-단기 기억 신경망(Long-Short Term Memory: LSTM)과 최근 각광받고 있는 회로형 순환 유닛(Gated Recurrent Units: GRU)이 있다.  
   * 장점 : 반복적이고 순차적인 데이터(Sequential data)학습에 특화, 현재의 학습과 과거의 학습이 연결, 과거의 정보를 통해 미래를 예측, 시간상의 순서가 있는 task에 적절, 이전의 정보를 현재의 문제 해결에 활용할 수 있음, 이벤트의 연속, 리스트에 관련된 문제를 해결하기 적절 
   * 단점 : 학습이 어려움, 시간이 오래 걸림, 장기 의존성 문제 해결 불가(처음 시작한 Weight의 값이 점차 학습이 될수록 상쇄됨) 

2. bi-LSTM, LSTM-GRNN
LSTM은 긴 순차적인 정보를 회로 메커니즘(gating mechanism)을 통해 저장하고 출력할 수 있따. 이 회로 메커니즘은 RNN의 학습을 방해하는 가장 큰 원인인 vanishing gradient 문제를 완화시켜 성능을 크게 향상시켰다. 
    * 장점 : cell gate를 통한 RNN의 장기 의존성 문제 해결(Weight를 계속 기억할지 결정하여 Gradient Vanishing 문제를 해결), 과거의 data를 계속해서 update 하므로, RNN보다 지속적, Cell State는 정보를 추가하거나 삭제하는 기능을 담당, 각각의 메모리 컨트롤 가능, 결과값 컨트롤 가능 
    * 단점 : 메모리가 덮어씌워질 가능성(더 자세히?), 연산속도가 느리다

3. CNN-char, CNN-word (Convolutional Neural Networks)
CNN은 사람의 시신경망에서 아이디어를 얻어 고안한 모델로, 다양한 패턴 인식 문제에 사용되고 있다. CNN은 컨볼루션 층, subsampling층(또는 max-pooling층)이라는 두 층을 번갈아가며 수행하다가 마지막에 있는 fully-connected층을 이용하여 분류를 수행한다. 컨볼루션 층은 입력에 대해 2차원 필터링을 수행하고, subsampling층은 매핑된 2차원 이미지에서 최댓값을 추출한다. 이러한 계층구조 속에서 역전파(backpropagation)을 이용, 오차를 최소화하는 방향으로 학습해나간다. 주로 비전 분야에서 얼굴 인식, 필기체 인식 등에 많이 사용되어 왔으나 최근에는 자연어 처리분야에서도 널리 활용되고 있다.
    * 장점 : 데이터에서 feature 추출하여 패턴 파악에 용이, layer size 감소로 Parameter 갯수 효과적으로 축소, 노이즈 상쇄, 미세한 부분에서 일관적인 특징을 제공
    * 단점 :

4. GRU(Gated Recurrent Unit)
2014년에 LSTM과 동일한 회로 메커니즘을 사용하지만 파라미터 수를 줄인 GRU가 제안되었다. GRU는 리셋 게이트와 업데이트 게이트로 구성되어 있으며, 두 게이트의 상호작용을 통해 학습한다. LSTM보다 적은 파라미터를 사용하기 때문에 이론적으로는 학습 속도가 조금 더 빠르고 완전한 학습에 필요한 데이터가 LSTM보다 적게 필요하다. 그러나 실제 성능으로는 특정 작업에서는 더 뛰어나기도 하고 뒤쳐지기도 한다.
    * 장점 : LSTM 개선하여 더 단순화됨(Update gate(과거의 상태를 반영하는 Gate)와 Reset gate(현시점 정보와 과거시점 정보의 반영 여부를 결정)를 추가하여 과거의 정보를 어떻게 반영할 것인지 결정), 연산속도가 빠르며 메모리가 LSTM처럼 덮여 씌여질 가능성이 없음, LSTM보다 학습속도가 조금 더 빠르고 완전한 학습에 필요한 데이터가 적게 필요하다.
    * 단점 : 메모리와 결과값의 컨트롤이 불가능

5. LEAM(Label Embedding Attentive Model)

6. PTE(Predictive Text Embeddings , Tang et al 2015)

7. SWEM

8. Bi-BloSAN

9. SVM(Support Vector Machine)(가장 기초적 classification, linear) TextFeatures
SVM은 각 클래스간 거리를 최대로 하는 경계선 또는 경계면(hyperplane)을 찾는다. 그리하여 새로운 데이터가 들어 왔을 때 일반화 오류를 최소화하는 모델이다. 이 때 각 클래스에서 데이터까지의 최소 거리를 마진(Margin), 그리고 경계선으로부터의 최소 거리인 데이터벡터를 서포트 벡터(Support Vector)라고 한다. 
    * 장점 : 해석 용이, 적은 Data에서도 적절한 결과가 나온다. 인공 신경망에도 크게 뒤지지 않는 성능을 낸다.
    * 단점 : 주로 단어의 빈도수를 feature로 사용하였기 때문에 해당 단어가 그 문장, 혹은 문단에서 어떤 의미로 쓰였는지 알기 힘들었다. 

10. HAN(Hierachical Attention Network) HN-ATT

11. BoW TFIDF, BoW 방법을 이용한 Naive Bayesian Classifier
    * 장점 : 
    * 단점 : 주로 단어의 빈도수를 feature로 사용하였기 때문에 해당 단어가 그 문장, 혹은 문단에서 어떤 의미로 쓰였는지 알기 힘들었다. 

12. Decision Tree

13. N-gram
    * 장점 : 단어의 의미적 문맥적 정보를 파악, 모든 문서 속 단어들의 의미적, 문맥적인 정보를 완벽하게 파악하지 않아도 적절한 성능이 나옴
    * 단점 : 단순히 여러 단어를 보는 것만으로는 텍스트의 모든 의미 파악하는데 한계 존재, N 값이 커질수록 계산량이 급격히 늘어나는 단점

14. Naive Bayesian Classifier
각 사건들이 서로 독립이라는 가정을 한 후, Bayes's theorem을 이용하여 확률을 계산, 분류하는 모델이다. 따라서 두 확률의 결합 확률(Joint Probability)을 두 확률의 곱으로 표현해버리지만, 상당히 강력한 성능을 보이고 있어서 널리 사용된다. Naive Bayesian Classifier는 feature들간의 조건부독립 성질을 이용하는 반면, Multinomial Naive Bayesian Classifier는 feature들이 다항 분표(multinomial distribution)를 따른다는 정보를 활용한다.
    * 장점 : 간단
    * 단점 : 독립이 아닐 수 있는 사건들을 독립으로 가정하므로 한계 존재

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

3. 실 험
 데이터 전처리와 Naive Bayesian Classifier, SVM은 Python2.7, CNN과 RNN은 Torch7의 nn, rnn패키지[16]를 이용하여 구현하였다.
 
 3.1 데이터
 대분류로는 9개, 소분류로는 68개 분야에 분포되어 있는 인터넷에서 수집한 623,303개의 뉴스 데이터를 준비하였으며 학습 데이터, 검증 데이터, 테스트 데이터는 각각 70%, 15%, 15%의 비율로 나누었다. 데이터의 분야별 분포는 표 1과 같다. 
 
3.2 설 계
 비교모델로 TF-IDF (Term Frequency-Inverse Document Frequency)[17]를 사용한 Multinomial Naive Bayesian classifier와 SVM을 사용하였다. CNN의 경우, 먼저 각 문서들을 형태소 분석기로 나눈 후 빈도수 기준 상위 n개의 단어로 lookup 테이블을 만든다. 그 다음 컨볼루션 커널을 슬라이드 하여 적절한 파라미터를 학습한다. 이때 커널의 가로와 세로 크기는 각각 단어 임베딩 크기, N-gram과 같이 동시에 학습하는 단어 크기와 같다. 활성 함수 (activation function)으로는 ReLU[18]를 사용했으며, logSoftMax를 이용하여 각 문서들이 특정 주제에 속할 확률을 출력하였다. 그 중 가장 높은 값을 가진 카테고리를 정답으로 예측하였다. RNN의 경우, 마찬가지로 lookup 테이블을 생성하고, 테이블을 단어 벡터 단위로 쪼개어 순환 신경망의 입력으로 넣는다. lookup 테이블을 업데이트하면서 계속 해서 학습, 최종 은닉 층을 출력한다. CNN과 동일하게 ReLU와 logSoftMax를 이용하여 예측을 수행하였다.

3.3 실험 결과
각 모델 정확도는 표 2, 표 3과 같다.
Model Accuracy (Top-1,3,5)
MNB 0.641 0.911 0.958
SVM 0.795 0.960 0.991
CNN 0.856 0.986 0.997
LSTM 0.811 0.965 0.994
GRU 0.886 0.992 0.999
표 2 모델 별 대주제 실험 정확도

Model Accuracy (Top-1,3,5)
MNB 0.399 0.679 0.794
SVM 0.614 0.851 0.906
CNN 0.700 0.920 0.962
LSTM 0.670 0.895 0.942
GRU 0.725 0.937 0.971
표 3 모델 별 소주제 실험 정확도

 실험 결과, GRU가 가장 뛰어난 성능을 보였다. LSTM은 SVM과 CNN 사이의 성능을 보였다.
 
4. 결 론

 본 논문은 인터넷에서 수집한 텍스트 문서를 여러 알고리즘을 이용하여 정해진 카테고리에 맞게 분류하는 내용을 담고 있다. 총 623,303개의 문서를 대분류 9개, 소분류 68개에 분류하였을 때, 분류 정확도는 Multinomial Naive Bayesian Classifier, SVM, LSTM, CNN, GRU의 순서로 나타났다. 이 결과는 다음과 같이 해석할 수 있다:
 
(1) LSTM보다 CNN의 성능이 더 뛰어난 것으로 보아 문서 분류 문제는 전체 글의 시퀀스를 학습하는 것 보다는 글의 feature를 통해 학습하는 것이 더 올바른 문제 접근법이라고 생각할 수 있다. 따라서, (2) LSTM과 GRU의 성능을 비교했을 때, LSTM에 비해 GRU가 feature를 더 잘 추출했다고 볼 수 있다. 마지막으로, (3) GRU가 CNN보다 성능이 더 좋은 것으로 보아, 문서 분류 문제는 feature와 시퀀스 두 가지를 모두 적절히 고려할 때 성능이 가장 잘 나온다는 것을 확인할 수 있었다. 추후 연구에서는 LSTM과 GRU에서 추출된 feature와 embedding 결과를 비교, 분석하여 어떤 차이가 GRU의 성능을 더 뛰어나게 만들었는지 확인해보고자 한다.

## 4. 논문 : Recent Trends in Deep Learning Based Natural Language Processing
https://arxiv.org/pdf/1708.02709.pdf
딥러닝 기반 자연어처리 연구트렌드 정리한 논문
한국어 번역 : ratsgo's blog : https://ratsgo.github.io/natural%20language%20processing/2017/08/16/deepNLP/



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



