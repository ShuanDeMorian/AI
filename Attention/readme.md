# Attention
<p align='center'>
<img src="./image/attention_example.jpg">           
<strong>그림을 보고 Captioning 하는 예시</strong>
</p> 
   
Attention이란 위의 그림과 같이 사람이 그림을 볼 때 특정 부분에 Attention을 하여 어떤 그림인지를 아는 것처럼 컴퓨터로 이를 구현해보고자 하는 것이다. 

<p align='center'>
<img width='100%' src="./image/attention_nlp_example.jpg">
</p>
<p align='center'>
<strong>Attention을 활용한 번역 visualization</strong>
</p>   
   
Attention은 번역에도 활용할 수 있다. 위의 그림은 번역을 할 때 attention을 visualization한건데 어떤 단어에 집중 또는 단어 간의 관계로 볼 수도 있다. 선이 진할수록 관계도가 높은 것이다.   
   
* Query   
* Key
* Value
Attention은 쿼리(Query)와 비슷한 값을 가진 키(Key)를 찾아서 그 값(Value)를 얻는 과정이다. 

Key-Value는 개념은 컴퓨터 자료구조에서 볼 수 있다. 그 외 여러 곳에서도 쓰이는데 python의 Dictionary를 예시로 들면
```dic = {'computer': 9, 'dog': 2, 'cat': 3}```
Key와 Value에 해당하는 값을 저장해놓고 Key를 통해 Value 값에 접근할 수 있다. Query를 주고 그 Key값에 따라 Value 값에 접근할 수 있다.
위의 작업을 함수로 나타낸다면 다음과 같이 표현할 수 있다.(이해를 돕기 위한 것으로 실제 파이썬 딕셔너리의 동작과는 다름)
```
def key_value_func(query):
   weights = []

   for key in dic.keys():
      weights += [is_same(key, query)]

   weight_sum = sum(weights)
   for i, w in enumerate(weights):
      weights[i] = weights[i] / weight_sum

   answer = 0

   for weight, value in zip(weights, dic.values()):
      answer += weight * value

   return answer

def is_same(key, query):
   if key == query:
      return 1.
   else:
      return .0
```
코드를 살펴보면, 순차적으로 'dic'변수 내부의 key값들과 query값을 비교하여, key가 같을 경우 'weights'변수에 1.0을 더하고, 다를 경우에는 0을 더한다. 그리고 각 'weights'를 'weights'의 총 합으로 나누어 그 합이 1이 되도록 만들어 준다. 다시 'dic'내부의 value 값들과 'weights'의 값에 대해서 곱하여 더해줍니다. 즉, 'weight'가 1.0인 경우에만 value'값을 'answer'에 더한다.


# Self-Attention

"Attention is all you need' 논문에서 나온 개념으로 기존의 Attention과는 달리 Query가 input이다. 즉 자기 자신을 제일 잘 표현할 수 있는 input(key, value) pair를 찾고 그 결과가 가장 좋은 embedding이 된다. 

# Attention Model의 장점
* <strong><font color="red">해석 가능하다(interpretable)!!!!!!</font></strong>(model이 어디에 attention을 줘서 그러한 결론을 내렸는지 알 수 있다)
* 각각 layer마다 필요로 하는 총 computing cost가 줄어든다.
* 병렬화가 가능한 computation이 늘어난다.(sequential operation을 필요로 하는 부분이 줄어든다)
* 신경망 내에서 long-range dependencies를 잇는 path length가 줄어든다.

참고 : path length란?
번역 문제 같은 sequence transduction problem에서는 input sequence와 output sequence가 길어지면 두 position간의 거리가 먼 경우에 그 의존성을 학습하기 힘들다는 문제가 있다. 이것을 Maximum Path Length를 이용해 표현하였다. 의존성을 학습하기 위해 거쳐야하는 connection이 최대 몇 단계가 필요한가를 나타내는 수치로서, 이 path의 길이가 짧을수록 position 간의 의존성을 학습하기 쉬워진다고 할 수 있다.

# 참고자료
1. [논문반] Self-Attention Generative Adversarial Networks   
http://www.modulabs.co.kr/DeepLAB_Paper/20167
2. Attention   
https://kh-kim.gitbook.io/natural-language-processing-with-pytorch/00-cover-9/03-attention
