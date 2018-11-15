# 한국어의 특수성에 도움될만한 방법들

## 논문 : Subword-level Word Vector Representations for Korean
http://aclweb.org/anthology/P18-1226

unknown 단어를 처리하기 위해 character level 단위가 제안되었다. 하지만 한국어의 특성상 character 단위 또한 충분히 잘게 쪼개지지 못 했다.
그래서 jamo level을 제안한다. jamo level이란 자음, 모음 단위로 끊는 것으로 다음과 같다. (초성, 중성, 종성으로 끊는다.)

character : '해', '달'
jamo : 'ㅎ,ㅐ,e', 'ㄷ,ㅏ,ㄹ'

해의 경우 마지막 종성이 없어서 이걸 표시해주기 위해 e를 넣어준다. 
