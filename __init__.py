
from intent import CNN # intent 폴더 안에 있는 CNN 파일을 가져옴
from intent import LSTM # intent 폴더 안에 있는 LSTM 파일을 가져옴

__ALL__ = [CNN, LSTM]

# 특정 디렉터리의 모듈을 *를 이용하여 import할 때에는 다음과 같이 해당 디렉터리의 __init__.py 
# 파일에 __all__이라는 변수를 설정하고 import할 수 있는 모듈을 정의해 주어야 합니다.
#__all__로 정의하지 않으면 인식되지 않습니다.