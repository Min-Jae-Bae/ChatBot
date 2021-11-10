from torch import nn # Torch로 구현하고 nn을 받기 (nn: Deep learning model에 필요한 모듈이 모아져 있는 패키지)
from torch import Tensor # Torch로 구현하고 Tensor을 받기
from decorators import intent # decorators 모듈에 intent 파일 가져오기
from convolution import Convolution # convolution 모듈에 Convolution 파일 가져오기

@intent
class CNN(nn.Module): # torch.nn의 Module을 상속받는다.

    def __init__(self, label_dict: dict, residual: bool = True): # 함수생성 self로 자기 자신을 받고, label_dict을 입력 받음, 나머지를 True로 받음
        super(CNN, self).__init__() # (사용할 함수 CNN을 정의하는 장소)
        self.label_dict = label_dict # label_dict을 입력 받은것을 self.label_dict에 할당한다.
        

        
        self.stem = Convolution(self.vector_size, self.d_model, kernel_size=1, residual=residual)
        self.hidden_layers = nn.Sequential(*[
            Convolution(self.d_model, self.d_model, kernel_size=1, residual=residual)
            for _ in range(self.layers)])

        # self.stem을 지정하고 Convolution(input값, ouput값, 커널의 크기, 나머지는 residual로) 받는다
        # self.hidden_layers로 이름을 지정하고 값을 받는다. 
        # Sequential클래스를 사용하면 명시적 클래스를 구축하지 않고도 PyTorch 신경망을 즉석에서 구축 
        # for 무한루프 self.layers 까지


    def forward(self, x: Tensor) -> Tensor:
        # (함수들을 사용하여 CNN의 forward를 정의하는 장소)
        x = x.permute(0, 2, 1)         # 차원 0->0, 1->2, 2->1로 변경 
        x = self.stem(x) # stem 값
        x = self.hidden_layers(x) # hidden_layers 값

        return x.view(x.size(0), -1) # ouput 지정 # 2 x 2 행렬을 1차원 벡터로 변형

# foward() 함수는 모델이 학습데이터를 입력받아서 forward 연산을 진행시키는 함수입니다.
# 입력 x로부터 예측된 값을 얻는 forward 연산
