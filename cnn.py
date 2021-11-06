from torch import nn # Torch로 구현하고 nn을 받기 (nn: Deep learning model에 필요한 모듈이 모아져 있는 패키지)
from torch import Tensor # Torch로 구현하고 Tensor을 받기
from decorators import intent # decorators 모듈에 intent 파일 가져오기
from convolution import Convolution # convolution 모듈에 Convolution 파일 가져오기

@intent
class CNN(nn.Module): # torch.nn의 Module을 상속받는다.

    def __init__(self, label_dict: dict, residual: bool = True): # 함수생성 self로 자기 자신을 받고, label_dict을 입력 받음, 나머지를 True로 받음
        super(CNN, self).__init__() #super(모델명, self) nn.Module의 서브 모든 메소드를 상속된다는 사실을 지칭한다
        self.label_dict = label_dict # label_dict을 입력 받은것을 self.label_dict에 할당한다.
        

        
        self.stem = Convolution(self.vector_size, self.d_model, kernel_size=1, residual=residual)
        self.hidden_layers = nn.Sequential(*[
            Convolution(self.d_model, self.d_model, kernel_size=1, residual=residual)
            for _ in range(self.layers)])

        # self.stem을 지정하고 Convolution(input값, ouput값, 커널의 크기, 나머지는 residual로) 받는다
        # self.hidden_layers로 이름을 지정하고 값을 받는다. 
        # Sequential클래스를 사용하면 명시적 클래스를 구축하지 않고도 PyTorch 신경망을 즉석에서 구축 
        # for 무한루프 self.layers 까지


    def forward(self, x: Tensor) -> Tensor:  # 순방향 전달 구현
        x = x.permute(0, 2, 1)
        x = self.stem(x)           # hidden lineal layer     
        x = self.hidden_layers(x)

        return x.view(x.size(0), -1)

        # forward() 는 모델이 학습데이터를 입력받아서 forward propagation을 진행시키는 함수이다
        ## 좋아요