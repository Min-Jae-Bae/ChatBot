import torch # torch로 구현
from torch import Tensor # torch를 사용하고 Tensor 가져오기
from torch import nn, autograd # nn, autograd 가져오기

from decorators import intent



class LSTM(nn.Module): # 클래스 생성하고 모듈 상속

    def __init__(self, label_dict: dict, bidirectional: bool = True):
           
        # label_dict: 라벨 딕셔너리 bidirectional 여부
        

        super().__init__() # (사용할 함수 CNN을 정의하는 장소)

        self.label_dict = label_dict
        self.direction = 2 if bidirectional else 1
        self.lstm = nn.LSTM(input_size=self.vector_size, # Input의 사이즈에 해당 하는 수를 입력
                            hidden_size=self.d_model, # 은닉층의 사이즈에 해당 하는 수를 입력
                            num_layers=self.layers, # RNN의 은닉층 레이어 갯수. 기본 값은 1
                            batch_first=True, # Output 값의 사이즈는 (batch, seq, feature)
                            bidirectional=bidirectional) # 양방향 RNN
        # 각각의 지정하는 인스턴스 생성                    

    def init_hidden(self, batch_size: int) -> autograd.Variable: # 자동 미분
        param1 = torch.randn(self.layers * self.direction, batch_size, self.d_model).to(self.device)
        param2 = torch.randn(self.layers * self.direction, batch_size, self.d_model).to(self.device)
        return autograd.Variable(param1), autograd.Variable(param2)
        
        # torch.randn()은 0 ~ 1 난수 생성
        # self.layers * self.direction – 출력 텐서.
        # size: 출력 텐서의 크기를 정의하는 정수 시퀀스
        # self.d_model).to(self.device) 장치
        # ouput param1, param2 출력

    def forward(self, x: Tensor) -> Tensor: # 다차원 행렬
        b, l, v = x.size() # tensor 각각의 변수 생성
        out, (h_s, c_s) = self.lstm(x, self.init_hidden(b)) # 값 대입
        return h_s[0]

        # output 생성
