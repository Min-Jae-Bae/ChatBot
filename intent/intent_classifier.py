from abc import ABCMeta, abstractmethod
from time import time
from typing import List

# abc = 추상 기본 클래스 / ABCMeta = 추상 기본 클래스(ABC)를 정의하기 위한 메타 클래스 / abstractmethod = 속성 및 설명자에 대한 추상 메서드 선언하는 데 사용
# 시간관련 정보 제공
# type 모듈 사용

import torch # torch로 구현
from torch import nn # Torch로 구현하고 nn을 받기 (nn: Deep learning model에 필요한 모듈이 모아져 있는 패키지)
from torch.nn import Parameter # 모듈 매개변수로 간주되는 일종의 Tensor



from decorators import intent # 파일 가져오기 및 사용
from torch_processor import TorchProcessor  # 파일 가져오기 및 사용


@intent

# class 생성 () TorchProcessor 가져오고 metaclass 키워드를 전달하고 ABCMeta직접 사용하여 추상 기본 클래스를 정의

class IntentClassifier(TorchProcessor, metaclass=ABCMeta):

    def __init__(self, model: nn.Module, parameters: Parameter or List[Parameter]):
        model = self.__add_classifier(model)
        super().__init__(model, parameters)
    # 내장 함수 만들기
    # model 속성을 만든다
    # 상위 클래스의 메소드 호출 (model 과 parameters)

    def fit(self, dataset: tuple, test: bool = True):
        """
        Pytorch 모델을 학습/테스트하고 모델의 출력값을 다양한 방법으로 시각화합니다.
        최종적으로 학습된 모델을 저장합니다.
        IntentClassifier는 OOD 데이터셋이 존재하면 추가적으로 Fallback Detector도 학습시킵니다.

        :param dataset: 학습할 데이터셋
        :param test: 테스트 여부
        """

        super().fit(dataset, test) # 상위 클래스의 dateset과 test를 가져온다.

        # ood 데이터가 있는 경우에 fallback detector 자동 학습/테스트
        if self.ood_train and self.ood_test:
            eta = time()
            self._ood_train_epoch()
            predicts, labels = self._ood_test_epoch()

            self.metrics.evaluate(labels, predicts, mode='ood')
            report, _ = self.metrics.report(['in_dist', 'out_dist'], mode='ood')
            report = report.drop(columns=['macro avg'])

            self.visualizer.draw_report(report, mode='ood')
            self._print(name=self.fallback_detector.__class__.__name__,
                        msg='Training, ETA : {eta} sec'.format(eta=round(time() - eta, 4)))

    @abstractmethod # 추상 메서드를 나타내는 데코레이터
    def _ood_train_epoch(self):
        raise NotImplementedError
    # 해당 메소드 생성시 에러 발생하게 함

    @abstractmethod # 추상 메서드를 나타내는 데코레이터
    def _ood_test_epoch(self):
        raise NotImplementedError
    # 해당 메소드 생성시 에러 발생하게 함

    @abstractmethod # 추상 메서드를 나타내는 데코레이터
    def _calibrate_msg(self, *args):
        raise NotImplementedError
    # 해당 메소드 생성시 에러 발생하게 함
    
    # 분류기 생성하기
    def __add_classifier(self, model): 
        sample = torch.randn(1, self.max_len, self.vector_size) # 난수를 받는다 (최대길이, 벡터 크기)
        sample = sample.to(self.device) # 장치
        output_size = model.to(self.device)(sample)   # 장치에서 받는 결과 값

        features = nn.Linear(output_size.shape[1], self.d_loss) # 특징 = 선형 (입력 샘플의 크기, 출력 샘플의 크기)
        classifier = nn.Linear(self.d_loss, len(model.label_dict)) # 분류 = 선형 (입력 샘플의 크기, 출력 샘플의 크기)
        setattr(model, 'features', features.to(self.device)) # setattr(객체, 이름, 값) -객체의 속상값 설정
        setattr(model, 'classifier', classifier.to(self.device)) # setattr(객체, 이름, 값)
        return model #모델 출력합니다
