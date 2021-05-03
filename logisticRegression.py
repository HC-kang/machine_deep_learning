# 라이브러리 불러오기
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
import numpy as np

# 데이터 생성
x = np.array([-2, -1.5, -1, 1.25, 1.62, 2])
y = np.array([0,0,0,1,1,1])

# 로지스틱 회귀 모델 만들기
# sigmoid(wx+b) 의 형태를 갖는 로지스틱 회귀 구현
model = Sequential()

# 입력 1개를 받아 출력 1개를 리턴하는 선형회귀 레이어 생성
model.add(Dense(input_dim=1, units=1))

# 선형회귀의 출력값을 시그모이드에 연결
model.add(Activation('sigmoid'))

# 크로스 엔트로피를 비용함수로 설정해 경사하강법으로 학습
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['binary_accuracy'])


# keras.layers.Dense() 파라미터
# - input_dim : 만들 모델의 입력 값의 개수(x값 하나 입력)
# - units : 만들 모델의 출력 값의 개수(y값 하나 출력)
# - activation : linear 로 설정해 선형성을 유지시킴.('sigmoid'도 가능)

# keras.Model.compile() 파라미터
# - loss : 최적화 과정에서 최소화될 손실함수를 설정하는 것으로, MSE(평균제곱오차)와 binary_crossentropy가 자주 사용.
# - optimizer : 훈련 과정을 설정하는 것으로, Adam, SGD 등이 있음.
# - metrics : 훈련을 모니터링하기 위해 사용. 
#             metrics = ['binary_accuracy']는 출력이 0.5 이상일 경우 출력을 1로 판단하고, 이하일 경우 0으로 판단


# 모델 학습
# 300번 반복학습을 통해 최적의 w와 b 찾기
model.fit(x, y, epochs=300, verbose=0)

# 학습 데이터에 따른 실제 모델의 출력값 확인
model.predict([-2, -1.5, -1, 1.25, 1.62, 2])

model.summary()

model.layers[0].get_weights()



#####
# And 연산 구현
###

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
import numpy as np

x = np.array([(0,0), (0,1), (1,0), (1,1)])
y = np.array([0,0,0,1])

# 로지스틱 회귀 모델 만들기
# sigmoid(w1x1 + w2x2 + b)의 형태를 띄는 간단한 로지스틱 회귀 구현
model = Sequential()

# 입력 2개를 받아 출력 1개를 리턴하는 선형회귀 레이어 생성
model.add(Dense(input_dim = 2, units=1))

# 선형회귀의 출력값을 시그모이드에 연결
model.add(Activation('sigmoid'))

# 크로스 엔트로피를 비용함수로 설정해 경사하강법으로 학습
model.compile(loss='binary_crossentropy', optimizer = 'sgd', metrics = ['binary_accuracy'])

model.fit(x, y, epochs=5000, verbose=0)
model.predict(x)

model.summary()

# 학습을 통해 구한 최적의 w1, w2, b
model.layers[0].get_weights()