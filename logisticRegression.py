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