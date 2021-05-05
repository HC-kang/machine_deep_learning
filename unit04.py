# 딥러닝을 구동하기 위한 케라스 모듈을 불러오기
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 필요한 라이브러리 불러오기
import numpy as np
import tensorflow as tf

# 실행할 때 마다 같은 결과를 출력하기 위해 시드 고정
np.random.seed(3)
tf.random.set_seed(3)

# 데이터 불러오기 / 적용
my_data = '/Users/heechankang/projects/pythonworkspace/git_study/machine_deep_learning/data/ThoraricSurgery.csv'
Data_set = np.loadtxt(my_data, delimiter=',')

# 환자의 기록과 수술 결과를 각각 X, Y에 저장
X = Data_set[:, :17]
Y = Data_set[:, 17]

# 딥러닝 구조 결정
model = Sequential()
model.add(Dense(30, input_dim = 17, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

# 딥러닝을 실행
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=100, batch_size=10)

