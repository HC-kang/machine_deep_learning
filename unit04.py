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

####################

# 딥러닝 구동에 필요한 케라스 호출
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 필요한 라이브러리 불러옴
import numpy as np
import tensorflow as tf

# 실행할 때마다 같은 결과를 출력하기 위해 설정하는 부분
np.random.seed(3)
tf.random.set_seed(3)

# 준비된 수술 환자 데이터를 불러오기
Data_set = np.loadtxt('/Users/heechankang/projects/pythonworkspace/git_study/machine_deep_learning/data/ThoraricSurgery.csv', delimiter = ',')

# 환자의 기록과 수술 결과를 X 와 Y로 구분하여 저장
X = Data_set[:,:17]
Y = Data_set[:,17]

# 딥러닝 구조 결정(모델 설정 및 실행)
model = Sequential()
model.add(Dense(30, input_dim = 17, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

# 딥러닝 실행
model.compile(loss='mean_squared_error', optimizer = 'adam', metrics = ['accuracy'])
model.fit(X, Y, epochs = 100, batch_size = 10)

#####################################
#####
# 피마 인디언 당뇨병 예측
###

# 판다스 불러오기
import pandas as pd

# 파일 불러오기
df = pd.read_csv('/Users/heechankang/projects/pythonworkspace/git_study/machine_deep_learning/data/pima-indians-diabetes.csv', names = ['pregnant', 'plasma', 'pressure', 'thickness', 'insulin', 'BMI', 'pedigree', 'age', 'class'])

# 데이터 형태 보기
df.head()

# 데이터 특성 알아보기
df.info()
# pregnant : 과거 임신횟수
# plasma : 포도당 부하 검사 2시간 후 공복 혈당 농도(mm Hg)
# pressure : 확장기 혈압
# thickness : 삼두근 피부 주름 두께
# insulin : 혈청 인슐린 농도
# BMI
# pedigree : 당뇨병 가족력
# age : 나이 
#  
# null 없음. class(당뇨병 유무) 가 int인게 좀 별로네

df.describe()

df[['pregnant', 'class']]

# 임신 횟수와 당뇨 발병 확률 관련성
print(df[['pregnant', 'class']].groupby(['pregnant'], as_index=False).mean().sort_values(by='pregnant', ascending=True))

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize = (12,12))

sns.heatmap(df.corr(), linewidths=0.1, vmax=0.5, cmap=plt.cm.gist_heat, linecolor = 'white', annot = True)
plt.show()

grid = sns.FacetGrid(df, col = 'class')
grid.map(plt.hist, 'plasma', bins = 10)
plt.show()

# seed값 생성
seed = 3
np.random.seed(seed)
tf.random.set_seed(seed)

model = Sequential()
model.add(Dense(12, input_dim = 8, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import tensorflow as tf

# seed 값 생성
np.random.seed(3)
tf.random.set_seed(3)

# 데이터 로드
dataset = numpy.loadtxt('/Users/heechankang/projects/pythonworkspace/git_study/machine_deep_learning/data/pima-indians-diabetes.csv', delimiter = ',')

X = dataset[:, :8]
Y = dataset[:, 8]

model = Sequential()
model.add(Dense(12, input_dim = 8, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

# 모델 컴파일
model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

# 모델 실행
model.fit(X, Y, epochs = 200, batch_size = 10)
# 결과 출력
print('\n Accuracy : %.4f' % (model.evaluate(X, Y)[1]))

#####
#연습 2

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import tensorflow as tf

seed = 3
np.random.seed(seed)
tf.random.set_seed(seed)

dataset = numpy.loadtxt('/Users/heechankang/projects/pythonworkspace/git_study/machine_deep_learning/data/pima-indians-diabetes.csv', delimiter = ',')
X = dataset[:, :8]
Y = dataset[:, 8]

model = Sequential()
model.add(Dense(12, input_dim = 8, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy',
              optimizer = 'RMSProp',
              metrics = ['accuracy'])

model.fit(X, Y, epochs=300, batch_size=10)

print('\n Accuracy : %.4f' %(model.evaluate(X, Y)[1]))


##################
#####
# IRIS
###
import pandas as pd
df = pd.read_csv('/Users/heechankang/projects/pythonworkspace/git_study/machine_deep_learning/data/iris.csv', names = ['sepal_length', 'sepal_width', 'petal_lenght', 'petal_width', 'species'])
df.head()

import seaborn as sns
import matplotlib.pyplot as plt
sns.pairplot(df, hue= 'species')
plt.show()

df = pd.read_csv('/Users/heechankang/projects/pythonworkspace/git_study/machine_deep_learning/data/iris.csv', names = ['sepal_lengh', 'sepal_width', 'petal_length', 'petal_width', 'species'])

dataset = df.values
X = dataset[:, :4].astype(float)
Y_obj = dataset[:,4]

from sklearn.preprocessing import LabelEncoder

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

from tensorflow.keras.utils import np_utils
Y_encoded = tf.keras.utils.to_categorical(Y)

Y_encoded

# softmax
model = Sequential()
model.add(Dense(16, input_dim = 4, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# seed setting
seed = 3
np.random.seed(seed)
tf.random.set_seed(seed)

df = pd.read_csv('/Users/heechankang/projects/pythonworkspace/git_study/machine_deep_learning/data/iris.csv', names = ['sepal_lenght', 'sepal_width', 'petal_length', 'petal_width', 'species'])

sns.pairplot(df, hue='species')
plt.show()

# data separating
dataset = df.values
X = dataset[:, :4].astype(float)
Y_obj = dataset[:,4]

# chr to int
e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)
Y_encoded = tf.keras.utils.to_categorical(Y)

model = Sequential()
model.add(Dense(16, input_dim = 4, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'SGD',
              metrics = ['accuracy'])

model.fit(X, Y_encoded, epochs=50, batch_size = 1)

print('\n Accuracy : %.4f' %(model.evaluate(X, Y_encoded)[1]))

############
# 2차 연습
from tensorflow.keras.models import Sequential
from tenserflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

seed = 3
np.random.seed(seed)
tf.random.set_seed(seed)

# 데이터 입력
df = pd.read_csv('/Users/heechankang/projects/pythonworkspace/git_study/machine_deep_learning/data/iris.csv', names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])

sns.pairplot(df, hue = 'species')
plt.show()

dataset = df.values
X = dataset[:, :4].astype(float)
Y_obj = dataset[:,4]

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)
Y_encoded = tf.keras.utils.to_categorical(Y)

model = Sequential()
model.add(Dense(16, input_dim=4, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'SGD',
              metrics = ['accuracy'])

model.fit(X, Y_encoded, epochs=50, batch_size = 1)

print('\n Accuracy : %.4f' %(model.evaluate(X, Y_encoded)[1]))


###############

import pandas as pd

df = pd.read_csv('/Users/heechankang/projects/pythonworkspace/git_study/machine_deep_learning/data/sonar.csv', header = None)
df.head()

from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np
import tensorflow as tf

seed = 3
np.random.seed(seed)
tf.random.set_seed(seed)

df = pd.read_csv('/Users/heechankang/projects/pythonworkspace/git_study/machine_deep_learning/data/sonar.csv', header = None)
df.tail()

dataset = df.values
X = dataset[:, :60].astype(float)
Y_obj = dataset[:,60]

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

model = Sequential()
model.add(Dense(24, input_dim = 60, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'mean_squared_error',
              optimizer = 'adam',
              metrics = ['accuracy'])

model.fit(X, Y, epochs=200, batch_size = 5)

print()



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size = 0.3, random_state = seed)


model.fit(X_train, y_train, epochs=130, batch_size=5)
print('\n Test Accuracy : %.4f'%(model.evaluate(X_test, y_test)[1]))

#####################

from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import tensorflow as tf

seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)
df = pd.read_csv('/Users/heechankang/projects/pythonworkspace/git_study/machine_deep_learning/data/sonar.csv', header = None)
dataset = df.values
X = dataset[:, :60].astype(float)
Y_obj = dataset[:,60]


e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size = 0.3, random_state = seed)

model = Sequential()
model.add(Dense(24, input_dim = 60, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'mean_squared_error',
              optimizer = 'adam',
              metrics = ['accuracy'])

model.fit(X_train, y_train, epochs=130, batch_size = 5)

import os
os.chdir('/Users/heechankang/projects/pythonworkspace/git_study/machine_deep_learning')

from keras.models import load_model
model.save('my_model.h5')

model = load_model('my_model.h5')

###################
# 2차 연습
from keras.models import Sequential, load_model
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np
import tensorflow as tf

seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)

df = pd.read_csv('/Users/heechankang/projects/pythonworkspace/git_study/machine_deep_learning/data/sonar.csv', header = None)

dataset = df.values
X = dataset[:, :60].astype(float)
Y_obj = dataset[:, 60]

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size = 0.3, random_state = seed
)

model = Sequential()
model.add(Dense(24, input_dim = 60, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'mean_squared_error',
              optimizer = 'adam',
              metrics = ['accuracy'])

model.fit(X_train, Y_train, epochs=130, batch_size=5)
model.save('my_model.h5')

del model

model = load_model('my_model.h5')

print('Test Accuracy : %.4f'%(model.evaluate(X_test, Y_test)[1]))
print('\n Test Accuracy : %.4f'%(model.evaluate(X_test, Y_test)[1]))

#################
#####
# k겹 교차 검증
###

from sklearn.model_selection import StratifiedKFold

n_fold = 10
skf = StratifiedKFold(n_splits = n_fold, shuffle=True, random_state = seed)

for train, test in skf.split(X, Y):
    model = Sequential()
    model.add(Dense(24, input_dim = 60, activation = 'relu'))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss = 'mean_squared_error',
                  optimizer = 'adam',
                  metrics = ['accuracy'])
    model.fit(X[train], Y[train], epochs = 100, batch_size = 5)


####################
accuracy = []

for train, test in skf.split(X, Y):
    model = Sequential()
    model.add(Dense(24, input_dim = 60, activation = 'relu'))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss = 'mean_squared_error',
                  optimizer = 'adam',
                  metrics = ['accuracy'])
    model.fit(X[train], Y[train], epochs = 100, batch_size = 5)
    k_accuracy = '%.4f'%(model.evaluate(X[test], Y[test])[1])
    accuracy.append(k_accuracy)

print('\n %.4f fold accuracy : ' % n_fold, accuracy)


#########
# 2차 연습

from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

import numpy as np
import pandas as pd
import tensorflow as tf

# seed setting
seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)

df = pd.read_csv('/Users/heechankang/projects/pythonworkspace/git_study/machine_deep_learning/data/sonar.csv', header = None)

dataset = df.values
X = dataset[:, :60].astype(float)
Y_obj = dataset[:, 60]

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

# 10개로 나누기
n_fold = 10
skf = StratifiedKFold(n_splits = n_fold, shuffle = True, random_state = seed)

accuracy = []

for train, test in skf.split(X, Y):
    model = Sequential()
    model.add(Dense(24, input_dim = 60, activation = 'relu'))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss = 'mean_squared_error',
                  optimizer = 'adam',
                  metrics = ['accuracy'])
    model.fit(X[train], Y[train], epochs = 100, batch_size = 5)
    k_accuracy = '%.4f' %(model.evaluate(X[test], Y[test])[1])
    accuracy.append(k_accuracy)

print('\n %.f fold accuracy : '%n_fold, accuracy)