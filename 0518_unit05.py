from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Input, Flatten, Reshape
from tensorflow.python.keras.backend import conv2d
from tensorflow.python.keras.layers.convolutional import UpSampling2D
import numpy as np
import matplotlib.pyplot as plt

# MNIST 데이터셋 호출
(X_train, _ ), (X_test, _) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') /255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') /255

# 생성자 모델 만들기
autoencoder = Sequential()

# 인코딩 부분
autoencoder.add(Conv2D(16, kernel_size = 3, padding = 'same', input_shape = (28, 28, 1), activation = 'relu'))
# 첫 은닉층에 input_shape = (28, 28, 1)이 있어야함. 28*28사이즈, channel 은 1(흑백)
autoencoder.add(MaxPooling2D(pool_size = 2, padding = 'same'))
autoencoder.add(Conv2D(8, kernel_size = 3, activation = 'relu', padding = 'same'))
autoencoder.add(MaxPooling2D(pool_size = 2, padding = 'same'))
autoencoder.add(Conv2D(8, kernel_size=3, strides=2, padding='same', activation='relu'))

# 디코딩 부분
autoencoder.add(Conv2D(9, kernel_size = 3, padding='same', activation='relu'))
autoencoder.add(UpSampling2D())
autoencoder.add(Conv2D(8, kernel_size=3, padding='same', activation='relu'))
autoencoder.add(UpSampling2D())
autoencoder.add(Conv2D(16, kernel_size=3, activation='relu'))
# 패딩이 없음 -> 이미지 크기를 줄여줌
autoencoder.add(UpSampling2D())
autoencoder.add(Conv2D(1, kernel_size=3, padding='same', activation='sigmoid'))

# 전체 구조 확인
autoencoder.summary()


# 컴파일 및 학습
autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy')
autoencoder.fit(X_train, X_train, epochs = 50, batch_size = 128, validation_data = (X_test, X_test))

# 학습결과 출력부분
random_test = np.random.randint(X_test.shape[0], size = 5)

# 테스트할 이미지를 랜덤으로 호출
ae_imgs = autoencoder.predict(X_test) # 앞서 만든 모델에 넣기

plt.figure(figsize = (7, 2))

for i, image_idx in enumerate(random_test):
    # 랜덤으로 뽑은 이미지 차례로 나열
    ax = plt.subplot(2, 7, i+1)
    # 테스트할 이미지를 먼저 그대로 보여줌
    plt.imshow(X_test[image_idx].reshape(28, 28))
    ax.axis('off')
    ax = plt.subplot(2, 7, 7+i+1)
    plt.imshow(ae_imgs[image_idx].reshape(28, 28))  #오토인코딩 결과를 다음열에 출력합니다.
    ax.axis('off')
plt.show()


#####
# 전이학습,, 소규모 데이터셋 활용법
###
trian_datagen = ImageDataGenerator(rescale = 1. / 255,
                                   horizontal_flip = True, 
                                   width_shift_range = 0.1,
                                   height_shift_range = 0.1
                                   rotation_range = 5,
                                   shear_range = 0.7,
                                   zoom_range = 1.2,
                                   vertical_flip = True,
                                   fill_mode = 'nearest')


# 테스트셋은 정규화만 진행
test_datagen = ImageDataGenerator(rescale = 1. / 255)

train_generator = train_datagen.flow_from_directory(
    './train',                # 이미지가 위치한 폴더 위치
    target_size = (150, 150), # 이미지 크기
    batch_size = 5,
    class_model = 'binary')   # 치매 / 정상 2진 분류이므로 바이너리 모드로 실행

# 같은 과정으로 테스트셋도 생성
test_generator = test_datagen.flow_from_directory(
    './test',
    target_size = (150, 150),
    batch_size = 5,
    class_mode = 'binary')

# 그냥 fit 쓰는걸로 바뀌었다는듯?
# model.fit_generatro( 
model.fit( 
    train_generator,        # 앞서 만들어진 train_generator를 학습모델로 활용
    steps_per_epoch = 100,  # 이미지 생성기에서 몇 개의 샘플을 뽑을 지 결정
    epochs = 20, 
    validation_data = test_generator, validation_steps = 4)
)   # 앞서 만들어진 test_generator를 테스트셋으로 사용



###########
#연습
import os 
os.chdir('/Users/heechankang/Downloads/drive-download-20210518T055205Z-001')

import itertools
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers, initializers, regularizers, metrics

np.random.seed(3)
tf.random.set_seed(3)

train_datagen = ImageDataGenerator(rescale = 1. /255,              
                                   horizontal_flip = True,      # 수평 대칭 이미지를 50% 확률로 추가
                                   width_shift_range = 0.1,     # 전체 크기의 10% 범위에서 좌우로 이동
                                   height_shift_range = 0.1,    # 마찬가지로 위아래 이동
                                                                # rotation_range = 5,
                                                                # shear_range = 0.7,
                                                                # zoom_range = [0.9, 2.2],
                                                                # vertical_flip = True 
                                   fill_mode = 'nearest')       

train_generator = train_datagen.flow_from_directory(
    './train',      # 학습셋이 있는 위치
    target_size = (150, 150),
    batch_size = 5,
    class_mode = 'binary')


# 테스트셋은 이미지 부풀리기를 진행 안함
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    './test',
    target_size = (150, 150),
    batch_size = 5,
    class_mode = 'binary')


# 앞서 배운 CNN 모델을 만들어 적용하기
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape = (150, 150, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('sigmoid'))

# 모델 컴파일
model.compile(loss = 'sparse_categorical_crossentropy', 
              optimizer = optimizers.Adam(learning_rate = 0.0002), metrics = 'accuracy')

# 모델 실행
history = model.fit(
    train_generator,
    steps_per_epoch = 31,
    epochs = 20,
    validation_data = test_generator,
    validation_steps = 4
)

# 결과를 그래프로 표현
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
y_vloss = history.history['val_loss']
y_loss = history.history['loss']

x_len = np.arange(len(y_loss))
plt.plot(x_len, acc, marker = '.', c = 'red', label = 'Trainset_acc')
plt.plot(x_len, val_acc, marker = '.', c = 'lightcoral', label = 'Testset_acc')
plt.plot(x_len, y_vloss, marker = '.', c = 'cornflowerblue', label = 'Testset_loss')
plt.plot(x_len, y_loss, marker = '.', c = 'blue', label = 'trainset_loss')
plt.legend(loc = 'upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss/acc')
plt.show()








############################
# 모델 성능 극대화하기