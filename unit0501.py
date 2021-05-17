from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, LeakyReLU, UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#이미지가 저장될 폴더가 없다면 만듭니다.
import os
os.chdir('/Users/heechankang/projects/pythonworkspace/git_study/machine_deep_learning')
if not os.path.exists("./gan_images"):
    os.makedirs("./gan_images")

np.random.seed(3)
tf.random.set_seed(3)

# 생성자 모델을 만듭니다.
generator = Sequential()
generator.add(Dense(128*7*7, input_dim=100, activation=LeakyReLU(0.2)))
        # Dense 의 128은 노드 수, 7*7은 이미지의 크기
        # input_dim 100 : 100개의 임의의 벡터 
        # 일반적인 relu 사용 시 학습이 불안정해서 LeakyReLU 활용 왜?
        # -> relu 사용 시 음수값에 대해서는 무조건 0이 되어 뉴런이 소실됨. 이를 막아 약간의 가중치를 남기기 위함.
generator.add(BatchNormalization())
        # 배치 정규화(평균 0, 분산 1)
generator.add(Reshape((7, 7, 128)))
generator.add(UpSampling2D())
        # UpSampling2D : 7x7 이미지를 14x14로 확장
generator.add(Conv2D(64, kernel_size=5, padding='same'))
        # 64: 노드 수, Conv2D로 축소된 크기를 padding = 'same'으로 다시 채워줌.
generator.add(BatchNormalization())
generator.add(Activation(LeakyReLU(0.2)))
generator.add(UpSampling2D())
        # 14x14 -> 28x28
generator.add(Conv2D(1, kernel_size=5, padding='same', activation='tanh'))
        # 출력층. 노드 수 : 1, 활성화함수 tanh 사용(-1~1)
        # 컴파일 없음.


# 판별자 모델을 만듭니다.
# 판별자에는 Pooling이 없음. 대신 padding이 있음
discriminator = Sequential()
discriminator.add(Conv2D(64, kernel_size=5, strides=2, input_shape=(28,28,1), padding="same"))
        # stride = 2칸씩 이동, input (28x28) : 이미지 크기, 1 : 흑백
discriminator.add(Activation(LeakyReLU(0.2)))
discriminator.add(Dropout(0.3))
        # 과적합 방지
discriminator.add(Conv2D(128, kernel_size=5, strides=2, padding="same"))
discriminator.add(Activation(LeakyReLU(0.2)))
discriminator.add(Dropout(0.3))
discriminator.add(Flatten())
        # 1차원으로 변경
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.compile(loss='binary_crossentropy', optimizer='adam')
        # 컴파일 있음.
discriminator.trainable = False
        # 판별자는 훈련한 가중치 학습 안함.

#생성자와 판별자 모델을 연결시키는 gan 모델을 만듭니다.
ginput = Input(shape=(100,))
        # 랜덤으로 100개의 벡터 생성, 생성자에 입력
dis_output = discriminator(generator(ginput))
        # 판별된 결과값임. generator -> discriminator 통과
gan = Model(ginput, dis_output)
gan.compile(loss='binary_crossentropy', optimizer='adam')
gan.summary()

#신경망을 실행시키는 함수를 만듭니다.
def gan_train(epoch, batch_size, saving_interval):
    # Test data 필요 없음. 훈련용 데이터가 곧 원본 데이터임.

  # MNIST 데이터 불러오기
  (X_train, _), (_, _) = mnist.load_data()  
        # 앞서 불러온 적 있는 MNIST를 다시 이용합니다. 단, 테스트과정은 필요없고 이미지만 사용할 것이기 때문에 X_train만 불러왔습니다.
  X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
  X_train = (X_train - 127.5) / 127.5  
        # 픽셀값은 0에서 255사이의 값입니다. 이전에 255로 나누어 줄때는 이를 0~1사이의 값으로 바꾸었던 것인데, 여기서는 127.5를 빼준 뒤 127.5로 나누어 줌으로 인해 -1에서 1사이의 값으로 바뀌게 됩니다.
        #X_train.shape, Y_train.shape, X_test.shape, Y_test.shape 중 첫번째만 받아오기

  true = np.ones((batch_size, 1))
        # 모두 참(1)인 배열 만들기
  fake = np.zeros((batch_size, 1))
        # 모두 거짓(0)인 배열 만들기

  for i in range(epoch):
          # 실제 데이터를 판별자에 입력하는 부분입니다.
          idx = np.random.randint(0, X_train.shape[0], batch_size) # batch_size : 인터벌
          imgs = X_train[idx]
          d_loss_real = discriminator.train_on_batch(imgs, true)

          #가상 이미지를 판별자에 입력하는 부분입니다.
          noise = np.random.normal(0, 1, (batch_size, 100))
                # random.normal : 정규분포 평균 : 0, 평균 : 1
          gen_imgs = generator.predict(noise)
          d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)

          #판별자와 생성자의 오차를 계산합니다.
          d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
          g_loss = gan.train_on_batch(noise, true)

          print('epoch:%d' % i, ' d_loss:%.4f' % d_loss, ' g_loss:%.4f' % g_loss)

        # 이부분은 중간 과정을 이미지로 저장해 주는 부분입니다. 본 장의 주요 내용과 관련이 없어
        # 소스코드만 첨부합니다. 만들어진 이미지들은 gan_images 폴더에 저장됩니다.
          if i % saving_interval == 0:
              #r, c = 5, 5
              noise = np.random.normal(0, 1, (25, 100))
              gen_imgs = generator.predict(noise)

              # Rescale images 0 - 1
              gen_imgs = 0.5 * gen_imgs + 0.5

              fig, axs = plt.subplots(5, 5)
              count = 0
              for j in range(5):
                  for k in range(5):
                      axs[j, k].imshow(gen_imgs[count, :, :, 0], cmap='gray')
                      axs[j, k].axis('off')
                      count += 1
              fig.savefig("gan_images/gan_mnist_%d.png" % i)

gan_train(4001, 32, 200)  #4000번 반복되고(+1을 해 주는 것에 주의), 배치 사이즈는 32,  200번 마다 결과가 저장되게 하였습니다.
