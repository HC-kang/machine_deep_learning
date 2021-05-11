# from keras.datasets import mnist
# from keras.utils import np_utils
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
# from keras.callbacks import ModelCheckpoint, EarlyStopping

# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import tensorflow as tf

# seed = 0
# np.random.seed(seed)
# tf.random.set_seed(3)

# (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')/255
# X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')/255
# Y_train = np_utils.to_categorical(Y_train)
# Y_test = np_utils.to_categorical(Y_test)

# model = Sequential()
# model.add(Conv2D(32, kernel_size = (3,3), input_shape=(28, 28, 1), activation = 'relu'))
# model.add(Conv2D(64, (3,3), activation = 'relu'))
# model.add(MaxPooling2D(pool_size = 2))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation = 'relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation = 'softmax'))

# model.compile(loss = 'categorical_crossentropy',
#               optimizer = 'adam',
#               metrics = 'accuracy')

# # 모델 최적화
# MODEL_DIR = './model/'
# if not os.path.exists(MODEL_DIR):
#     os.mkdir(MODEL_DIR)

# modelpath = './model/{epoch:02d} - {val_loss:.4f}.hdf5'
# checkpointer = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss', verbose = 1, save_best_only = True)
# early_stopping_callback = EarlyStopping(monitor = 'val_loss', patience = 10)

# # 모델 실행
# history = model.fit(X_train, Y_train, validation_data = (X_test, Y_test), epochs = 30, batch_size = 200, verbose = 1, callbacks = [early_stopping_callback, checkpointer])

# # 테스트 정확도 출력
# print('\n Test Accuracy: %.4f'%(model.evaluate(X_test, Y_test)[1]))

# # 테스트셋 오차
# y_vloss = history.history['val_loss']

# # 학습셋의 오차
# y_loss = history.history['loss']

# # 그래프 표현
# x_len = np.arange(len(y_loss))
# plt.plot(x_len, y_vloss, marker = '.', c = 'red', label = 'Testset_loss')
# plt.plot(x_len, y_loss, marker = '.', c = 'blue', label = 'Trainset_loss')

# plt.legend(loc = 'upper right')
# plt.grid()
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.show()


# #####################

# from tensorflow.keras.preprocessing.text import text_to_word_sequence

# text = '해보지 않으면 해낼 수 없다'
# result = text_to_word_sequence(text)
# print(result)


# from tensorflow.keras.preprocessing.text import Tokenizer

# docs = ['먼저 텍스트의 각 단어를 나누어 토큰화합니다.',
#         '텍스트의 단어로 토큰화해야 딥러닝에서 인식됩니다.',
#         '토큰화한 결과는 딥러닝에서 사용할 수 있습니다.']

# token = Tokenizer()         # 토큰화 함수 지정
# token.fit_on_texts(docs)    # 토큰화 함수에 문장 적용
# print(token.word_counts)    # 단어의 빈도 수를 계산한 결과 출력

# print(token.document_count) # 문장의 수를 출력

# print(token.word_docs)      # 각 단어들이 몇개의 문장에 나오는지 출력

# print(token.word_index)     # 각 단어에 매겨진 인덱스값 출력

# from tensorflow.keras.preprocessing.text import text_to_word_sequence

# text = '해보지 않으면 해낼 수 없다'

# # 해당 텍스트 토큰화
# result = text_to_word_sequence(text)
# print('\n원문:\n', text)
# print('\n토큰화:\n', result)

# # 텍스트 전처리 함수 Tokenizer() 호출
# from keras.preprocessing.text import Tokenizer

# # 전처리하려는 세 개의 문장 정하기
# docs = ['먼저 텍스트의 각 단어를 나누어 토큰화합니다.',
#  '텍스트의 단어로 토큰화해야 딥러닝에서 인식됩니다.',
#  '토큰화한 결과는 딥러닝에서 사용할 수 있습니다.']

# # 토큰화 함수를 이용해 전처리하는 과정
# token = Tokenizer()         # 토큰화 함수 지정
# token.fit_on_texts(docs)    # 토큰화 함수에 문장 적용하기

# # 각 옵션에 맞춰 단어의 빈도 수 계산 결과 출력
# print('\n단어 카운트:\n', token.word_counts)

# # 출력되는 순서는 랜덤
# print('\n문장 카운트: ', token.document_count)
# print('\n각 단어가 몇 개의 문장에 포함되어 있는가:\n', token.word_docs)
# print('\n각 단어에 매겨진 인덱스 값:', token.word_index)


# from tensorflow.keras.preprocessing.text import Tokenizer

# text = '오랫동안 꿈꾸는 이는 그 꿈을 닮아간다'

# token = Tokenizer()
# token.fit_on_texts([text])
# print(token.word_index)

# x = token.texts_to_sequences([text])
# print(x)

# from keras.utils import np_utils

# # 인덱스 수에 하나를 추가해서 원핫 인코딩 배열 만들기

# word_size = len(token.word_index)+1
# x = np_utils.to_categorical(x, num_classes = word_size)
# x

# from keras.layers import Embedding

# model = Sequential()
# model.add(Embedding(16, 4))
# import pandas as pd
# import numpy as np

# docs = [
#     '너무 재밌네요', '최고에요', '참 잘 만든 영화에요', '추천하고 싶은 영화입니다', '한번 더 보고싶네요',
#     '글쎄요', '별로에요', '생각보다 지루하네요', '연기가 어색해요', '재미없어요'
# ]

# class = array([1,1,1,1,1,0,0,0,0,0])

# # 토큰화
# token = Tokenizer()
# token.fit_on_texts(docs)
# print(token.word_index)


# x = token.texts_to_sequences(docs)
# print(x)


# padded_x = pad_sequence(x, 4) # 서로 다른 길이의 데이터를 4로 맞추기
# print(padded_x)

# Embedding(word_size, 8, input_length = 4)

# model = Sequential()
# model.add(Embedding(word_size, 8, input_length = 4))
# model.add(Flatten)
# model.add(Dense(1, activation = 'sigmoid'))
# model.compile(optimizer = 'adam',  loss = 'binary_crossentropy', metircs = 'accuracy')
# model.fit(padded_x, labels, epochs = 20)
# print('\n Accuracy : %.4f'%(model.evaluate(padded_x, labels)[1]))

# ######################
# import numpy as np
# import tensorflow as tf
# from numpy import array
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten, Embedding

# docs = [
#     '너무 재밌네요', '최고에요', '참 잘 만든 영화에요', '추천하고 싶은 영화입니다', '한번 더 보고싶네요',
#     '글쎄요', '별로에요', '생각보다 지루하네요', '연기가 어색해요', '재미없어요'
# ]

# labels = array([1,1,1,1,1,0,0,0,0,0])

# # 토큰화   
# token = Tokenizer()
# token.fit_on_texts(docs)
# print(token.word_index)

# x = token.texts_to_sequences(docs)
# print(x)

# padded_x = pad_sequences(x, 4)
# '\n패딩결과\n', print( padded_x)

# word_size = len(token.word_index)+1

# model = Sequential()
# model.add(Embedding(word_size, 8, input_length=4))
# model.add(Flatten())

# model.add(Dense(1, activation = 'sigmoid'))
# model.compile(optimizer = 'adam', loss = 'binary_crossentropy',
# metrics = 'accuracy')

# model.fit(padded_x, labels, epochs = 20)

# print('\n Accuracy : %.4f' % (model.evaluate(padded_x, labels)[1]))

#########
# NLP

import numpy as np
import tensorflow as tf
from numpy import array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding

from tensorflow.keras.preprocessing.text import text_to_word_sequence

text = '해보지 않으면 해낼 수 없다'

result = text_to_word_sequence(text)
print(text)
print(result)

docs = ['먼저 텍스트의 각 단어를 나누어 토큰화합니다.',
        '텍스트의 단어로 토큰화해야 딥러닝에서 인식됩니다.',
        '토큰화한 결과는 딥러닝에서 사용할 수 있습니다.']

token = Tokenizer()
token.fit_on_texts(docs)

print(token.word_counts)
print(token.document_count)
print(token.word_docs)


docs2 = [
    '너무 재밌네요', '최고에요', '참 잘 만든 영화에요', '추천하고 싶은 영화입니다', '한번 더 보고싶네요',
    '글쎄요', '별로에요', '생각보다 지루하네요', '연기가 어색해요', '재미없어요'
]

classes = array([1,1,1,1,1,0,0,0,0,0])

token2 = Tokenizer()
token2.fit_on_texts(docs2)

print(token2.word_index)
x = token2.texts_to_sequences(docs2)
x

padded_x = pad_sequences(x, 4)
padded_x

word_size = len(token2.word_index) +1
word_size

model = Sequential()
model.add(Embedding(word_size, 8, input_length = 4))
model.add(Flatten())
model.add(Dense(1, activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = 'accuracy')
model.fit(padded_x, classes, epochs = 20)
print('\n Accuracy : %.4f'%(model.evaluate(padded_x, classes)[1]))