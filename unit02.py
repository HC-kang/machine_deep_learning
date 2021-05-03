import numpy as np

x = [2,4,6,8]
y = [81,93, 91, 97]

mx = np.mean(x)
my = np.mean(y)
print('x의 평균값 : ', mx)
print('y의 평균값 : ', my)

# 기울기 공식의 분모
divisor = sum([(i-mx)**2 for i in x])
# divisor = sum([(i-mx)**2 for i in x])

# 기울기 공식의 분자
def top(x, mx, y ,my):
    d = 0
    for i in range(len(x)):
        d += (x[i] - mx) * (y[i] - my)
    return d
dividend = top(x, mx, y, my)
dividend

def top(x, mx, y, my):
    d = 0
    for i in range(len(x)):
        d += (x[i]-mx) * (y[i]-my)
    return d
dividend = top(x, mx, y, my)

# 분모와 분자를 계산하여 기울기 a 구하기
a = dividend / divisor

# 이부분은 노트에 증명 해놨으니 확인.


# a를 가지고 b 구하기
b = my - (mx*a)

# 출력으로 확인
print('기울기 a = ', a)
print('y 절편 b = ', b)


# MSE 구해보기
fake_a_b = [3, 76]

data = [[2, 81], [4,93], [6, 91],[8,97]]
x = [i[0] for i in data]
y = [i[1] for i in data]

# predict 라는 함수로 y = ax+b를 구현
def predict(x):
    return fake_a_b[0]*x + fake_a_b[1]

평균제곱근 공식을 그대로 옮기기
def mse(y_hat, y):
    return ((y_hat-y) **2).mean()

# 데이터를 대입해서 최종값을 구하는 함수
def mse_val(predict_result, y):
    return mse(np.array(predict_result), np.array(y))

# 예측값이 들어갈 리스트
predict_result=[]

# 모든 x값 대입하여 리스트 완성
for i in range(len(x)):
    predict_result.append(predict(x[i]))
    print('공부시간 = %.f, 실제 점수=%.f, 예측점수=%.f'%(x[i], y[i], predict(x[i])))

# 최종 MSE 출력
print('MSE 최종값 : ' + str(mse_val(predict_result,y)))

################################################

y_pred = a * x_data + b     # 오차 함수인 y = ax + b를 정의
error = y_data - y_pred     # 실제값 - 예측값, 즉 오차를 구하는 식

# 평균 제곱 오차를 a를 미분한 결과
a_diff = -(2 / len(x_data)) * sum(x_data * (error))
# 평균 제곱 오차를 b로 미분한 결과
b_diff = -(2 / len(x_data)) * sum(y_data - y_pred)

a = a - lr * a_diff     # 미분 결과에 학습률을 곱한 후 기존의 a값을 업데이트
b = b - lr * b_diff     # 미분 결과에 학습률을 곱한 후 기존의 b값을 업데이트

#####
# 경사하강법 실습
###

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 공부시간 X와 성적 Y의 리스트를 만들기
data = [[2, 81], [4, 93],[6,91], [8,97]]
x = [i[0] for i in data]
y = [i[1] for i in data]

# 그래프로 나타내기
plt.figure(figsize = (8, 5))
plt.scatter(x, y)
plt.show()

# 리스트로 되어 있는 x와 y값을 넘파이 배열로 바꾸기
# (인덱스를 주어 하나씩 불러와 계산이 가능하게 하기 위함) 
x_data = np.array(x)
y_data = np.array(y)

# 기울기 a와 b의 값 초기화
a = 0
b = 0

# 학습률 정하기
lr = 0.05

# 몇 번 반복될지 설정(0부터 세므로 원하는 반복 횟수에 +1)
epochs = 2001

# 경사하강법 시작
for i in range(epochs):         # 에포크 수만큼 반복
    y_pred = a * x_data + b     # y를 구하는 식 세우기
    error = y_data - y_pred     # 오차를 구하는 식
    # 오차 함수를 a로 미분한 값
    a_diff = -(1/len(x_data))*sum(x_data * (error))
    # 오차 함수를 b로 미분한 값
    b_diff = -(1/len(x_data))*sum(y_data-y_pred)
    a = a-lr*a_diff     # 학습률을 곱해 기존의 a값 업데이트
    b = b-lr*b_diff     # 학습률을 곱해 기존의 b값 업데이트

    if i % 100 == 0:    # 100번 반복될 때마다 현재의 a, b값 출력∂
        print('epoch = %.f, 기울기 = %.04f, 절편 = %.04f'%(i, a, b))
        plt.scatter(x, y)
        plt.plot([min(x_data), max(x_data)], [min(y_pred), max(y_pred)])
        plt.show()

# 앞서 구한 기울기와 절편을 이용해 그래프를 다시 그리기
y_pred = a*x_data + b
plt.scatter(x, y)
plt.plot([min(x_data), max(x_data)], [min(y_pred), max(y_pred)])
plt.show()

################################
data = [[2,0,81], [4,4,93], [6,2,91], [8,3,97]]
x1 = [i[0] for i in data]
x2 = [i[1] for i in data]
y = [i[2] for i in data]

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
# 3D 그래프를 그리는 라이브러리 가져오기

ax = plt.axes(projection='3d')      # 그래프 유형 정하기
ax.set_xlabel('study_hours')
ax.set_ylabel('private_class')
ax.set_zlabel('Score')
ax.scatter(x1, x2, y)
plt.show()

y_pred= a1 * x1_data + a2 * x2_data + b # y를 구하는 식을 세우기
error = y_data - y_pred
a1_diff = -(1/len(x1_data))*sum(x1_data * (error))
# 오차함수를 a1으로 미분
a2_diff = -(1/len(x2_data))*sum(x2_data * (error))
# 오차함수를 a2로 미분

b_new = -(1/len(x1_data))*sum(y_data - y_pred)
# 오차함수를 b로 미분

a1 = a1 - lr * a1_diff  # 학습률을 곱해 기존의 a1값 업데이트
a2 = a2 - lr * a2_diff  # 학습률을 곱해 기존의 a2값 업데이트
b = b - lr * b_diff     # 학습률을 곱해 기존의 b 값 업데이트

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# 공부시간 X와 성적 Y의 리스트 만들기
data = [[2,0,81], [4,4,93], [6,2,91], [8,3,97]]
x1 = [i[0] for i in data]
x2 = [i[1] for i in data]
y = [i[2] for i in data]


ax = plt.axes(projection='3d')      # 그래프 유형 정하기
ax.set_xlabel('study_hours')
ax.set_ylabel('private_class')
ax.set_zlabel('Score')
ax.dist = 11
ax.scatter(x1, x2, y)
plt.show()

x1_data = np.array(x1)
x2_data = np.array(x2)
y_data = np.array(y)

# 기울기와 절편 초기화
a1 = 0
a2 = 0
b = 0

# 학습률
lr = 0.05

# 몇 번 반복할 지 설정
epochs = 2001

# 경사하강법 시작
for i in range(epochs):
    y_pred = a1 * x1_data + a2 * x2_data + b
    error = y_data - y_pred
    # 오차함수를 a1으로 미분한 값
    a1_diff = -(1/len(x1_data))*sum(x1_data * (error))
    # 오차함수를 a2으로 미분한 값
    a2_diff = -(1/len(x2_data))*sum(x2_data * (error))
    # 오차함수를 b로 미분한 값
    b_new = -(1/len(x1_data))*sum(y_data-y_pred)
    a1 = a1 - lr * a1_diff
    a2 = a2 - lr * a2_diff
    b = b - lr * b_new

    if i % 100==0:
        print('epoch = %.f, 기울기1 = %.04f, 기울기2 = %.04f, 절편 = %.04f'%(i, a1, a2, b))
        # plt.scatter(x, y)
        # plt.plot([min(x_data), max(x_data)], [min(y_pred), max(y_pred)])
        # plt.show()

########################################
# 참고 - 다중 선형회귀 예측평면 3D로 보기
import statsmodels.api as statm
import statsmodels.formula.api as statfa
import pandas as pd

X = [i[0:2] for i in data]
y = [i[2] for i in data]

X_1=statm.add_constant(X)
results=statm.OLS(y,X_1).fit()

hour_class=pd.DataFrame(X,columns=['study_hours','private_class'])
hour_class['Score']=pd.Series(y)

model = statfa.ols(formula='Score ~ study_hours + private_class', data=hour_class)

print(model)

results_formula = model.fit()

이어서
#####################################
import numpy as np
import matplotlib.pyplot as plt

data = [[2,0], [4,0], [6,0], [8,1], [10,1], [12,1], [14,1]]
x_data = [i[0] for i in data]
y_data = [i[1] for i in data]


plt.scatter(x_data, y_data)
plt.xlim(0, 15)
plt.ylim(-.1, 1.1)

a = 0
b = 0
lr = 0.05

def sigmoid(x):
    return 1/(1+np.e**(-x))

for i in range(2001):
    for x_data, y_data in data:
        a_diff = x_data*(sigmoid(a*x_data + b) - y_data)
        b_diff = sigmoid(a*x_data + b) -y_data
        a = a - lr * a_diff
        b = b - lr * b_diff
        if i % 1000 ==0:
            print('epoch = %.f, 기울기 = %.04f, 절편 = %.04f' %(i, a, b))

plt.scatter(x_data, y_data)
plt.xlim(0, 15)
plt.ylim(-.1, 1.1)
x_range = np.arange(0, 15, 0.1)
plt.plot(np.arange(0, 15, 0.1), np.array([sigmoid(a*x+b) for x in x_range]))
plt.show()