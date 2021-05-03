import numpy as np

x = [2,4,6,8]
y = [81,93, 91, 97]

mx = np.mean(x)
my = np.mean(y)
print('x의 평균값 : ', mx)
print('y의 평균값 : ', my)

# 기울기 공식의 분모
divisor = sum([(i-mx)**2 for i in x])
divisor

# 기울기 공식의 분자
def top(x, mx, y ,my):
    d = 0
    for i in range(len(x)):
        d += (x[i] - mx) * (y[i] - my)
    return d
dividend = top(x, mx, y, my)
dividend

# 분모와 분자를 계산하여 기울기 a 구하기
a = dividend / divisor

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

def predict(x):
    return fake_a_b[0]*x + fake_a_b[1]


def mse(y_hat, y):
    return ((y_hat-y) **2).mean()


def mse_val(predict_result, y):
    return mse(np.array(predict_result), np.array(y))

# 예측값이 들어갈 리스트
predict_result=[]

# 모든 x값 대입하여 리스트 완성
for i in range(len(x)):
    predict_result.append(predict(x[i]))
    print('공부시간 = %.f, 실제 점수=%.f, 예측점수=%.f'%(x[i], y[i], predict(x[i])))