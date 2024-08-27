import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input, Add
import numpy as np

# 生成带有16个特征的时间序列数据
def generate_data(seq_length, num_samples, num_features):
    X = []
    y = []
    for _ in range(num_samples):
        seq = np.random.rand(seq_length, num_features)
        X.append(seq[:-1])  # 输入序列
        y.append(seq[1:])   # 目标序列，向右平移一位
    return np.array(X), np.array(y)

# 超参数
seq_length = 10
num_samples = 1000
num_features = 16  # 每个时间步的特征数
batch_size = 32
epochs = 20

# 生成数据
X, y = generate_data(seq_length, num_samples, num_features)

# 将数据重塑为 [samples, time steps, features] 格式
X = X.reshape((X.shape[0], X.shape[1], num_features))
y = y.reshape((y.shape[0], y.shape[1], num_features))

# 定义输入层
inputs = Input(shape=(seq_length-1, num_features))

# LSTM 层
x = LSTM(50, return_sequences=True)(inputs)

# Dense 层
x = Dense(num_features)(x)

# 残差连接：将输入直接加到输出上
outputs = Add()([x, inputs])

# 构建模型
model = Model(inputs=inputs, outputs=outputs)

# 编译模型
model.compile(optimizer='adam', loss='mse')

# 训练模型
model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=2)

# 进行预测
test_input = np.random.rand(seq_length - 1, num_features)
test_input = test_input.reshape((1, seq_length-1, num_features))
predicted = model.predict(test_input, verbose=0)

print("输入序列:")
print(test_input.flatten().reshape((seq_length-1, num_features)))
print("预测输出:")
print(predicted.flatten().reshape((seq_length-1, num_features)))
