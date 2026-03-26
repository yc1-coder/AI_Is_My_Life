import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# 三层神经网络：2 输入（BMI、平均血压），3 隐藏，1 输出 + ReLU + Sigmoid 激活
class NeuralNetwork:
    def __init__(self):
        # 网络结构
        # 输入层 (2) → 隐藏层 (3) → 输出层 (1)

        # 输入 → 隐藏 权重 (2×3)
        self.W1 = np.random.randn(2, 3) * 0.5

        # 隐藏 → 输出 权重 (3×1)
        self.W2 = np.random.randn(3, 1) * 0.5

        # 隐藏层偏置 (3 个)
        self.b1 = np.zeros((1, 3))

        # 输出层偏置 (1 个)
        self.b2 = np.zeros((1, 1))

    # ReLU 激活函数
    def relu(self, x):
        return np.maximum(0, x)

    # ReLU 导数（反向传播用）
    def relu_deriv(self, x):
        return np.where(x > 0, 1, 0)

    # Sigmoid 激活函数（用于输出层）
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    # Sigmoid 导数
    def sigmoid_deriv(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    # 前向传播
    def forward(self, X):
        # 隐藏层：输入×权重 + 偏置，然后 ReLU 激活
        self.z1 = np.dot(X, self.W1) + self.b1
        self.hidden = self.relu(self.z1)

        # 输出层：隐藏×权重 + 偏置，然后 Sigmoid 激活
        self.z2 = np.dot(self.hidden, self.W2) + self.b2
        self.output = self.sigmoid(self.z2)

        return self.output

    # 反向传播
    def backward(self, X, y, output, lr=0.001):
        # 样本数量
        n = X.shape[0]

        # 输出层梯度：误差 × Sigmoid 导数
        error_output = output - y
        delta_output = error_output * self.sigmoid_deriv(self.z2)

        # 隐藏层梯度
        delta_hidden = np.dot(delta_output, self.W2.T) * self.relu_deriv(self.z1)

        # 更新权重
        dW2 = np.dot(self.hidden.T, delta_output) / n
        dW1 = np.dot(X.T, delta_hidden) / n

        # 更新偏置
        db2 = np.mean(delta_output, axis=0, keepdims=True)
        db1 = np.mean(delta_hidden, axis=0, keepdims=True)

        # 梯度下降
        self.W2 -= lr * dW2
        self.W1 -= lr * dW1
        self.b2 -= lr * db2
        self.b1 -= lr * db1

    # 训练
    def train(self, X, y, epochs=10000, lr=0.001):
        for i in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, lr)

            # 每 1000 轮打印损失
            if i % 1000 == 0:
                loss = np.mean(np.square(y - output))
                print(f"轮数 {i:4d} | 损失：{loss:.4f}")

    # 预测概率
    def predict_proba(self, X):
        return self.forward(X)

    # 预测类别
    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    # 评估准确率
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)


# ===================== 测试 =====================


if __name__ == "__main__":
    # 加载糖尿病数据集
    print("正在加载糖尿病数据集...")
    diabetes = load_diabetes()

    # 数据集说明
    print(f"\n数据集特征：{diabetes.feature_names}")
    print(f"数据样本数：{diabetes.data.shape[0]}")
    print(f"特征数量：{diabetes.data.shape[1]}")

    # 提取 BMI 和平均血压数据
    # diabetes 数据集的 feature_names:
    # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
    # bmi 是索引 2, bp (平均血压) 是索引 3
    X = diabetes.data[:, [2, 3]]  # BMI 和平均血压

    # 创建二分类标签：根据疾病进展指标（target）
    # target 是连续值，我们将其转换为二分类（高于中位数为 1，否则为 0）
    median_target = np.median(diabetes.target)
    y = (diabetes.target > median_target).astype(int).reshape(-1, 1)

    print(f"\n特征：BMI 和平均血压")
    print(f"正样本（患病）数量：{np.sum(y)}")
    print(f"负样本（健康）数量：{len(y) - np.sum(y)}")

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # 创建神经网络
    nn = NeuralNetwork()

    print("\n开始训练神经网络...")
    nn.train(X_train, y_train, epochs=20000, lr=0.01)

    # 训练集评估
    train_acc = nn.score(X_train, y_train)
    print(f"\n训练集准确率：{train_acc:.4f}")

    # 测试集评估
    test_acc = nn.score(X_test, y_test)
    print(f"测试集准确率：{test_acc:.4f}")

    # 预测结果展示
    print("\n部分样本预测结果：")
    predictions = nn.predict(X_test)
    probabilities = nn.predict_proba(X_test)

    for i in range(min(10, len(X_test))):
        true_label = y_test[i][0]
        pred_label = predictions[i][0]
        prob = probabilities[i][0]
        print(f"样本{i + 1}: 真实={true_label}, 预测={pred_label}, 概率={prob:.4f}")

    # 可视化特征
    print("\n特征统计信息：")
    print(f"BMI - 均值：{X[:, 0].mean():.2f}, 标准差：{X[:, 0].std():.2f}")
    print(f"平均血压 - 均值：{X[:, 1].mean():.2f}, 标准差：{X[:, 1].std():.2f}")

