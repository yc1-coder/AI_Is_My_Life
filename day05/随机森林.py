
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

#1.数据处理
#1.1加载数据集
digits = load_digits()
X = digits.data
y = digits.target

#1.2数据预处理
X = X/15
X = StandardScaler().fit_transform(X)

#2.学习过程
#2.1选择模型
cl = Cl(RandomForestClassifier(),X,y)

#2.2训练模型
cl.show_val_curve(n_estimators = range(10,101,10))

