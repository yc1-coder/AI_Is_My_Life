import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
#1.数据处理
#1.1读取数据
df = pd.read_csv('monsters.csv',encoding='gbk',index_col=0)
df.columns = ['name','X1','X2','y']
#print(df)

#1.2数据预处理
lb = LabelEncoder()
#df.iloc[:,'X1'] = df['X1']
df['X1'] = lb.fit_transform(df['X1'])
df['X2'] = lb.fit_transform(df['X2'])
df['y'] = lb.fit_transform(df['y'])
print(df)

#1.3数据集划分
X = df.loc[:,['X1','X2']]
y = df.loc[:,'y']

#2.学习过程
#2.1选择模型
model = DecisionTreeClassifier()
#2.2 训练模型
#我们现在的决策树，在构建时会默认按照基尼不纯度进行分裂
model.fit(X,y)
#2.3 评估模型
pred_y = model.predict(X)

print(f'acc:{accuracy_score(y,pred_y)}')
print(f'confusion_matrix:\n{confusion_matrix(y,pred_y)}')
print(f'classification_report:\n{classification_report(y,pred_y)}')

#3.图形树
from sklearn.tree import plot_tree
plt.figure(figsize=(16,10),dpi=100)
plot_tree(model,feature_names=['背景','实力'],class_names=lb.classes_,rounded=True,fontsize=30,filled=True)
plt.show()



























