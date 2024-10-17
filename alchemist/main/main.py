import matplotlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from ase.io import read
from pprint import pprint
import matplotlib.pyplot as plt
from dscribe.descriptors import SOAP
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
zhfont1 = matplotlib.font_manager.FontProperties(fname='./font/SourceHanSansSC-Bold.otf')

class CustomRandomForestRegressor:
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
        self.trees = []
        
    def fit(self, X, y):
        for _ in range(self.n_estimators):
            # 对样本进行有放回抽样
            indices = np.random.choice(range(X.shape[0]), size=X.shape[0], replace=True)
            X_sample = X[indices]
            y_sample = y[indices]
            tree = self._build_tree(X_sample, y_sample)
            self.trees.append(tree)
    
    def predict(self, X):
        # 投票机制
        predictions = np.array([self._predict_tree(x) for x in X for tree in self.trees])
        final_predictions = np.mean(predictions.reshape(-1, self.n_estimators), axis=1)
        return final_predictions

    def _build_tree(self, X, y):
        # 这里简单实现树的构建，可以添加更复杂的逻辑
        return (X, y)

    def _predict_tree(self, x):
        # 在树上预测
        return np.mean([tree[1] for tree in self.trees])

# 读取数据
with open('./data/supecon.csv', 'r') as file:
    data = pd.read_csv(file)
    pattern = r'([A-Z][a-z]?)(\d*)'
    elements = data['Formula'].str.findall(pattern).apply(lambda x: [match[0] for match in x])
    all_elements = set()
    all_elements.update(elements.explode().unique())
    all_elements = list(all_elements)

    cifs = data['cif']
    la = data['la']
    wlog = data['wlog']
    Tc_AD = data['Tc_AD']
    features = []
    labels = []
    
    # 创建标准化器对象
    scaler = StandardScaler()
    
    # 创建SOAP描述符对象
    soap = SOAP(
        species=all_elements,
        r_cut=5,
        n_max=3,
        l_max=5,
        sigma=0.2,
        compression={"mode":"off","species_weighting":None},
        sparse=False,
        dtype='float32'
    )
    
    max_data = min(80, len(cifs))
    for i in tqdm(range(max_data), desc=f"读取并用ase解析cif文件中{max_data}条数据,生成SOAP描述符"):
        cif_file = data.loc[i, "cif"]
        with open('temp_file.cif', 'w') as cif_output:
            cif_output.write(cif_file)
        atoms = read('temp_file.cif')
        soap_descriptors = scaler.fit_transform(np.array(soap.create(atoms), dtype=np.float32))  
        features.append(soap_descriptors)
        labels.append(data.loc[i, "Tc_AD"])
        
    print(len(features))

    # 计算最大长度和宽度
    max_length = max(feature.shape[0] for feature in features)
    max_width = max(feature.shape[1] for feature in features)

    # 使用resize替代填充0
    resized_features = np.empty((len(features), max_length, max_width), dtype=np.float32)
    for i, feature in enumerate(features):
        resized_features[i, :feature.shape[0], :feature.shape[1]] = feature

    pprint(f'Resized features shape: {resized_features.shape}')
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(resized_features, labels, test_size=0.3, random_state=42, shuffle=True)

    # 将列表转换为 numpy 数组
    X_train = np.resize(X_train, (int(max_data * 0.7), -1))
    X_test = np.resize(X_test, (int(max_data * 0.3), -1))
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    # 创建自定义随机森林模型
    model = CustomRandomForestRegressor(n_estimators=10)

    # 训练模型
    model.fit(X_train, y_train)

    # 进行预测
    y_pred_train = model.predict(X_train)
    mse_train = mean_squared_error(y_train, y_pred_train)
    
    # 评估模型性能
    y_pred_test = model.predict(X_test)
    mse_test = mean_squared_error(y_test, y_pred_test)

    # 输出均方误差
    print(f"训练集均方误差: {mse_train:.4f}")
    print(f"测试集均方误差: {mse_test:.4f}")

    # 绘制拟合曲线
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_test, color='yellow', label='预测值') 
    plt.scatter(y_test, y_test, color='blue', label='实际值') 
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='理想预测')
    plt.xlabel('实际 Tc_AD')
    plt.ylabel('预测 Tc_AD')
    plt.title('测试集 Tc_AD 预测值与实际值比较')
    plt.legend()
    plt.grid()
    plt.show()

    # 打印均方误差
    pprint(f"测试集均方误差: {mse_test}")

