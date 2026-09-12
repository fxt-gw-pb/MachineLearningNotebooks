"""经人工检查的案例修复。键为原章节/文件序号及 Python 代码块序号。"""
OVERRIDES = {}
NOTES = {}

def add(key, block, code, note=None):
    OVERRIDES[(key, block)] = code.strip()
    if note:
        NOTES.setdefault(key, []).append(note)

add('1/12', 0, '''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
rng = np.random.default_rng(42)
X_train = np.linspace(0, 1, 60).reshape(-1, 1)
y_train = np.sin(2 * np.pi * X_train[:, 0]) + rng.normal(0, 0.15, len(X_train))
X_test = np.linspace(-0.15, 1.15, 250).reshape(-1, 1)
kernel = ConstantKernel(1.0, (0.01, 100)) * RBF(0.2, (0.01, 10)) + WhiteKernel(0.02, (1e-5, 1))
model = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                n_restarts_optimizer=3, random_state=42)
model.fit(X_train, y_train)
mean, std = model.predict(X_test, return_std=True)
print('拟合后的核：', model.kernel_)
print('对数边际似然：', model.log_marginal_likelihood_value_)
plt.figure(figsize=(10, 5))
plt.scatter(X_train[:, 0], y_train, s=20, label='Observations')
plt.plot(X_test[:, 0], mean, label='Posterior mean')
plt.plot(X_test[:, 0], np.sin(2 * np.pi * X_test[:, 0]), '--', label='True function')
plt.fill_between(X_test[:, 0], mean - 1.96 * std, mean + 1.96 * std,
                 alpha=0.2, label='Approx. 95% predictive interval')
plt.xlabel('x'); plt.ylabel('y'); plt.legend(); plt.show()
''', '高斯过程仍是原来的带噪正弦回归任务，改用 scikit-learn 的 GaussianProcessRegressor，免去 PyTorch/GPyTorch 依赖；区间包含 WhiteKernel 观测噪声。')

add('1/15', 0, '''
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
X = np.linspace(0, 10, 100)
y = np.sin(X) + 0.3 * np.random.randn(100)
''')
add('1/15', 2, '''
def locally_weighted_regression(X, y, tau, query_points):
    predictions = []
    for q in query_points:
        W = get_weights(q, X, tau)
        theta = np.linalg.pinv(X.T @ W @ X) @ X.T @ W @ y
        predictions.append(q @ theta)
    return np.asarray(predictions)
''', '将矩阵伪代码改为 NumPy 运算，并补齐先使用后导入的依赖。')

add('8/07', 0, '''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
X, y = make_regression(n_samples=600, n_features=8, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# 增加常数列，让最后一个参数表示截距。
A = np.column_stack([X_train, np.ones(len(X_train))])
B = np.column_stack([X_test, np.ones(len(X_test))])
weights = np.zeros(A.shape[1])
m = np.zeros_like(weights); v = np.zeros_like(weights)
beta1, beta2, learning_rate, epsilon = 0.9, 0.999, 0.3, 1e-8
train_losses = []
for step in range(1, 1501):
    residual = A @ weights - y_train
    gradient = 2 * A.T @ residual / len(A)
    m = beta1 * m + (1 - beta1) * gradient
    v = beta2 * v + (1 - beta2) * gradient**2
    m_hat = m / (1 - beta1**step)
    v_hat = v / (1 - beta2**step)
    weights -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    train_losses.append(np.mean(residual**2))
print('测试集 MSE:', mean_squared_error(y_test, B @ weights))
print('最小二乘基准 MSE:', mean_squared_error(y_test, B @ np.linalg.lstsq(A, y_train, rcond=None)[0]))
plt.plot(train_losses); plt.yscale('log')
plt.xlabel('Iteration'); plt.ylabel('Training MSE'); plt.title('Adam for linear regression')
plt.show()
''', '保留 Adam 理论，用 NumPy 实现一阶矩、二阶矩与偏差修正，并在线性回归上演示；原深层网络训练案例不转写。')

add('8/13', 0, '''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)
pipeline = make_pipeline(StandardScaler(), SVC())
param_grid = {'svc__C': [0.1, 1, 10], 'svc__gamma': [0.001, 0.01, 0.1]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='f1', n_jobs=1,
                      error_score='raise', return_train_score=True)
search.fit(X_train, y_train)
print('最佳参数:', search.best_params_)
print('训练数据内部的最佳交叉验证 F1:', search.best_score_)
print(classification_report(y_test, search.predict(X_test)))
results = pd.DataFrame(search.cv_results_)
heat = results.pivot(index='param_svc__C', columns='param_svc__gamma', values='mean_test_score')
sns.heatmap(heat, annot=True, fmt='.3f'); plt.title('Cross-validation F1'); plt.show()
ConfusionMatrixDisplay.from_estimator(search, X_test, y_test); plt.show()
''', '原神经网络网格搜索改为 SVM；预处理放入 Pipeline，使用训练集交叉验证选参，再对独立测试集评估。')

add('11/08', 0, '''
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# 使用本地图片演示几何与颜色增强，不下载 CIFAR-10 或训练神经网络。
original = Image.open(DATA_DIR / 'lenna.jpg').convert('RGB').resize((192, 192))
variants = [original, original.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
            original.rotate(15, resample=Image.Resampling.BILINEAR),
            ImageEnhance.Brightness(original).enhance(1.25)]
fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for ax, img, title in zip(axes, variants, ['Original', 'Horizontal flip', 'Rotation', 'Brightness']):
    ax.imshow(img); ax.set_title(title); ax.axis('off')
plt.tight_layout(); plt.show()
''', '数据增强保留图片变换与训练集增强思想；改用本地图片和传统分类器，移除 CIFAR-10 下载及 CNN 训练。增强是否保持标签语义需要结合具体任务判断。')
add('11/08', 1, '''
# 一个独立的表格分类实验：先划分，再只对训练集添加小幅噪声。
X, y = make_classification(n_samples=600, n_features=8, n_informative=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train); X_test = scaler.transform(X_test)
rng = np.random.default_rng(42)
X_augmented = np.vstack([X_train, X_train + rng.normal(0, 0.05, X_train.shape)])
y_augmented = np.tile(y_train, 2)
''')
add('11/08', 2, '''
baseline = LogisticRegression(max_iter=1000).fit(X_train, y_train)
augmented = LogisticRegression(max_iter=1000).fit(X_augmented, y_augmented)
print('原始训练集准确率:', accuracy_score(y_test, baseline.predict(X_test)))
print('增强训练集准确率:', accuracy_score(y_test, augmented.predict(X_test)))
print('增强不保证提高性能；噪声尺度应在训练集内部验证。')
''')
for i in range(3,8):
    add('11/08', i, '# 深度学习训练步骤已由上面的传统分类实验替代。')

add('7/11', 6, '''
from sklearn.decomposition import TruncatedSVD
# 用潜在语义分析替代 BERT：仍然比较文本表示如何影响 MDS。
lsa = TruncatedSVD(n_components=5, random_state=42)
lsa_emb = lsa.fit_transform(tfidf_matrix)
dist_lsa = pairwise_distances(lsa_emb, metric='euclidean')
mds_lsa = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
coords_lsa = mds_lsa.fit_transform(dist_lsa)
plt.figure(figsize=(9, 6))
plt.scatter(coords_lsa[:, 0], coords_lsa[:, 1])
for label, point in zip(labels, coords_lsa):
    plt.annotate(label, point)
plt.title('LSA + MDS'); plt.show()
''', 'MDS 的 BERT 扩展示例替换为 TF-IDF + TruncatedSVD（潜在语义分析），避免深度学习和模型下载。10 篇文本的 t-SNE perplexity 改为 3。')

add('13/08', 0, '''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel
X, y = make_circles(n_samples=600, factor=0.5, noise=0.12, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
# 权重也属于超参数，因此只能用训练集内部的验证集选择。
Xa, Xv, ya, yv = train_test_split(X_train, y_train, stratify=y_train, random_state=7)
kernels = [lambda A, B: rbf_kernel(A, B, gamma=0.5),
           lambda A, B: rbf_kernel(A, B, gamma=2),
           lambda A, B: polynomial_kernel(A, B, degree=3, gamma=1, coef0=1)]
weights = [(a / 4, b / 4, (4-a-b) / 4) for a in range(5) for b in range(5-a)]
records = []
for w in weights:
    K = sum(wi * kernel(Xa, Xa) for wi, kernel in zip(w, kernels))
    Kv = sum(wi * kernel(Xv, Xa) for wi, kernel in zip(w, kernels))
    clf = SVC(kernel='precomputed').fit(K, ya)
    records.append((accuracy_score(yv, clf.predict(Kv)), w))
validation_score, mu = max(records, key=lambda item: item[0])
K = sum(wi * kernel(X_train, X_train) for wi, kernel in zip(mu, kernels))
Kt = sum(wi * kernel(X_test, X_train) for wi, kernel in zip(mu, kernels))
clf = SVC(kernel='precomputed').fit(K, y_train)
print('验证集选择的核权重:', mu, '验证集准确率:', validation_score)
print('独立测试集准确率:', accuracy_score(y_test, clf.predict(Kt)))
plt.bar(['RBF 0.5', 'RBF 2', 'Polynomial 3'], mu)
plt.ylabel('Kernel weight'); plt.show()
xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 100), np.linspace(-1.5, 1.5, 100))
grid = np.c_[xx.ravel(), yy.ravel()]
Kg = sum(wi * kernel(grid, X_train) for wi, kernel in zip(mu, kernels))
plt.contourf(xx, yy, clf.decision_function(Kg).reshape(xx.shape), levels=20, alpha=0.6)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='white')
plt.title('Nonnegative multiple-kernel SVM'); plt.show()
''', '原代码未闭合，且倒数梯度更新不等同于 SimpleMKL。案例改为验证集选择非负核组合权重，明确它是多核学习的教学实现，并非 SimpleMKL 优化器。')

# 以原代码为基础的局部修复，便于追溯改动。
from pathlib import Path
import re

def source_block(key, index):
    chapter, number = key.split('/')
    root = Path(__file__).resolve().parents[2]
    directory = next(root.glob(chapter + '. *'))
    source = next(directory.glob(number + '.*.md')).read_text()
    return re.findall(r'^```Python\n(.*?)^```', source, re.M | re.S)[index].strip()

s = source_block('1/11', 0)
s = s.replace('knn = KNeighborsRegressor()', 'from sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\nknn = make_pipeline(StandardScaler(), KNeighborsRegressor())')
s = s.replace("'n_neighbors'", "'kneighborsregressor__n_neighbors'")
s = s.replace("'param_n_neighbors'", "'param_kneighborsregressor__n_neighbors'")
s = s.replace("].data", "].to_numpy()")
add('1/11', 0, s, 'KNN 的特征标准化放入 Pipeline，避免大尺度特征主导距离，并修复 pandas Series.data 接口。')

s = source_block('1/01', 4)
s = s.replace('X_new = selector.fit_transform(X, y)\nX_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y, test_size=0.2, random_state=42)', 'X_train_new = selector.fit_transform(X_train, y_train)\nX_test_new = selector.transform(X_test)\ny_train_new, y_test_new = y_train, y_test')
add('1/01', 4, s, '监督式特征选择只在训练集拟合，测试集仅应用相同变换。')

s = source_block('2/01', 5)
s = s.replace("'C':", "'logistic__C':").replace("'solver':", "'logistic__solver':")
s = s.replace("grid = GridSearchCV(LogisticRegression(random_state=42, max_iter=1000), param_grid, cv=5, scoring='roc_auc')", "from imblearn.pipeline import Pipeline\ncv_pipeline = Pipeline([('scale', StandardScaler()), ('smote', SMOTE(random_state=42)),\n                        ('logistic', LogisticRegression(random_state=42, max_iter=1000))])\ngrid = GridSearchCV(cv_pipeline, param_grid, cv=5, scoring='roc_auc')")
s = s.replace('grid.fit(X_train_smote, y_train_smote)', 'grid.fit(X_train, y_train)').replace('best_model.fit(X_train_smote, y_train_smote)', 'best_model.fit(X_train, y_train)').replace('best_model.predict(X_test_scaled)', 'best_model.predict(X_test)').replace('best_model.predict_proba(X_test_scaled)', 'best_model.predict_proba(X_test)')
add('2/01', 5, s, 'SMOTE 属于过采样。调参时用 imbalanced-learn Pipeline 在每个训练折内标准化并过采样，防止合成样本泄漏到验证折。')

add('2/02', 2, '''
# 本地字段为 Survived；不使用 PassengerId、姓名、票号等标识列。
# 数值缺失值在 Pipeline 中按每一训练折的中位数填补。
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
df = df.drop(columns=['PassengerId', 'Cabin', 'Name', 'Ticket'])
print(df.head())
''', '统一本地 Survived 字段名；缺失值填补移入 Pipeline，并删除乘客标识字段。')
s=source_block('2/02',3).replace("'survived'", "'Survived'").replace('random_state=42)', 'random_state=42, stratify=y)')
add('2/02',3,s)
s=source_block('2/02',4).replace('clf = DecisionTreeClassifier(random_state=42)', "from sklearn.pipeline import make_pipeline\nfrom sklearn.impute import SimpleImputer\nclf = make_pipeline(SimpleImputer(strategy='median'), DecisionTreeClassifier(max_depth=4, random_state=42))")
add('2/02',4,s)
add('2/02',6,source_block('2/02',6).replace('plot_tree(clf,','plot_tree(clf.named_steps["decisiontreeclassifier"],'))
s=source_block('2/02',7).replace("'max_depth'", "'decisiontreeclassifier__max_depth'").replace("'min_samples_split'", "'decisiontreeclassifier__min_samples_split'").replace("'min_samples_leaf'", "'decisiontreeclassifier__min_samples_leaf'").replace("'criterion'", "'decisiontreeclassifier__criterion'")
s=s.replace('GridSearchCV(DecisionTreeClassifier(random_state=42),','GridSearchCV(clf,')
add('2/02',7,s)
add('2/02',9,source_block('2/02',9).replace('plot_tree(best_clf,','plot_tree(best_clf.named_steps["decisiontreeclassifier"],'))

s=source_block('2/10',1)
s=re.sub(r'# 处理数值型特征缺失值.*?# 预处理数值型特征和分类型特征', '# 缺失值填补与编码都在 Pipeline 内拟合。\nfrom sklearn.impute import SimpleImputer\n# 预处理数值型特征和分类型特征',s,flags=re.S)
s=s.replace("('scaler', StandardScaler())", "('imputer', SimpleImputer(strategy='median')),\n    ('scaler', StandardScaler())")
s=s.replace("('onehot', OneHotEncoder(handle_unknown='ignore'))", "('imputer', SimpleImputer(strategy='constant', fill_value='missing')),\n    ('onehot', OneHotEncoder(handle_unknown='ignore'))")
add('2/10',1,s,'LightGBM 房价案例的缺失值处理移入 Pipeline，训练和预测共用训练集统计量。')
s=source_block('2/10',7)
s=re.sub(r'^feature_names = .*$', 'feature_names = grid_search.best_estimator_.named_steps["preprocessor"].get_feature_names_out()',s,flags=re.M)
add('2/10',7,s)

s=source_block('3/03',0)
start=s.index('# 在交叉验证上进行网格搜索')
end=s.index('# 对测试集进行预测',start)
s=s[:start]+'''# 对完整 StackingRegressor 调参；元特征由内部交叉验证产生。
param_grid = {'final_estimator__alpha': [0.1, 1.0, 10.0]}
grid_search = GridSearchCV(stacking_model, param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
optimized_stacking_model = grid_search.best_estimator_
print('最佳元学习器参数:', grid_search.best_params_)

'''+s[end:]
add('3/03',0,s,'堆叠调参改为对完整 StackingRegressor 交叉验证，避免用基模型对其训练数据的拟合值调优元模型。')

s=source_block('3/09',0)
s=s.replace('                 early_stopping_rounds=20,\n','')
s=s.replace('eval_set=[(X_test, y_test)],','eval_set=[(X_train, y_train)],')
s=s.replace('# 为防止过拟合，加入 early_stopping_rounds 参数，利用验证集监控模型性能', '# 轮数已在训练集交叉验证选定；测试集不参与早停或调参。')
s=s.replace("target_counts = df['target'].value_counts()", "target_counts = df['target'].value_counts().sort_index()")
add('3/09',0,s,'XGBoost 移除旧 fit 早停参数；树数由训练集交叉验证选择，测试集不参与训练。饼图按类别编码顺序排列。')

add('7/05',4, '''
from sklearn.model_selection import GridSearchCV
from sklearn.manifold import trustworthiness
# 在验证数据上调用 transform，评估其局部邻域保真度。
# 原式比较不同样本数的距离矩阵会报错；距离差也不等于 LLE 重建误差。
def validation_trustworthiness(estimator, X_validation):
    embedding = estimator.transform(X_validation)
    return trustworthiness(X_validation, embedding, n_neighbors=5)
param_grid = {'n_neighbors': [12, 18], 'reg': [0.01, 0.05]}
lle = LocallyLinearEmbedding(n_components=2, method='modified', random_state=42)
grid = GridSearchCV(lle, param_grid, cv=3, scoring=validation_trustworthiness,
                    error_score='raise')
grid.fit(X_scaled)
print('Best Parameters:', grid.best_params_)
print('Validation trustworthiness:', grid.best_score_)
''','LLE 调参改为在验证样本的 transform 结果上评估 trustworthiness，修复训练/验证距离矩阵维度不一致的问题。')

# 数据平衡的分步代码补齐导入和占位；选 k 仅使用训练集内部验证集。
imports='''from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, auc
import random
'''
add('11/09',0,imports+'\n'+source_block('11/09',0))
add('11/09',6,'''fig, axs = plt.subplots(1, 3, figsize=(15, 4))
for ax, (Xb, yb, label) in zip(axs, [(X_rus, y_rus, 'Undersampling'), (X_ros, y_ros, 'Oversampling'), (X_smt, y_smt, 'SMOTE')]):
    ax.scatter(Xb[:, 0], Xb[:, 1], c=yb, s=8, alpha=0.4)
    ax.set_title(label)
plt.show()
''')
s=source_block('11/09',7).replace('evaluate_and_plot(...)  # 分别对四种训练集调用', "for Xb, yb, label in [(X_train, y_train, 'Original'), (X_rus, y_rus, 'Undersampling'), (X_ros, y_ros, 'Oversampling'), (X_smt, y_smt, 'SMOTE')]:\n    evaluate_and_plot(Xb, yb, label)")
add('11/09',7,s)
k_search='''# 只在训练数据内部选择 SMOTE 邻居数。
Xa, Xv, ya, yv = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train, random_state=7)
best_auc, best_k = -np.inf, None
for k in [3, 5, 7, 9]:
    X_opt, y_opt = custom_smote(Xa, ya, k=k)
    candidate = LogisticRegression(max_iter=1000).fit(X_opt, y_opt)
    score = auc(*roc_curve(yv, candidate.predict_proba(Xv)[:, 1])[:2])
    print(f'k={k}, validation AUC={score:.4f}')
    if score > best_auc:
        best_auc, best_k = score, k
X_final, y_final = custom_smote(X_train, y_train, k=best_k)
final_model = LogisticRegression(max_iter=1000).fit(X_final, y_final)
print('Selected k:', best_k)
print('Held-out test AUC:', auc(*roc_curve(y_test, final_model.predict_proba(X_test)[:, 1])[:2]))
'''
add('11/09',8,k_search,'补齐分步案例中的导入、占位调用和空图；SMOTE 邻居数改为在训练集内部选取。')
s=source_block('11/09',9)
s=s[:s.index('# 9. 算法优化')]+k_search
add('11/09',9,s)

add('13/05',5, '''
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
# 包装原始特征到核矩阵的转换，避免 SVC 的 pairwise 标签使 CV 误切矩阵。
class LapSVC(ClassifierMixin, BaseEstimator):
    def __init__(self, sigma=1.0, C=1.0):
        self.sigma = sigma
        self.C = C
    def fit(self, X, y):
        self.X_fit_ = np.asarray(X)
        self.estimator_ = SVC(kernel='precomputed', C=self.C)
        self.estimator_.fit(laplacian_kernel(self.X_fit_, self.X_fit_, self.sigma), y)
        self.classes_ = self.estimator_.classes_
        self.n_features_in_ = self.X_fit_.shape[1]
        return self
    def predict(self, X):
        return self.estimator_.predict(laplacian_kernel(np.asarray(X), self.X_fit_, self.sigma))
param_grid = {'sigma': [0.1, 0.5, 1.0, 2.0], 'C': [0.1, 1.0, 10.0, 100.0]}
model = GridSearchCV(LapSVC(), param_grid, cv=5, error_score='raise')
model.fit(X_train, y_train)
''','拉普拉斯核分类器用标准估计器包装，保证交叉验证按原始特征拆分，随后分别构造训练核和验证核。')

add('7/03',0,'from sklearn.decomposition import FastICA\nimport seaborn as sns\n'+source_block('7/03',0))
s=source_block('7/03',4).replace('S_chunk = ica.fit_transform(X_chunk)', 'S_chunk = ica.transform(X_chunk)')
add('7/03',4,s,'ICA 在完整数据上拟合一次后分批 transform；各批独立重新拟合会使分量的顺序、符号和尺度不一致，不能直接拼接。')
s=source_block('7/03',5).replace('return ica.fit_transform(X_chunk)', 'return ica.transform(X_chunk)')
add('7/03',5,s)
add('7/04',0,source_block('7/04',0).replace('from sklearn.decomposition import FactorAnalysis','from sklearn.decomposition import FactorAnalysis, PCA'), '补齐 PCA 导入。模拟数据的各列原本独立，载荷仅用于演示，不能据此认定“学习能力”“社交活跃度”等真实潜因子。')

s=source_block('16/02',0).replace('nonlin = (y ** 3).astype(float) + 0.5 * np.random.randn(n_samples)', 'nonlin = X[:, 0] ** 3 + 0.5 * np.random.randn(n_samples)').replace('# 增加一个非线性特征：y 的三次方加噪声', '# 从已有输入特征构造非线性特征；不使用目标标签构造输入。')
s+='\nX_train_all, X_test_all, y_train_all, y_test_all = train_test_split(df_X, series_y, test_size=0.3, stratify=series_y, random_state=42)\n'
add('16/02',0,s,'非线性输入特征改由原始 X 构造，避免将标签泄漏进特征；互信息筛选仅使用训练集，单特征实现改用公开分类互信息接口。')
add('16/02',1,"mi = mutual_info_classif(X_train_all, y_train_all, discrete_features=False, n_neighbors=3, random_state=42)")
add('16/02',2,source_block('16/02',2).replace('df_X.values, series_y.values','X_train_all.values, y_train_all.values'))
s=source_block('16/02',4)
s=re.sub(r'X_train, X_test, y_train, y_test = train_test_split\(.*?\)\n', 'X_train, X_test = X_train_all[selected_features], X_test_all[selected_features]\ny_train, y_test = y_train_all, y_test_all\n',s,count=1,flags=re.S)
s=s.replace('train_eval(*train_test_split(df_X, series_y, test_size=0.3, random_state=42))','train_eval(X_train_all, X_test_all, y_train_all, y_test_all)')
add('16/02',4,s)
s=source_block('16/02',5)
start=s.index('from sklearn.feature_selection._mutual_info')
end=s.index('# 并行调用',start)
s=s[:start]+'''def mi_single(x, y, n_neighbors=3):
    return mutual_info_classif(x.reshape(-1, 1), y, discrete_features=False,
                               n_neighbors=n_neighbors, random_state=42)[0]

'''+s[end:]
s=s.replace('df_X[col].values, series_y.values','X_train_all[col].values, y_train_all.values')
add('16/02',5,s)
add('16/02',6,source_block('16/02',6).replace('df_X[col].values, series_y.values','X_train_all[col].values, y_train_all.values'))
add('18/09',7,source_block('18/09',7).replace('mean, std','mean, std_dev'))
add('18/09',9,'''fig, axes = plt.subplots(1, 2, figsize=(10, 4))
probplot(data['sqrt_accidents'], dist='norm', plot=axes[0])
sns.histplot(data['sqrt_accidents'], kde=True, ax=axes[1])
plt.tight_layout(); plt.show()
''','补齐复述代码块中的 std_dev 名称和绘图数据参数。')

s=source_block('1/13',3)
s+='''
# 使用独立测试集检查泛化误差；正值预测由对数链接保证。
from sklearn.metrics import mean_absolute_error, mean_squared_error
X_test_const = sm.add_constant(X_test_scaled, has_constant='add')
y_pred = result.predict(X_test_const)
print('Test MAE:', mean_absolute_error(y_test, y_pred))
print('Test RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
assert np.isfinite(y_pred).all() and (y_pred > 0).all()
'''
add('1/13',3,s)

s=source_block('9/06',0).replace('multichannel=True','channel_axis=-1')
add('9/06',0,s,'scikit-image 的 multichannel 参数改为 channel_axis=-1。')

s=source_block('12/01',0).replace("data.set_index('Date', inplace=True)", "data = data.sort_values('Date').set_index('Date')")
add('12/01',0,s,'股票记录按日期排序后再做前后切分；交易日序列按观测步长建模。')
s=source_block('12/03',0)
s=s.replace("pm_data = city_data['PM2.5'].fillna(method='ffill')", "pm_data = city_data.groupby(level=0)['PM2.5'].mean().sort_index().asfreq('D').ffill().dropna()")
add('12/03',0,s,'空气质量先按日期汇总重复记录、排序并补齐日频；仅向前填补缺口，再按时间划分训练和测试。')
s=source_block('12/04',0)
s=s.replace("plt.fill_between(df.index, df['ARIMA_Prediction']*0.9, df['ARIMA_Prediction']*1.1, color='crimson', alpha=0.1)", "prediction_ci = arima_result.get_prediction(start=0, end=len(df)-1).conf_int()\nplt.fill_between(df.index, prediction_ci.iloc[:, 0], prediction_ci.iloc[:, 1], color='crimson', alpha=0.1, label='95% model interval')")
s+='''
# 单独保留末尾 90 天做样本外预测，区别于上面的样本内拟合图。
from sklearn.metrics import mean_squared_error
train_series, test_series = df['Sales'].iloc[:-90], df['Sales'].iloc[-90:]
heldout_model = ARIMA(train_series, order=(5, 1, 2)).fit()
heldout_forecast = heldout_model.forecast(steps=len(test_series))
print('Held-out RMSE:', np.sqrt(mean_squared_error(test_series, heldout_forecast)))
'''
add('12/04',0,s,'原任意的 ±10% 阴影替换为模型预测区间；明确全量拟合图是样本内结果，另外保留末尾 90 天评估样本外误差。')
s=source_block('12/05',0).replace('model = VAR(df)', 'model = VAR(df.diff().dropna())')
s=s.replace('forecast = results.forecast(df.values[-lag_order:], steps=12)', 'forecast_diff = results.forecast(model.endog[-lag_order:], steps=12)\nforecast = df.iloc[-1].to_numpy() + np.cumsum(forecast_diff, axis=0)')
add('12/05',0,s,'模拟原始序列是随机游走；VAR 在一阶差分上拟合，再累加还原到原量纲。数据是合成演示，不能据此解释实际经济因果关系。')
s=source_block('12/06',0).replace('for p in range(1,4):','for p in range(1,3):').replace('for q in range(1,4):','for q in [1]:')
s=s.replace('except:\n            continue', "except (ValueError, np.linalg.LinAlgError) as exc:\n            print(f'Candidate {(p,q)} failed: {exc}')")
s=s.replace("cpi_lower = forecast_ci.iloc[:, 0]", "cpi_lower = forecast_ci['lower CPI']").replace("cpi_upper = forecast_ci.iloc[:, 1]", "cpi_upper = forecast_ci['upper CPI']")
s=s.replace("ir_lower = forecast_ci.iloc[:, 2]", "ir_lower = forecast_ci['lower InterestRate']").replace("ir_upper = forecast_ci.iloc[:, 3]", "ir_upper = forecast_ci['upper InterestRate']")
add('12/06',0,s,'VARMA 采用小型阶数候选集；失败候选显示原因。预测区间按列名读取，避免将两个变量的上下界混用；图中预测量为差分值。')
s=source_block('12/10',0).replace('test_size=0.2, random_state=42)', 'test_size=0.2, shuffle=False)')
add('12/10',0,s,'时间序列按先后顺序切分，修复随机切分后曲线与日期错位的问题。这是滚动一步预测：每个测试日使用当时已观测到的过去 7 天销量；不是一次性预测未来多天。')

s=source_block('8/11',1)
s=s.replace("['Name', 'Ticket', 'Cabin']", "['PassengerId', 'Name', 'Ticket', 'Cabin']")
s=s[:s.index('# 为了模拟大数据集')]+'''# 保留真实的独立乘客，不先重复抽样再切分。
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
# 随机森林不要求特征标准化。
'''
add('8/11',1,s,'删除在切分前重复抽样到 10 万行的步骤，避免同一个乘客同时进入训练集和测试集；森林无需标准化。')
s=source_block('8/11',4).replace("['auto', 'sqrt', 'log2']", "['sqrt', 'log2']").replace('Integer(100, 1000)','Integer(50, 200)').replace('n_iter=30','n_iter=12')
add('8/11',4,s,'更新 max_features 的合法候选，并使用 12 次贝叶斯搜索与 50–200 棵树作为教学规模。')

s=source_block('11/03',0).replace('col.startswith(original_feature)', 'col.startswith(original_feature + "_")').replace("OneHotEncoder(handle_unknown='ignore', sparse=True)", "OneHotEncoder(handle_unknown='ignore', sparse_output=True)")
add('11/03',0,s,'相关性图只选择编码后的数值列，避免把原始字符串列混入；更新稀疏独热编码参数。')
s=source_block('13/01',4).replace('results.pivot("param_C", "param_gamma", "mean_test_score")', 'results.pivot(index="param_C", columns="param_gamma", values="mean_test_score")')
add('13/01',4,s)

s=source_block('2/03',0)
s=s.replace('from sklearn.metrics import accuracy_score','from sklearn.metrics import accuracy_score\nfrom scipy.stats import entropy')
s=s.replace('le = LabelEncoder()\nfor column in df.columns:\n    df[column] = le.fit_transform(df[column])','encoders = {}\nfor column in df.columns:\n    encoders[column] = LabelEncoder()\n    df[column] = encoders[column].fit_transform(df[column])')
s=s.replace('test_data[column] = le.fit_transform(test_data[column])','test_data[column] = encoders[column].transform(test_data[column])')
s=s.replace('H_S = -np.sum((np.bincount(y) / len(y)) * np.log2(np.bincount(y) / len(y)))','H_S = entropy(np.bincount(y), base=2)')
s=s.replace('(-np.sum((np.bincount(subset) / len(subset)) * np.log2(np.bincount(subset) / len(subset))))','entropy(np.bincount(subset), base=2)')
s+='''
# 真正的离散多分支 ID3：按信息增益选择特征，每个类别取值对应一个分支。
def fit_id3(X_data, labels):
    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    majority = classes[np.argmax(counts)]
    if len(classes) == 1 or X_data.shape[1] == 0:
        return int(majority)
    base = entropy(counts, base=2)
    gains = {}
    for feature in X_data.columns:
        conditional = 0.0
        for value in X_data[feature].unique():
            mask = (X_data[feature] == value).to_numpy()
            conditional += mask.mean() * entropy(np.unique(labels[mask], return_counts=True)[1], base=2)
        gains[feature] = base - conditional
    best = max(gains, key=gains.get)
    if gains[best] <= 1e-12:
        return int(majority)
    branches = {}
    for value in X_data[best].unique():
        mask = (X_data[best] == value).to_numpy()
        branches[int(value)] = fit_id3(X_data.loc[mask].drop(columns=best), labels[mask])
    return {'feature': best, 'default': int(majority), 'branches': branches}

def predict_id3(tree, row):
    if not isinstance(tree, dict):
        return tree
    child = tree['branches'].get(int(row[tree['feature']]), tree['default'])
    return predict_id3(child, row)

id3_tree = fit_id3(X, y)
id3_predictions = np.array([predict_id3(id3_tree, row) for _, row in test_data.iterrows()])
print('ID3 multiway tree:', id3_tree)
print('ID3 predictions:', encoders['PlayTennis'].inverse_transform(id3_predictions))
assert np.isfinite(info_gain).all()
'''
add('2/03',0,s,'信息熵正确处理 0·log(0)，训练与预测复用同一编码。scikit-learn 的 entropy 决策树是二叉 CART；末尾另附离散多分支 ID3 实现，明确两者区别。')

s=source_block('1/02',0)
s=s.replace('from sklearn.model_selection import GridSearchCV', 'from sklearn.model_selection import GridSearchCV\nfrom sklearn.pipeline import Pipeline')
s=s.replace('scaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)', '# 缩放将在每个交叉验证训练折内拟合。')
s=s.replace('train_test_split(X_scaled, y,','train_test_split(X, y,')
s=s.replace('alphas = np.logspace(-4, 4, 50)', 'alphas = np.logspace(-4, 4, 20)')
s=s.replace('ridge = Ridge()', "ridge = Pipeline([('scale', StandardScaler()), ('model', Ridge())])")
s=s.replace("'alpha'", "'model__alpha'").replace("'l1_ratio'", "'model__l1_ratio'")
s=s.replace('best_ridge.coef_', "best_ridge.named_steps['model'].coef_")
s=s.replace('estimator=Lasso()', "estimator=Pipeline([('scale', StandardScaler()), ('model', Lasso(max_iter=10000))])")
s=s.replace('estimator=ElasticNet()', "estimator=Pipeline([('scale', StandardScaler()), ('model', ElasticNet(max_iter=10000))])")
s=s.replace('np.linspace(0, 1, 10)', 'np.linspace(0.1, 1, 4)')
add('1/02',0,s,'Ridge/Lasso/ElasticNet 比较的标准化统一移入 Pipeline；ElasticNet 的 l1_ratio 不取 0，纯 L2 的情况由 Ridge 演示，避免坐标下降退化。')

for i in range(9):
    add('20/01',i,source_block('20/01',i).replace("'GrLivArea'", "'Gr Liv Area'"))
NOTES.setdefault('20/01',[]).append('统一 AmesHousing 中实际存在的 Gr Liv Area 字段。')
add('20/06',6,"sns.histplot(df['price'], kde=True)\nplt.show()")
add('20/06',7,"sns.scatterplot(x=np.arange(len(df)), y=df['z_score'], hue=df['is_outlier'])\nplt.show()")
add('19/01',4,'''plt.figure(figsize=(10, 5))
plt.hist(data, bins=30, density=True, alpha=0.5, label='Observed')
plt.plot(x_vals, expon.pdf(x_vals, scale=1/lambda_mle), label='MLE fitted density')
plt.legend(); plt.show()
''')
add('19/01',5,"print(f'MLE lambda = {lambda_mle:.5f}; closed-form = {1 / np.mean(data):.5f}')\nassert np.isclose(lambda_mle, 1 / np.mean(data), rtol=1e-4)")
s=source_block('19/03',0).replace('SSR[0]','SSR').replace('R_squared[0]','R_squared').replace('beta_hat = np.linalg.inv(XTX) @ XTy','beta_hat = np.linalg.lstsq(X_design, y_vec, rcond=None)[0]')
add('19/03',0,s,'修复标量下标；实际求解用数值更稳定的 lstsq，正规方程保留用于推导对照。')
add('17/09',2,'X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)')
s=source_block('17/04',0).replace('precision_score(y_test, y_pred)', 'precision_score(y_test, y_pred, pos_label=0)').replace('classification_report(y_test, y_pred, output_dict=True)', 'classification_report(y_test, y_pred, target_names=data.target_names, output_dict=True)')
add('17/04',0,s,'乳腺癌数据中 0 表示恶性；精确率使用 pos_label=0，报告键显式使用类别名称。')

add('1/13',4,'''plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x='HouseStyle', y='SalePrice', hue='CentralAir')
plt.title('Ames: house style, central air and observed sale price')
plt.xlabel('House style'); plt.ylabel('Sale price (USD)')
plt.xticks(rotation=30); plt.tight_layout(); plt.show()
''')
add('1/13',5,'''plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='GrLivArea', y='SalePrice', hue='CentralAir', alpha=0.35, s=15)
# 描述性分箱均值，不将逐点连线误称为拟合模型。
bins = pd.qcut(df['GrLivArea'], q=15, duplicates='drop')
observed_means = df.groupby(bins, observed=True)[['GrLivArea', 'SalePrice']].mean()
plt.plot(observed_means['GrLivArea'], observed_means['SalePrice'], color='black', label='Observed bin means')
plt.title('居住面积与房价')
plt.xlabel('Above-grade living area (sq ft)'); plt.ylabel('Sale price (USD)')
plt.legend(title='Central air / summary'); plt.tight_layout(); plt.show()
''')

s=source_block('15/01',0)
s=s[:s.index('# 将分类变量转为哑变量')]+'''# 先划分；填补、编码和缩放都在训练折内部拟合。
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
'''
add('15/01',0,s,'缺失值、编码和标准化放入 Pipeline；调参使用更适合目标量纲的 alpha 范围。Lasso 路径使用 warm_start，并仅展示系数幅度最大的 12 个特征。')
add('15/01',1,'''from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
numeric_columns = X.select_dtypes(include='number').columns
categorical_columns = X.select_dtypes(exclude='number').columns
preprocessor = ColumnTransformer([
    ('numeric', Pipeline([('fill', SimpleImputer(strategy='median')),
                          ('scale', StandardScaler())]), numeric_columns),
    ('categorical', Pipeline([('fill', SimpleImputer(strategy='constant', fill_value='missing')),
                              ('encode', OneHotEncoder(handle_unknown='ignore'))]), categorical_columns)
])
''')
add('15/01',2,'''from sklearn.base import clone
estimators = {'Ridge': Ridge(), 'Lasso': Lasso(max_iter=50000),
              'ElasticNet': ElasticNet(max_iter=50000)}
models, results = {}, {}
param_grid = {'model__alpha': np.logspace(1, 4, 10)}
for name, estimator in estimators.items():
    pipeline = Pipeline([('preprocessor', clone(preprocessor)), ('model', estimator)])
    search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error',
                          error_score='raise')
    search.fit(X_train, y_train)
    model = search.best_estimator_
    models[name] = model
    y_train_pred, y_test_pred = model.predict(X_train), model.predict(X_test)
    results[name] = {'Train MSE': mean_squared_error(y_train, y_train_pred),
                     'Test MSE': mean_squared_error(y_test, y_test_pred),
                     'Train R2': r2_score(y_train, y_train_pred),
                     'Test R2': r2_score(y_test, y_test_pred),
                     'Best Alpha': search.best_params_['model__alpha']}
ridge_best, lasso_best, elastic_net_best = models['Ridge'], models['Lasso'], models['ElasticNet']
results_df = pd.DataFrame(results).T
print(results_df)
''')
add('15/01',4,'''# 在同一训练集上展示正则化路径；从强正则化走向弱正则化。
fitted_preprocessor = lasso_best.named_steps['preprocessor']
X_path = fitted_preprocessor.transform(X_train)
feature_names = fitted_preprocessor.get_feature_names_out()
alphas = np.logspace(4, 1, 25)
path_model = Lasso(max_iter=50000, warm_start=True)
coefs = []
for alpha in alphas:
    path_model.set_params(alpha=alpha)
    path_model.fit(X_path, y_train)
    coefs.append(path_model.coef_.copy())
coefs = np.asarray(coefs)
selected = np.argsort(np.max(np.abs(coefs), axis=0))[-12:]
plt.figure(figsize=(12, 6))
for j in selected:
    plt.plot(alphas, coefs[:, j], label=feature_names[j])
plt.xscale('log'); plt.xlabel('Alpha'); plt.ylabel('Coefficient')
plt.title('Lasso paths: 12 largest coefficients')
plt.legend(fontsize=8, ncol=2); plt.tight_layout(); plt.show()
''')
add('15/02',1,source_block('15/02',1).replace("('regressor', ElasticNet())", "('regressor', ElasticNet(max_iter=20000))"))
add('15/02',2,source_block('15/02',2).replace('np.logspace(-4, 0, 50)','np.logspace(-4, 0, 12)').replace('np.linspace(0, 1, 10)','np.linspace(0.1, 1, 4)'), 'ElasticNet 搜索去掉 l1_ratio=0 的退化边界，增加迭代上限并缩小用于热力图的候选网格。')

add('11/04',2,'''X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
''','先保留独立测试集，再进行标准化及监督特征选择；特征集合的比较使用同一个测试集。Lasso 的超参数搜索也在完整预处理 Pipeline 内执行。')
for i in [3,4]:add('11/04',i,source_block('11/04',i).replace('fit_transform(X_scaled, y)', 'fit_transform(X_scaled, y_train)'))
s=source_block('11/04',5)
s=s.replace('lasso = LassoCV(cv=5, random_state=0)\nlasso.fit(X_scaled, y)', '''from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
lasso_search = GridSearchCV(make_pipeline(StandardScaler(), Lasso(max_iter=20000)),
                            {'lasso__alpha': np.logspace(-3, -1, 12)},
                            cv=5, scoring='neg_mean_squared_error', error_score='raise')
lasso_search.fit(X_train, y_train)
lasso = lasso_search.best_estimator_.named_steps['lasso']''')
add('11/04',5,s)
s=source_block('11/04',7)
s=re.sub(r'    X_train, X_test, y_train, y_test = train_test_split\(.*?\)\n', '    columns = [X.columns.get_loc(f) for f in features]\n    X_subset_train, X_subset_test = X_scaled[:, columns], X_test_scaled[:, columns]\n',s,flags=re.S)
s=s.replace('model.fit(X_train, y_train)','model.fit(X_subset_train, y_train)').replace('model.predict(X_test)','model.predict(X_subset_test)').replace('model.predict_proba(X_test)','model.predict_proba(X_subset_test)')
add('11/04',7,s)

s=OVERRIDES[('12/06',0)].replace('model.fit(disp=False)', 'model.fit(disp=False, maxiter=500)')
s=s.replace('if results.aic < best_aic:', "if results.mle_retvals.get('converged', False) and np.isfinite(results.aic) and results.aic < best_aic:")
s=s.replace("print(f'Best VARMA order by AIC: {best_order}, AIC={best_aic:.2f}')", "assert best_order is not None, '没有收敛的候选模型，请检查数据和阶数。'\nprint(f'Best VARMA order by AIC: {best_order}, AIC={best_aic:.2f}')")
s=s.replace('print(results.summary())', "assert results.mle_retvals.get('converged', False), '最终模型未收敛。'\nprint(results.summary())")
add('12/06',0,s,'VARMA 增加迭代上限，只比较已经收敛的候选模型，并显式检查最终模型的收敛状态。')
