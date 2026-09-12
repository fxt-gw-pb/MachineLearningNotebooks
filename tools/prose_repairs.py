"""同步改写案例说明，避免保留已替换实现的陈旧解释。"""
import re

CASES = {
 '1/12': ('## Python案例', '## 应用场景', '''本例对带噪正弦曲线做高斯过程回归。先定义 RBF 核与观测噪声核，再通过对数边际似然优化核参数。图中同时显示真实函数、后验均值以及包含观测噪声的近似 95% 预测区间。尝试改变训练点数量与噪声大小，观察区间如何变化。'''),
 '1/13': ('## 完整案例', '## 模型分析', '''本例使用本地 AmesHousing.csv 预测房屋成交价 SalePrice。输入为居住面积 GrLivArea、质量评分 OverallQual、房屋类型 HouseStyle 和中央空调 CentralAir。目标为正值，因此用 Gamma 分布与对数链接作为演示模型；是否适合真实数据仍须检查残差和验证误差。

依次读取数据、独热编码、划分训练集与测试集、在训练集上标准化、拟合 GLM。随后查看系数摘要、测试集误差与数据分布。分组图显示观测差异，不支持因果结论。'''),
 '5/06': ('## 完整案例', '## 模型分析', r'''本例生成 A、B 两组独立的模拟评分，每组 100 个观测，演示贝叶斯均值比较。数据和先验均为教学设定。

分别为两组均值设置正态先验，为标准差设置半正态先验。使用 PyMC 5 运行两条 NUTS 链，每条预热 1000 次并保留 1000 个后验样本；通过 ArviZ 的轨迹、有效样本量和 $\hat{R}$ 检查采样。

最后计算 $P(\mu_A>\mu_B\mid D)$，即后验样本中 A 组均值大于 B 组均值的比例。这个量位于 0 到 1 之间，**不是贝叶斯因子**。贝叶斯因子需要比较两个明确假设下的边际似然；本例没有执行该计算。可以更改先验和样本量，观察后验概率的敏感性。'''),
 '8/07': ('## 完整案例', '## 应用场景', '''本例用 NumPy 手写 Adam 来拟合线性回归。每一步计算均方误差梯度，更新梯度的一阶矩和二阶矩，进行偏差修正，再更新权重。特征标准化仅在训练集上拟合。

训练结束后绘制训练损失，并将独立测试集 MSE 与最小二乘解比较。尝试修改学习率、beta1 和 beta2，观察收敛速度及稳定性。原案例中的神经网络训练按范围要求移除。'''),
 '8/13': ('## Python案例', None, '''本例使用乳腺癌小数据集，通过网格搜索选择 RBF SVM 的 C 与 gamma。先保留独立测试集，再在训练集内部做分层交叉验证；StandardScaler 放入 Pipeline，保证每折分别拟合。

热力图显示各参数组合的交叉验证 F1。最佳模型在训练集重新拟合后，才在独立测试集上计算分类报告和混淆矩阵。测试集不参与选参。'''),
 '11/08': ('## 完整案例', '## 模型分析', '''本例首先使用本地 lenna.jpg 展示水平翻转、旋转与亮度调整。图像变换是否保留标签取决于任务，例如数字翻转可能改变类别，不能机械地应用所有增强。

随后在模拟表格分类数据上演示小幅高斯噪声增强。先划分数据，并只用训练集拟合标准化器；噪声样本只添加到训练集。原始训练集和增强训练集各训练一个逻辑回归模型，在相同的原始测试集上比较准确率。数据增强不保证提高性能。'''),
 '13/08': ('## 完整案例', '## 模型分析', '''本例在环形数据上组合两个 RBF 核和一个多项式核，所有权重非负且和为 1。先在训练集内部划分验证集，搜索一个小型权重网格，选择验证准确率最高的组合；随后在完整训练集重新拟合预计算核 SVM，最后评估独立测试集。

这是展示“学习核组合权重”的简化实现。它使用验证集网格搜索，不是前面介绍的 SimpleMKL 投影梯度优化器。图表给出选中的权重与对应分类边界。'''),
}


def repair_prose(cells, key):
    if key in CASES:
        start, end, explanation = CASES[key]
        active=False;done=False
        for c in cells:
            if c.cell_type=='markdown':
                s=c.source
                if not active and not done and start in s:
                    before=s.split(start,1)[0]
                    c.source=before+start+'\n\n'+explanation
                    active=True
                elif active:
                    if end and end in s:
                        c.source=end+s.split(end,1)[1];active=False;done=True
                    else:c.source=''
            elif c.source.startswith('# 深度学习训练步骤已'):
                c.source=''
    for c in cells:
        if c.cell_type!='markdown':continue
        s=c.source
        s=s.replace('`fetch_california_housing()`','`load_local_housing()`').replace('`fetch_california_housing`','`load_local_housing`')
        s=s.replace('`sklearn.datasets.fetch_california_housing()`','`load_local_housing()`（本地 CSV）')
        s=s.replace('共 20640 条样本，8 个特征','原数据约 20640 行、8 个特征；本例删除缺失行后固定抽样 2500 行')
        s=s.replace('使用 `fetch_openml` 从 OpenML 加载 MNIST 数据集','使用 `load_local_mnist()` 从本地 IDX 文件读取 MNIST 教学样本')
        s=s.replace('使用`fetch_openml`从OpenML加载MNIST数据集','使用 `load_local_mnist()` 从本地 IDX 文件读取 MNIST 教学样本')
        if key=='1/01':
            s=s.replace('y = 2x + 0', 'y = x + 100').replace('每平方米的大小对应的价格是2万，而且即使房子大小为零，起价也是0万', '面积每增加一平方米，价格增加1万，截距为100万')
        if key=='2/01':s=s.replace('SMOTE 进行欠采样','SMOTE 进行过采样')
        if key=='7/11':s=s.replace('BERT','LSA（TruncatedSVD）')
        c.source=s
    return [c for c in cells if c.source.strip()]
