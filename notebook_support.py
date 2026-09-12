"""可随整个文件夹迁移的数据读取工具。这里没有替换或修改第三方库的行为。"""
from pathlib import Path
import gzip
import struct
import numpy as np
import pandas as pd
from sklearn.utils import Bunch

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'dataset'


def load_local_housing(*, as_frame=False, return_X_y=False, sample_size=2500):
    """从 housing.csv 构造 sklearn 加州房价的 8 个特征；目标单位为十万美元。

    本地 CSV 有缺失的 total_bedrooms，故删除这些行，不用全数据统计量填补。
    默认固定抽样 2500 行以方便教学；sample_size=None 可使用全部有效行。
    """
    raw = pd.read_csv(DATA_DIR / 'housing.csv').dropna(subset=['total_bedrooms'])
    if sample_size is not None and len(raw) > sample_size:
        raw = raw.sample(sample_size, random_state=42)
    frame = pd.DataFrame({
        'MedInc': raw.median_income, 'HouseAge': raw.housing_median_age,
        'AveRooms': raw.total_rooms / raw.households,
        'AveBedrms': raw.total_bedrooms / raw.households,
        'Population': raw.population, 'AveOccup': raw.population / raw.households,
        'Latitude': raw.latitude, 'Longitude': raw.longitude,
    }).reset_index(drop=True)
    target = (raw.median_house_value / 100000).reset_index(drop=True).rename('MedHouseVal')
    X, y = (frame, target) if as_frame else (frame.to_numpy(), target.to_numpy())
    if return_X_y:
        return X, y
    return Bunch(data=X, target=y, feature_names=frame.columns.tolist(),
                 target_names=['MedHouseVal'], frame=frame.assign(MedHouseVal=target),
                 DESCR='本地加州房价；删除缺失行；固定教学样本；目标单位为十万美元。')


def load_local_boston(*, return_X_y=False):
    """读取原案例对应的历史 Boston CSV，避免调用已经删除的 load_boston。"""
    frame = pd.read_csv(DATA_DIR / 'BostonHousing.csv')
    X = frame.drop(columns='medv')
    y = frame.medv.to_numpy()
    if return_X_y:
        return X.to_numpy(), y
    return Bunch(data=X.to_numpy(), target=y,
                 feature_names=X.columns.str.upper().tolist())


def load_local_mnist(*, sample_size=2000):
    """直接解析本地 IDX gzip；按类别分层抽样用于传统降维实验。"""
    base = DATA_DIR / 'MNIST' / 'raw'
    with gzip.open(base / 'train-images-idx3-ubyte.gz', 'rb') as f:
        magic, n, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic == 2051
        X = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows * cols)
    with gzip.open(base / 'train-labels-idx1-ubyte.gz', 'rb') as f:
        magic, count = struct.unpack('>II', f.read(8))
        assert magic == 2049 and count == n
        y = np.frombuffer(f.read(), dtype=np.uint8)
    if sample_size is not None and sample_size < n:
        from sklearn.model_selection import train_test_split
        idx, _ = train_test_split(np.arange(n), train_size=sample_size,
                                  stratify=y, random_state=42)
        X, y = X[idx], y[idx]
    return Bunch(data=X.astype(np.float32) / 255, target=y)


def load_creditcard(*, sample_size=30000):
    """保留原始类别比例的固定分层教学样本；None 表示读取全部交易。"""
    df = pd.read_csv(DATA_DIR / 'creditcard.csv')
    if sample_size is not None and len(df) > sample_size:
        from sklearn.model_selection import train_test_split
        df, _ = train_test_split(df, train_size=sample_size, stratify=df.Class,
                                  random_state=42)
    return df.reset_index(drop=True)
