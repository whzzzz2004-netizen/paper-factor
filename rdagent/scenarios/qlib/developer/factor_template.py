"""
正确的因子计算代码模板
这个模板展示了如何处理 Qlib 的 MultiIndex 数据格式
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path


DATA_DIR = Path(os.environ.get("FACTOR_DATA_DIR") or os.environ.get("RDAGENT_FACTOR_DATA_DIR") or ".")


def calculate_volume_imbalance():
    """
    成交量失衡因子 - 当日成交量与过去20日平均成交量的差值
    公式: Vt - AVG(Vt-n) 其中 n=20
    """
    # 1. 读取数据
    # 可以按任务选择不同粒度的数据文件，例如：
    # - daily_pv.h5
    # - minute_pv.h5
    df = pd.read_hdf(DATA_DIR / "daily_pv.h5", key="data")
    
    # 2. 确保成交量是数值类型（注意列名可能是 'volume' 或 '$volume'）
    # 根据数据描述，列名是 '$volume'
    volume = df['$volume']
    
    # 3. 处理异常值（转换为数值类型，错误值变为 NaN）
    volume = pd.to_numeric(volume, errors='coerce')
    
    # 4. 按 instrument 分组计算 20 日移动平均
    # 使用 rolling(window=20, min_periods=1) 保持索引对齐
    avg_volume = volume.groupby(level='instrument').transform(
        lambda x: x.rolling(window=20, min_periods=1).mean()
    )
    
    # 5. 计算成交量失衡因子
    volume_imbalance = volume - avg_volume
    
    # 6. 处理无穷值
    volume_imbalance = volume_imbalance.replace([np.inf, -np.inf], np.nan)
    
    # 7. 删除 NaN 值（可选，根据需求保留或删除）
    volume_imbalance = volume_imbalance.dropna()
    
    # 8. 转换为 DataFrame 并设置列名
    result_df = volume_imbalance.to_frame(name='volume_imbalance')
    
    # 9. 保存结果
    result_df.to_hdf("result.h5", key="data")
    
    return result_df


def calculate_volume_imbalance_alternative():
    """
    成交量失衡因子的另一种写法 - 使用 rolling + reset_index
    """
    df = pd.read_hdf(DATA_DIR / "daily_pv.h5", key="data")
    
    # 获取成交量列
    volume = pd.to_numeric(df['$volume'], errors='coerce')
    
    # 按 instrument 分组计算滚动平均
    avg_volume = volume.groupby(level='instrument').rolling(window=20, min_periods=1).mean()
    
    # 重置索引，去掉 groupby 添加的额外索引层
    avg_volume = avg_volume.reset_index(level=0, drop=True)
    
    # 计算失衡值
    volume_imbalance = volume - avg_volume
    
    # 处理异常值
    volume_imbalance = volume_imbalance.replace([np.inf, -np.inf], np.nan)
    volume_imbalance = volume_imbalance.dropna()
    
    # 保存结果
    result_df = volume_imbalance.to_frame(name='volume_imbalance')
    result_df.to_hdf("result.h5", key="data")
    
    return result_df


def calculate_volume_imbalance_simple():
    """
    成交量失衡因子的简单写法 - 直接使用 rolling + mean
    注意：这种写法需要确保索引正确对齐
    """
    df = pd.read_hdf(DATA_DIR / "daily_pv.h5", key="data")
    
    # 计算成交量
    volume = pd.to_numeric(df['$volume'], errors='coerce')
    
    # 计算 20 日均量
    avg_volume = volume.groupby(level='instrument').rolling(window=20).mean()
    
    # 重置索引以匹配原 DataFrame
    avg_volume = avg_volume.droplevel(0)
    
    # 计算失衡因子
    volume_imbalance = volume - avg_volume
    
    # 清理数据
    volume_imbalance = volume_imbalance.replace([np.inf, -np.inf], np.nan)
    volume_imbalance = volume_imbalance.dropna()
    
    # 保存
    result_df = volume_imbalance.to_frame(name='volume_imbalance')
    result_df.to_hdf("result.h5", key="data")
    
    return result_df


if __name__ == '__main__':
    # 测试模板
    calculate_volume_imbalance()
