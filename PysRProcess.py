from pysr import PySRRegressor
import numpy as np
from collections import defaultdict
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

SYMBOLIC_LOSS_THRESHOLD = 1e-4


def symbolic_regression(target_var: str, datadictionary,
                        loss_threshold: float = SYMBOLIC_LOSS_THRESHOLD):
    data_dict = defaultdict(list)
    for data in datadictionary:
        # Time/timestamp columns are not valid symbolic-regression features.
        # The previous string comparison against '<M8[ns]' missed pandas'
        # Timestamp dtype and caused Sorting/Num_Blue to fail on X conversion.
        if (data.dtype != 'bool' and data.dtype != 'char'
                and not is_datetime64_any_dtype(data.dtype)):
            data_dict[data.name] = data.body

    target_data = data_dict[target_var]
    
    change_indices = [i for i in range(1, len(target_data)) if target_data[i] != target_data[i-1]]
    if not change_indices:
        return None, float("inf"), change_indices
    
    # 因变量 y：变化后的值
    y = np.array([target_data[i] for i in change_indices])
    
    # 3. 构造自变量 X
    X = []
    feature_names = []
    
    for idx in change_indices:
        row = []
        
        # (a) 目标变量变化前的值
        row.append(target_data[idx-1])
        
        # (b) 其他变量在 idx 及 idx-1 的值
        for var, values in data_dict.items():
            if var == target_var:
                continue
            row.append(values[idx])      # 当前值
            # row.append(values[idx-1])    # 前一个值
        
        X.append(row)
    
    X = np.array(X)
    
    # 构造特征名（便于输出表达式）
    feature_names = [f"{target_var}_prev"]
    for var in data_dict.keys():
        if var == target_var:
            continue
        feature_names.append(f"{var}_curr")
        # feature_names.append(f"{var}_prev")
    
    # 4. 使用 PySR 进行符号回归
    model = PySRRegressor(
        niterations=10,              # 迭代次数，可调
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["square", "exp", "log"],
        # Final selection is performed explicitly from model.equations_.
        model_selection="accuracy",
    )
    print(X)
    print(y)
    model.fit(X, y, variable_names=feature_names)
    
    # 5. Explicit accuracy-constrained minimum-complexity selection.
    # PySR returns a loss-complexity Pareto table. Retain candidates that
    # satisfy the symbolic-state loss threshold and choose the least complex
    # expression; loss resolves ties of equal complexity. If none qualify,
    # the caller retains the output as discrete observed states.
    candidates = model.equations_
    accepted = candidates[candidates["loss"] <= loss_threshold]
    if accepted.empty:
        return None, float("inf"), change_indices
    selected = accepted.sort_values(
        by=["complexity", "loss"], ascending=[True, True]
    ).iloc[0]
    best_eq = str(selected["equation"])
    best_loss = float(selected["loss"])

    if 'prev' in best_eq:
        best_eq = best_eq.replace('_prev', '')
    if 'curr' in best_eq:
        best_eq = best_eq.replace('_curr', '')
    best_eq = target_var + " = " + best_eq
    return best_eq, best_loss, change_indices
