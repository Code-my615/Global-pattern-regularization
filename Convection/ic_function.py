import numpy as np

def function(u0: str):
    """Initial condition, string --> function."""

    # if u0 == 'sin(x)':
    #     u0 = lambda x: np.sin(x)
    # return u0
    # 将字符串转化为可执行的函数
    try:
        return eval(f"lambda x: {u0}", {"sin": np.sin})
    except Exception as e:
        raise ValueError(f"Error parsing function string: {e}")