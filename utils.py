import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

def simple_returns(prices: ArrayLike, *, axis: int | None = None) -> np.ndarray:
    """
    Calculate simple returns.

    Parameters
    ----------

    prices: array_like
        An array of prices.
    axis: int or None, optional
        Axis along which to calculate the returns. If None, the array
        is flattened before calculations.

    Returns
    -------
    simple_returns: ndarray
        Array of simple returns, with almost the same shape as prices,
        except for the selected axis, where there is one less element.
    """
    prices = np.asarray(prices)
    n = prices.shape[axis] if axis is not None else prices.size()
    
    return prices.take(range(1, n),axis=axis)/prices.take(range(0, n-1), axis=axis)-1.0

def simple_returns_dataframe(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate simple returns for a DataFrame

    Parameters
    ----------

    prices: DataFrame
        A DataFrame of prices.

    Returns
    -------
    simple_returns: DataFrame
        DataFrame of simple returns, with the first day removed
        compared to prices.
    """
    index = prices.index[1:]
    columns = prices.columns
    returns = simple_returns(prices, axis=0)
    returns = pd.DataFrame(returns, index=index, columns=columns)
    return returns

def log_returns(prices: ArrayLike, *, axis: int | None = None) -> np.ndarray:
    """
    Calculate log returns.

    Parameters
    ----------

    prices: array_like
        An array of prices.
    axis: int or None, optional
        Axis along which to calculate the returns. If None, the array
        is flattened before calculations.

    Returns
    -------
    log_returns: ndarray
        Array of log returns, with almost the same shape as prices,
        except for the selected axis, where there is one less element.
    """
    prices = np.asarray(prices)
    n = prices.shape[axis] if axis is not None else prices.size()
    
    return np.log(prices.take(range(1, n),axis=axis)/prices.take(range(0, n-1), axis=axis))

def log_returns_dataframe(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate log returns for a DataFrame

    Parameters
    ----------

    prices: DataFrame
        A DataFrame of prices.

    Returns
    -------
    log_returns: DataFrame
        DataFrame of log returns, with the first day removed
        compared to prices.
    """
    index = prices.index[1:]
    columns = prices.columns
    returns = simple_returns(prices, axis=0)
    returns = pd.DataFrame(returns, index=index, columns=columns)
    return returns
