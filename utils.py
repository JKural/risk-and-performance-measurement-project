import numpy as np
import pandas as pd
import datetime as dt
import scipy as sp
from numpy.typing import ArrayLike

def simple_returns(
        prices: ArrayLike | pd.Series | pd.DataFrame,
        axis: int = None) -> ArrayLike | pd.Series | pd.DataFrame:
    """
    Calculate simple returns.

    Parameters
    ----------

    prices: ArrayLike | pd.Series | pd.DataFrame
        An array of prices.
    axis: int or None, optional
        Axis along which to calculate the returns.

    Returns
    -------
    simple_returns: ArrayLike | pd.Series | pd.DataFrame
        Array of simple returns, with almost the same shape as prices,
        except for the selected axis, where there is one less element.
    """
    if isinstance(prices, pd.Series):
        if axis is not None:
            raise ValueError("For pd.Series input axis has to be None")
        return prices.pct_change().dropna()
    if isinstance(prices, pd.DataFrame):
        if axis is None:
            axis = 0
        return prices.pct_change().dropna()
    prices = np.asarray(prices)
    n = np.size(prices, axis)
    return np.diff(prices, axis=axis) / prices.take(range(0, n-1), axis=axis)

def _value_at_risk(
        x: ArrayLike,
        alpha: float | ArrayLike,
        axis: int | None) -> np.ndarray:
    """Calculate VaR without respecting x's structure
    """
    x = np.sort(x, axis=axis)
    # if axis is None, x has been flattened and we can set axis = 0
    if axis is None:
        axis = 0
    if axis < 0:
        axis += x.ndim
    assert axis in range(0, x.ndim)
    alpha = np.asarray(alpha)
    n = np.size(x, axis=axis)
    k = np.int64(alpha*n)
    lower_quantiles = x.take(np.maximum(k-1,0), axis=axis)
    upper_quantiles = x.take(k, axis=axis)
    # move axis, so that the shape of alpha is in the front
    lower_quantiles = np.moveaxis(
        lower_quantiles,
        axis+np.arange(k.ndim),
        np.arange(k.ndim))
    upper_quantiles = np.moveaxis(
        upper_quantiles,
        axis+np.arange(k.ndim),
        np.arange(k.ndim))
    # expand dims at the back, so that broadcasting works correctly
    # with quantiles, that is initial dimensions of alpha and k
    # match initial dimensions of lower_quantiles and upper_quantiles
    alpha = np.expand_dims(alpha, axis=list(range(1-x.ndim, 0)))
    k = np.expand_dims(k, axis=list(range(1-x.ndim, 0)))
    t = alpha*n-k
    return -((1-t)*lower_quantiles + t*upper_quantiles)
    # almost equivalent:
    # x = np.asarray(x)
    # return -np.quantile(x, alpha, axis=axis, method='interpolated_inverted_cdf')

def value_at_risk(
        x: ArrayLike | pd.Series | pd.DataFrame,
        alpha: float | ArrayLike,
        axis: int | None = None) -> ArrayLike | pd.Series | pd.DataFrame:
    """
    Calculate VaR of a sample with a given threshold
    
    Parameters
    ----------

    x: ArrayLike | pd.Series | pd.DataFrame
        An array of P&Ls
    alpha: float | ArrayLike
        A parameter (or an array of parameters) between 0.0 and 1.0
    axis: int or None, optional
        Axis along which to calculate VaR. If None, the array
        is flattened before calculations
    
    Returns
    -------

    VaR: np.ndarray
        Estimated VaR with treshold alpha of sample x
    """
    if isinstance(x, pd.Series):
        if axis is not None:
            raise ValueError("For pd.Series input axis has to be None")
        try:
            return pd.Series(_value_at_risk(x, alpha), name=name, index=alpha)
        except TypeError:
            return _value_at_risk(x, alpha, axis=0)
    if isinstance(x, pd.DataFrame):
        try:
            if axis is None:
                axis = 0
            if axis == 0:
                return pd.DataFrame(
                    _value_at_risk(x, alpha, axis=axis),
                    index=alpha,
                    columns=x.columns)
            elif axis == 1:
                return pd.DataFrame(
                    _value_at_risk(x, alpha, axis=axis),
                    index=x.index,
                    columns=alpha)
            else:
                raise ValueError("For pd.Dataframe input axis has to be 0, 1 or None")
        except TypeError:
            return pd.Series(
                _value_at_risk(x, alpha, axis=0),
                index=x.columns)
    return _value_at_risk(x, alpha, axis=axis)

def _expected_shortfall(
        x: ArrayLike,
        alpha: float | ArrayLike,
        axis: int | None) -> np.ndarray:
    """Calculate ES without respecting x's structure
    """
    x = np.sort(x, axis=axis)
    # if axis is None, x has been flattened and we can set axis = 0
    if axis is None:
        axis = 0
    if axis < 0:
        axis += x.ndim
    assert axis in range(0, x.ndim)
    alpha = np.asarray(alpha)
    n = np.size(x, axis=axis)
    k = np.int64(alpha*n)
    cumsums = np.cumsum(x, axis=axis)
    sums = np.where(np.expand_dims(k, axis=list(range(1-x.ndim, 0))) > 0, cumsums.take(k-1, axis=axis), 0)
    upper_quantiles = x.take(k, axis=axis)
    # move axis, so that the shape of alpha is in the front
    sums = np.moveaxis(
        sums,
        axis+np.arange(k.ndim),
        np.arange(k.ndim))
    upper_quantiles = np.moveaxis(
        upper_quantiles,
        axis+np.arange(k.ndim),
        np.arange(k.ndim))
    # expand dims at the back, so that broadcasting works correctly
    # with quantiles, that is initial dimensions of alpha and k
    # match initial dimensions of lower_quantiles and upper_quantiles
    alpha = np.expand_dims(alpha, axis=list(range(1-x.ndim, 0)))
    k = np.expand_dims(k, axis=list(range(1-x.ndim, 0)))
    t = alpha*n-k
    return -1/(alpha*n)*(sums + t*upper_quantiles)

def expected_shortfall(
        x: ArrayLike | pd.Series | pd.DataFrame,
        alpha: float | ArrayLike,
        axis: int | None = None) -> ArrayLike | pd.Series | pd.DataFrame:
    """
    Calculate ES of a sample with a given treshold

    Parameters
    ----------

    x: ArrayLike | pd.Series | pd.DataFrame
        An array of P&Ls
    alpha: float | ArrayLike
        A parameter (or an array of parameters) between 0.0 and 1.0
    axis: int or None, optional
        Axis along which to calculate ES. If None, the array
        is flattened before calculations
    
    Returns
    -------

    ES: float
        Estimated ES with treshold alpha of sample x
    """
    if isinstance(x, pd.Series):
        if axis is not None:
            raise ValueError("For pd.Series input axis has to be None")
        try:
            return pd.Series(_expected_shortfall(x, alpha), name=name, index=alpha)
        except TypeError:
            return _expected_shortfall(x, alpha, axis=0)
    if isinstance(x, pd.DataFrame):
        try:
            if axis is None:
                axis = 0
            if axis == 0:
                return pd.DataFrame(
                    _expected_shortfall(x, alpha, axis=axis),
                    index=alpha,
                    columns=x.columns)
            elif axis == 1:
                return pd.DataFrame(
                    _expected_shortfall(x, alpha, axis=axis),
                    index=x.index,
                    columns=alpha)
            else:
                raise ValueError("For pd.Dataframe input axis has to be 0, 1 or None")
        except TypeError:
            return pd.Series(
                _expected_shortfall(x, alpha, axis=0),
                index=x.columns)
    return _expected_shortfall(x, alpha, axis=axis)
    
def sharpe_ratio(
        x: ArrayLike | pd.Series | pd.DataFrame,
        axis: int | None = None) -> ArrayLike | pd.Series | pd.DataFrame:
    """
    Calculate Sharpe Ratio

    Parameters
    ----------

    x: ArrayLike | pd.Series | pd.DataFrame
        Input array
    axis: int | None, optional
        An axis along which to calculate the Sharpe Ratio

    Returns
    -------

    SR: ArrayLike | pd.Series | pd.DataFrame
        A Sharpe Ratio for the given array
    
    """
    if isinstance(x, pd.Series):
        if axis is not None:
            raise ValueError("For pd.Series input axis has to be None")
        return x.mean()/x.std(ddof=1)
    if isinstance(x, pd.DataFrame) and axis is None:
        axis = 0
    return x.mean(axis=axis)/x.std(ddof=1, axis=axis)

def gain_loss_ratio(
        x: ArrayLike | pd.Series | pd.DataFrame,
        axis: int | None = None) -> ArrayLike | pd.Series | pd.DataFrame:
    """
    Calculate Gain Loss Ratio

    Parameters
    ----------

    x: ArrayLike | pd.Series | pd.DataFrame
        Input array
    axis: int | None, optional
        An axis along which to calculate the Gain Loss Ratio

    Returns
    -------

    GLR: ArrayLike | pd.Series | pd.DataFrame
        A Sharpe Ratio for the given array
    
    """
    if isinstance(x, pd.Series):
        if axis is not None:
            raise ValueError("For pd.Series input axis has to be None")
        return np.maximum(-x.mean()/x[x < 0].mean(), 0)
    if isinstance(x, pd.DataFrame) and axis is None:
        axis = 0
    return np.maximum(-x.mean(axis=axis)/x[x < 0].mean(axis=axis), 0)

def sortino_ratio(
        x: ArrayLike | pd.Series | pd.DataFrame,
        axis: int | None = None) -> ArrayLike | pd.Series | pd.DataFrame:
    """
    Calculate Sortino Ratio

    Parameters
    ----------

    x: ArrayLike | pd.Series | pd.DataFrame
        Input array
    axis: int | None, optional
        An axis along which to calculate the Sortino Ratio

    Returns
    -------

    GLR: ArrayLike | pd.Series | pd.DataFrame
        A Sharpe Ratio for the given array
    
    """
    if isinstance(x, pd.Series):
        if axis is not None:
            raise ValueError("For pd.Series input axis has to be None")
        return np.maximum(x.mean()/np.sqrt((x[x < 0]**2).mean()), 0)
    if isinstance(x, pd.DataFrame) and axis is None:
        axis = 0
    return np.maximum(x.mean(axis=axis)/np.sqrt((x[x < 0]**2).mean(axis=axis)), 0)

# def _mean_variance_criterion(
#         x: ArrayLike,
#         gamma: float | ArrayLike,
#         axis: int | None) -> np.ndarray:
#     """Calculate MV without respecting x's structure
#     """
#     x = np.asarray(x)
#     # if axis is None, we flatten x so that we can set axis = 0
#     if axis is None:
#         x = x.reshape(-1)
#         axis = 0
#     gamma = np.asarray(gamma)
#     # expand dims to match dims of x.std()
#     gamma = np.expand_dims(gamma, axis=list(range(1-x.ndim, 0)))
#     mv = x.mean(axis=axis) + gamma/2*x.std(ddof=1, axis=axis)
#     return mv

# def mean_variance_criterion(
#         x: ArrayLike | pd.Series | pd.DataFrame,
#         gamma: float | ArrayLike,
#         axis: int | None = None) -> ArrayLike | pd.Series | pd.DataFrame:
#     """
#     Calculate mean-variance criterion

#     Parameters
#     ----------

#     x: ArrayLike | pd.Series | pd.DataFrame
#         Input Array
#     gamma: float | ArrayLike
#         Risk aversion coefficient
#     axis: int | None = None
#         Axis along which to calculate the criterion

#     Returns
#     -------

#     MV: ArrayLike | pd.Series | pd.DataFrame
#         Mean-variance criterion for the given array
#     """
#     if isinstance(x, pd.Series):
#         if axis is not None:
#             raise ValueError("For pd.Series input axis has to be None")
#         try:
#             return pd.Series(_mean_variance_criterion(x, gamma), name=name, index=gamma)
#         except TypeError:
#             return _mean_variance_criterion(x, gamma, axis=0)
#     if isinstance(x, pd.DataFrame):
#         try:
#             if axis is None:
#                 axis = 0
#             if axis == 0:
#                 return pd.DataFrame(
#                     _mean_variance_criterion(x, gamma, axis=axis),
#                     index=gamma,
#                     columns=x.columns)
#             elif axis == 1:
#                 return pd.DataFrame(
#                     _mean_variance_criterion(x, gamma, axis=axis),
#                     index=x.index,
#                     columns=gamma)
#             else:
#                 raise ValueError("For pd.Dataframe input axis has to be 0, 1 or None")
#         except TypeError:
#             return pd.Series(
#                 _mean_variance_criterion(x, gamma, axis=0),
#                 index=x.columns)
#     return _mean_variance_criterion(x, gamma, axis=axis)

def _mean_variance_criterion(
        x: ArrayLike,
        gamma: float | ArrayLike,
        axis: int | None) -> np.ndarray:
    """Calculate MV without respecting x's structure
    """
    x = np.asarray(x)
    # if axis is None, we flatten x so that we can set axis = 0
    if axis is None:
        x = x.reshape(-1)
        axis = 0
    gamma = np.asarray(gamma)
    # expand dims to match dims of x.var()
    gamma = np.expand_dims(gamma, axis=list(range(1-x.ndim, 0)))
    mv = x.mean(axis=axis) + gamma/2*x.var(ddof=1, axis=axis)
    return mv

def mean_variance_criterion(
        x: ArrayLike | pd.Series | pd.DataFrame,
        gamma: float | ArrayLike,
        axis: int | None = None) -> ArrayLike | pd.Series | pd.DataFrame:
    """
    Calculate mean-variance criterion

    Parameters
    ----------

    x: ArrayLike | pd.Series | pd.DataFrame
        Input Array
    gamma: float | ArrayLike
        Risk aversion coefficient
    axis: int | None = None
        Axis along which to calculate the criterion

    Returns
    -------

    MV: ArrayLike | pd.Series | pd.DataFrame
        Mean-variance criterion for the given array
    """
    if isinstance(x, pd.Series):
        if axis is not None:
            raise ValueError("For pd.Series input axis has to be None")
        try:
            return pd.Series(_mean_variance_criterion(x, gamma), name=name, index=gamma)
        except TypeError:
            return _mean_variance_criterion(x, gamma, axis=0)
    if isinstance(x, pd.DataFrame):
        try:
            if axis is None:
                axis = 0
            if axis == 0:
                return pd.DataFrame(
                    _mean_variance_criterion(x, gamma, axis=axis),
                    index=gamma,
                    columns=x.columns)
            elif axis == 1:
                return pd.DataFrame(
                    _mean_variance_criterion(x, gamma, axis=axis),
                    index=x.index,
                    columns=gamma)
            else:
                raise ValueError("For pd.Dataframe input axis has to be 0, 1 or None")
        except TypeError:
            return pd.Series(
                _mean_variance_criterion(x, gamma, axis=0),
                index=x.columns)
    return _mean_variance_criterion(x, gamma, axis=axis)

def describe_pnl(pnl: pd.Series | pd.DataFrame) -> pd.DataFrame:
    """
    Returns selected statistics for P&L

    Parameters
    ----------

    pnl: pd.Series | pd.DataFrame
        A Series or DataFrame of P&Ls

    Returns
    -------

    Statistics: pd.DataFrame
        Selected statistics for the P&Ls
    """
    statistics = [
        ("mean", pnl.mean()),
        ("std", pnl.std(ddof=1)),
        ("VaR 1%", value_at_risk(pnl, 0.01)),
        ("ES 2.5%", expected_shortfall(pnl, 0.025)),
        ("SR", sharpe_ratio(pnl)),
        ("GLR", gain_loss_ratio(pnl))]
    names, stats = zip(*statistics)
    if isinstance(pnl, pd.DataFrame) and len(pnl.columns) > 1:
        return pd.DataFrame(
            list(stats),
            index=list(names),
            columns=pnl.columns)
    else:
        try:
            return pd.Series(
                list(stats),
                index=list(names),
                name=pnl.name)
        except AttributeError:
            return pd.Series(
                list(stats),
                index=list(names))

def rolling_array(
        x: ArrayLike,
        lookback: int,
        start: int | None = None,
        end: int | None = None,
        axis: int | None = 0) -> np.ndarray:
    """
    Returns an array, where the projection onto (axis, axis+1)
    dimensions is equal to roughly
    [[x[start-lookback], ..., x[start-1]],
     ...
     [x[end-lookback],   ..., x[end-1]]]

    Parameters
    ----------

    x: ArrayLike
        An array to be rolled
    lookback: int
        Amount to roll the array
    start: int | None, optional
        Starting index from which to roll the array
    end: int | None, optional
        Ending index from which to roll the array
    axis: int | None, optional
        Axis along which to roll the array

    Returns
    -------

    rolled_array: np.ndarray
        Rolled array
    """
    x = np.asarray(x)
    # if axis is None, we flatten returns so that we can set axis = 0
    if axis is None:
        x = x.reshape(-1)
        axis = 0
    if start is None:
        start = lookback
    if end is None:
        end = np.size(x, axis)
    indices = np.arange(start, end)[:, np.newaxis] + np.arange(-lookback, 0)
    return x.take(indices, axis=axis)
    
def var_backtest(
        returns: pd.Series | pd.DataFrame,
        lookback: int,
        start: pd.Index,
        end: pd.Index,
        alpha: float) -> pd.Series | pd.DataFrame:
    """
    Performs a VaR backtest on given returns

    Parameters
    ----------

    returns: pd.Series | pd.DataFrame
        Returns of a portfolio of stocks
    lookback: int
        Lookback period for the backtest
    start: pd.Index
        Starting index from which to perform the backtest
    end: pd.Index
        Ending index from which to perform the backtest
    alpha: float
        VaR threshold

    Returns
    -------

    realized_returns: pd.Series | pd.DataFrame
        Realized returns in the given timeframe
    vars: pd.Series | pd.DataFrame
        Calculated VaRs
    """
    realized_returns = returns.loc[start:end]
    var = value_at_risk(
        rolling_array(
            returns,
            lookback=lookback,
            start=returns.index.get_slice_bound(start, 'left'),
            end=returns.index.get_slice_bound(end, 'right')),
        alpha=alpha, axis=1)
    if isinstance(returns, pd.Series):
        var = pd.Series(var, name=realized_returns.name, index=realized_returns.index)
    elif isinstance(returns, pd.DataFrame):
        var = pd.DataFrame(var, columns=realized_returns.columns, index=realized_returns.index)
    return realized_returns, var

def pit_backtest(
        returns: pd.Series | pd.DataFrame,
        lookback: int,
        start: pd.Index,
        end: pd.Index) -> pd.Series | pd.DataFrame:
    """
    Performs a PIT backtest on given returns

    Parameters
    ----------

    returns: pd.Series | pd.DataFrame
        Returns of a portfolio of stocks
    lookback: int
        Lookback period for the backtest
    start: pd.Index
        Starting index from which to perform the backtest
    end: pd.Index
        Ending index from which to perform the backtest

    Returns
    -------

    realized_returns: pd.Series | pd.DataFrame
        Realized returns in the given timeframe
    PIT: pd.Series | pd.DataFrame
        Empirical probabilities of realized returns with respect to
        historical simulation
    """
    realized_returns = returns.loc[start:end]
    historical_simulations = rolling_array(
            returns,
            lookback=lookback,
            start=returns.index.get_slice_bound(start, 'left'),
            end=returns.index.get_slice_bound(end, 'right'))
    probabilities = np.zeros_like(realized_returns)
    if isinstance(returns, pd.Series):
        for i in range(historical_simulations.shape[0]):
            ecdf = sp.stats.ecdf(historical_simulations[i, :]).cdf
            probabilities[i] = ecdf.evaluate(realized_returns.iloc[i])
        probabilities = pd.Series(probabilities, name=realized_returns.name, index=realized_returns.index)
    elif isinstance(returns, pd.DataFrame):
        for i in range(historical_simulations.shape[0]):
            for t in range(historical_simulations.shape[2]):
                ecdf = sp.stats.ecdf(historical_simulations[i, :, t]).cdf
                probabilities[i, t] = ecdf.evaluate(realized_returns.iloc[i, t])
        probabilities = pd.DataFrame(probabilities, columns=realized_returns.columns, index=realized_returns.index)
    return realized_returns, probabilities