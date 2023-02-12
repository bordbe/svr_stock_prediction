from __future__ import annotations
from typing import List, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from retrotester import Retrotester
from math import log, exp, sqrt
from statistics import variance, stdev, mean
import operator


def division(x: Union[int, float], y: Union[int, float]):
    """True division of x by y

    Parameters
    ----------
    x : Union[int, float]
        dividend
    y : Union[int, float]
        divisor

    Returns
    -------
    Union[int, float]
        quotient
    """
    if not all((x, y)):
        # return 0 instead of throwing ZeroDivisionError or TypeError
        return 0
    else:
        return operator.truediv(x, y)


def compute_statistics_backtest(backtest: Retrotester, rf: float = 0.0):
    """Implementation of https://github.com/kernc/backtesting.py/blob/master/backtesting/_stats.py,
    without using pandas

    Parameters
    ----------
    backtest : Retrotester
        retrotester object
    rf : float, optional
        risk-free rate, by default 0.0

    Returns
    -------
    dict
        dictionary containing the different stats
    """

    def drawdown_duration_peaks(l: List[float]):
        """Compute maximum drawdown, duration of drawdowns and the peak drawdown for a list of returns"""
        max_accum = [m := x if idx == 0 else max(m, x) for idx, x in enumerate(l)]  # https://stackoverflow.com/a/71833205
        dd = list(map(lambda level, max_: 1 - level / max_, l, max_accum))
        curr = [i for i, e in enumerate(dd) if e == 0]
        curr += [len(dd) - 1]
        prev = [float("nan")] + curr[:-1]
        index = {c: p for c, p in zip(curr, prev) if c > p + 1}
        duration = [c - p for c, p in index.items()]
        peaks = [max(dd[p : c + 1]) for c, p in index.items()]
        return -max(dd), duration, peaks

    def geometric_mean(returns: List[float]):
        """Geometric mean of returns"""
        returns = list(map(lambda x: x + 1, returns))
        returns = [x if x > 0 else 0 for x in returns]
        logret = list(map(log, returns))
        return exp(sum(logret) / len(logret)) - 1

    def compute_returns(levels: List[float]):
        """Compute returns for a list of float, their geometric mean and annual return"""
        strat_ret = list(map(lambda x, y: x / y - 1, levels[1:], levels[:-1]))
        gmean_ret = geometric_mean(strat_ret)
        ann_ret = (1 + gmean_ret) ** 252 - 1
        return strat_ret, gmean_ret, ann_ret

    s = {}
    s["Start"] = str(backtest._config.start_ts)
    s["End"] = str(backtest._config.end_ts)
    s["Duration"] = str(backtest._config.end_ts - backtest._config.start_ts)
    s["Exposure [%]"] = 100 * len(list(filter(lambda w: w.value > 0, backtest._strategy._weight_by_pk.values()))) / len(backtest._strategy._weight_by_pk.values())
    levels = [l.close for l in backtest._level_by_ts.values()]
    s["Equity Final"] = levels[-1]
    s["Equity Peak"] = max(levels)
    s["Return [%]"] = 100 * (levels[-1] / levels[0] - 1)
    strat_ret, gmean_ret, ann_ret = compute_returns(levels)
    s["Return (Ann.) [%]"] = 100 * ann_ret
    s["Volatility (Ann.) [%]"] = sqrt((variance(strat_ret) + (1 + gmean_ret) ** 2) ** 252 - (1 + gmean_ret) ** (2 * 252)) * 100
    s["Sharpe Ratio"] = (s["Return (Ann.) [%]"] - rf) / (s["Volatility (Ann.) [%]"] or float("nan"))
    s["Sortino Ratio"] = (ann_ret - rf) / (sqrt(mean(list(map(lambda x: (max(min(x, 0), float("-inf"))) ** 2, strat_ret)))) * sqrt(252))
    max_dd, dd_dur, dd_peaks = drawdown_duration_peaks(levels)
    s["Calmar Ratio"] = ann_ret / (-max_dd or float("nan"))
    s["Max. Drawdown [%]"] = max_dd * 100
    s["Avg. Drawdown [%]"] = -mean(dd_peaks) * 100
    s["Max. Drawdown Duration"] = max(dd_dur)
    s["Avg. Drawdown Duration"] = mean(dd_dur)
    return s
