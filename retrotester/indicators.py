from __future__ import annotations
from .dataobj import Quote
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from retrotester import Config
from typing import List, Callable
from statistics import mean, stdev
from itertools import islice
import operator
from .mathfunc import division


class Indicator:
    """
    This is object is a base class for representing an indicator.
    Extend this class and override method: `retrotester.indicators.Indicator.compute_values`,
    to define the calculation of the indicator
    """

    def __init__(self, config: Config, name: str):
        self._config = config
        self._name = name

    def compute_values(self, data: List[Quote]):
        """
        Compute the indicator values for each quote
        Override this method
        """
        raise NotImplementedError


class MovingIndicator(Indicator):
    """
    Moving indicator base class
    """

    def __init__(self, config: Config, name: str, n: int = 2):
        super().__init__(config, name)
        self.window_size = n

    @staticmethod
    def window(seq: list, lag: int):
        """Returns a sliding window (of width lag) over data from the iterable
        This method is taken from itertools examples (https://docs.python.org/release/2.3.5/lib/itertools-example.html)
        """
        it = iter(seq)
        result = tuple(islice(it, lag))
        if len(result) == lag:
            yield result
        for elem in it:
            result = result[1:] + (elem,)
            yield result

    def apply(self, func: Callable, seq: list, lag: int = None):
        """Apply a function on a sliding window of width lag to a list of data (seq)

        Parameters
        ----------
        func : Callable
            function to apply
        seq : list
            data to apply the function on
        lag : int, optional
            size the window, by default None

        Returns
        -------
        list :
            output of the function
        """
        lag = self.window_size if lag is None else lag
        results = list(map(func, self.window(seq, lag)))
        return [None] * (len(seq) - len(results)) + results


class SimpleMovingAverage(MovingIndicator):
    """
    Simple moving average indicator,
    see (https://en.wikipedia.org/wiki/Moving_average#Simple_moving_average)
    """

    def compute_values(self, data: List[Quote]):
        quotes = list(map(lambda x: getattr(x, self._config.quote_period), data))
        ma = self.apply(mean, quotes)
        for quote, a in zip(data, ma):
            setattr(quote, self._name, a)


class WeightedMovingAverage(MovingIndicator):
    """
    Weighted moving average indicator,
    see (https://en.wikipedia.org/wiki/Moving_average#Weighted_moving_average)
    """

    def compute_values(self, data: List[Quote]):
        def weighted_average(quotes: List[float]):
            weights = list(range(1, self.window_size + 1))
            return sum([x * y for x, y in zip(quotes, weights)]) / (self.window_size * (self.window_size + 1) / 2)

        quotes = list(map(lambda x: getattr(x, self._config.quote_period), data))
        wma = self.apply(weighted_average, quotes)
        for quote, a in zip(data, wma):
            setattr(quote, self._name, a)


class AccumulationDistributionOscillator(Indicator):
    """
    Accumulation distribution indicator,
    see (https://www.investopedia.com/terms/a/accumulationdistribution.asp)
    """

    def compute_values(self, data: List[Quote]):
        setattr(data[0], self._name, None)
        for curr_quote, prec_quote in zip(data[1:], data[:-1]):
            ado = division((curr_quote.high - prec_quote.close), (curr_quote.high - curr_quote.low))
            setattr(curr_quote, self._name, ado)


class RelativeStrenghtIndex(MovingIndicator):
    """
    Relative strengh index with simple moving average
    see (https://en.wikipedia.org/wiki/Relative_strength_index)
    """

    def compute_values(self, data: List[Quote]):
        def difference(quotes: List[float]):
            return list(map(operator.sub, quotes[1:], quotes[:-1]))

        quotes = list(map(lambda x: getattr(x, self._config.quote_period), data))
        delta = difference(quotes)
        ups = list(map(lambda x: max(x, 0), delta))
        downs = list(map(lambda x: -1 * min(x, 0), delta))
        ups_avg = self.apply(mean, ups)
        downs_avg = self.apply(mean, downs)
        res = list(map(division, ups_avg, downs_avg))
        setattr(data[0], self._name, None)
        for quote, rs in zip(data[1:], res):
            if rs:
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = None
            setattr(quote, self._name, rsi)


class TrueRange(Indicator):
    """
    True range indicator
    see (https://en.wikipedia.org/wiki/Average_true_range)
    """

    def compute_values(self, data: List[Quote]):
        setattr(data[0], self._name, None)
        for curr_quote, prec_quote in zip(data[1:], data[:-1]):
            tr = max(curr_quote.high - curr_quote.low, abs(curr_quote.high - prec_quote.close), abs(curr_quote.low - prec_quote.close))
            setattr(curr_quote, self._name, tr)


class AverageTrueRange(MovingIndicator):
    """
    Average true range indicator
    see (https://en.wikipedia.org/wiki/Average_true_range)
    """

    def compute_values(self, data: List[Quote]):
        def truerange_average(data: List[Quote]):
            list_atr = [None] * (self.window_size - 1)
            list_tr = [quote.tr for quote in data]
            first_atr = mean(list_tr[: self.window_size])
            list_atr.append(first_atr)
            prev_atr = first_atr
            for quote in data[self.window_size :]:
                curr_atr = (prev_atr * (self.window_size - 1) + quote.tr) / self.window_size
                list_atr.append(curr_atr)
                prev_atr = curr_atr
            return list_atr

        true_range = TrueRange(self._config, "tr")
        true_range.compute_values(data)
        setattr(data[0], self._name, None)
        list_atr = truerange_average(data[1:])
        for quote, atr in zip(data[1:], list_atr):
            setattr(quote, self._name, atr)
