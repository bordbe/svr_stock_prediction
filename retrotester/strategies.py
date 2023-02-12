from __future__ import annotations
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from retrotester import Config
from datetime import datetime
from .indicators import Indicator
from .dataobj import Data, Quote, Weight


class BaseStrategy:
    """
    A trading strategy base class. Extend this class and override methods:
        - retrotester.strategies.BaseStrategy.construct
        - retrotester.strategies.BaseStrategy.update
        - retrotester.strategies.BaseStrategy.compute_signals
        - retrotester.strategies.BaseStrategy.compute_performance
    to define the strategy
    """

    def __init__(self, config: Config, data: Data):
        self._config = config
        self.strategy_code = config.strategy_code
        self._universe = config.universe
        self._data = data
        self.indicators = dict()

    def construct(self):
        """
        Initialize the strategy
        Override this method
        Declare indicators (with `retrotester.retrotester.BaseStrategy.add_indicators`)
        """
        raise NotImplementedError

    def add_indicators(self, indicators: List[Indicator]):
        """Add indicators to the strategy"""
        if type(indicators) is not list:
            indicators = [indicators]
        for ind in indicators:
            self.indicators[ind._name] = ind

    def update(self, data: List[Quote]):
        """
        Update the strategy with data
        Override this method
        """
        raise NotImplementedError

    def compute_signals(self, quotes: List[Quote]):
        """
        Compute the signals of the quotes
        Override this method
        """
        raise NotImplementedError

    def compute_performance(self, ts: datetime) -> float:
        """
        Compute strategy's performance between at ts
        Override this method
        """
        raise NotImplementedError


class WeightStrategy(BaseStrategy):
    """
    A trading strategy based on portfolio rebalancing
    """

    def __init__(self, config: Config, data: Data):
        super().__init__(config, data)
        self._weight_by_pk = dict()

    def get_weight(self, underlying_code: str, ts: datetime) -> Weight:
        """Return weight object for a given underlying_code at ts"""
        return self._weight_by_pk.get((self.strategy_code, underlying_code, ts))

    def compute_performance(self, ts: datetime) -> float:
        perf_ = 0.0
        for underlying_code in self._universe:
            weight = self.get_weight(underlying_code, ts)
            if weight is not None:
                value = weight.value
                current_quote = self._data.quotes_by_pk.get((underlying_code, ts))
                # retrieve last available quote
                prev_key = self._data.quotes_by_pk.get_prev_key((underlying_code, ts))
                previous_quote = self._data.quotes_by_pk.get(prev_key)
                if current_quote is not None and previous_quote is not None:
                    perf_ += value * (current_quote.close / previous_quote.close - 1)
                else:
                    raise ValueError(f"Missing Quote for {underlying_code} at {ts}")
        return perf_


class EquiWeightedStrategy(WeightStrategy):
    def update(self, data: List[Quote]):
        sum_ = sum(map(lambda x: abs(x.signal), data))
        for quote in data:
            value = quote.signal / sum_ if sum_ != 0 else 0
            next_quote = self._data.quotes_by_pk.get_next_key((quote.symbol, quote.ts))
            if next_quote is not None:
                next_key = (self.strategy_code, *next_quote)
                # affect a weight to the next available quote for the symbol
                self._weight_by_pk[next_key] = Weight(product_code=self.strategy_code, underlying_code=quote.symbol, ts=next_key[-1], value=value)
