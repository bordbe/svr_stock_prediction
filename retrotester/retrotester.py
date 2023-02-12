from .mathfunc import compute_statistics_backtest
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List
import plotly.graph_objs as go
from bisect import bisect_left, bisect
from .strategies import BaseStrategy
from .dataobj import Data, Quote, Frequency
from tqdm import tqdm


@dataclass
class Config:
    """Configuration object to pass on to the Retrotester"""

    universe: List[str]
    start_ts: datetime
    end_ts: datetime
    strategy_code: str
    frequency: Frequency
    basis: int = 100
    model_parameters: dict = None

    def __post_init__(self):
        if self.model_parameters:
            self.__dict__.update(self.model_parameters)
        self.universe = list(map(lambda x: x.upper(), self.universe))
        if self.start_ts >= self.end_ts:
            raise ValueError("self.start_ts must be before self.end_ts")
        if len(self.universe) == 0:
            raise ValueError("self.universe should contains at least one element")

    @property
    def timedelta(self):
        if self.frequency == Frequency.HOURLY:
            return timedelta(hours=1)
        elif self.frequency == Frequency.DAILY:
            return timedelta(days=1)


class Retrotester:
    """
    Backtest a strategy on particular data.
    Upon initialization, call method `backtesting.backtesting.Backtest.run` to run a backtest
    """

    # TODO: implement an optimization method

    def __init__(self, data: Data, strategy: BaseStrategy, config: Config):
        self._data = data
        self._quotes_by_pk = data.quotes_by_pk
        self._config = config
        self._strategy = strategy(config, data)
        self._universe = config.universe
        self._timedelta = config.timedelta
        self._level_by_ts = dict()

    def _create_strategy(self):
        """Create the strategy by computing indicators' values and signals for all symbols in the universe"""
        self._strategy.construct()
        try:
            for underlying_code in tqdm(self._universe, desc="Creating strategy"):
                data = self._data.quotes_by_symbol[underlying_code]
                for ind in self._strategy.indicators.values():
                    ind.compute_values(data)
                self._strategy.compute_signals(data)
        except Exception as e:
            raise RuntimeError(f"Problem when creating strategy with {underlying_code} symbol") from e

    @property
    def _calendar(self) -> List[datetime]:
        """Return the dates available between:
        - retrotester.retrotester.Config.start_ts and,
        - retrotester.retrotester.Config.end_ts
        """
        if not self._data:
            raise ValueError("No data uploaded")
        dates, start, end = self._data.dates, self._config.start_ts, self._config.end_ts
        return dates[bisect_left(dates, start) : bisect(dates, end)]

    def _get_ts_before(self, ts: datetime, dt: int = 1):
        """Return the previous available date before ts"""
        if ts == self._data.dates[0]:
            raise ValueError(f"No date before {ts} uploaded")
        return self._data.dates[bisect_left(self._data.dates, ts) - dt]

    def _update_strategy(self, ts: datetime):
        data = self._data.quotes_by_ts[ts]
        return self._strategy.update(data)

    def run(self) -> List[Quote]:
        """Run the backtest

        Returns
        -------
        List[Quote]
            levels of the strategy
        """
        self._create_strategy()
        self._level_by_ts[self._calendar[0]] = Quote(close=self._config.basis, ts=self._calendar[0])
        for ts in tqdm(self._calendar[1:], desc="Backtesting"):
            prev_ts = self._get_ts_before(ts, 1)
            self._update_strategy(prev_ts)
            perf = self._strategy.compute_performance(ts)
            close = self._level_by_ts.get(prev_ts).close * (1 + perf)
            quote = Quote(close=close, ts=ts)
            self._level_by_ts[ts] = quote
        return list(self._level_by_ts.values())

    @property
    def stats(self, riskfree_rate: float = 0.0):
        return compute_statistics_backtest(self, riskfree_rate)

    def plot(self):
        """
        Plot the levels of the strategy
        """
        x = [quote.ts for quote in self._level_by_ts.values()]
        y = [quote.close for quote in self._level_by_ts.values()]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y))
        fig.update_layout(title_text=f"Strategy '{self._config.strategy_code}' levels", template="simple_white")
        fig.show()
