from schema import Schema
from enum import Enum
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Tuple
from itertools import groupby
import operator
from functools import cached_property
from collections import OrderedDict
import bisect


class Frequency(Enum):
    HOURLY = "Hourly"
    MONTHLY = "Monthly"
    DAILY = "Daily"


class _QuotesByPk(OrderedDict):
    """
    This object represents quotes arranged in dict,
    with a tuple (quote.symbol, quote.ts) as key and a quote as value
    """

    @cached_property
    def dict_keys(self) -> list:
        return list(self.keys())

    def get_next_key(self, k: Tuple[str, datetime]):
        """Get the next key after k"""
        i = bisect.bisect_right(self.dict_keys, k)
        if self.dict_keys[i][1] > k[1]:
            # check if next key date is after
            if self.dict_keys[i - 1] != k:
                raise KeyError(k)
            return self.dict_keys[i]
        else:
            # if k is last key
            return None

    def get_prev_key(self, k: Tuple[str, datetime]):
        """Get the previous key before k"""
        i = bisect.bisect_left(self.dict_keys, k)
        if self.dict_keys[i - 1][1] < k[1]:
            # check if prev key date is before
            if self.dict_keys[i] != k:
                raise KeyError(k)
            return self.dict_keys[i - 1]
        else:
            # if k is first key
            return None


@dataclass
class Quote:
    """
    This object represents a market quote
    """

    symbol: str = None
    ts: datetime = None
    open: float = None
    high: float = None
    low: float = None
    close: float = None
    adj_close: float = None
    volume: float = None
    signal: float = None

    def __str__(self) -> str:
        return f"Quote({', '.join([f'{k}={v}' for k, v in self.__dict__.items()])}"

    def __repr__(self) -> str:
        return f"Quote({', '.join([f'{k}={v}' for k, v in self.__dict__.items()])}"


@dataclass
class Weight:
    """
    This object represents a portfolio weight
    """

    product_code: str = None
    underlying_code: str = None
    ts: datetime = None
    value: float = None


class Data:
    """
    Object representing the data fed to the backtest
    """

    def __init__(self, data: List[Quote] = None):
        self.quotes = []
        if data:
            self.load(data)

    def load(self, data: List[Quote]):
        """load quotes to the data

        Parameters
        ----------
        data : List[Quote]
            list of quotes
        """
        self.quotes += self._check_data(data)

    def _check_data(self, data: List[Quote]) -> List[Quote]:
        """Check if the data uploaded is a list of Quote objects"""
        schema = Schema([Quote])
        return schema.validate(data)

    @cached_property
    def dates(self) -> List[datetime]:
        """Return the dates of all the quotes passed"""
        return list(sorted(set([quote.ts for quote in self.quotes])))

    @cached_property
    def quotes_by_pk(self) -> Dict[str, List[Quote]]:
        iter_ = groupby(self.quotes, lambda quote: (quote.symbol, quote.ts))
        group = {key: list(group)[0] for key, group in iter_}
        return _QuotesByPk(sorted(group.items()))

    def _group_by_attr(self, attr: List[str]):
        """Group data.quotes by attributes of retrotester.dataobj.Quote

        Parameters
        ----------
        attr : List[str]
            list of attribute(s) to group on

        Returns
        -------
        dict
            attr as keys and corresponding Quotes as values
        """
        get_attr = operator.attrgetter(*attr)
        return {k: list(g) for k, g in groupby(sorted(self.quotes, key=get_attr), get_attr)}

    @cached_property
    def quotes_by_symbol(self) -> Dict[str, List[Quote]]:
        return self._group_by_attr(["symbol"])

    @cached_property
    def quotes_by_ts(self) -> Dict[str, List[Quote]]:
        return self._group_by_attr(["ts"])

    @staticmethod
    def filter_quotes_by_signal(data: List[Quote], value: int) -> List[Quote]:
        """Filter quotes by their signal attribute

        Parameters
        ----------
        data : List[Quote]
            list of quotes
        value : int
            signal value to filter

        Returns
        -------
        List[Quote]
            quotes which signal are equal to value
        """
        return list(filter(lambda q: (q.signal == value), data))
