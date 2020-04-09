from collections import defaultdict
import abc
from typing import Callable


class Priceable(abc.ABC):
    """Generic class for priceable securities.
    """
    def __init__(self,
                 term_sheet: defaultdict = defaultdict(),
                 discount_curve: Callable = lambda: None,
                 pricing_date: float = 0.0,
                 ) -> None:
        self._term_sheet = term_sheet
        self._discount_curve = discount_curve
        self._pricing_date = pricing_date

    @property
    def term_sheet(self) -> defaultdict:
        return self._term_sheet

    @term_sheet.setter
    def term_sheet(self, new_term_sheet: defaultdict) -> None:
        self._term_sheet = new_term_sheet

    @property
    def discount_curve(self) -> Callable:
        return self._discount_curve

    @discount_curve.setter
    def discount_curve(self, new_discount_curve: Callable) -> None:
        self._discount_curve = new_discount_curve

    @property
    def pricing_date(self) -> float:
        return self._pricing_date

    @pricing_date.setter
    def pricing_date(self, new_pricing_date: float) -> None:
        self._pricing_date = new_pricing_date

    @property
    @abc.abstractmethod
    def price(self) -> float:
        pass
