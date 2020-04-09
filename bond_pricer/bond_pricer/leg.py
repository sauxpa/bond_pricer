import numpy as np
from collections import defaultdict
from typing import Callable
from .priceable import Priceable


class CashflowLeg(Priceable):
    def __init__(self,
                 term_sheet: defaultdict = defaultdict(),
                 discount_curve: Callable = lambda: None,
                 pricing_date: float = 0.0,
                 ) -> None:
        super().__init__(
            term_sheet=term_sheet,
            discount_curve=discount_curve,
            pricing_date=pricing_date,
            )

    @property
    def cashflow_dates(self) -> np.ndarray:
        return self.term_sheet.get('cashflow_dates')

    @property
    def cashflows(self) -> np.ndarray:
        return self.term_sheet.get('cashflows')

    @property
    def day_count_fractions(self) -> np.ndarray:
        return np.diff(self.cashflow_dates, prepend=self.pricing_date)

    @property
    def discount_factors(self) -> np.ndarray:
        """self.discount_curve needs to be vectorized.
        """
        return (
            1 + self.discount_curve(self.cashflow_dates)
            ) ** (-self.cashflow_dates / self.day_count_fractions)

    @property
    def discounted_cashflows(self) -> np.ndarray:
        return self.cashflows * self.discount_factors

    @property
    def price(self) -> float:
        return np.sum(self.discounted_cashflows)
