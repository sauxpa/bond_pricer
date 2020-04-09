import numpy as np
from scipy.optimize import root
from collections import defaultdict
import abc
from typing import Callable


class Security(abc.ABC):
    """Generic class.
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


class CashflowLeg(Security):
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


class FixedCouponBond(Security):
    def __init__(self,
                 term_sheet: defaultdict = defaultdict(),
                 ) -> None:
        super().__init__(
            term_sheet=term_sheet
            )

    @property
    def coupon(self) -> float:
        return self.term_sheet.get('coupon')

    @property
    def principal(self) -> float:
        return self.term_sheet.get('principal', 1.0)

    @property
    def maturity(self) -> float:
        return self.term_sheet.get('maturity')

    @property
    def coupon_frequency(self) -> float:
        return self.term_sheet.get('coupon_frequency')

    @property
    def day_count_fraction(self) -> float:
        return 1 / self.coupon_frequency

    @property
    def coupon_dates(self) -> np.ndarray:
        """Dates are relative: 4.5 means in 4.5 years from
        current pricing date.
        """
        coupon_dates = np.arange(
            self.maturity % self.day_count_fraction,
            self.maturity + self.day_count_fraction / 2,
            self.day_count_fraction
            )
        if coupon_dates[0] == 0:
            return coupon_dates[1:]
        else:
            return coupon_dates

    @property
    def coupon_cashflows(self) -> np.ndarray:
        return self.day_count_fraction * self.coupon * np.ones(
            len(self.coupon_dates)
            )

    @property
    def coupon_leg(self):
        coupon_term_sheet = defaultdict(
            None,
            {
                'cashflow_dates': self.coupon_dates,
                'cashflows': self.coupon_cashflows,
                }
            )
        return CashflowLeg(
            term_sheet=coupon_term_sheet,
            pricing_date=self.pricing_date,
            )

    @property
    def principal_leg(self):
        principal_curve = np.zeros(len(self.coupon_dates))
        principal_curve[-1] = self.principal
        principal_term_sheet = defaultdict(
            None,
            {
                'cashflow_dates': self.coupon_dates,
                'cashflows': principal_curve
            }
        )
        return CashflowLeg(
            term_sheet=principal_term_sheet,
            pricing_date=self.pricing_date,
            )

    def flat_yield_curve_factory(self, Y: float) -> Callable:
        return np.vectorize(lambda t: Y)

    def yield_to_coupon_cashflows(self, Y: float) -> float:
        """Yield is annualized before passed to coupon leg.
        """
        coupon_leg = self.coupon_leg
        coupon_leg.discount_curve = self.flat_yield_curve_factory(
            Y * self.day_count_fraction
            )
        return coupon_leg.discounted_cashflows

    def yield_to_coupon_price(self, Y: float) -> float:
        """Yield is annualized before passed to coupon leg.
        """
        coupon_leg = self.coupon_leg
        coupon_leg.discount_curve = self.flat_yield_curve_factory(
            Y * self.day_count_fraction
            )
        return coupon_leg.price

    def yield_to_principal_cashflows(self, Y: float) -> float:
        """Yield is annualized before passed to principal leg.
        """
        principal_leg = self.principal_leg
        principal_leg.discount_curve = self.flat_yield_curve_factory(
            Y * self.day_count_fraction
            )
        return principal_leg.discounted_cashflows

    def yield_to_principal_price(self, Y: float) -> float:
        """Yield is annualized before passed to principal leg.
        """
        principal_leg = self.principal_leg
        principal_leg.discount_curve = self.flat_yield_curve_factory(
            Y * self.day_count_fraction
            )
        return principal_leg.price

    def yield_to_price(self, Y: float) -> float:
        """Dirty price.
        """
        return self.yield_to_coupon_price(Y) + self.yield_to_principal_price(Y)

    def price_to_yield(self, P: float) -> float:
        def f(y: float) -> float:
            return self.yield_to_price(y) - P
        sol = root(f, self.coupon)
        if sol.success:
            return sol.x[0]
        else:
            raise Exception('Cannot solve for price {}'.format(P))

    def duration(self, P: float) -> float:
        Y = self.price_to_yield(P)
        coupon_cashflows = self.yield_to_coupon_cashflows(Y)
        principal_cashflows = self.yield_to_principal_cashflows(Y)

        dur_coupon_cashflows = coupon_cashflows * self.coupon_dates
        dur_principal_cashflows = principal_cashflows * self.maturity

        return np.sum(dur_coupon_cashflows) + np.sum(dur_principal_cashflows)

    def macaulay_duration(self, P: float) -> float:
        return self.duration(P) / P

    def dv01(self, P: float) -> float:
        Y = self.price_to_yield(P)
        return self.duration(P) / (1 + self.day_count_fraction * Y)

    def modified_duration(self, P: float) -> float:
        return self.dv01(P) / P
