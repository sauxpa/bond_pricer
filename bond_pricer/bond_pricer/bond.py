import numpy as np
from scipy.optimize import root
from collections import defaultdict
from typing import Callable
from .leg import CashflowLeg
from .priceable import Priceable


class FixedCouponBond(Priceable):
    def __init__(self,
                 term_sheet: defaultdict = defaultdict(),
                 marking_mode: str = 'price',
                 mark: float = 0.0,
                 ) -> None:
        self.check_marking_mode(marking_mode)
        self._marking_mode = marking_mode.lower()
        self._mark = mark
        super().__init__(
            term_sheet=term_sheet
            )

    def check_marking_mode(self, marking_mode: str) -> None:
        msg = '{:s} not an available marking mode'.format(marking_mode)
        assert marking_mode.lower() in {'price', 'ytm'}, msg

    @property
    def marking_mode(self) -> str:
        return self._marking_mode

    @marking_mode.setter
    def marking_mode(self, new_marking_mode: str) -> None:
        self.check_marking_mode(new_marking_mode)
        self._marking_mode = new_marking_mode.lower()

    @property
    def mark(self) -> float:
        return self._mark

    @mark.setter
    def mark(self, new_mark: float) -> None:
        self._mark = new_mark

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

    @property
    def price(self) -> float:
        if self.marking_mode == 'price':
            return self.mark
        else:
            return self.yield_to_price(self.mark)

    @property
    def ytm(self) -> float:
        if self.marking_mode == 'ytm':
            return self.mark
        else:
            return self.price_to_yield(self.mark)

    def duration_calc(self, P: float) -> float:
        Y = self.price_to_yield(P)
        coupon_cashflows = self.yield_to_coupon_cashflows(Y)
        principal_cashflows = self.yield_to_principal_cashflows(Y)

        dur_coupon_cashflows = coupon_cashflows * self.coupon_dates
        dur_principal_cashflows = principal_cashflows * self.maturity

        return np.sum(dur_coupon_cashflows) + np.sum(dur_principal_cashflows)

    def macaulay_duration_calc(self, P: float) -> float:
        return self.duration(P) / P

    def dv01_calc(self, P: float) -> float:
        Y = self.price_to_yield(P)
        return self.duration(P) / (1 + self.day_count_fraction * Y)

    def modified_duration_calc(self, P: float) -> float:
        return self.dv01(P) / P

    @property
    def macaulay_duration(self) -> float:
        return self.duration_calc(self.price) / self.price

    @property
    def dv01(self) -> float:
        return self.duration_calc(self.price) / (1 + self.day_count_fraction * self.ytm)

    @property
    def modified_duration(self) -> float:
        return self.dv01 / self.price
