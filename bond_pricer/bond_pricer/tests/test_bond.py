import numpy as np
from collections import defaultdict
from typing import Type
from copy import deepcopy
from pprint import pprint
from bond_pricer.utils import ONE_PCT, ONE_BP
from bond_pricer.bond import FixedCouponBond


def test_bond(verbose: bool = False):
    """Test various bond analytics and edge cases.
    """
    def test_par_to_yield(bond_: Type[FixedCouponBond]):
        """Fixed coupon bond at par should have yield to maturity equal
        to the fixed coupon rate.
        """
        bond = deepcopy(bond_)
        bond.marking_mode = 'price'
        bond.mark = bond.principal
        msg = 'Par yield is not the coupon rate.'
        assert np.isclose(
            bond.ytm,
            bond.coupon,
            atol=1e-8
            ), msg

    def test_yield_to_par(bond_: Type[FixedCouponBond]):
        """Fixed coupon bond at yield to maturity equal to the fixed coupon rate
        should be priced at par.
        """
        bond = deepcopy(bond_)
        bond.marking_mode = 'ytm'
        bond.mark = bond.coupon
        msg = 'Price is not par when YTM = coupon rate.'
        assert np.isclose(
            bond.price,
            bond.principal,
            atol=1e-8
            ), msg

    def test_yield_to_price(bond_: Type[FixedCouponBond]):
        """Assert yield -> price -> yield.
        """
        bond = deepcopy(bond_)
        bond.marking_mode = 'ytm'
        bond.mark = bond.coupon * 1.1
        msg = 'Price <-> YTM is broken.'
        assert np.isclose(
            bond.price_to_yield(bond.price),
            bond.ytm,
            atol=1e-8
            ), msg

    def test_dv01(bond_: Type[FixedCouponBond]):
        """Test the theoretical formula for DV01 vs finite difference.
        """
        bond = deepcopy(bond_)

        bond.marking_mode = 'price'

        prices = [
            bond.principal,
            0.8 * bond.principal,
            1.2 * bond.principal,
            ]
        msgs = [
            'Inaccurate DV01 at par: ',
            'Inaccurate DV01 below par: ',
            'Inaccurate DV01 above par: ',
            ]

        for price, msg in zip(prices, msgs):
            bond.mark = price
            dv01 = bond.dv01
            approx_dv01 = (
                bond.yield_to_price(bond.ytm - ONE_BP)
                - bond.yield_to_price(bond.ytm)
                ) / ONE_BP
            assert np.isclose(
                dv01,
                approx_dv01,
                atol=1e-1
                ), msg + 'dv01: {:.6f}\napprox: {:.6f}'.format(
                    dv01,
                    approx_dv01
                    )

    def test_macaulay_zc(bond_: Type[FixedCouponBond]):
        """Test that macaulay_duration for zero coupon bond is
        indeed the maturity.
        """
        bond = deepcopy(bond_)
        bond.marking_mode = 'price'
        bond.mark = bond.principal

        if bond.coupon != 0.0:
            pass
        else:
            msg = 'Macaulay duration != maturity for ZC.'
            assert np.isclose(
                bond.macaulay_duration,
                bond.maturity,
                atol=1e-8
                ), msg

    coupons = [0.0, 2 * ONE_PCT, 10 * ONE_PCT]
    principals = [100 * ONE_PCT]
    coupon_frequencies = [1, 2, 4]
    maturities = [1, 5, 10, 30]

    for coupon in coupons:
        for principal in principals:
            for coupon_frequency in coupon_frequencies:
                for maturity in maturities:
                    term_sheet = defaultdict(
                        None,
                        {
                            'coupon': coupon,
                            'principal': principal,
                            'coupon_frequency': coupon_frequency,
                            'maturity': maturity,
                        }
                    )

                    if verbose:
                        pprint(term_sheet)

                    bond = FixedCouponBond(term_sheet)

                    test_par_to_yield(bond)
                    test_yield_to_par(bond)
                    test_yield_to_price(bond)
                    test_dv01(bond)
                    if bond.coupon == 0.0:
                        test_macaulay_zc(bond)
