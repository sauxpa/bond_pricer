import numpy as np
import pandas as pd
from scipy.optimize import root
from collections import defaultdict
from typing import Callable, Union
from ito_diffusions import Vasicek_multi_d, BlackKarasinski_multi_d
from .leg import CashflowLeg
from .priceable import Priceable


class FixedCouponBond(Priceable):
    def __init__(self,
                 term_sheet: defaultdict = defaultdict(),
                 marking_mode: str = 'price',
                 mark: float = 0.0,
                 recovery_rate: float = 0.4,
                 funding_rate: float = 0.0,
                 sim_config: defaultdict = defaultdict(float),
                 ) -> None:
        self.check_marking_mode(marking_mode)
        self._marking_mode = marking_mode.lower()
        self._mark = mark
        self._recovery_rate = recovery_rate
        self._funding_rate = funding_rate
        self._sim_config = sim_config
        self._gen_path = None
        self.paths = pd.DataFrame()

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
    def recovery_rate(self) -> float:
        """Fraction of principal recoverd in case of default.
        Used in model price only.
        """
        return self._recovery_rate

    @recovery_rate.setter
    def recovery_rate(self, new_recovery_rate: float) -> None:
        self._recovery_rate = new_recovery_rate

    @property
    def funding_rate(self) -> float:
        """Constant extra discounting on top of the short rate.
        Used in model price only.
        """
        return self._funding_rate

    @funding_rate.setter
    def funding_rate(self, new_funding_rate: float) -> None:
        self._funding_rate = new_funding_rate

    @property
    def sim_config(self) -> defaultdict:
        return self._sim_config

    @sim_config.setter
    def sim_config(self, new_sim_config: defaultdict) -> None:
        self._sim_config = new_sim_config

    @property
    def coupon(self) -> float:
        return float(self.term_sheet.get('coupon'))

    @property
    def principal(self) -> float:
        return float(self.term_sheet.get('principal', 1.0))

    @property
    def maturity(self) -> float:
        return float(self.term_sheet.get('maturity'))

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
        return self.duration_calc(self.price) / (
            1 + self.day_count_fraction * self.ytm
            )

    @property
    def modified_duration(self) -> float:
        return self.dv01 / self.price

    @property
    def pm_name(self) -> str:
        return self.sim_config.get('pm_name')

    @property
    def init_ir(self) -> float:
        """Starting point for short interest rate.
        """
        return self.sim_config.get('init_ir')

    @property
    def init_cd(self) -> float:
        """Starting point for short credit hazard rate.
        """
        return self.sim_config.get('init_cd')

    @property
    def scheme_steps(self) -> int:
        return self.sim_config.get('scheme_steps')

    @property
    def n_mc_sim(self) -> int:
        """Number of Monte Carlo paths to simulate.
        """
        return self.sim_config.get('n_mc_sim')

    @property
    def model_params(self) -> defaultdict:
        """Dict of parameters for the diffusion model.
        """
        return self.sim_config.get('model_params')

    def init_gen_path(self) -> None:
        """Initialize the path generator
        for the rates (IR) and credit (CD) factors.
        """
        mgr = BondPM_mgr()
        self._gen_path = mgr.model(self)

    def check_default(self,
                      intensity: pd.Series,
                      tstart: float,
                      tend: float
                      ) -> Union[float, None]:
        """Jump to default is an inhomogeneous Poisson process.
        Check if there is a jump between tstart and tend.
        """
        default = np.where(
            np.random.poisson(intensity.loc[tstart:tend]) > 0
            )[0]
        if len(default):
            return intensity.index[default[0]]
        else:
            return None

    @property
    def model_dates(self) -> np.ndarray:
        """Dates that need to be knots in the diffusion grid.
        """
        return self.coupon_dates

    def insert_dates_index(self,
                           df_: pd.DataFrame,
                           interp_method='linear',
                           ) -> pd.DataFrame:
        """Coupon dates may no coincide with diffusion scheme knots.
        Interpolate between scheme knots to add coupon dates.
        Interpolate in short rates space, i.e before computing
        the discount factors (more precise than interpolating the discount
        factors);
        """
        new_idx = set(np.concatenate([self.model_dates, df_.index]))
        return df_.reindex(index=new_idx).sort_index().interpolate(
            method=interp_method,
            axis=0,
            )

    def simulate_with_discount_factor(self,
                                      gen_: Callable,
                                      idx=0,
                                      ) -> pd.DataFrame:
        df = gen_.simulate()
        df = self.insert_dates_index(df)

        df_discount = np.exp(
            -gen_.scheme_step * df[['IR', 'CD']].cumsum().iloc[:-1]
        )
        df_discount.index = df.index[1:]
        df['ir_discount_factor_{}'.format(idx)] = df_discount['IR']
        df['survival_prob_{}'.format(idx)] = df_discount['CD']
        df['fr_discount_factor_{}'.format(idx)] = np.exp(
            -df.index * self.funding_rate
            )
        df['ir_discount_factor_{}'.format(idx)].iloc[0] = 1.0
        df['survival_prob_{}'.format(idx)].iloc[0] = 1.0
        df['total_discount_factor_{}'.format(idx)] \
            = df['ir_discount_factor_{}'.format(idx)] \
            * df['fr_discount_factor_{}'.format(idx)]
        df = df.rename(
            columns={
                'IR': 'IR_{}'.format(idx),
                'CD': 'CD_{}'.format(idx),
                }
            )
        return df

    def PV_bullet_on_path(
        self,
        df_: pd.DataFrame,
        tstart: float,
        tend: float,
        default_date: None = None,
        discount_factor_name: str = 'total_discount_factor',
    ) -> float:
        """For simplicity, assume no accrued coupon is recovered
        in case of default, only the principal.
        Path is on (tstart, tend] if default_date is None,
        otherwise it is on (tstart, default_date].
        """
        tend = default_date if default_date is not None else tend
        recovered_principal = self.recovery_rate * self.principal \
            if default_date is not None else self.principal
        remaining_coupons = self.coupon_dates[
            (self.coupon_dates > tstart) * (self.coupon_dates <= tend)
        ]
        PV_bullet_coupon = df_[discount_factor_name].loc[
            remaining_coupons
        ].sum() * self.coupon * self.day_count_fraction
        PV_bullet_principal = df_[discount_factor_name].loc[
            tend
        ] * recovered_principal
        return PV_bullet_coupon + PV_bullet_principal

    def model_price_paths(self) -> np.ndarray:
        PVs = np.empty(self.n_mc_sim)
        paths = []
        for i in range(self.n_mc_sim):
            df = self.simulate_with_discount_factor(self._gen_path, idx=i)
            # Check for credit default on this path on
            # [pricing_date, maturity]
            default_date = self.check_default(
                df['CD_{}'.format(i)] * self._gen_path.scheme_step,
                self.pricing_date,
                self.maturity,
            )
            PVs[i] = self.PV_bullet_on_path(
                df,
                self.pricing_date,
                self.maturity,
                default_date=default_date,
                discount_factor_name='total_discount_factor_{}'.format(i)
            )
            paths.append(df)
        self.paths = pd.concat(paths, axis='columns')
        return PVs

    @property
    def model_price(self) -> float:
        self.init_gen_path()
        PVs = self.model_price_paths()
        return np.mean(PVs)


class BondPM_mgr():
    def __init__(self):
        pass

    def model(self, bond: 'FixedCouponBond') -> Callable:
        if bond.pm_name == 'Vasicek':
            assert len(bond.model_params) == 7
            cov_matrix = [
                [
                    bond.model_params['vol_ir'],
                    0.0
                ],
                [
                    bond.model_params['vol_cd']
                    * bond.model_params['corr_ir_cd'],
                    bond.model_params['vol_cd']
                    * np.sqrt(1 - bond.model_params['corr_ir_cd'])
                ]
            ]
            return Vasicek_multi_d(
                x0=[bond.init_ir, bond.init_cd],
                T=bond.maturity,
                mean_reversion=[bond.model_params['mean_reversion_ir'],
                                bond.model_params['mean_reversion_cd']],
                long_term=[bond.model_params['long_term_ir'],
                           bond.model_params['long_term_cd']],
                vol=cov_matrix,
                scheme_steps=bond.scheme_steps,
                keys=['IR', 'CD'],
                barrier=[None, 0.0],
                barrier_condition='absorb'
                )
        elif bond.pm_name == 'BK':
            assert len(bond.model_params) == 7
            cov_matrix = [
                [
                    bond.model_params['vol_ir'],
                    0.0
                ],
                [
                    bond.model_params['vol_cd']
                    * bond.model_params['corr_ir_cd'],
                    bond.model_params['vol_cd']
                    * np.sqrt(1 - bond.model_params['corr_ir_cd'])
                ]
            ]
            return BlackKarasinski_multi_d(
                x0=[bond.init_ir, bond.init_cd],
                T=bond.maturity,
                mean_reversion=[bond.model_params['mean_reversion_ir'],
                                bond.model_params['mean_reversion_cd']],
                long_term=[bond.model_params['long_term_ir'],
                           bond.model_params['long_term_cd']],
                vol=cov_matrix,
                scheme_steps=bond.scheme_steps,
                keys=['IR', 'CD'],
                barrier=[None, 0.0],
                barrier_condition='absorb'
                )
        else:
            raise Exception(
                'Unknown model name: {:s}'.format(bond.pm_name)
                )
