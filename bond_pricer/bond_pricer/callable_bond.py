import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Callable
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from .bond import FixedCouponBond, BondPM_mgr
from .utils import ONE_PCT


class SingleCallableFixedCouponBond(FixedCouponBond):
    def __init__(self,
                 term_sheet: defaultdict = defaultdict(),
                 marking_mode: str = 'price',
                 mark: float = 0.0,
                 recovery_rate: float = 0.4,
                 funding_rate: float = 0.0,
                 sim_config: defaultdict = defaultdict(float),
                 ) -> None:
        super().__init__(
            term_sheet=term_sheet,
            marking_mode=marking_mode,
            mark=mark,
            recovery_rate=recovery_rate,
            funding_rate=funding_rate,
            sim_config=sim_config,
            )

    @property
    def n_mc_call_sim(self) -> int:
        """Number of paths to learn from in the
        Longstaff-Schwartz algorithm.
        """
        return self.sim_config.get('n_mc_call_sim')

    @property
    def call_date(self) -> float:
        return self.term_sheet.get('call_date')

    def init_gen_call_path(self, x0: np.ndarray) -> None:
        """Initialize the path generator
        for the rates (IR) and credit (CD) factors.
        """
        mgr = BondPM_mgr()
        self._gen_call_path = mgr.model(self)
        self._gen_call_path.x0 = x0
        self._gen_call_path.T = self.maturity - self.call_date

    @property
    def model_dates(self) -> np.ndarray:
        """Dates that need to be knots in the diffusion grid.
        """
        return np.unique(np.append(self.coupon_dates, [self.call_date]))

    def generate_paths_after_call(self, x0: np.ndarray) -> tuple:
        """Generates short rates paths and corresponding
        no call PV to be used as training example for the
        regresion step in the Longstaff-Schwartz algorithm.
        """
        PV_no_calls = np.empty(self.n_mc_call_sim)

        self.init_gen_call_path(x0)

        for i in range(self.n_mc_call_sim):
            df = self.simulate_with_discount_factor(self._gen_call_path, idx=i)
            discount_factor_name = 'total_discount_factor_{}'.format(i)

            default_date = self.check_default(
                            df['CD_{}'.format(i)]
                            * self._gen_path.scheme_step,
                            self.pricing_date,
                            self.call_date,
                        )

            PV_no_call = self.PV_bullet_on_path(
                df,
                self.call_date,
                self.maturity,
                default_date=default_date,
                discount_factor_name=discount_factor_name,
            )

            PV_no_calls[i] = PV_no_call
        return PV_no_calls

    def issuer_calls(self,
                     x0: np.ndarray,
                     ) -> tuple:
        """Estimate whether the issuer calls.
        """

        PV_no_calls = self.generate_paths_after_call(x0)
        PV_call = self.principal
        return PV_call < np.mean(PV_no_calls), np.mean(PV_call < PV_no_calls)

    def model_price_paths(self) -> np.ndarray:
        PVs = np.empty(self.n_mc_sim)
        calls = []  # to estimate the call probability given survival
        paths = []

        for i in range(self.n_mc_sim):
            df = self.simulate_with_discount_factor(self._gen_path, idx=i)
            discount_factor_name = 'total_discount_factor_{}'.format(i)
            # Check for credit default on this path on
            # [pricing_date, call_date]
            default_date = self.check_default(
                df['CD_{}'.format(i)] * self._gen_path.scheme_step,
                self.pricing_date,
                self.call_date,
            )

            if default_date:
                PVs[i] = self.PV_bullet_on_path(
                        df,
                        self.pricing_date,
                        self.call_date,
                        default_date=default_date,
                        discount_factor_name=discount_factor_name,
                    )
            else:
                short_rates_on_call = np.array(
                    [
                        df['IR_{}'.format(i)].loc[self.call_date],
                        df['CD_{}'.format(i)].loc[self.call_date]
                    ]
                )
                call, prob = self.issuer_calls(short_rates_on_call)
                calls.append(prob)
                if call:
                    # No default, PV on [pricing_date, call_date],
                    # principal paid at call_date
                    PVs[i] = self.PV_bullet_on_path(
                        df,
                        self.pricing_date,
                        self.call_date,
                        discount_factor_name=discount_factor_name,
                    )
                else:
                    calls.append(False)
                    default_date = self.check_default(
                        df['CD_{}'.format(i)] * self._gen_path.scheme_step,
                        self.call_date,
                        self.maturity,
                    )

                    # If default, PV on [pricing_date, default_date],
                    # principal partially recovered at default_date
                    PVs[i] = self.PV_bullet_on_path(
                        df,
                        self.pricing_date,
                        self.maturity,
                        default_date=default_date,
                        discount_factor_name=discount_factor_name,
                        )
            paths.append(df)
        self.paths = pd.concat(paths, axis='columns')
        self.call_prob = np.mean(calls)
        return PVs


class SingleCallableFixedCouponBondLS(FixedCouponBond):
    """Single callable with Longstaff-Schwartz call model.
    """
    def __init__(self,
                 term_sheet: defaultdict = defaultdict(),
                 marking_mode: str = 'price',
                 mark: float = 0.0,
                 recovery_rate: float = 0.4,
                 funding_rate: float = 0.0,
                 sim_config: defaultdict = defaultdict(float),
                 ) -> None:
        super().__init__(
            term_sheet=term_sheet,
            marking_mode=marking_mode,
            mark=mark,
            recovery_rate=recovery_rate,
            funding_rate=funding_rate,
            sim_config=sim_config,
            )

    @property
    def n_ls_sim(self) -> int:
        """Number of paths to learn from in the
        Longstaff-Schwartz algorithm.
        """
        return self.sim_config.get('n_ls_sim')

    @property
    def ls_degree(self) -> int:
        """Polynonial regression degree in the
        Longstaff-Schwartz algorithm.
        """
        return self.sim_config.get('ls_degree')

    @property
    def call_date(self) -> float:
        return self.term_sheet.get('call_date')

    @property
    def model_dates(self) -> np.ndarray:
        """Dates that need to be knots in the diffusion grid.
        """
        return np.unique(np.append(self.coupon_dates, [self.call_date]))

    @property
    def _poly_features(self) -> Callable:
        return PolynomialFeatures(degree=self.ls_degree)

    def _generate_ls_paths(self) -> tuple:
        """Generates short rates paths and corresponding
        no call PV to be used as training example for the
        regresion step in the Longstaff-Schwartz algorithm.
        """
        PV_no_calls = np.empty(self.n_ls_sim)
        short_rates = np.empty((self.n_ls_sim, 2))

        for i in range(self.n_ls_sim):
            df = self.simulate_with_discount_factor(self._gen_path, idx=i)
            discount_factor_name = 'total_discount_factor_{}'.format(i)

            default_date = self.check_default(
                            df['CD_{}'.format(i)]
                            * self._gen_path.scheme_step,
                            self.pricing_date,
                            self.call_date,
                        )

            PV_no_call = self.PV_bullet_on_path(
                df,
                self.call_date,
                self.maturity,
                default_date=default_date,
                discount_factor_name=discount_factor_name,
            )

            short_rates[i] = [
                df['IR_{}'.format(i)].loc[self.call_date],
                df['CD_{}'.format(i)].loc[self.call_date]
            ]
            PV_no_calls[i] = PV_no_call
        return short_rates, PV_no_calls

    def ls_learning_phase(self) -> Callable:
        """Learn a log-polynomial approximator of the call decision
        function. Training is done on generated short rates paths.
        """
        short_rates, PV_no_calls = self._generate_ls_paths()
        # Transform the short rates data into polynomial features.
        poly_short_rates = self._poly_features.fit_transform(short_rates)

        log_PV_no_call_approximator = LinearRegression(
            fit_intercept=True,
            normalize=True,
            )

        log_PV_no_call_approximator.fit(
            poly_short_rates / ONE_PCT,
            np.log(PV_no_calls),
            )
        return log_PV_no_call_approximator

    def issuer_calls(self,
                     short_rates: np.ndarray,
                     log_PV_no_call_approximator: Callable,
                     ) -> bool:
        """Estimate whether the issuer calls.
        """
        PV_no_call_pred = np.exp(
            log_PV_no_call_approximator.predict(
                self._poly_features.fit_transform([short_rates / ONE_PCT])
                )
            )
        PV_call = self.principal
        return PV_call < PV_no_call_pred

    def model_price_paths(self) -> np.ndarray:
        log_PV_no_call_approximator = self.ls_learning_phase()

        PVs = np.empty(self.n_mc_sim)
        calls = []  # to estimate the call probability given survival
        paths = []

        for i in range(self.n_mc_sim):
            df = self.simulate_with_discount_factor(self._gen_path, idx=i)
            discount_factor_name = 'total_discount_factor_{}'.format(i)
            # Check for credit default on this path on
            # [pricing_date, call_date]
            default_date = self.check_default(
                df['CD_{}'.format(i)] * self._gen_path.scheme_step,
                self.pricing_date,
                self.call_date,
            )

            if default_date:
                PVs[i] = self.PV_bullet_on_path(
                        df,
                        self.pricing_date,
                        self.call_date,
                        default_date=default_date,
                        discount_factor_name=discount_factor_name,
                    )
            else:
                short_rates = np.array(
                    [
                        df['IR_{}'.format(i)].loc[self.call_date],
                        df['CD_{}'.format(i)].loc[self.call_date]
                    ]
                )

                if self.issuer_calls(short_rates, log_PV_no_call_approximator):
                    calls.append(True)
                    # No default, PV on [pricing_date, call_date],
                    # principal paid at call_date
                    PVs[i] = self.PV_bullet_on_path(
                        df,
                        self.pricing_date,
                        self.call_date,
                        discount_factor_name=discount_factor_name,
                    )
                else:
                    calls.append(False)
                    default_date = self.check_default(
                        df['CD_{}'.format(i)] * self._gen_path.scheme_step,
                        self.call_date,
                        self.maturity,
                    )

                    # If default, PV on [pricing_date, default_date],
                    # principal partially recovered at default_date
                    PVs[i] = self.PV_bullet_on_path(
                        df,
                        self.pricing_date,
                        self.maturity,
                        default_date=default_date,
                        discount_factor_name=discount_factor_name,
                        )
            paths.append(df)
        self.paths = pd.concat(paths, axis='columns')
        self.call_prob = np.mean(calls)
        return PVs
