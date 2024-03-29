import numpy as np
import pandas as pd
from collections import defaultdict
import itertools
from bond_pricer import FixedCouponBond, ONE_PCT, ONE_BP,\
    SingleCallableFixedCouponBond, SingleCallableFixedCouponBondLS

from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, Panel
from bokeh.models.widgets import TextInput, Tabs, Div, Slider
from bokeh.layouts import layout, WidgetBox
from bokeh.palettes import Dark2_5
from bokeh.plotting import figure
from bokeh.models import NumeralTickFormatter


N_MARKING_MODES = 2
MARKING_MODE_MAP = {
    1: 'price',
    2: 'ytm',
}
CASHFLOW_WIDTH = 0.05
N_PM_NAMES = 2
N_CALL_PM_NAMES = 2


def extract_numeric_input(s: str) -> int:
    try:
        int(s)
        return int(s)
    except:  # noqa
        try:
            float(s)
            return float(s)
        except:  # noqa
            raise Exception('{:s} must be numeric.'.format(s))


def make_dataset(marking_mode_int,
                 mark,
                 coupon,
                 principal,
                 maturity,
                 coupon_frequency,
                 div_,
                 ):
    """Creates a ColumnDataSource object with data to plot.
    """
    term_sheet = defaultdict(
        None,
        {
            'coupon': coupon * ONE_PCT,
            'principal': principal * ONE_PCT,
            'maturity': maturity,
            'coupon_frequency': coupon_frequency,
        }
    )

    marking_mode = MARKING_MODE_MAP[marking_mode_int]
    mark *= ONE_PCT

    bond = FixedCouponBond(
        term_sheet,
        marking_mode,
        mark,
        )

    df = pd.DataFrame(
        {
            'coupon_dates': bond.coupon_dates,
            'coupon_dates_left': bond.coupon_dates - CASHFLOW_WIDTH,
            'coupon_dates_right': bond.coupon_dates + CASHFLOW_WIDTH,
            'coupons': bond.coupon_leg.cashflows,
            'maturity': bond.maturity,
            'maturity_left': bond.maturity - CASHFLOW_WIDTH,
            'maturity_right': bond.maturity + CASHFLOW_WIDTH,
            'principal': bond.principal,
        }
    ).set_index('coupon_dates')

    macaulay_duration = bond.macaulay_duration
    df_duration = pd.DataFrame(
        {
            'x': [macaulay_duration, macaulay_duration],
            'y': [0.0, bond.principal],
        }
    )

    params_text = 'Marking mode: <b>{:s}</b><ul>\
    <li>Price = {:.2%}</li>\
    <li>YTM = {:.2%}</li>\
    <li>Macaulay Duration = {:.2f}</li>\
    <li>Modified Duration = {:.2f}</li>\
    <li>DV01 = {:.2f}</li>\
    </ul>'.format(
        marking_mode,
        bond.price,
        bond.ytm,
        macaulay_duration,
        bond.modified_duration,
        bond.dv01,
        )
    div_.text = params_text

    # Convert dataframe to column data source#
    return ColumnDataSource(df), ColumnDataSource(df_duration)


def make_dataset_model(coupon,
                       principal,
                       maturity,
                       coupon_frequency,
                       recovery_rate,
                       funding_rate,
                       init_ir,
                       init_cd,
                       scheme_steps,
                       n_mc_sim,
                       pm_name_int,
                       mr_ir,
                       mr_cd,
                       lt_ir,
                       lt_cd,
                       vol_ir,
                       vol_cd,
                       corr_ir_cd,
                       div_,
                       ):
    """Creates a ColumnDataSource object with data to plot.
    """
    term_sheet = defaultdict(
        None,
        {
            'coupon': coupon * ONE_PCT,
            'principal': principal * ONE_PCT,
            'maturity': maturity,
            'coupon_frequency': coupon_frequency,
        }
    )

    recovery_rate *= ONE_PCT
    funding_rate *= ONE_BP

    init_ir *= ONE_BP
    init_cd *= ONE_BP

    lt_ir *= ONE_BP
    lt_cd *= ONE_BP

    vol_ir *= ONE_BP
    vol_cd *= ONE_BP
    corr_ir_cd *= ONE_PCT

    if pm_name_int == 1:
        pm_name = 'Vasicek'
    elif pm_name_int == 2:
        pm_name = 'BK'
    else:
        raise ValueError('Unknown PM name.')

    model_params = {
        'mean_reversion_ir': mr_ir,
        'mean_reversion_cd': mr_cd,
        'long_term_ir': lt_ir,
        'long_term_cd': lt_cd,
        'vol_ir': vol_ir,
        'vol_cd': vol_cd,
        'corr_ir_cd': corr_ir_cd,
    }

    sim_config = defaultdict(
        None,
        {
            'pm_name': pm_name,
            'init_ir': init_ir,
            'init_cd': init_cd,
            'scheme_steps': scheme_steps,
            'n_mc_sim': n_mc_sim,
            'model_params': model_params,
        }
    )

    bond = FixedCouponBond(
        term_sheet,
        recovery_rate=recovery_rate,
        funding_rate=funding_rate,
        sim_config=sim_config,
        )

    # Fix random seed (for reproducibility of MC simulations)
    np.random.seed(2)

    params_text = 'Pricing model: {:s}<ul>\
    <li>Price = <b>{:.2%}</b></li>\
    </ul>'.format(
        bond.pm_name,
        bond.model_price,
        )
    div_.text = params_text

    # Warning! Call bond.paths AFTER calling bond.model_price
    df = bond.paths
    df = df.assign(t=df.index).copy()

    # Convert dataframe to column data source#
    return ColumnDataSource(df)


def make_dataset_call_model(coupon,
                            call_date,
                            maturity,
                            coupon_frequency,
                            recovery_rate,
                            funding_rate,
                            init_ir,
                            init_cd,
                            scheme_steps,
                            n_mc_sim,
                            n_mc_call_sim,
                            ls_degree,
                            pm_name_int,
                            call_pm_name_int,
                            mr_ir,
                            mr_cd,
                            lt_ir,
                            lt_cd,
                            vol_ir,
                            vol_cd,
                            corr_ir_cd,
                            div_,
                            ):
    """Creates a ColumnDataSource object with data to plot.
    """
    term_sheet = defaultdict(
        None,
        {
            'coupon': coupon * ONE_PCT,
            'principal': 1.0,
            'call_date': call_date,
            'maturity': maturity,
            'coupon_frequency': coupon_frequency,
        }
    )

    recovery_rate *= ONE_PCT
    funding_rate *= ONE_BP

    init_ir *= ONE_BP
    init_cd *= ONE_BP

    lt_ir *= ONE_BP
    lt_cd *= ONE_BP

    vol_ir *= ONE_BP
    vol_cd *= ONE_BP
    corr_ir_cd *= ONE_PCT

    if pm_name_int == 1:
        pm_name = 'Vasicek'
    elif pm_name_int == 2:
        pm_name = 'BK'
    else:
        raise ValueError('Unknown PM name.')

    if call_pm_name_int == 1:
        call_pm_name = 'MC'
    elif call_pm_name_int == 2:
        call_pm_name = 'Longstaff-Schwartz'
    else:
        raise ValueError('Unknown Call PM name.')

    model_params = {
        'mean_reversion_ir': mr_ir,
        'mean_reversion_cd': mr_cd,
        'long_term_ir': lt_ir,
        'long_term_cd': lt_cd,
        'vol_ir': vol_ir,
        'vol_cd': vol_cd,
        'corr_ir_cd': corr_ir_cd,
    }

    sim_config = defaultdict(
        None,
        {
            'pm_name': pm_name,
            'init_ir': init_ir,
            'init_cd': init_cd,
            'scheme_steps': scheme_steps,
            'n_mc_sim': n_mc_sim,
            'model_params': model_params,
        }
    )

    if call_pm_name == 'MC':
        sim_config['n_mc_call_sim'] = n_mc_call_sim
        bond = SingleCallableFixedCouponBond(
            term_sheet,
            recovery_rate=recovery_rate,
            funding_rate=funding_rate,
            sim_config=sim_config,
            )
    elif call_pm_name == 'Longstaff-Schwartz':
        sim_config['n_ls_sim'] = n_mc_call_sim
        sim_config['ls_degree'] = ls_degree
        bond = SingleCallableFixedCouponBondLS(
            term_sheet,
            recovery_rate=recovery_rate,
            funding_rate=funding_rate,
            sim_config=sim_config,
            )

    # Fix random seed (for reproducibility of MC simulations)
    np.random.seed(2)

    params_text = '<ul>\
    <li>PM Name: <b>{:s}</b></li>\
    <li>Call PM Name: <b>{:s}</b></li>\
    <li>Price = <b>{:.2%}</b></li>\
    <li>Call probability = <b>{:.2%}</b></li>\
    </ul>'.format(
        pm_name,
        call_pm_name,
        bond.model_price,
        bond.call_prob,
        )
    div_.text = params_text

    # Warning! Call bond.paths AFTER calling bond.model_price
    df = bond.paths
    df['t'] = df.index

    # Convert dataframe to column data source#
    return ColumnDataSource(df)


def make_plot(src, src_duration):
    """Create a figure object to host the plot.
    """
    fig = figure(
        plot_width=800,
        plot_height=400,
        title='Bond cashflows',
        x_axis_label='time',
        y_axis_label='',
        )

    fig.quad(
        top='coupons',
        bottom=0,
        left='coupon_dates_left',
        right='coupon_dates_right',
        source=src,
        fill_color='navy',
        line_color='white',
        alpha=0.5,
        legend='Coupons',
        )

    fig.quad(
        top='principal',
        bottom=0,
        left='maturity_left',
        right='maturity_right',
        source=src,
        fill_color='orange',
        line_color='white',
        alpha=0.1,
        legend='Principal',
        )

    fig.line(
        'x',
        'y',
        source=src_duration,
        line_color='red',
        line_dash='dashed',
        alpha=0.3,
        legend='Macaulay Duration',
        )

    fig.yaxis.formatter = NumeralTickFormatter(format='0%')
    fig.legend.click_policy = 'hide'
    fig.legend.location = 'top_left'

    return fig


def make_plot_model(src_model):
    """Create a figure object to host the plot.
    """
    # This won't be updated with source, so it will only show
    # initial n_mc_sim paths.
    n_mc_sim = extract_numeric_input(n_mc_sim_select.value)

    fig_ir = figure(
        plot_width=900,
        plot_height=400,
        title='Interest rates paths',
        x_axis_label='time',
        y_axis_label='IR short rate',
        )

    colors = itertools.cycle(Dark2_5)
    for i, color in zip(range(n_mc_sim), colors):
        fig_ir.line(
            't',
            'IR_{}'.format(i),
            source=src_model,
            alpha=0.8,
            color=color,
            )

    fig_ir.yaxis.formatter = NumeralTickFormatter(format='0.00%')

    fig_cd = figure(
        plot_width=900,
        plot_height=400,
        title='Hazard rates paths',
        x_axis_label='time',
        y_axis_label='CD short rate',
        )

    colors = itertools.cycle(Dark2_5)

    for i, color in zip(range(n_mc_sim), colors):
        fig_cd.line(
            't',
            'CD_{}'.format(i),
            source=src_model,
            alpha=0.8,
            color=color,
            )

    fig_cd.yaxis.formatter = NumeralTickFormatter(format='0.00%')

    return fig_ir, fig_cd


def make_plot_call_model(src_model_call):
    """Create a figure object to host the plot.
    """
    # This won't be updated with source, so it will only show
    # initial n_mc_sim paths.
    n_mc_sim = extract_numeric_input(n_mc_sim_call_select.value)

    fig_ir = figure(
        plot_width=900,
        plot_height=400,
        title='Interest rates paths',
        x_axis_label='time',
        y_axis_label='IR short rate',
        )

    colors = itertools.cycle(Dark2_5)
    for i, color in zip(range(n_mc_sim), colors):
        fig_ir.line(
            't',
            'IR_{}'.format(i),
            source=src_call_model,
            alpha=0.8,
            color=color,
            )

    fig_ir.yaxis.formatter = NumeralTickFormatter(format='0.00%')

    fig_cd = figure(
        plot_width=900,
        plot_height=400,
        title='Hazard rates paths',
        x_axis_label='time',
        y_axis_label='CD short rate',
        )

    colors = itertools.cycle(Dark2_5)

    for i, color in zip(range(n_mc_sim), colors):
        fig_cd.line(
            't',
            'CD_{}'.format(i),
            source=src_call_model,
            alpha=0.8,
            color=color,
            )

    fig_cd.yaxis.formatter = NumeralTickFormatter(format='0.00%')

    return fig_ir, fig_cd


def update(attr, old, new):
    """Update ColumnDataSource object.
    """
    # Change parameters to selected values
    marking_mode = marking_mode_select.value
    mark = extract_numeric_input(mark_select.value)
    coupon = extract_numeric_input(coupon_select.value)
    principal = extract_numeric_input(principal_select.value)
    maturity = extract_numeric_input(maturity_select.value)
    coupon_frequency = extract_numeric_input(coupon_frequency_select.value)

    new_src, new_src_duration = make_dataset(
        marking_mode,
        mark,
        coupon,
        principal,
        maturity,
        coupon_frequency,
        div,
        )

    # Update the data on the plot
    src.data.update(new_src.data)
    src_duration.data.update(new_src_duration.data)


def update_model(attr, old, new):
    """Update ColumnDataSource object.
    """
    # Change parameters to selected values
    coupon_model = extract_numeric_input(coupon_model_select.value)
    principal_model = extract_numeric_input(principal_model_select.value)
    maturity_model = extract_numeric_input(maturity_model_select.value)
    coupon_frequency_model = extract_numeric_input(
        coupon_frequency_model_select.value
    )

    recovery_rate = extract_numeric_input(recovery_rate_select.value)
    funding_rate = extract_numeric_input(funding_rate_select.value)

    init_ir = extract_numeric_input(init_ir_select.value)
    init_cd = extract_numeric_input(init_cd_select.value)

    scheme_steps = extract_numeric_input(scheme_steps_select.value)
    n_mc_sim = extract_numeric_input(n_mc_sim_select.value)

    pm_name = pm_name_select.value

    mr_ir = extract_numeric_input(mr_ir_select.value)
    mr_cd = extract_numeric_input(mr_cd_select.value)
    lt_ir = extract_numeric_input(lt_ir_select.value)
    lt_cd = extract_numeric_input(lt_cd_select.value)
    vol_ir = extract_numeric_input(vol_ir_select.value)
    vol_cd = extract_numeric_input(vol_cd_select.value)
    corr_ir_cd = extract_numeric_input(corr_ir_cd_select.value)

    new_src_model = make_dataset_model(
        coupon_model,
        principal_model,
        maturity_model,
        coupon_frequency_model,
        recovery_rate,
        funding_rate,
        init_ir,
        init_cd,
        scheme_steps,
        n_mc_sim,
        pm_name,
        mr_ir,
        mr_cd,
        lt_ir,
        lt_cd,
        vol_ir,
        vol_cd,
        corr_ir_cd,
        div_model,
        )

    # Update the data on the plot
    src_model.data.update(new_src_model.data)


def update_call_model(attr, old, new):
    """Update ColumnDataSource object.
    """
    # Change parameters to selected values
    coupon = extract_numeric_input(coupon_call_model_select.value)
    call_date = extract_numeric_input(call_date_model_select.value)
    maturity = extract_numeric_input(maturity_call_model_select.value)
    coupon_freq = extract_numeric_input(coupon_frequency_model_select.value)

    recovery_rate = extract_numeric_input(recovery_rate_call_select.value)
    funding_rate = extract_numeric_input(funding_rate_call_select.value)

    init_ir = extract_numeric_input(init_ir_call_select.value)
    init_cd = extract_numeric_input(init_cd_call_select.value)

    scheme_steps = extract_numeric_input(scheme_steps_call_select.value)
    n_mc_sim = extract_numeric_input(n_mc_sim_call_select.value)
    n_mc_call_sim = extract_numeric_input(n_mc_call_sim_select.value)
    ls_degree = extract_numeric_input(ls_degree_call_select.value)

    pm_name = pm_name_call_select.value
    call_pm_name = call_pm_name_call_select.value

    mr_ir = extract_numeric_input(mr_ir_call_select.value)
    mr_cd = extract_numeric_input(mr_cd_call_select.value)
    lt_ir = extract_numeric_input(lt_ir_call_select.value)
    lt_cd = extract_numeric_input(lt_cd_call_select.value)
    vol_ir = extract_numeric_input(vol_ir_call_select.value)
    vol_cd = extract_numeric_input(vol_cd_call_select.value)
    corr_ir_cd = extract_numeric_input(corr_ir_cd_call_select.value)

    new_src_call_model = make_dataset_call_model(
        coupon,
        call_date,
        maturity,
        coupon_freq,
        recovery_rate,
        funding_rate,
        init_ir,
        init_cd,
        scheme_steps,
        n_mc_sim,
        n_mc_call_sim,
        ls_degree,
        pm_name,
        call_pm_name,
        mr_ir,
        mr_cd,
        lt_ir,
        lt_cd,
        vol_ir,
        vol_cd,
        corr_ir_cd,
        div_call_model,
        )

    # Update the data on the plot
    src_call_model.data.update(new_src_call_model.data)


######################################################################
# BOND QUOTING
######################################################################
# Marking
marking_mode_select = Slider(start=1,
                             end=N_MARKING_MODES,
                             step=1,
                             title='Marking mode',
                             value=1,
                             )
mark_select = TextInput(value='100.00', title='Mark')

# Bond term sheet
coupon_select = TextInput(value='5.00', title='Coupon (%)')
principal_select = TextInput(value='100', title='Principal')
maturity_select = TextInput(value='5', title='Maturity (y)')
coupon_frequency_select = TextInput(
    value='1',
    title='Coupon Frequency (per year)'
    )

# Update the plot when parameters are changed
marking_mode_select.on_change('value', update)
mark_select.on_change('value', update)
coupon_select.on_change('value', update)
principal_select.on_change('value', update)
maturity_select.on_change('value', update)
coupon_frequency_select.on_change('value', update)

marking_mode = marking_mode_select.value
mark = extract_numeric_input(mark_select.value)
coupon = extract_numeric_input(coupon_select.value)
principal = extract_numeric_input(principal_select.value)
maturity = extract_numeric_input(maturity_select.value)
coupon_frequency = extract_numeric_input(coupon_frequency_select.value)

div = Div(text='<b></b><br>', width=250, height=150)

src, src_duration = make_dataset(
    marking_mode,
    mark,
    coupon,
    principal,
    maturity,
    coupon_frequency,
    div,
    )

controls = WidgetBox(
    coupon_select,
    principal_select,
    maturity_select,
    coupon_frequency_select,
    marking_mode_select,
    mark_select,
    div,
    width=250,
    height=550,
    )

fig = make_plot(src, src_duration)

# Create a row layout
layout_quoting = layout(
    [
        [controls, fig],
    ],
)

# Make a tab with the layout
tab = Panel(child=layout_quoting, title='Bond pricer')


######################################################################
# BOND MODEL PRICE
######################################################################
# Bond term sheet
coupon_model_select = TextInput(value='5.00', title='Coupon (%)')
principal_model_select = TextInput(value='100', title='Principal')
maturity_model_select = TextInput(value='5', title='Maturity (y)')
coupon_frequency_model_select = TextInput(
    value='1',
    title='Coupon Frequency (per year)'
    )

recovery_rate_select = TextInput(value='40.0', title='Recovery rate (%)')
funding_rate_select = TextInput(value='0.0', title='Funding rate (bps)')
scheme_steps_select = TextInput(value='100', title='Number of diffusion steps')
n_mc_sim_select = TextInput(value='10', title='Number of MC simulations')

init_ir_select = TextInput(value='50.0', title='Spot IR (bp)')
init_cd_select = TextInput(value='150.0', title='Spot CD (bp)')

pm_name_select = Slider(start=1,
                        end=N_PM_NAMES,
                        step=1,
                        title='PM name',
                        value=1,
                        )

mr_ir_select = TextInput(value='1.0', title='Mean reversion IR')
mr_cd_select = TextInput(value='1.0', title='Mean reversion CD')
lt_ir_select = TextInput(value='50.0', title='Long-term IR (bp)')
lt_cd_select = TextInput(value='150.0', title='Long-term CD (bp)')
vol_ir_select = TextInput(value='20.0', title='Vol IR (bps)')
vol_cd_select = TextInput(value='50.0', title='Vol CD (bps)')
corr_ir_cd_select = TextInput(value='0.0', title='Corr IR-CD (%)')

# Update the plot when parameters are changed
coupon_model_select.on_change('value', update_model)
principal_model_select.on_change('value', update_model)
maturity_model_select.on_change('value', update_model)
coupon_frequency_model_select.on_change('value', update_model)

recovery_rate_select.on_change('value', update_model)
funding_rate_select.on_change('value', update_model)
scheme_steps_select.on_change('value', update_model)
n_mc_sim_select.on_change('value', update_model)

init_ir_select.on_change('value', update_model)
init_cd_select.on_change('value', update_model)

pm_name_select.on_change('value', update_model)

mr_ir_select.on_change('value', update_model)
mr_cd_select.on_change('value', update_model)
lt_ir_select.on_change('value', update_model)
lt_cd_select.on_change('value', update_model)
vol_ir_select.on_change('value', update_model)
vol_cd_select.on_change('value', update_model)
corr_ir_cd_select.on_change('value', update_model)

coupon_model = extract_numeric_input(coupon_model_select.value)
principal_model = extract_numeric_input(principal_model_select.value)
maturity_model = extract_numeric_input(maturity_model_select.value)
coupon_frequency_model = extract_numeric_input(
    coupon_frequency_model_select.value
)

recovery_rate = extract_numeric_input(recovery_rate_select.value)
funding_rate = extract_numeric_input(funding_rate_select.value)
scheme_steps = extract_numeric_input(scheme_steps_select.value)
n_mc_sim = extract_numeric_input(n_mc_sim_select.value)

init_ir = extract_numeric_input(init_ir_select.value)
init_cd = extract_numeric_input(init_cd_select.value)

pm_name = pm_name_select.value

mr_ir = extract_numeric_input(mr_ir_select.value)
mr_cd = extract_numeric_input(mr_cd_select.value)
lt_ir = extract_numeric_input(lt_ir_select.value)
lt_cd = extract_numeric_input(lt_cd_select.value)
vol_ir = extract_numeric_input(vol_ir_select.value)
vol_cd = extract_numeric_input(vol_cd_select.value)
corr_ir_cd = extract_numeric_input(corr_ir_cd_select.value)

div_model = Div(text='<b></b><br>', width=250, height=150)

src_model = make_dataset_model(
    coupon_model,
    principal_model,
    maturity_model,
    coupon_frequency_model,
    recovery_rate,
    funding_rate,
    init_ir,
    init_cd,
    scheme_steps,
    n_mc_sim,
    pm_name,
    mr_ir,
    mr_cd,
    lt_ir,
    lt_cd,
    vol_ir,
    vol_cd,
    corr_ir_cd,
    div_model,
    )

controls_model = WidgetBox(
    coupon_model_select,
    principal_model_select,
    maturity_model_select,
    coupon_frequency_model_select,
    recovery_rate_select,
    funding_rate_select,
    init_ir_select,
    init_cd_select,
    pm_name_select,
    width=350,
    height=700,
    )

controls_model_params = WidgetBox(
    scheme_steps_select,
    n_mc_sim_select,
    mr_ir_select,
    mr_cd_select,
    lt_ir_select,
    lt_cd_select,
    vol_ir_select,
    vol_cd_select,
    corr_ir_cd_select,
    width=250,
    height=700,
    )

controls_model_price = WidgetBox(
    div_model,
    width=250,
    height=700,
)


fig_ir, fig_cd = make_plot_model(src_model)

# Create a row layout
layout_model = layout(
    [
        [controls_model, controls_model_params, controls_model_price],
        [fig_ir],
        [fig_cd],
    ],
)

# Make a tab with the layout
tab_model = Panel(child=layout_model, title='Bond model')

######################################################################
# CALLABLE BOND MODEL PRICE
######################################################################
# Bond term sheet
coupon_call_model_select = TextInput(value='5.00', title='Coupon (%)')
call_date_model_select = TextInput(value='3', title='Call date (y)')
maturity_call_model_select = TextInput(value='5', title='Maturity (y)')
coupon_frequency_call_model_select = TextInput(
    value='1',
    title='Coupon Frequency (per year)'
    )

recovery_rate_call_select = TextInput(value='40.0', title='Recovery rate (%)')
funding_rate_call_select = TextInput(value='0.0', title='Funding rate (bps)')
scheme_steps_call_select = TextInput(
    value='100', title='Number of diffusion steps'
)
n_mc_sim_call_select = TextInput(value='10', title='Number of MC simulations')
n_mc_call_sim_select = TextInput(
    value='10',
    title='Number of call MC simulations'
    )
ls_degree_call_select = TextInput(value='3', title='Degree of LS regression')

init_ir_call_select = TextInput(value='50.0', title='Spot IR (bp)')
init_cd_call_select = TextInput(value='150.0', title='Spot CD (bp)')

pm_name_call_select = Slider(start=1,
                             end=N_PM_NAMES,
                             step=1,
                             title='PM name',
                             value=1,
                             )

call_pm_name_call_select = Slider(start=1,
                                  end=N_CALL_PM_NAMES,
                                  step=1,
                                  title='Call PM name',
                                  value=1,
                                  )

mr_ir_call_select = TextInput(value='1.0', title='Mean reversion IR')
mr_cd_call_select = TextInput(value='1.0', title='Mean reversion CD')
lt_ir_call_select = TextInput(value='50.0', title='Long-term IR (bp)')
lt_cd_call_select = TextInput(value='150.0', title='Long-term CD (bp)')
vol_ir_call_select = TextInput(value='20.0', title='Vol IR (bps)')
vol_cd_call_select = TextInput(value='50.0', title='Vol CD (bps)')
corr_ir_cd_call_select = TextInput(value='0.0', title='Corr IR-CD (%)')

# Update the plot when parameters are changed
coupon_call_model_select.on_change('value', update_call_model)
call_date_model_select.on_change('value', update_call_model)
maturity_call_model_select.on_change('value', update_call_model)
coupon_frequency_call_model_select.on_change('value', update_call_model)

recovery_rate_call_select.on_change('value', update_call_model)
funding_rate_call_select.on_change('value', update_call_model)
scheme_steps_call_select.on_change('value', update_call_model)
n_mc_sim_call_select.on_change('value', update_call_model)
n_mc_call_sim_select.on_change('value', update_call_model)
ls_degree_call_select.on_change('value', update_call_model)

init_ir_call_select.on_change('value', update_call_model)
init_cd_call_select.on_change('value', update_call_model)

pm_name_call_select.on_change('value', update_call_model)
call_pm_name_call_select.on_change('value', update_call_model)

mr_ir_call_select.on_change('value', update_call_model)
mr_cd_call_select.on_change('value', update_call_model)
lt_ir_call_select.on_change('value', update_call_model)
lt_cd_call_select.on_change('value', update_call_model)
vol_ir_call_select.on_change('value', update_call_model)
vol_cd_call_select.on_change('value', update_call_model)
corr_ir_cd_call_select.on_change('value', update_call_model)

coupon_call_model = extract_numeric_input(coupon_call_model_select.value)
call_date_model = extract_numeric_input(call_date_model_select.value)
maturity_call_model = extract_numeric_input(maturity_call_model_select.value)
coupon_frequency_call_model = extract_numeric_input(
    coupon_frequency_call_model_select.value
)

recovery_rate_call = extract_numeric_input(recovery_rate_call_select.value)
funding_rate_call = extract_numeric_input(funding_rate_call_select.value)
scheme_steps_call = extract_numeric_input(scheme_steps_call_select.value)
n_mc_sim_call = extract_numeric_input(n_mc_sim_call_select.value)
n_mc_call_sim = extract_numeric_input(n_mc_call_sim_select.value)
ls_degree_call = extract_numeric_input(ls_degree_call_select.value)

init_ir_call = extract_numeric_input(init_ir_call_select.value)
init_cd_call = extract_numeric_input(init_cd_call_select.value)

pm_name_call = pm_name_call_select.value
call_pm_name_call = call_pm_name_call_select.value

mr_ir_call = extract_numeric_input(mr_ir_call_select.value)
mr_cd_call = extract_numeric_input(mr_cd_call_select.value)
lt_ir_call = extract_numeric_input(lt_ir_call_select.value)
lt_cd_call = extract_numeric_input(lt_cd_call_select.value)
vol_ir_call = extract_numeric_input(vol_ir_call_select.value)
vol_cd_call = extract_numeric_input(vol_cd_call_select.value)
corr_ir_cd_call = extract_numeric_input(corr_ir_cd_call_select.value)

div_call_model = Div(text='<b></b><br>', width=250, height=150)

src_call_model = make_dataset_call_model(
    coupon_call_model,
    call_date_model,
    maturity_call_model,
    coupon_frequency_call_model,
    recovery_rate_call,
    funding_rate_call,
    init_ir_call,
    init_cd_call,
    scheme_steps_call,
    n_mc_sim_call,
    n_mc_call_sim,
    ls_degree_call,
    pm_name_call,
    call_pm_name_call,
    mr_ir_call,
    mr_cd_call,
    lt_ir_call,
    lt_cd_call,
    vol_ir_call,
    vol_cd_call,
    corr_ir_cd_call,
    div_call_model,
    )

controls_call_model = WidgetBox(
    coupon_call_model_select,
    call_date_model_select,
    maturity_call_model_select,
    coupon_frequency_call_model_select,
    recovery_rate_call_select,
    funding_rate_call_select,
    init_ir_call_select,
    init_cd_call_select,
    pm_name_call_select,
    call_pm_name_call_select,
    width=350,
    height=700,
    )

controls_call_model_params = WidgetBox(
    scheme_steps_call_select,
    n_mc_sim_call_select,
    n_mc_call_sim_select,
    ls_degree_call_select,
    mr_ir_call_select,
    mr_cd_call_select,
    lt_ir_call_select,
    lt_cd_call_select,
    vol_ir_call_select,
    vol_cd_call_select,
    corr_ir_cd_call_select,
    width=250,
    height=700,
    )

controls_call_model_price = WidgetBox(
    div_call_model,
    width=250,
    height=700,
)


fig_ir_call, fig_cd_call = make_plot_call_model(src_call_model)

# Create a row layout
layout_call_model = layout(
    [
        [
            controls_call_model,
            controls_call_model_params,
            controls_call_model_price
        ],
        [fig_ir_call],
        [fig_cd_call],
    ],
)

# Make a tab with the layout
tab_call_model = Panel(child=layout_call_model, title='Callable bond model')

tabs = Tabs(tabs=[tab, tab_model, tab_call_model])

curdoc().add_root(tabs)
