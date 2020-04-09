import pandas as pd
from collections import defaultdict
from bond_pricer import FixedCouponBond, ONE_PCT

from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, Panel
from bokeh.models.widgets import TextInput, Tabs, Div, Slider
from bokeh.layouts import layout, WidgetBox
from bokeh.plotting import figure
from bokeh.models import NumeralTickFormatter


N_MARKING_MODES = 2
MARKING_MODE_MAP = {
    1: 'price',
    2: 'ytm',
}
CASHFLOW_WIDTH = 0.05


def extract_numeric_input(s: str) -> int:
    try:
        int(s)
        return int(s)
    except:  # noqa
        try:
            float(s)
            return float(s)
        except:  # noqa
            raise Exception('{:s} must be numeric.')


def make_dataset(marking_mode_int,
                 mark,
                 coupon,
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
            'principal': 1.0,
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


def update(attr, old, new):
    """Update ColumnDataSource object.
    """
    # Change parameters to selected values
    marking_mode = marking_mode_select.value
    mark = extract_numeric_input(mark_select.value)
    coupon = extract_numeric_input(coupon_select.value)
    maturity = extract_numeric_input(maturity_select.value)
    coupon_frequency = extract_numeric_input(coupon_frequency_select.value)

    new_src, new_src_duration = make_dataset(
        marking_mode,
        mark,
        coupon,
        maturity,
        coupon_frequency,
        div,
        )

    # Update the data on the plot
    src.data.update(new_src.data)
    src_duration.data.update(new_src_duration.data)


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
maturity_select = TextInput(value='5', title='Maturity (y)')
coupon_frequency_select = TextInput(
    value='1',
    title='Coupon Frequency (per year)'
    )

# Update the plot when parameters are changed
marking_mode_select.on_change('value', update)
mark_select.on_change('value', update)
coupon_select.on_change('value', update)
maturity_select.on_change('value', update)
coupon_frequency_select.on_change('value', update)

marking_mode = marking_mode_select.value
mark = extract_numeric_input(mark_select.value)
coupon = extract_numeric_input(coupon_select.value)
maturity = extract_numeric_input(maturity_select.value)
coupon_frequency = extract_numeric_input(coupon_frequency_select.value)

div = Div(text='<b></b><br>', width=250, height=150)

src, src_duration = make_dataset(
    marking_mode,
    mark,
    coupon,
    maturity,
    coupon_frequency,
    div,
    )

controls = WidgetBox(
    coupon_select,
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
layout = layout(
    [
        [controls, fig],
    ],
)

# Make a tab with the layout
tab = Panel(child=layout, title='Bond pricer')

tabs = Tabs(tabs=[tab])

curdoc().add_root(tabs)
