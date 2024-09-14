from math import pi

import pandas as pd

from bokeh.palettes import Category20c
from bokeh.plotting import figure, show
from bokeh.transform import cumsum
from bokeh.layouts import column

x_without = {
    'Training': 358.56,
    'Test': 90.26,
}

x_with = {
    'Training': 301.27,
    'Test': 90.26,
    'Validierung': 57.29
}

ownColorPalette = [
    '#008080', # category10: teal
    '#ff8000', # category10: orange
    '#800080', # category10: violet
    '#2ca02c', # category10: green
    '#d62728', # category10: red
    '#8c564b', # category10: brown
    '#1f77b4', # category10: blue
    '#7f7f7f', # category10: gray
    '#bcbd22', # category10: lime
    '#17becf'  # category10: cyan
    ] 

data = pd.Series(x_with).reset_index(name='value').rename(columns={'index': 'set'})
data['angle'] = data['value']/data['value'].sum() * 2*pi
data['color'] = ownColorPalette[0:len(x_with)]

p1 = figure(title="Einteilung des Datensatzes mit Validierungssatz", x_range=(-0.5, 1.0))

p1.wedge(x=0, y=1, radius=0.4,
        start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
        line_color="white", fill_color='color', legend_field='set', source=data)

p1.axis.axis_label = None
p1.axis.visible = False
p1.grid.grid_line_color = None

data = pd.Series(x_without).reset_index(name='value').rename(columns={'index': 'set'})
data['angle'] = data['value']/data['value'].sum() * 2*pi
data['color'] = ownColorPalette[0:len(x_without)]

p2 = figure(title="Einteilung des Datensatzes ohne Validierungssatz", x_range=(-0.5, 1.0))

p2.wedge(x=0, y=1, radius=0.4,
        start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
        line_color="white", fill_color='color', legend_field='set', source=data)

p2.axis.axis_label = None
p2.axis.visible = False
p2.grid.grid_line_color = None

plot = column([p1, p2], sizing_mode='fixed', width=350, height=210)

show(plot)