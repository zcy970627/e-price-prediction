import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import pathlib
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
from datetime import date
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
from app import app
from apps import side_and_nav_bars

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../dataset").resolve()
data_dictionary=["Max temperature in C°","Min temperature in C°","UV index","Heat index in C°", "Average Temprature in C°","Wind Gust in Kmph",
                 "Humidity","Wind speed in Kmph ","Electricity price in Euro","Brent Oil price in Dollar","WIT Oil price in Dollar","Natural Gas price in Dollar"]
data_in_csv_dictionary=['maxtempC','mintempC','uvIndex','HeatIndexC','tempC','WindGustKmph','humidity','windspeedKmph','elec_price','Brent_Prices','WTI_Prices','Henry_Hub_Price']
data_csv = pd.read_csv(DATA_PATH.joinpath("historical_data_all.csv"))
data_dates = pd.to_datetime(data_csv['Unnamed: 0']).dt.strftime('%Y-%m-%d %H:%M:%S').to_list()
fig=go.Figure(data=[go.Scatter(x=data_dates[-5:], y=[])],layout = go.Layout(height=700))
fig.update_xaxes(title_text="Time")
matrix = np.load(DATA_PATH.joinpath("corr.npy"))


labels =['maxtempC', 'mintempC', 'uvIndex', 'HeatIndexC', 'WindGustKmph', 'humidity', 'tempC', 'windspeedKmph', 'Brent_Prices', 'Henry_Hub_Price', 'WTI_Prices', 'elec_price']
X = [labels[k] for k in range(12)]
hovertext = [[f'corr({X[i]}, {X[j]})= {matrix[i][j]:.2f}' if i > j else '' for j in range(12)] for i in range(12)]
sns_colorscale = [[0.0, '#3f7f93'],
                  [0.071, '#5890a1'],
                  [0.143, '#72a1b0'],
                  [0.214, '#8cb3bf'],
                  [0.286, '#a7c5cf'],
                  [0.357, '#c0d6dd'],
                  [0.429, '#dae8ec'],
                  [0.5, '#f2f2f2'],
                  [0.571, '#f7d7d9'],
                  [0.643, '#f2bcc0'],
                  [0.714, '#eda3a9'],
                  [0.786, '#e8888f'],
                  [0.857, '#e36e76'],
                  [0.929, '#de535e'],
                  [1.0, '#d93a46']]

heat = go.Heatmap(z=matrix,
                  x=X,
                  y=X,
                  xgap=1, ygap=1,
                  colorscale=sns_colorscale,
                  colorbar_thickness=20,
                  colorbar_ticklen=3,
                  hovertext=hovertext,
                  hoverinfo='text'
                  )


layout = go.Layout(width=700, height=700,
                   xaxis_showgrid=False,
                   yaxis_showgrid=False,
                   yaxis_autorange='reversed')

figCorr = go.Figure(data=[heat], layout=layout)




#fig_heatmap=go.Figure(data=go.Heatmap(),layout = go.Layout(height=700))
historical_content = html.Div(id="page-2-content", className="content-style",
children=[
dbc.Card(
    [
     dbc.CardBody(
        [
        html.Div([
        html.H1('Data Overview', id="Overview"),
            html.P(["After the extraction,  the correlation between the history of elecricity price and other data. A threshold of 0,1 must be achieved "
                    "for each type of data in order to keep it. In this graph below we visualize the data that  that have correlation factor greater then 0.1. The total type of data is 12."
                    "Each type of data can be visualized individually or alongside the electricity price"]),
            html.Div([
                html.Div([
                    html.Pre(children="Historical data", style={"fontSize": "150%"}),
                    dcc.Dropdown(
                        id='data-type', value='', clearable=True,
                        persistence=True, persistence_type='session',
                        options=[{'label': x, 'value': x} for x in data_dictionary]
                    )
                ],className='dropdown-style'),

                html.Div([
                    html.Pre(children="Period of time", style={"fontSize": "150%"}),
                    dcc.DatePickerRange(
                            id='my-date-picker-range',
                            display_format='YYYY-MM-DD',
                            min_date_allowed=date(2016, 1, 1),
                            max_date_allowed=date(2021, 8, 1),
                            initial_visible_month=date(2021, 1, 1),
                    ),
                ], ),

                html.Div([
                    html.Pre(children="Visualization type", style={"fontSize": "150%"}),
                    dcc.Dropdown(
                        id='type-visual', value='', clearable=True,
                        persistence=True, persistence_type='session',
                        options=[{'label': x, 'value': x} for x in ["One Graph", "One Graph + electricity price"]]
                    )
                ], className='dropdown-style'),
            ],className='row')
        ], ),
        dcc.Graph(id='visual-graph', figure=fig),
        html.Div([
            html.H1('Data correlation',id="coorelation"),
            html.P(["The heat map below demonstrates the correlation factor between the elecricity price and each of type of data with 0,1. The data are Max temperature in "
                    "C°, Min temperature in C°UV index, Heat index in C°, Average Temprature in C°, Wind Gust in Kmph,Humidity, Wind speed in Kmph, Electricity price in Euro, "
                    "Brent Oil price in Dollar, WIT Oil price in Dollar and Natural Gas price in Dollar. "

            ], className='row')
        ], ),
        dcc.Graph(id='heatmap-graph', figure=figCorr, className="figure-center"),
        html.Div([
            html.P(["As a next step we can predit the future of electricity price. For that we need either one dimensional model or multiple dimensional. In case of only one input"
                    "it is self explanatory that we use the historical data of electricity price. Asd for the mutlivariante model we opted for the 5 best correlated data with the "
                    "electricity price and these are and Electricity price in Euro, average Temprature in C°,Wind speed in Kmph, Electricity price in Euro, "
                    "Brent Oil price in Dollar, WIT Oil price in Dollar and Natural Gas price in Dollar  "

            ], className='row')
        ], ),
        ])
    ])
])
layout = html.Div([historical_content, side_and_nav_bars.navbar, side_and_nav_bars.sidebar])

@app.callback(
    Output('visual-graph', component_property='figure'),
    [Input('my-date-picker-range', 'start_date'),
     Input('my-date-picker-range', 'end_date'),
    Input('data-type', 'value'),
    Input('type-visual', 'value')])

def update_graph(start_date, end_date,value,visual_type):
    First_day_hour = " 00:00:00"
    Last_day_hour = " 23:00:00"
    fig=go.Figure(data=[go.Scatter(x=data_dates[-5:], y=[])],layout = go.Layout(height=700))
    fig.update_xaxes(title_text="<b>Time<b>")
    if (start_date is not None) and (end_date is not None):
        start_date_object = date.fromisoformat(start_date)
        start_date_string = start_date_object.strftime('%Y-%m-%d')
        start_date_string= start_date_string + First_day_hour
        end_date_object = date.fromisoformat(end_date)
        end_date_string = end_date_object.strftime('%Y-%m-%d')
        end_date_string = end_date_string + Last_day_hour
        for counter in range (0,12):
            if value ==data_dictionary[counter]:
                label=data_in_csv_dictionary[counter]
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(x=data_dates[data_dates.index(start_date_string):data_dates.index(end_date_string)],
                       y=data_csv[label][data_dates.index(start_date_string):data_dates.index(end_date_string)], name=label),
            secondary_y=False,
        )
        if (visual_type != "One Graph"):
            fig.add_trace(
                go.Scatter(x=data_dates[data_dates.index(start_date_string):data_dates.index(end_date_string)],
                           y=data_csv['elec_price'][data_dates.index(start_date_string):data_dates.index(end_date_string)], name='elec_price'),
                secondary_y=(visual_type != "One Graph"),
            )

        fig.update_xaxes(title_text="<b>Time<b>")
        fig.update_yaxes(
            title_text="<b>"+value+"<b>",
            secondary_y=False)
        fig.update_yaxes(
            title_text="<b>Electricity price in euro<b>",
            secondary_y=True)

    # layout = {"xaxis": {"title": x}, "yaxis": {"title": y}}

    return  fig


