import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import pathlib
import pandas as pd
from plotly.subplots import make_subplots
from datetime import date
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
from datetime import datetime, timedelta
import numpy as np


from apps import prediction


from app import app
from apps import side_and_nav_bars



list_prediction_type =  ['Hourly', 'Daily']
list_model_type = ["Only historical electricity prices", "Best 6 Correlated Data"]
list_prediction_period = ["One hour", "One day","One Week"]

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../dataset").resolve()

data_dictionary=["Electricity price in Euro"]
data_in_csv_dictionary=['elec_price']

#Load daily and hourly data
data_daily = pd.read_csv(DATA_PATH.joinpath("hourly_historical_data.csv"))
data_hourly = pd.read_csv(DATA_PATH.joinpath("hourly_historical_data.csv"))

data_list_daily=list(data_daily)
data_list_hourly=list(data_hourly)

data_list_daily=data_list_daily[-1:]+ data_list_daily[6:-1]
data_list_hourly=data_list_hourly[-1:]+ data_list_hourly[6:-1]
data_dates_hourly = pd.to_datetime(data_hourly['Unnamed: 0']).to_list()
data_dates_daily = pd.to_datetime(data_daily['Unnamed: 0']).to_list()
predict_period_dates = pd.date_range(data_dates_hourly[-1] + timedelta(hours=1),periods=24*7, freq='1H').tolist()
new_dates = data_dates_hourly[-31*24:]+predict_period_dates
figure=go.Figure(data=[go.Scatter(x=new_dates, y=data_hourly['elec_price'][-31*24:])],layout = go.Layout(height=700))
figure.update_xaxes(title_text="Time")

prediction_content = html.Div(id="page-3-content", className="content-style",
children=[
dbc.Card(
    [
     dbc.CardBody(
        [
        html.Div([
        html.H1('Prediction results'),
            html.Div([
                html.Div([
                    html.Pre(children="Hourly/Daily", style={"fontSize": "150%"}),
                    dcc.Dropdown(
                        id='prediction-type', value='', clearable=True,
                        persistence=True, persistence_type='session',
                        options=[{'label': x, 'value': x} for x in list_prediction_type]
                    )
                ],className='dropdown-style'),
                html.Div([
                    html.Pre(children="Model input", style={"fontSize": "150%"}),
                    dcc.Dropdown(
                        id='model-type', value='', clearable=True,
                        persistence=True, persistence_type='session',
                        options=[{'label': x, 'value': x} for x in list_model_type]
                    )
                ], className='dropdown-style'),

                html.Div([
                    html.Pre(children="Prediction period", style={"fontSize": "150%"}),
                    dcc.Dropdown(
                        id='prediction-period', value='', clearable=True,
                        persistence=True, persistence_type='session',
                        options=[{'label': x, 'value': x} for x in list_prediction_period]
                    )
                ], className='dropdown-style'),
            ],className='row')
        ], ),
        dcc.Graph(id='visual-graph-1', figure=figure, className="figure-plot"),
        ])
    ])
])
layout = html.Div([prediction_content, side_and_nav_bars.navbar, side_and_nav_bars.sidebar])

@app.callback(
    Output('visual-graph-1', component_property='figure'),
    [Input('prediction-type', 'value'),
     Input('model-type', 'value'),
    Input('prediction-period', 'value')])

def use_prediction(prediction_type,model_type,prediction_period):
    fig=figure
    predicted=[]
    data_figure=data_hourly
    if ((prediction_period in list_prediction_period) and (model_type in list_model_type) and (prediction_type in list_prediction_type)):
        if model_type == "Only historical electricity prices":
           if prediction_period =="One hour":
                predicted=prediction.one_hour()
           elif prediction_period == "One day":
               if (prediction_type=="Hourly"):
                   predicted=prediction.one_day()
               elif(prediction_type=="Daily"):
                    predicted = prediction.one_day()
           elif prediction_period == "One Week":
               if(prediction_type == "Hourly"):
                    predicted=prediction.one_week()
               elif (prediction_type == "Daily"):
                   predicted = prediction.one_week()
        elif model_type == "Best 6 Correlated Data":
            if prediction_period == "One hour":
                predicted = prediction.one_hour(mutivariate_model=True)
            elif prediction_period == "One day":
                if (prediction_type == "Hourly"):
                    predicted = prediction.one_day(mutivariate_model=True)
                elif (prediction_type == "Daily"):
                    predicted = prediction.one_daymutivariate_model(mutivariate_model=True, daily=True)
                    data_figure=data_daily
            elif prediction_period == "One Week":
                if (prediction_type == "Hourly"):
                    predicted = prediction.one_week(mutivariate_model=True)
                elif (prediction_type == "Daily"):
                    predicted = prediction.one_week(mutivariate_model=True, daily=True)
                    data_figure = data_daily
        fig = go.Figure(data=
                        [go.Scatter(x=new_dates, y=data_figure['elec_price'][-31*24:]),
                        go.Scatter(x=predict_period_dates, y=predicted[0,:], line=dict(color='orange') )],
                        layout = go.Layout(height=700))

        fig.update_xaxes(title_text="<b>Time<b>")
        fig.update_yaxes(
            title_text="<b>Electricity price in euro<b>")
    return  fig


