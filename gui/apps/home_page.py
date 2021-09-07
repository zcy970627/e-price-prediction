import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import pathlib
from dash.dependencies import Input, Output, State

from app import app
from apps import side_and_nav_bars



# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()

index_content=html.Div(id="page-1-content",className="content-style",
    children=[
   dbc.Card(
    [

        dbc.CardBody(
            [
                html.H1("Introduction", className="card-title"),
                html.P(
                    "With the ever increasing demand for energy, the world is already consuming too much energy. "
                    "Knowing the current prices of electricity and predicting the trends in the "
                    "future might give more and more individuals a better idea of the huge demand for electricity and as a "
                    "result entice them to consume it in a more responsible way. Moreover, seeing the correlations "
                    "between the electricity price and other factors such as oil price temperatures might push governments and companies"
                    " to adopt more environmentally friendly ways of producing electricty. The following graph highlights the distribution of utilized ressources for the electricity production  "
                    "in Germany during 2016.",
                    style={'textAlign': 'justify'},
                    className="card-text",
                ),
            ]
        ),
        dbc.CardImg(src=app.get_asset_url("germany_206.png"),
                    className="image-center",
                    title='Primary energy consumption in Germany 2016',
                    bottom=True),
        dbc.CardBody(
            [
                html.H1("Source of Data ", className="card-title"),
                html.Label(style={'textAlign': 'justify'},
                    children=[
                    "We extracted the data from several sources. We utlized the API montel whose access was given by the Chair for Data Processing. Additional Data were also collected throughout"
                    "several APIs such as ",
                        html.A("World weather online for developers", href="https://www.worldweatheronline.com/developer/"),
                        " For weather data in Leibzig. The Oil and gaz information are available from the ,",
                        html.A(" US energy Information administration", href="'https://www.eia.gov/dnav/"),
                        " and were extracted daily in contraction to all the other data "
                    "We extracted the data from several sources. We utlized the API montel whose access was given by the Chair for Data Processing. Additional data were also collected throughout "
                    "several APIs such as the one from ",
                        html.A("World Weather Online", href="https://www.worldweatheronline.com/developer/"),
                        " containing weather data for the German city Leipzig. The oil and gas information are available from the ",
                        html.A("US energy Information administration", href="https://www.eia.gov/"),
                        " and were extracted daily in contraction to all the other data."
                ]),
            ]
        ),
    ],
),

    ],
)
home_layout = html.Div([
    index_content,
    side_and_nav_bars.navbar,
    side_and_nav_bars.sidebar,

])

