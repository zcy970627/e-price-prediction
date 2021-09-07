import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State

from app import app
from app import server


navbar = dbc.Navbar(
    [
        html.Div(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(dbc.NavbarBrand("Future of electricity in Germany: Prediction of the Prices on the basis of neural network", className="ml-2")),
                ],
                align="center",
                no_gutters=True,
            )
        ),
        dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
        dbc.Collapse(
            id="navbar-collapse", navbar=True, is_open=False
        ),
    ],
    color="dark",
    className= "navbar-style",
    dark=True,
)


submenu_1 = [
    html.Li(
        dbc.Row(
            [
                dbc.Col(dbc.NavLink("Home", href="/home", className="link-style font-weight-bold")),
                dbc.Col(
                    html.I(className="fas fa-chevron-right mr-3",id="chevron-1"),
                    width="auto",

                ),
            ],
            className="my-1",
        ),
        style={"cursor": "pointer"},
        id="submenu-1",
    ),
    dbc.Collapse(
            [
                dbc.NavLink("Introduction", href="/home", id="intro"),
                dbc.NavLink("Source of Data", href="/home", id="data"),
            ],
            id="submenu-1-collapse",
        ),
]

submenu_2 = [
    html.Li(
        # use Row and Col components to position the chevrons
        dbc.Row(
            [
                dbc.Col(dbc.NavLink("Historical data", href="/historical_data", className="link-style font-weight-bold")),
                dbc.Col(html.I(className="fas fa-chevron-right mr-3",
                               id="chevron-2")
                        ,width="auto",
                ),
            ],
            className="my-1",
        ),
        style={"cursor": "pointer"},
        id="submenu-2",
    ),
    # we use the Collapse component to hide and reveal the navigation links
                    dbc.Collapse(
                        [
                            dbc.NavLink("Data Overview", href="/historical_data"),
                            dbc.NavLink("Data Correlation", href="/historical_data"),
                        ],
                        id="submenu-2-collapse",
                    ),
]

submenu_3 = [
    html.Li(
        dbc.Row(
            [
                dbc.Col(dbc.NavLink("Prediction", href="/prediction", className="link-style font-weight-bold")),
                dbc.Col(
                    html.Span(className="fas fa-chevron-right mr-3",
                           id="chevron-3",
                           ),
                    width="auto",
                ),
            ],
            className="my-1",
        ),
        style={"cursor": "pointer"},
        id="submenu-3",
    ),
    dbc.Collapse(
        [
            dbc.NavLink("Prediction Results", href="/prediction"),
        ],
        id="submenu-3-collapse",
    ),

]


sidebar = html.Div(
    [
        html.H3("Applied Machine Intelligence Project", className="display-12"),
        html.H3(
            "Group 4", className="lead"
        ),
        html.Hr(),
        dbc.Nav(submenu_1 + submenu_2 + submenu_3, vertical=True),
    ],
    className="sidebar-style",
    id="sidebar",
)

def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open
def set_navitem_class(is_open):
    if is_open:
        return "fas fa-chevron-down mr-3"
    return "fas fa-chevron-right mr-3"

for i in [1,2,3]:
    app.callback(
        Output(f"submenu-{i}-collapse", "is_open"),
        [Input(f"chevron-{i}", "n_clicks")],
        [State(f"submenu-{i}-collapse", "is_open")],
    )(toggle_collapse)

    app.callback(
        Output(f"chevron-{i}", "className"),
        [Input(f"submenu-{i}-collapse", "is_open")],
    )(set_navitem_class)

