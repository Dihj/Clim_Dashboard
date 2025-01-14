"""
You need to run this cript with python3 app.py
and then, open the url : http://127.0.0.1:8050/
You can see there all the modification made till now. 
"""
import dash
import dash_leaflet as dl
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import os
import yaml
#import function
import pandas as pd
import numpy as np
import seaborn as sns
from netCDF4 import Dataset
import xarray as xr
import plotly.express as px
from matplotlib import cm, colors
import scipy.stats as stats
from scipy.stats import t, norm
import function as fc

CONFIG = fc.load_config(os.environ["CONFIG"])


# Sample NetCDF file path
#netcdf_filepath = "/run/media/dhj/01D55DC66499B380/ASA/RESEARCH/Project/DASH_Python/data/rr_Mdg_ENACTv3_monthly_025deg.nc"
netcdf_filepath = CONFIG["rainfall_data"]
latitudes, longitudes, monthly_climatology, rainfall_df = fc.load_netcdf_data(netcdf_filepath)

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout of the app
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Interactive Map and Graphs Dashboard for Madagascar"), width=12)
    ], justify="center", style={'marginTop': 20, 'marginBottom': 20}),
    
    dbc.Row([
        dcc.Location(id="location", refresh=True),
        dbc.Col([
            html.H2("About this!"),
            html.P("""Description information: 
                This app is for monitoring and visualizing all information about climate over a grid point and spatial map Madagascar. 
                This should help users to identify their local climate and other climate parameters.
                This app offer a fitting option i.e the PDF and CDF option view for each grid point selected.
                And also a view for seasonal variability."""),

            html.Div([
                html.H4("Enter your coordinates"),
                dbc.Input(id="input-lat", placeholder="Latitude", type="number", step="any", style={'marginBottom': 10}),
                dbc.Input(id="input-lon", placeholder="Longitude", type="number", step="any", style={'marginBottom': 10}),
                dbc.Button("Submit", id="submit-coords", color="primary", style={'marginTop': 10}),
            ], style={'marginTop': 20}),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='cdf_graph'),
                    dcc.Graph(id='pdf_graph'),
                ], width=12)
            ])
        ], width=4, style={'backgroundColor': '#f8f9fa', 'padding': 20}),
        
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H4("Interactive Map"),
                        html.Div(  # try month
                            [
                                "Select Month:"
                            ],
                            style={
                                "color": "black",
                                "position": "relative",
                                "display": "inline-block",
                                "vertical-align": "top",
                                "padding": "5px",
                            }
                        ),
                        html.Div(  
                            [
                                dcc.Dropdown(
                                    id="monthy",
                                    clearable=False,
                                    options=[
                                        dict(label="January", value=1),
                                        dict(label="February", value=2),
                                        dict(label="March", value=3),
                                        dict(label="April", value=4),
                                        dict(label="May", value=5),
                                        dict(label="June", value=6),
                                        dict(label="July", value=7),
                                        dict(label="August", value=8),
                                        dict(label="September", value=9),
                                        dict(label="October", value=10),
                                        dict(label="November", value=11),
                                        dict(label="December", value=12),
                                    ],
                                    value=1,
                                )
                            ],
                            style={
                                "display": "inline-block",
                                "vertical-align": "top",
                                "width": "130px",
                                "padding": "5px",
                            }
                        ), # end try to month tab

                        dl.Map(center=[-19.99, 46.11], zoom=5, children=[
                        dl.LayersControl(
                            [
                                dl.BaseLayer(
                                    dl.TileLayer(
                                        url="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"
                                    ),
                                    name="Topo",
                                    checked=True,
                                ),
                                dl.BaseLayer(
                                    dl.TileLayer(
                                        url="https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png"
                                    ),
                                    name="Street",
                                    checked=False,
                                ),
                                dl.Overlay(
                                    dl.LayerGroup(
                                       # opacity=1,
                                        id="data_layer"
                                    ),
                                    name="Climate",
                                    checked=True,
                                ),
                                dl.Overlay(
                                    dl.Marker(
                                        opacity=1,
                                        id="marker",
                                        position=[-19.99, 46.11]
                                    ),
                                    name="Marker",
                                    checked=True,
                                ),
                                html.Div(id="legend", style={
                                'position': 'absolute',
                                'bottom': '10px',
                                'left': '10px',
                                'background': 'white',
                                'padding': '10px',
                                'border-radius': '4px',
                                'box-shadow': '0 0 15px rgba(0, 0, 0, 0.2)',
                                'z-index': '1000'
                                })
                            ],
                            position="topleft",
                            id="layers_control",
                        ),
                        ], style={'width': '100%', 'height': '400px'}, id="map"),
                    
                    ], style={'border': '1px solid #d3d3d3', 'padding': 10, 'borderRadius': 5})
                ], width=12)
            ], style={'marginBottom': 20}),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='graph1')
                ], width=6),
                dbc.Col([
                    dcc.Graph(id='graph2')
                ], width=6)
            ])
        ], width=8)
    ])
], fluid=True)

# Helper function to get rainfall climatology for a specific coordinate
def get_climatology_for_coordinate(lat, lon):
    lat_idx = (np.abs(latitudes - lat)).argmin()
    lon_idx = (np.abs(longitudes - lon)).argmin()
    climatology = monthly_climatology.iloc[:, lat_idx * len(longitudes) + lon_idx]
    return climatology.values

def get_timeseries_for_coordinate(lat, lon):
    lat_idx = (np.abs(latitudes - lat)).argmin()
    lon_idx = (np.abs(longitudes - lon)).argmin()
    timeseries = rainfall_df.iloc[:, lat_idx * len(longitudes) + lon_idx]
    return timeseries


# Callback to update marker position and graphs based on the selected coordinates
@app.callback(
    [Output('marker', 'position'),
     Output('graph1', 'figure'),
     Output('graph2', 'figure'),
     Output('cdf_graph', 'figure'),
     Output('pdf_graph', 'figure'),
     Output('input-lat', 'value'),
     Output('input-lon', 'value')],
    [Input('map', 'click_lat_lng'),
     Input('submit-coords', 'n_clicks'),
     Input("monthy", "value")],
    [State('input-lat', 'value'),
     State('input-lon', 'value')]
)

def update_marker_and_graphs(click_lat_lng, n_clicks, monthy, input_lat, input_lon):
    ctx = dash.callback_context

    # Determine which input triggered the callback
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    triggered_by = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_by == 'submit-coords' and input_lat is not None and input_lon is not None:
        lat, lon = input_lat, input_lon
    elif triggered_by == 'map' and click_lat_lng is not None:
        lat, lon = click_lat_lng
    else:
        lat, lon = -19.99, 46.11  # Default coordinates

    # Round the coordinates to 2 decimal places
    lat = round(lat, 2)
    lon = round(lon, 2)

    # Get climatology data for the selected coordinates
    climatology = get_climatology_for_coordinate(lat, lon)
    timeseries = get_timeseries_for_coordinate(lat, lon)
    months = np.arange(1, 13)  # January to December

    # Create histogram for monthly climatology
    data1 = go.Bar(x=months, y=climatology, name='Monthly Climatology')
    fig1 = go.Figure(data=[data1])
    fig1.update_layout(title=f"Average Monthly rainfall for [{lat},{lon}]", xaxis_title="Month", yaxis_title="Rainfall (mm)")

    # Create boxplot for monthly climatology
    data2 = go.Box(x=timeseries.index.month, y=timeseries.values)
   # data2 = go.Box( x=months, y=timeseries, name='Monthly Climatology')
    fig2 = go.Figure(data=[data2])
    fig2.update_layout(title=f"Monthly Climatology for [{lat}, {lon}]", yaxis_title="Rainfall (mm)")

################################################
        # Get timeseries data for the selected coordinates
    ts = get_timeseries_for_coordinate(lat, lon)
    tsm = ts[ts.index.month == monthy]
    tsm = tsm.to_xarray()
    # Calculate the CDF
    # This is the improvement here about CDF
    #################################################
    # CDF from 499 quantiles
    quantiles = np.arange(1, 500) / 500
    quantiles = xr.DataArray(
        quantiles, dims="percentile", coords={"percentile": quantiles}
    )

    # Obs CDF
    obs_q, obs_mu = xr.broadcast(quantiles, tsm.mean(dim="index"))
    obs_stddev = tsm.std(dim="index")
    obs_ppf = xr.apply_ufunc(
        norm.ppf,
        obs_q,
        kwargs={"loc": obs_mu, "scale": obs_stddev},
    ).rename("obs_ppf")
    # Obs quantiles
    obs_quant = tsm.quantile(quantiles, dim="index")
    obs_ppf = obs_ppf.clip(min=0)
    poe = obs_ppf["percentile"] * -1 + 1
    cdf_graph = go.Figure()
    cdf_graph.add_trace(
        go.Scatter(
            x=obs_ppf.values,
            y=poe,
            hovertemplate="%{y:.0%} chance of exceeding"
            + "<br>%{x:.1f} "
            + "mm",
            name="Parametric",
            line=go.scatter.Line(color="red"),
        )
    )
    cdf_graph.add_trace(
        go.Scatter(
            x=obs_quant.values,
            y=poe,
            hovertemplate="%{y:.0%} chance of exceeding"
            + "<br>%{x:.1f} "
            + "mm",
            name="Empirical",
            line=go.scatter.Line(color="blue"),
        )
    )
    cdf_graph.update_traces(mode="lines", connectgaps=False)
    cdf_graph.update_layout(
        xaxis_title=f"Rainfall (mm)",
        yaxis_title="Probability of exceeding",
        title=f'Inverse CDF for {monthy} at ({lat}, {lon})',
    )

    # PDF from 499 ppf values
    obs_pdf = xr.apply_ufunc(
        norm.pdf,
        obs_ppf,
        kwargs={"loc": obs_mu, "scale": obs_stddev},
    ).rename("obs_pdf")
    # Graph for PDF
    pdf_graph = go.Figure()
    pdf_graph.add_trace(
        go.Scatter(
            x=obs_ppf.values,
            y=obs_pdf.values,
            customdata=poe,
            hovertemplate="%{customdata:.0%} chance of exceeding"
            + "<br>%{x:.1f} "
            + "mm",
            name="obs",
            line=go.scatter.Line(color="blue"),
        )
    )
    pdf_graph.update_traces(mode="lines", connectgaps=False)
    pdf_graph.update_layout(
        xaxis_title=f"Rainfall (mm)",
        yaxis_title="Probability density",
        title=f'PDF for {monthy} at ({lat}, {lon})',
    )
#    return cdf_graph, pdf_graph

    return [lat, lon], fig1, fig2, cdf_graph, pdf_graph, lat, lon

# To plot CDF probability for a month
    # For figure 3 about CDF, probability of exceedence
    # this will be based on timeseries dataset


@app.callback(
    [Output("data_layer", "children"), Output("legend", "children")],
    Input("monthy", "value"),
)
def update_map(monthy):
    # Filter data for selected month
    ds = xr.open_dataset(netcdf_filepath)
    data4 = ds.precip.groupby('time.month').mean(dim="time")
    #monthly_climatology_df = data4.to_dataframe()
    data3 = data4[data4['month'] == monthy]
    df = data3.to_dataframe().reset_index()

    max_precip = df['precip'].max()

    # Create a list of CircleMarkers for each data point
    markers = [
        dl.CircleMarker(
            center=(row['lat'], row['lon']),
            radius=1.9,
            color=fc.get_color(row['precip'], max_precip),
            fill=True,
            fillColor=fc.get_color(row['precip'], max_precip),
            fillOpacity=0.6 if not np.isnan(row['precip']) else 0,
            children=[
                dl.Tooltip(f"Precip: {row['precip']:.2f} mm") if not np.isnan(row['precip']) else None
            ]
        ) for _, row in df.iterrows()
    ]

    # Create the legend as a list of div elements
    legend_colors = [fc.get_color(v, max_precip) for v in np.linspace(0, max_precip, 6)]
    legend_labels = [f'{v:.2f} mm' for v in np.linspace(0, max_precip, 6)]
    legend = [
        html.Div([
            html.Span(style={'backgroundColor': color, 'width': '20px', 'height': '20px', 'display': 'inline-block', 'marginRight': '5px'}),
            html.Span(f'{label}')
        ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '5px'}) for color, label in zip(legend_colors, legend_labels)
    ]

    return markers, legend


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
