
import pandas as pd
import numpy as np
import plotly.express as px
from dash import Dash, dcc, html, Input, Output

# -------------------------------
# Data Loading & Preparation
# -------------------------------
DATA_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/historical_automobile_sales.csv"

df = pd.read_csv(DATA_URL)

# Some cleaning / feature engineering
if {"Year", "Month"}.issubset(df.columns):
    df["Date"] = pd.to_datetime(dict(year=df["Year"], month=df["Month"], day=1))
else:
    # fallback if Month is absent
    df["Date"] = pd.to_datetime(df["Year"].astype(str) + '-01-01')

# Provide some safe defaults for missing columns used in charts
for col, default in [
    ("Vehicle_Type", "Unknown"),
    ("Recession", 0),
    ("Automobile_Sales", 0.0),
    ("GDP", np.nan),
    ("Seasonality_Weight", 1.0),
    ("Advertising_Expenditure", 0.0),
    ("Unemployment_Rate", np.nan),
    ("Consumer_Confidence", np.nan),
    ("Avg_Vehicle_Price", np.nan),
]:
    if col not in df.columns:
        df[col] = default

df["Recession"] = df["Recession"].astype(int)

# Basic ranges / options
min_year, max_year = int(df["Year"].min()), int(df["Year"].max())
vehicle_types = sorted(df["Vehicle_Type"].dropna().unique().tolist())

# -------------------------------
# Dash App
# -------------------------------
app = Dash(__name__)
app.title = "Automobile Sales Dashboard"

app.layout = html.Div(
    [
        html.H1("Automobile Sales Dashboard (Plotly + Dash)", style={"textAlign": "center"}),
        html.P(
            "Interactively explore automobile sales, recession effects, and related economic indicators."
        ),
        html.Hr(),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Recession Filter"),
                        dcc.Checklist(
                            id="recession-check",
                            options=[
                                {"label": "Recession (1)", "value": 1},
                                {"label": "Non‑Recession (0)", "value": 0},
                            ],
                            value=[0, 1],
                            inline=True,
                        ),
                        html.Br(),
                        html.Label("Vehicle Types"),
                        dcc.Dropdown(
                            id="vehicle-types",
                            options=[{"label": v, "value": v} for v in vehicle_types],
                            value=vehicle_types,
                            multi=True,
                        ),
                        html.Br(),
                        html.Label("Metric for Time Series"),
                        dcc.RadioItems(
                            id="metric-radio",
                            options=[
                                {"label": "Automobile Sales", "value": "Automobile_Sales"},
                                {"label": "GDP", "value": "GDP"},
                                {"label": "Advertising Expenditure", "value": "Advertising_Expenditure"},
                            ],
                            value="Automobile_Sales",
                            inline=True,
                        ),
                        html.Br(),
                        html.Label("Year Range"),
                        dcc.RangeSlider(
                            id="year-range",
                            min=min_year,
                            max=max_year,
                            value=[min_year, max_year],
                            marks={y: str(y) for y in range(min_year, max_year + 1, max(1, (max_year-min_year)//10))},
                            step=1,
                            allowCross=False,
                        ),
                    ],
                    style={"flex": "1", "minWidth": "280px", "paddingRight": "24px"},
                ),
                html.Div(
                    [
                        dcc.Tabs(
                            id="tabs",
                            value="tab-timeseries",
                            children=[
                                dcc.Tab(label="Time Series", value="tab-timeseries"),
                                dcc.Tab(label="Vehicle Type Comparison", value="tab-vehicletype"),
                                dcc.Tab(label="Seasonality Bubble", value="tab-bubble"),
                                dcc.Tab(label="Advertising Pie", value="tab-pie"),
                                dcc.Tab(label="Unemployment vs Sales", value="tab-unemp"),
                            ],
                        ),
                        html.Div(id="tabs-content"),
                    ],
                    style={"flex": "3", "minWidth": "400px"},
                ),
            ],
            style={"display": "flex", "flexWrap": "wrap"},
        ),
        html.Hr(),
        html.Footer(
            "DV0101EN Final Assignment Part 2 — Sample Solution using Plotly Dash",
            style={"textAlign": "center", "fontSize": "0.9em", "color": "#555"},
        ),
    ],
    style={"margin": "2%"},
)


# -------------------------------
# Callbacks
# -------------------------------

def get_filtered_df(recession_values, vehicle_values, year_range):
    dff = df.copy()
    if recession_values:
        dff = dff[dff["Recession"].isin(recession_values)]
    if vehicle_values:
        dff = dff[dff["Vehicle_Type"].isin(vehicle_values)]
    if year_range and len(year_range) == 2:
        dff = dff[(dff["Year"] >= year_range[0]) & (dff["Year"] <= year_range[1])]
    return dff


@app.callback(
    Output("tabs-content", "children"),
    Input("tabs", "value"),
    Input("recession-check", "value"),
    Input("vehicle-types", "value"),
    Input("metric-radio", "value"),
    Input("year-range", "value"),
)
def render_tab_content(tab, recession_values, vehicle_values, metric, year_range):
    dff = get_filtered_df(recession_values, vehicle_values, year_range)

    if tab == "tab-timeseries":
        # Time series by Year for chosen metric, optionally separated by Vehicle_Type
        agg = (
            dff.groupby(["Year", "Vehicle_Type"], as_index=False)[metric]
               .mean()
        )

        fig = px.line(
            agg,
            x="Year",
            y=metric,
            color="Vehicle_Type",
            markers=True,
            title=f"Time Series of {metric.replace('_',' ')} by Vehicle Type",
        )
        fig.update_layout(hovermode="x unified")
        return dcc.Graph(figure=fig)

    elif tab == "tab-vehicletype":
        # Bar comparison recession vs non-recession, by vehicle type (average Automobile Sales)
        metric2 = "Automobile_Sales"
        agg = (
            dff.groupby(["Recession", "Vehicle_Type"], as_index=False)[metric2]
               .mean()
               .rename(columns={metric2: "Avg_Automobile_Sales"})
        )
        agg["Phase"] = agg["Recession"].map({0: "Non‑Recession", 1: "Recession"})
        fig = px.bar(
            agg,
            x="Vehicle_Type",
            y="Avg_Automobile_Sales",
            color="Phase",
            barmode="group",
            title="Average Automobile Sales per Vehicle Type — Recession vs Non‑Recession",
        )
        fig.update_layout(xaxis_tickangle=-45)
        return dcc.Graph(figure=fig)

    elif tab == "tab-bubble":
        # Bubble plot (Seasonality impact) — non-recession subset
        dff_nr = dff[dff["Recession"] == 0].copy()
        if "Month" not in dff_nr.columns:
            return html.Div("Month column not found, cannot build bubble plot.")
        bubble = (
            dff_nr.groupby("Month", as_index=False)
                  .agg({"Automobile_Sales": "mean", "Seasonality_Weight": "mean"})
        )
        fig = px.scatter(
            bubble,
            x="Month",
            y="Automobile_Sales",
            size="Seasonality_Weight",
            hover_name="Month",
            title="Seasonality impact on Automobile Sales (Non‑recession)",
        )
        return dcc.Graph(figure=fig)

    elif tab == "tab-pie":
        # Pie chart of total advertising expenditure by vehicle type during recession
        dff_rec = dff[dff["Recession"] == 1].copy()
        agg = (
            dff_rec.groupby("Vehicle_Type", as_index=False)["Advertising_Expenditure"]
                  .sum()
                  .rename(columns={"Advertising_Expenditure": "Total_Ad_Expenditure"})
        )
        if agg.empty:
            return html.Div("No data for recession period with current filters.")
        fig = px.pie(
            agg,
            values="Total_Ad_Expenditure",
            names="Vehicle_Type",
            title="Total Advertising Expenditure by Vehicle Type (Recession)",
            hole=0.3,
        )
        return dcc.Graph(figure=fig)

    elif tab == "tab-unemp":
        # Lineplot: unemployment vs sales by vehicle type (recession)
        dff_rec = dff[dff["Recession"] == 1].copy()
        if dff_rec.empty:
            return html.Div("No data for recession period with current filters.")
        fig = px.line(
            dff_rec.sort_values("Unemployment_Rate"),
            x="Unemployment_Rate",
            y="Automobile_Sales",
            color="Vehicle_Type",
            markers=True,
            title="Effect of Unemployment Rate on Vehicle Type and Sales (Recession)",
        )
        return dcc.Graph(figure=fig)

    return html.Div("Tab not found")


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    app.run_server(debug=False)
