# %%
# Imports
import pandas as pd
import numpy as np

import plotly.graph_objects as go

import matplotlib.cm as cm
import matplotlib.colors as colors
import json
from datetime import datetime, timedelta
from tqdm import tqdm
import pathlib
import requests

from download_data import download_data

# %%
# Define helper functions
def get_color_string(val):
    # Get string 'rgba(r, g, b, a)' from tuple(r, g, b, a)
    val = np.array(val) * 255
    return f"rgba({val[0]*255:.2f},{val[1]*255:.2f},{val[2]*255:.2f},{val[3]*255:.2f})"

def get_color_strings(vals):
    # Get list of strings 'rgba(r, g, b, a)' from list of tuples(r, g, b, a)
    return [get_color_string(val) for val in vals]

def string_from_date(date):
    # Convert datetime object to string
    return date.strftime("%Y-%m-%d")

def dayname_from_date(date):
    # Get day of week as name from datetime object
    return date.strftime("%A")

def get_name_from_code(boundaries, code):
    # Get ltla name from area code
    return boundaries.loc[code].areaName

def get_code_from_name(boundaries, name):
    # Get area code from ltla name
    return boundaries.loc[boundaries.areaName == name].index.values[0]

def get_number_on_date(df, date, code):
    # Get number of cases on date for area code
    return df.loc[date, code]

def get_name_format(ltla_name, area_prefix, number, area_suffix):
    # Format area name including prefix, suffix and case numbers
    return f"{ltla_name}<br>{area_prefix}{number:.0f}{area_suffix}"

# %%
# Plot boundaries
def plot_map(df, title, area_prefix="", area_suffix="", filename=None, show=False):
    boundaries = pd.read_feather("LTLA/Transformed/ltla_uk_islands.feather").set_index("areaCode")

    vmax = np.abs([df.min().min(), df.max().max()]).max()
    vmin = -vmax

    norm = colors.SymLogNorm(linthresh=1, vmin=vmin, vmax=vmax)
    cmap = cm.coolwarm

    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    # Plotly boilerplate
    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Date: ",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 300, "easing": "linear"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }

    fig_dict = {
        "data": [],
        "layout": {},
        "frames": []
    }

    fig_dict["layout"] = {"showlegend": False, "title": title}
    fig_dict["layout"]["xaxis"] = {"range": [-100_000, 700_000]}
    fig_dict["layout"]["yaxis"] = {"scaleanchor": "x", "scaleratio": 1}
    fig_dict["layout"]["hovermode"] = "closest"
    fig_dict["layout"]["updatemenus"] = [
        {
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 500, "redraw": False},
                                    "fromcurrent": True, "transition": {"duration": 300, "easing": "linear"}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                      "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }
    ]

    # Put in boundaries
    daily_df = df.iloc[0][boundaries.index]
    strings = get_color_strings(m.to_rgba(daily_df))
    for i, (ltla_code, (x, y, ltla_name)) in enumerate(boundaries.iterrows()):
        fig_dict["data"].append(go.Scatter(
            x=list(x), 
            y=list(y),
            line=dict(
                width=0.1,
                color="rgb(0,0,0)",
            ),
            fill="toself", 
            name=get_name_format(ltla_name, area_prefix, daily_df[i], area_suffix),
            fillcolor=strings[i],
        ))

    num_locs = boundaries.shape[0]
    num_frames = df.shape[0]

    # Pre-allocate frames and steps lists for efficiency
    fig_dict['frames'] = [None] * num_frames
    sliders_dict['steps'] = [None] * num_frames

    for j, date in enumerate(tqdm(df.index, desc="Generating frames")):
        frame = {"data": [None]*num_locs, "name": string_from_date(date)}
        daily_df = df.loc[date][boundaries.index]
        strings = get_color_strings(m.to_rgba(daily_df))
        for i, (ltla_code, (x, y, ltla_name)) in enumerate(boundaries.iterrows()):
            col = strings[i]

            data_dict = dict(
                fillcolor=strings[i],
                name=get_name_format(ltla_name, area_prefix, daily_df[i], area_suffix),
            )
            frame["data"][i] = data_dict

        # Add frame to the slider
        fig_dict["frames"][j] = frame
        sliders_dict["steps"][j] = {
            "args": [
                [string_from_date(date)],
                {"frame": {"duration": 300, "redraw": False},
                "mode": "immediate",
                "transition": {"duration": 300}}
            ],
            "label": f"{string_from_date(date-timedelta(days=6))} to {string_from_date(date)}",
            "method": "animate"
        }

    # Add slider to layout
    fig_dict["layout"]["sliders"] = [sliders_dict]

    # Make the figure
    fig = go.Figure(fig_dict)
    if show:
        fig.show()
    
    # Save the figure
    if filename is not None:
        html_path = pathlib.Path(filename)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(html_path.with_suffix(".html"))
    
    return fig

# %%
# Plot
def main():
    # %%
    # Load population data
    population = pd.read_csv("ONS-population_2021-08-05.csv")
    pop_lookup = pd.Series(dtype=int)
    for i, group in population.groupby("areaCode"):
        pop_lookup[i] = group[group.category == 'ALL'].population.values[0]

    # Ingest case data (downloading if needed)
    dfs = []

    case_data_path = download_data()
    case_data_path = "data/other_data.json"
    with open(case_data_path) as f:
        data = json.load(f)
    df = pd.DataFrame(data['body'])

    for i, group in tqdm(df.groupby("date"), desc="Transforming case data"):
        dfs.append(
            group
            .set_index("areaCode")
            .loc[:, ['newCasesByPublishDate']].T
            .rename({'newCasesByPublishDate': datetime(*[int(x) for x in i.split("-")])})
        )

    cases_df = pd.concat(dfs).fillna(0)

    # Select grouper to use (current day or Sunday to match govt map)
    if True:
        GRP = pd.Grouper(freq=f'W-{dayname_from_date(cases_df.index[-1])[:3].upper()}')
    else:
        GRP = pd.Grouper(freq=f'W-SUN')
    
    # Weekly case data
    weekly_cases_df = (
        cases_df
        .groupby(GRP)
        .sum()
    )
    
    # Make frames
    weekly_cases_dif = weekly_cases_df.diff()

    plot_map(cases_df, "Daily Covid Cases", filename=f"HTML/Daily-cases-{string_from_date(datetime.now())}", show=True)

    # Plot weekly difference
    plot_map(weekly_cases_dif, "Weekly Covid Cases Change", filename=f"HTML/Weekly-dif-{string_from_date(datetime.now())}", show=True)
    
    # Plot weekly percentage change
    weekly_cases_percent = pd.DataFrame(
        weekly_cases_dif*100/weekly_cases_df.iloc[1:],
        index=weekly_cases_df.index[1:], 
        columns=weekly_cases_df.columns
    )

    # Remove infinities/nans
    weekly_cases_percent = weekly_cases_percent.replace([np.inf, -np.inf], 0).fillna(0)

    plot_map(weekly_cases_percent, "Weekly Covid Cases Percentage Change", area_suffix="%", filename=f"HTML/Weekly-rate-{string_from_date(datetime.now())}", show=True)

# %%
main()