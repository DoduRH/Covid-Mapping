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
import time

from download_data import download_data

# %%
# Define helper functions
class MidpointLogNorm(colors.SymLogNorm):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)
    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))

    All arguments are the same as SymLogNorm, except for midpoint    
    """
    def __init__(self, linthresh, linscale=1, midpoint=None, vmin=None, vmax=None):
        self.midpoint = midpoint
        self.lin_thres = linthresh
        self.lin_scale = linscale
        #fraction of the cmap that the linear component occupies
        self.linear_proportion = (linscale / (linscale + 1)) * 0.5
        # print(self.linear_proportion)

        colors.SymLogNorm.__init__(self, linthresh, linscale, vmin, vmax)

    def __get_value__(self, v, log_val, clip=None):
        if v < -self.lin_thres or v > self.lin_thres:
            return log_val
        
        x = [-self.lin_thres, self.midpoint, self.lin_thres]
        y = [0.5 - self.linear_proportion, 0.5, 0.5 + self.linear_proportion]
        interpol = np.interp(v, x, y)
        return interpol

    def __call__(self, value, clip=None):
        log_val = colors.SymLogNorm.__call__(self, value)

        out = [0] * len(value)
        for i, v in enumerate(value):
            out[i] = self.__get_value__(v, log_val[i])

        return np.ma.masked_array(out)



def get_color_string(val):
    # Get string 'rgba(r, g, b, a)' from tuple(r, g, b, a)
    val = np.array(val) * 255
    return f"rgba({val[0]*255:.2f},{val[1]*255:.2f},{val[2]*255:.2f},{val[3]*255:.2f})"

def get_color_strings(series, m):
    # Get dataframe of strings 'rgba(r, g, b, a)' from list of tuples(r, g, b, a)
    return pd.Series([get_color_string(val) for val in m.to_rgba(series)], index=series.index)

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

def get_name_format(ltla_name, area_prefix, number, area_suffix, number_format=".0f"):
    # Format area name including prefix, suffix and case numbers
    return f"{ltla_name}<br>{area_prefix}{number:{number_format}}{area_suffix}"

# %%
# Plot boundaries
def plot_map(df, title, area_prefix="", area_suffix="", filename=None, show=False, frame_time=100, normalisation="lognorm", time_period="week", number_format=".0f"):
    boundaries = pd.read_feather("LTLA/Transformed/ltla_uk_islands_low_points_connected_mixed.feather").set_index("areaCode")

    # Get colour map
    df_min = df.min().min()
    df_max = df.max().max()

    vmax = np.abs([df_min, df_max]).max()
    vmin = -vmax

    if normalisation == "lognorm":
        norm = colors.SymLogNorm(linthresh=0.1, vmin=vmin, vmax=vmax)
    else:
        norm = MidpointLogNorm(linthresh=0.1, vmin=df_min, midpoint=0, vmax=df_max)
    cmap = cm.coolwarm

    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    # TODO: Clean up this dictionary setup, json??
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
        "transition": {"duration": frame_time, "easing": "linear"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }

    fig_dict = {
        "data": [],
        "layout": {},
        "frames": [],
    }

    fig_dict["layout"] = {"showlegend": False, "title": title}
    fig_dict["layout"]["xaxis"] = {"range": [-100_000, 700_000]}
    fig_dict["layout"]["yaxis"] = {"scaleanchor": "x", "scaleratio": 1}
    fig_dict["layout"]["hovermode"] = "closest"
    fig_dict["layout"]["updatemenus"] = [
        {
            "buttons": [
                {
                    "args": [None, {
                        "frame": {"duration": 500, "redraw": False},
                        "fromcurrent": True, 
                        "transition": {"duration": frame_time}
                    }],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None],
                    {
                        "mode": "immediate"
                    }],
                    "label": "Pause",
                    "method": "animate",
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
    # idx is needed to ensure the order of dataframes is maintained
    idx = list(set(boundaries.index) & set(df.columns))

    daily_df = df.iloc[-1][idx]
    strings = get_color_strings(daily_df, m)

    for i, (ltla_code, (x, y, ltla_name)) in enumerate(boundaries.loc[idx].iterrows()):
        fig_dict["data"].append(go.Scatter(
            x=list(x), 
            y=list(y),
            line=dict(
                width=0.1,
                color="rgb(0,0,0)",
            ),
            fill="toself", 
            name=get_name_format(ltla_name, area_prefix, daily_df[ltla_code], area_suffix, number_format),
            fillcolor=strings[ltla_code],
            mode="lines",
        ))
    
    num_locs = boundaries.loc[idx].shape[0]
    num_frames = df.shape[0]
    fig_dict['frames'] = [None] * num_frames
    sliders_dict['steps'] = [None] * num_frames
    for j, date in enumerate(tqdm(df.index, desc=f"Generating '{title}'")):
        frame = {"data": [None]*num_locs, "name": string_from_date(date)}
        daily_df = df.loc[date][idx]
        strings = get_color_strings(daily_df, m)
        for i, (ltla_code, (x, y, ltla_name)) in enumerate(boundaries.loc[idx].iterrows()):
            data_dict = dict(
                fillcolor=strings[ltla_code],
                name=get_name_format(ltla_name, area_prefix, daily_df[ltla_code], area_suffix, number_format),
            )
            frame["data"][i] = data_dict

        fig_dict["frames"][j] = frame
        sliders_dict["steps"][j] = {
            "args": [
                [string_from_date(date)],
                {"frame": {"duration": frame_time, "redraw": False},
                "mode": "immediate",
                "transition": {"duration": frame_time}}
            ],
            "method": "animate"
        }
        if time_period == "week":
            sliders_dict["steps"][j]["label"] = f"{string_from_date(date-timedelta(days=6))} to {string_from_date(date)}"
        elif time_period == "day":
            sliders_dict["steps"][j]["label"] = string_from_date(date)

    sliders_dict['active'] = len(fig_dict["frames"]) - 1
    fig_dict["layout"]["sliders"] = [sliders_dict]

    fig = go.Figure(fig_dict)
    if show:
        fig.show(config={"scrollZoom": True,})

    if filename is not None:
        html_path = pathlib.Path(filename)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(html_path.with_suffix(".html"), config={"scrollZoom": True,})
    
    return fig

# %%
# Plot
def main():
    # %%
    # Load population data
    population = pd.read_csv("Population/ONS-population_2021-08-05.csv")
    pop_lookup = pd.Series(dtype=int)
    for i, group in population.groupby("areaCode"):
        pop_lookup[i] = group[group.category == 'ALL'].population.values[0]

    # Ingest case data (downloading if needed)
    dfs = {}

    case_data_dir = download_data()
    
    # Read in all case data
    for file in case_data_dir.glob("*.feather"):
        df = pd.read_feather(file).fillna(0).set_index("index")
        # Remove zero values at the end of the dataframe
        i = 0
        while df.iloc[i-1].sum() == 0:
            i -= 1
        if i < 0:
            df = df.iloc[:i]

        dfs[file.stem] = df

    # Select grouper to use (current day or Sunday to match govt map)
    if True:
        GRP = pd.Grouper(freq=f'W-{dayname_from_date(dfs["newCasesBySpecimenDate"].index[-1])[:3].upper()}')
    else:
        GRP = pd.Grouper(freq=f'W-SUN')


    # Weekly case data
    weekly_cases_df = (
        dfs['newCasesBySpecimenDate']
        .groupby(GRP)
        .sum()
    )

    # Weekly test data
    weekly_tests_df = (
        dfs['newVirusTestsBySpecimenDate']
        .groupby(GRP)
        .sum()
    )

    # Weekly positive rate
    weekly_positive_rate_df = (
        (weekly_cases_df/weekly_tests_df)
        .replace([np.inf, -np.inf], 0)
    )*100

    # Weekly positive test rate, rate of change
    weekly_positive_rate_diff_df = weekly_positive_rate_df.diff()

    output_folder = pathlib.Path(f"HTML/{string_from_date(datetime.now())}")

    # Should the plots be opened after generation?
    show = True

    # Daily Plot
    if True:
        plot_map(
            dfs['newCasesBySpecimenDate'], 
            "Daily Covid Cases", 
            filename=output_folder.joinpath("Daily-cases"), 
            show=show, 
            time_period="day"
        )
    
    # Plot weekly difference
    weekly_cases_dif = weekly_cases_df.diff()
    if True:
        plot_map(weekly_cases_dif,
        "Weekly Covid Cases Change",
        filename=output_folder.joinpath("Weekly-dif"),
        show=show,
        number_format="+.0f"
    )

    # Plot weekly changes
    # Remove infinities/nans
    weekly_cases_percent = (
        weekly_cases_df
        .pct_change()
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )*100

    if True:
        plot_map(
            weekly_cases_percent, 
            "Weekly Covid Cases Percentage Change", 
            area_suffix="%", 
            filename=output_folder.joinpath("Weekly-rate"), 
            show=show, 
            normalisation="norm", 
            number_format="+.2f"
        )
    
    # Plot weekly positive rate
    if True:
        plot_map(
            weekly_positive_rate_df, 
            "Weekly Positive Rate", 
            area_suffix="%", 
            filename=output_folder.joinpath("Weekly-positive"), 
            show=show, 
            normalisation="norm"
        )
    
    # Plot weekly positive rate change
    if True:
        plot_map(
            weekly_positive_rate_diff_df, 
            "Weekly Positive Rate Rate of Change", 
            area_suffix="%", 
            filename=output_folder.joinpath("Weekly-positive-diff"), 
            show=show, 
            normalisation="norm",
            number_format="+.2f"
        )
# %%
if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"--- {time.time() - start_time} seconds ---")