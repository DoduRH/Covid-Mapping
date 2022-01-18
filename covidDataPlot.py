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

# %%
# Load data
population = pd.read_csv("Population/ONS-population_2021-08-05.csv")
pop_lookup = pd.Series(dtype=int)
for i, group in population.groupby("areaCode"):
    pop_lookup[i] = group[group.category == 'ALL'].population.values[0]

boundaries = pd.read_feather("LTLA/Transformed/ltla_uk.feather").set_index("areaCode")

# %%
# Define helper functions
def get_color_string(val):
    # Get string 'rgba(r, g, b, a)' from tuple(r, g, b, a)
    val = np.array(val) * 255
    val[3] = 1
    return f"rgba({val[0]},{val[1]},{val[2]},{val[3]})"

def get_color_strings(vals):
    # Get list of strings 'rgba(r, g, b, a)' from list of tuples(r, g, b, a)
    return [get_color_string(val) for val in vals]

def get_name_from_code(code):
    # Get ltla name from area code
    return boundaries.loc[code].areaName

def string_from_date(date):
    # Convert datetime object to string
    return date.strftime("%Y-%m-%d")

# %%
# Transform data
dfs = []

requested_date = datetime(2021, 12, 24)
today_path = pathlib.Path(f"data/ltla_{string_from_date(requested_date)}.json")
if not today_path.exists():
    today_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading data for {string_from_date(requested_date)}")
    url = "https://api.coronavirus.data.gov.uk/v2/data?areaType=ltla&metric=newCasesByPublishDate&format=json"
    with open(today_path, "w") as f:
        json.dump(requests.get(url).json(), f)

with open(today_path) as f:
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

# %%
# Plot boundaries
PER_HUNDRED = False
DIF = True

placename_map = {
    "Buckinghamshire": "South Bucks",
    "Na h-Eileanan Siar": "Comhairle nan Eilean Siar",
    'Cornwall': "Cornwall and Isles of Scilly",
    'Isles of Scilly': "Cornwall and Isles of Scilly",
    'City of London': "Hackney and City of London",
    'Hackney': "Hackney and City of London",
}

placecode_map = {
    "E06000060": "E07000006",
    'E06000053': "E06000052",
    'E09000001': "E09000012",
}

# Group the cases by week
cases_df = (
    cases_df
    .groupby(pd.Grouper(freq='W-MON'))
    .sum()
)

# Make frames
cases_dif = cases_df.diff()
if DIF:
    vmax = np.abs([cases_dif.min().min(), cases_dif.max().max()]).max()
    vmin = -vmax
else:
    vmin = 0
    vmax = cases_df.max().max()

# cases_dif += vmax
norm = colors.SymLogNorm(linthresh=1, vmin=vmin, vmax=vmax)
# norm = colors.Normalize(vmin=vmin, vmax=vmax)
cmap = cm.coolwarm

m = cm.ScalarMappable(norm=norm, cmap=cmap)

# Plotly biolerplate
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
    "transition": {"duration": 300},
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

# fill in most of layout
fig_dict["layout"] = {"showlegend": False}
fig_dict["layout"]["xaxis"] = {"range": [-100_000, 700_000]}
fig_dict["layout"]["yaxis"] = {"scaleanchor": "x", "scaleratio": 1}
fig_dict["layout"]["hovermode"] = "closest"
fig_dict["layout"]["updatemenus"] = [
    {
        "buttons": [
            {
                "args": [None, {"frame": {"duration": 500, "redraw": False},
                                "fromcurrent": True, "transition": {"duration": 300,}}],
                "label": "Play",
                "method": "animate"
            },
            {
                "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                  "transition": {"duration": 300}}],
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
for ltla_code, (x, y, ltla_name) in boundaries.iterrows():
    fig_dict["data"].append(go.Scatter(
        x=list(x), 
        y=list(y),
        line=dict(
            width=0.1,
            color="rgb(0,0,0)",
        ),
        fill="toself", 
        name=f"{ltla_name}",
        hovertext=ltla_code,
        fillcolor="rgb(255,255,255)",
    ))

num_locs = boundaries.shape[0]
num_frames = cases_df.shape[0]
# Pre-allocate frames and steps lists for efficiency
fig_dict['frames'] = [None] * num_frames
sliders_dict['steps'] = [None] * num_frames

# Colour the map
for j, date in enumerate(tqdm(cases_df.index, desc="Generating frames")):
    frame = {"data": [None]*num_locs, "name": string_from_date(date)}
    daily_cases = cases_dif.loc[date][boundaries.index]
    strings = get_color_strings(m.to_rgba(daily_cases))
    for i, (ltla_code, (x, y, ltla_name)) in enumerate(boundaries.iterrows()):
        col = strings[i]

        data_dict = dict(
            fillcolor=strings[i],
            name=f"{ltla_name}<br>{daily_cases[i]:.0f}",
        )
        frame["data"][i] = data_dict

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

fig_dict["layout"]["sliders"] = [sliders_dict]

# Show the plot
fig = go.Figure(fig_dict)
fig.show()

# Save the map as HTML
html_path = pathlib.Path(f"HTML/{string_from_date(requested_date)}2.html")
html_path.parent.mkdir(parents=True, exist_ok=True)
fig.write_html(html_path)

print("Done")