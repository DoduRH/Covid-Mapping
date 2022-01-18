url = "https://api.coronavirus.data.gov.uk/v2/data?areaType=ltla&metric=newCasesByPublishDate&format=json"

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

# %%
# Load data
population = pd.read_csv("Population/ONS-population_2021-08-05.csv")
pop_lookup = pd.Series(dtype=int)
for i, group in population.groupby("areaCode"):
    pop_lookup[i] = group[group.category == 'ALL'].population.values[0]

with open("data/ltla_2021-12-22.json") as f:
    data = json.load(f)
df = pd.DataFrame(data['body'])

boundaries = pd.read_feather("LTLA/Transformed/ltla_uk.feather").set_index("areaCode")


# %%
# Transform data
dfs = []

for i, group in tqdm(df.groupby("date"), desc="Transforming case data"):
    dfs.append(
        group
        .set_index("areaCode")
        .loc[:, ['newCasesByPublishDate']].T
        .rename({'newCasesByPublishDate': datetime(*[int(x) for x in i.split("-")])})
    )

cases_df = pd.concat(dfs).fillna(0)
print(cases_df.shape)

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

# Map of place codes
placecode_map = {
    "E06000060": "E07000006",
    'E06000053': "E06000052",
    'E09000001': "E09000012",
}


def get_color_string(val):
    # Get string 'rgba(r, g, b, a)' from tuple(r, g, b, a)
    val = val[0]
    return f"rgba({val[0]},{val[1]},{val[2]},{val[3]})"

def get_color_strings(vals):
    # Get list of strings 'rgba(r, g, b, a)' from list of tuples(r, g, b, a)
    return [get_color_string(val) for val in vals]

def get_name_from_code(code):
    # Get ltla readable name from area code
    return boundaries.loc[code].areaName

# Fix dates
date_current = datetime(2021, 12, 15)
date_yest = date_current - timedelta(days=1)

scatters = {}
# Place all of the boundaries traces
for ltla_code, (x, y, ltla_name) in tqdm(boundaries.iterrows(), total=boundaries.shape[0], desc="Generating boundaries"):
    if DIF:
        case_num = cases_df[ltla_code][date_current] - cases_df[ltla_code][date_yest]
    else:
        case_num = cases_df[ltla_code][date_current]

    if PER_HUNDRED:
        case_num = case_num / pop_lookup[ltla_code] * 100_000

    # col = m.to_rgba(case_num)
    
    # plt.fill(points[:,0], points[:,1], c=col, linewidth=1)
    scatters[ltla_code] = go.Scatter(
        x=list(x), 
        y=list(y),
        line=dict(
            width=0.1,
            color="rgb(0,0,0)",
        ),
        fill="toself", 
        # fillcolor=get_color_string(col), 
        name=f"{ltla_name}",
        hovertext=ltla_code,
    )
    

# Color them based on a day
def color_date(date):
    cases_dif = cases_df.diff()[cases_df.index == date]
    if DIF:
        vmax = np.abs([cases_dif.min().min(), cases_dif.max().max()]).max()
        vmin = -vmax
    else:
        vmin = 0
        vmax = cases_df.max().max()

    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.coolwarm

    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    for scatter in fig.select_traces():
        ltla_code = scatter.hovertext
        if DIF:
            case_num = cases_dif[ltla_code]
        else:
            case_num = cases_df[ltla_code][date]

        if PER_HUNDRED:
            case_num = case_num / pop_lookup[ltla_code] * 100_000

        col = m.to_rgba(case_num)

        scatter.fillcolor = get_color_string(col)
        scatter.name = f"{scatter.name}<br>{case_num.values[0]:.0f} cases"

import time
start = time.time()

fig = go.Figure(list(scatters.values()))
fig.update_yaxes(
    scaleanchor="x",
    scaleratio=1,
)
fig.update_layout(
    showlegend=False,
)
fig.show()

# %%
# save frames as images
p = pathlib.Path("images")
p.mkdir(exist_ok=True)

# Set up dimensions
scale = 2
width = 1920//scale//2
height = 1080//scale

fig_data = {}

progress_bar = tqdm(cases_df.index[1:], desc=f"Generating images {cases_df.index[1].strftime('%Y-%m-%d')}")
for date in progress_bar:
    date_str = date.strftime('%Y-%m-%d')
    filepath = p.joinpath(f"{date_str}.png")
    # Only render the frame if it doesnt exist
    if not filepath.exists():
        progress_bar.set_description(f"Generating images {date_str}")
        color_date(date)
        fig.write_image(filepath, width=width, height=height)

print("Done")