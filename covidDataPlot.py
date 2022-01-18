url = "https://api.coronavirus.data.gov.uk/v2/data?areaType=ltla&metric=newCasesByPublishDate&format=json"

# %%
# Imports
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import json
import shapefile
from datetime import datetime, timedelta
from tqdm import tqdm

# %%
# Load data
population = pd.read_csv("Population/ONS-population_2021-08-05.csv")
pop_lookup = pd.Series(dtype=int)
for i, group in population.groupby("areaCode"):
    pop_lookup[i] = group[group.category == 'ALL'].population.values[0]

with open("data/ltla_2021-12-22.json") as f:
    data = json.load(f)
df = pd.DataFrame(data['body'])
# Orkney Islands not on 22nd dec 2021, assumed to be 0
# df.loc[len(df)] = {
#     "areaType": "ltla",
#     "areaCode": "S12000023",
#     "areaName": "Orkney Islands",
#     "date": "2021-12-22",
#     "newCasesByPublishDate": "0",
# }
boundaries = shapefile.Reader("LTLA/ltla_uk.shp")

# %%
# Transform data
# cases_df = pd.DataFrame(columns=df.areaCode.unique())

dfs = []

for i, group in tqdm(df.groupby("date")):
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
    # "Na h-Eileanan Siar": "Comhairle nan Eilean Siar",
    'E06000053': "E06000052",
    'E09000001': "E09000012",
}


date_current = datetime(2021, 12, 15)
date_yest = date_current - timedelta(days=1)

if DIF:
    # Calculate the rate of change of cases
    d = cases_df.loc[date_current] - cases_df.loc[date_yest]
    if d.min() < 0:
        vmin = -np.abs([d.min(), d.max()]).max()
    else:
        vmin = 0
    vmax = np.abs([d.min(), d.max()]).max()
    print(f"vmin: {vmin}, vmax: {vmax}")
else:
    vmin = 0
    vmax = cases_df.loc[date_current].max()

# Generate color map
norm = colors.Normalize(vmin=vmin, vmax=vmax)
cmap = cm.coolwarm

m = cm.ScalarMappable(norm=norm, cmap=cmap)

def get_color_strings(vals):
    # tuple(r,g,b,a) to string 'rgba(r,g,b,a)'
    return [f"rgba({r},{g},{b},{a})" for r, g, b, a in vals]

scatters = []
for zone in tqdm(boundaries.shapeRecords()):
    # Extract ltla name and code
    ltla_code = zone.record[1]
    ltla_code = placecode_map.get(ltla_code, ltla_code)
    ltla_name = zone.record[2]
    ltla_name = placename_map.get(ltla_name, ltla_name)

    # Get case count
    cases = cases_df[ltla_code] # all of the cases in this area since start
    
    # Get number to show (rate of change or absolute)
    if DIF:
        case_num = cases_df[ltla_code][date_current] - cases_df[ltla_code][date_yest]
    else:
        case_num = cases_df[ltla_code][date_current]

    # Divide by population if desired
    if PER_HUNDRED:
        case_num = case_num / pop_lookup[ltla_code] * 100_000

    # Get color
    col = m.to_rgba(case_num)
    
    # Get shape points and plot
    points = np.array(zone.shape.points)
    scatters.append(
        go.Scatter(
            x=points[:,0], 
            y=points[:,1],
            line=dict(
                width=0.1,
                color="rgb(0,0,0)",
            ),
            fill="toself", 
            fillcolor=f"rgba({','.join([str(x) for x in col])})", 
            name=f"{ltla_name}\n{case_num:.0f}",
        )
    )

# Generate figure
fig = go.Figure(scatters)
fig.update_yaxes(
    scaleanchor="x",
    scaleratio=1,
)
fig.update_layout(
    showlegend=False,
)
colorbar_trace  = go.Scatter(
    x=[None],
    y=[None],
    mode='markers',
    marker=dict(
        colorscale=get_color_strings(m.to_rgba(np.linspace(vmin, vmax, 100))),
        showscale=True,
        cmin=-5,
        cmax=5,
        colorbar=dict(thickness=5, tickvals=[-5, 5], ticktext=['Low', 'High'], outlinewidth=0)
    ),
    hoverinfo='none'
)
fig.add_trace(colorbar_trace)

fig.show()
print("Done")
