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
pop_lookup = pd.Series()
for i, group in population.groupby("areaCode"):
    pop_lookup[i] = group[group.category == 'ALL'].population.values[0]

with open("data/ltla_2021-12-22.json") as f:
    data = json.load(f)
df = pd.DataFrame(data['body'])
# Orkney Islands not on 22nd dec 2021, assumed to be 0
df.loc[len(df)] = {
    "areaType": "ltla",
    "areaCode": "S12000023",
    "areaName": "Orkney Islands",
    "date": "2021-12-22",
    "newCasesByPublishDate": "0",
}
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

cases_df = pd.concat(dfs)
cases_df.head()

# %%
# Plot boundaries
PER_HUNDRED = False
DIF = True


place_map = {
    "Buckinghamshire": "South Bucks",
    "Na h-Eileanan Siar": "Comhairle nan Eilean Siar",
    'Cornwall': "Cornwall and Isles of Scilly",
    'Isles of Scilly': "Cornwall and Isles of Scilly",
    'City of London': "Hackney and City of London",
    'Hackney': "Hackney and City of London",
}


current_day = datetime(2021, 12, 21)
date_str = current_day.strftime("%Y-%m-%d")
date_yest = (current_day - timedelta(days=1))
date_yest_str = date_yest.strftime("%Y-%m-%d")
current_df = df[np.logical_or(df.date == date_str, df.date == date_yest_str)]
expected = 2

if DIF:
    d = (current_df[current_df.date == date_yest_str].newCasesByPublishDate.values - current_df[current_df.date == date_str].newCasesByPublishDate.values)
    vmin = d.min()
    vmax = d.max()
else:
    vmin = 0
    vmax = current_df.newCasesByPublishDate.values.max()

norm = colors.Normalize(vmin=vmin, vmax=vmax)
cmap = cm.jet

m = cm.ScalarMappable(norm=norm, cmap=cmap)

for zone in tqdm(boundaries.shapeRecords()):
    ltla_code = zone.record[1]
    ltla_name = zone.record[2]
    ltla_name = place_map.get(ltla_name, ltla_name)

    cases = current_df[current_df.areaName == ltla_name]
    
    days = 0
    while len(cases) == 0:
        days += 1
        print(f"No cases for {ltla_name} on {date_str}")
        date = (current_day - timedelta(days=days)).strftime("%Y-%m-%d")
        cases = df[np.logical_and(current_df.areaName == ltla_name, df.date == date)]
    
    if len(cases) > expected:
        cases = cases[cases.areaName == ltla_name]
    
    if DIF:
        case_num = (cases[cases.date == date_yest_str].newCasesByPublishDate.values - cases[cases.date == date_str].newCasesByPublishDate.values)[0]
    else:
        case_num = cases.newCasesByPublishDate.values[0]

    if PER_HUNDRED:
        case_num = case_num / pop_lookup[ltla_code] * 100_000

    col = m.to_rgba(case_num)
    
    points = np.array(zone.shape.points)
    plt.fill(points[:,0], points[:,1], c=col, linewidth=1)

    cases_num = cases.newCasesByPublishDate.values[0]

plt.gca().set_aspect('equal', adjustable='box')
plt.show()