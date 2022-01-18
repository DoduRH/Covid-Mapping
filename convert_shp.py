# Script to reduce the number of points in the shapefile
# Loads the shape file 'LTLA/ltla_uk.shp' and reduces the number of points before
# saving the shapefile as 'LTLA/Transformed/ltla_uk_islands_low_points.feather'

# %%
# Imports
import shapefile
import pandas as pd
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

# %%
# Load shapefile
boundaries = shapefile.Reader("LTLA/ltla_uk.shp")

# %%
# Name maps
placecode_map = {
    "E06000060": "E07000006",
    'E06000053': "E06000052",
    'E09000001': "E09000012",
}

placename_map = {
    "Buckinghamshire": "South Bucks",
    "Na h-Eileanan Siar": "Comhairle nan Eilean Siar",
    'Cornwall': "Cornwall and Isles of Scilly",
    'Isles of Scilly': "Cornwall and Isles of Scilly",
    'City of London': "Hackney and City of London",
    'Hackney': "Hackney and City of London",
}

# %%
# define functions

def distance(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def norm(a):
    """Normalise vector a"""
    return a/(np.sqrt((a**2).sum()))

def iscloseangle(angle1, angle2, thresh=0.2):
    """Check if angle a is close to angle b"""
    if angle1 > angle2:
        angle1, angle2 = angle2, angle1

    # angle2 > angle1
    return abs(angle1 - angle2) < thresh or abs(angle1 + 2*np.pi - angle2) < thresh

def point_to_line_distance(p1, p2, p3):
    """Distance from p3 to the line segment p1-p2"""
    # In some cases p1 == p2 for some reason, just return zeros
    if (p1 - p2).sum() == 0:
        return np.zeros(shape=p3.shape[0])
    return np.abs((np.cross(p2-p1, p3-p1)/np.linalg.norm(p2-p1)))

def angle(a):
    """Angle between vector a and the x-axis"""
    return np.arctan2(a[1], a[0])

# %%
# Convert boundaries
df = pd.DataFrame(columns=['areaCode', 'x', 'y', 'areaName'])

tot_old_points = 0
tot_new_points = 0

for i, zone in enumerate(tqdm(boundaries.shapeRecords(), desc="Converting boundaries")):
    ltla_code = zone.record[1]
    ltla_code = placecode_map.get(ltla_code, ltla_code)
    ltla_name = zone.record[2]
    ltla_name = placename_map.get(ltla_name, ltla_name)

    points = np.array(zone.shape.points)
    x, y = points.T

    # Sort out 'islands'
    new_x = np.zeros(shape=x.shape)
    new_y = np.zeros(shape=y.shape)
    new_x[0] = points[0][0]
    new_y[0] = points[0][1]
    offset = 0
    for i, point in enumerate(zip(x[1:], y[1:]), start=1):
        prev_point = (x[i-1], y[i-1])

        # If the distance is >10000 then there is an island.  Plotly plots 2 seperate
        # shapes if there is a null element in the list.
        if distance(*prev_point, *point) > 10000:
            # Put None in the current location
            new_x[i+offset] = None
            new_y[i+offset] = None
            # Add an extra element to the arrays
            new_x = np.append(new_x, 0)
            new_y = np.append(new_y, 0)
            # Increment offset
            offset += 1
        new_x[i+offset] = point[0]
        new_y[i+offset] = point[1]
    
    # Vector simplification algorithm
    line_start = 0
    line_end = 1
    threshold = 150
    points = np.array([new_x, new_y]).T

    new_points = [list(points[0])]

    # Iterate over all the points, making a line from 'start' to 'end'.  
    # Increase the 'end' point until there are points on the line further than 'threshold' away.
    # Careful to seperate islands that are denoted by the 'None' values
    while line_end < len(new_x):
        while line_end < len(new_x) and \
            not np.isnan(points[line_end][0]) and \
            (point_to_line_distance(points[line_start], points[line_end], points[line_start+1:line_end-1]) < threshold).all():

            line_end += 1
        
        # Add last point to the array
        new_points.append(points[line_end-1])
        # If the the value is None, leave them inplace
        if line_end < len(new_x) and np.isnan(points[line_end][0]):
            new_points.append([None, None])
            line_end += 1

        line_start = line_end

    # Cast to numpy array
    xs, ys = np.array(new_points).T

    tot_old_points += len(new_x)
    tot_new_points += len(new_points)

    # Save the new points to the dataframe
    if ltla_code in df.areaCode.values:
        df.loc[df.areaCode == ltla_code, 'x'].values[0].extend([None, *xs])
        df.loc[df.areaCode == ltla_code, 'y'].values[0].extend([None, *ys])
    else:
        df = df.append({'areaCode': ltla_code, 'x': list(xs), 'y': list(ys), 'areaName': ltla_name}, ignore_index=True)

print(f"Total old points: {tot_old_points}")
print(f"Total new points: {tot_new_points}")
df.to_feather('LTLA/Transformed/ltla_uk_islands_low_points.feather')

