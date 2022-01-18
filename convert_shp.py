"""
Script to reduce points in shapefile

OG Data https://geoportal.statistics.gov.uk/datasets/ons::local-authority-districts-december-2018-boundaries-gb-bfc/explore - use this for Northampton as it seems ok?
Takes shape file 'LTLA/Local_Authority_Districts_(May_2021)_UK_BFE_V3/LAD_MAY_2021_UK_BFE_V2.shp' and reduces the number of points in them
https://geoportal.statistics.gov.uk/datasets/local-authority-districts-may-2021-uk-bfe/explore
Takes approximately 35 minutes to run
"""

# FIXME: South bucks engulfs several other areas
# FIXME: (Linked with above?) South northamptonshire text shows in the wrong place
# FIXME: Cambridge (the outer bit) doesnt cannot be hovered

# %%
# Imports
import shapefile
import pandas as pd
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree, KDTree
from plotly import graph_objects as go
import json
import pathlib

# %%
# Name maps
placecode_map = {
    "E06000060": "E07000006",
    'E06000053': "E06000052",
    'E09000001': "E09000012",
}

inv_placecode_map = {v: k for k, v in placecode_map.items()}

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

def distance(p1, p2):
    p1 = np.asarray(p1).reshape(-1, 2)
    return ((p1[:,0] - p2[0]) ** 2 + (p1[:,1] - p2[1]) ** 2) ** 0.5

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

def find_nearest(array, point):
    array = np.asarray(array)
    idx = (distance(array, point)).argmin()
    return array[idx]

def get_index_from_code(ltla_code):
    """Get the ltla code from the name"""
    for record in main_boundaries:
        if record.record[1] == ltla_code:
            return record.record[0]-1
    raise ValueError(f"Could not find ltla index for {ltla_code}")

def get_index_from_name(ltla_name):
    """Get the ltla code from the name"""
    for record in main_boundaries:
        if record.record[2] == ltla_name:
            return record.record[0]-1 # -1 because the indexing is off by one
    raise ValueError(f"Could not find ltla index for {ltla_name}")

def simplify_line(points, progress=False):
    """Simplify a line using to use fewer points"""

    # In some cases points is empty, return empty array
    if points.shape[0] == 0:
        return []

    # Sort out 'islands'
    new_x = np.zeros(shape=points.shape[0])
    new_y = np.zeros(shape=points.shape[0])
    new_x[0] = points[0][0]
    new_y[0] = points[0][1]
    offset = 0

    # TODO: This may be able to be replaced with more efficient solution (numpy)
    for i, point in enumerate(points[1:], start=1):
        old_point = points[i-1]
        if distance(old_point, point) > 10000:
            # Distance between points is too high
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

    if progress:
        # Show progress bar
        t = tqdm(desc="Simplifying boundaries", total=len(new_x))

    # Iterate over all the points, making a line from 'start' to 'end'.  
    # Increase the 'end' point until there are points on the line further than 'threshold' away.
    # Careful to seperate islands that are denoted by the 'nan' values
    while line_end < len(new_x):
        while line_end < len(new_x) and \
            not np.isnan(points[line_end][0]) and \
            (point_to_line_distance(points[line_start], points[line_end], points[line_start+1:line_end-1]) < threshold).all():

            line_end += 1
        
        # Add last point to the array
        new_points.append(points[line_end-1])
        # If the the value is 'nan', replace them with 'None' for plotly
        if line_end < len(new_x) and np.isnan(points[line_end][0]):
            new_points.append([None, None])
            line_end += 1

        line_start = line_end
        if progress:
            t.n = line_end
            t.update(0)
    if progress:
        t.close()

    return new_points

def array_slice_wrap(arr, start, end):
    """Slice an array, wrapping around the end"""
    if start >= end:
        return np.append(arr[start:], arr[:end], axis=0)
    else:
        return arr[start:end]

def parse_ltla(ltla_zone):
    ltla_code = ltla_zone.record[1]
    current_zone = ltla_zone
    ltla_name = current_zone.record[2]

    return current_zone, ltla_code, ltla_name

# TODO: Split this into functions
def simplify_boundary(data, neighbors, ltla_zone, progress=False):
    """Simplifies the boundary of an ltla
        Data is a reference to the dictionary containing the completed boundaries
        Neighbors is a reference to the dictionary containing the neighbors of the completed boundaries
        Both will be modified by the function
    """
    # Get current zone from boundaries
    current_zone, ltla_code, ltla_name = parse_ltla(ltla_zone)

    # get self zone info
    self_point_array = np.array(current_zone.shape.points)

    # Setup KDTree to get closest points
    tree = cKDTree(self_point_array)

    # own_shared is a dict of indexes of the neighbors shared with the current zone
    # and the corresponding boundary (key is the minimum of the two self indexes self shares with other)
    own_shared = {}
    if progress:
        b = tqdm(main_boundaries, desc=f"{ltla_name} finding neighbors")
    else:
        b = main_boundaries

    neighbor_exists = False
    for zone in b:
        # Skip current zone
        if zone.record[0] == current_zone.record[0]:
            continue
        
        closest_points = tree.query(zone.shape.points, k=1, distance_upper_bound=1e-8, workers=-1)
        
        # Check if any points are equal
        if (closest_points[0] == 0).any():
            neighbor_exists = True
            other_point_array = np.array(zone.shape.points)

            # Get the indexes where the points go from being equal to not being equal
            other_changes_idx = np.where(np.diff(closest_points[0] == 0, prepend=False, append=False))[0]
            # Reshape to give 'pairs' of cords where the points are equal
            other_shared_indexes = other_changes_idx.reshape(-1, 2)
            other_shared_indexes[:,1] -= 1

            # Get corners where the boundaries attach/detach
            corners = other_point_array[other_shared_indexes.flatten()]

            # Get indexes of the corners of current boundary
            self_changes_idx = np.zeros(shape=other_changes_idx.shape)
            for i, corner in enumerate(corners):
                self_changes_idx[i] = np.where(np.all(self_point_array == corner, axis=1))[0][0]

            self_shared_idx = self_changes_idx.reshape(-1, 2).astype(int)
            for i, r in enumerate(self_shared_idx):
                # Ensure the start and end points are correct before continuing
                assert (other_point_array[other_shared_indexes[i]] == self_point_array[r]).all(), "Points do not match"

                # Work out the direction of the boundary
                for a in [0, 1]:
                    for b in [0, 1]:
                        if closest_points[1][other_shared_indexes[i][a]] == r[b]:
                            matching = a,b
                            break
                
                if abs(closest_points[1][(other_shared_indexes[i][matching[0]]+1) % len(closest_points[1])] - r[matching[1]]) == 1:
                    middle_in_range = closest_points[1][(other_shared_indexes[i][matching[0]]+1) % len(closest_points[1])]
                else:
                    middle_in_range = closest_points[1][(other_shared_indexes[i][matching[0]]-1) %  len(closest_points[1])]

                r.sort()
                if not r[0] < middle_in_range < r[1]:
                    r = r[::-1]

                own_shared[min(r)] = {
                    "Indexes": r,
                    "other_indexes": other_shared_indexes[i],
                    "other_code": zone.record[1],
                    "other_name": zone.record[2],
                }

    # Simplify each 'line' segment
    # Initialise the self_boundary list
    if neighbor_exists:
        # self_shared_indexes = np.sort(np.array([x['Indexes'] for x in own_shared.values()]))
        # if self_shared_indexes[0][0] > 0:
        #     self_boundary = simplify_line(self_point_array[:self_shared_indexes[0][0]])
        # else:
        self_boundary = []
        sorted_keys = np.sort(list(own_shared.keys()))
        
        # For each pair of points, extend self_boundary with the simplified version
        for i, key in enumerate(sorted_keys):
            pts = own_shared[key]['Indexes']

            # Get points from start or end of current segment
            neighbor = neighbors.get(
                f"{ltla_code}-{pts[0]}",
                neighbors.get(
                    f"{ltla_code}-{pts[1]}"
                )
            )

            # if f"{ltla_code}-{pts[0]}" in neighbors.keys():
            #     # The neighbor has already determined this section of boundary, copy it
            #     self_boundary.extend(
            #         neighbors[f"{ltla_code}-{pts[0]}"]['new_boundary']
            #     )
            # elif f"{ltla_code}-{pts[1]}" in neighbors.keys():
            #     self_boundary.extend(
            #         neighbors[f"{ltla_code}-{pts[1]}"]['new_boundary'][::-1]
            #     )

            # FIXME: This may be the cause of some bugs later,maybe better to check which end the point is at?  
            #   self_point_array[pts[0]] == neighbors[f"{ltla_code}-{pts[1]}"]['new_boundary'][0]
            # The neighbor has already determined this section of boundary, copy it but maybe need to revese it
            
            if neighbor is not None and np.allclose(neighbor.get("Indexes"), pts):
                boundary_line = neighbor.get("new_boundary")
                if distance(boundary_line[0], self_point_array[pts[0]]) < distance(boundary_line[-1], self_point_array[pts[0]]):
                    self_boundary.extend(boundary_line)
                else:
                    self_boundary.extend(boundary_line[::-1])

            else:
                # We need to determine this section of boundary and save it for others
                other_idx = own_shared[key]['other_indexes']

                # Make the new boundary
                new_boundary = simplify_line(array_slice_wrap(self_point_array, pts[0], pts[1]))
                # Add it to own boundary
                self_boundary.extend(new_boundary)
                
                # Save the boundary to the other neighbor
                other_ltla = own_shared[key]['other_code']
                assert f"{other_ltla}-{other_idx[0]}" not in neighbors.keys()
                neighbors[f"{other_ltla}-{other_idx[0]}"] = {"Indexes": other_idx, "Code": ltla_code, "new_boundary": new_boundary}

            # Do the section to the next shared boundary (if there is any to do)
            if i + 1 < len(sorted_keys) and pts[1] != sorted_keys[i+1]:
                self_boundary.extend(
                    simplify_line(array_slice_wrap(self_point_array, pts[1], sorted_keys[i+1]))
                )
        
        # Add the last bit of the boundary
        if pts[1] != own_shared[sorted_keys[0]]['Indexes'][0]:
            a=1
            self_boundary.extend(
                simplify_line(array_slice_wrap(self_point_array, pts[1], sorted_keys[0]))
            )
    else:
        self_boundary = simplify_line(self_point_array)

    x, y = np.array(self_boundary).T
    # Save data to data dict to be converted to the following dataframe
    # pd.DataFrame(columns=['areaCode', 'x', 'y', 'areaName'])
    ltla_code = placecode_map.get(ltla_code, ltla_code)
    if ltla_code in data.keys():
        data[ltla_code] = {
            "x": np.append(data[ltla_code]["x"], [np.nan, *x]),
            "y": np.append(data[ltla_code]["y"], [np.nan, *y]),
            "areaName": placename_map.get(ltla_name, ltla_name),
        }
    else:
        data[ltla_code] = {
            "x": x,
            "y": y,
            "areaName": placename_map.get(ltla_name, ltla_name),
        }

# %%
# Run simplification algorithm on all the boundaries
data = dict()
neighbors = dict()

def prt(x):
    print(x.record[2])
    return x


# Get boundaries
# Northampton recently changed the boundaries so we need to get the latest
northampton_areas = [
    "Corby",
    "Daventry",
    "East Northamptonshire",
    "Kettering",
    "Northampton",
    "South Northamptonshire",
    "Wellingborough",
]

removed_areas = [
    'North Northamptonshire',
    'West Northamptonshire'
]

main_shapes = shapefile.Reader("LTLA/Local_Authority_Districts_(May_2021)_UK_BFE_V3/LAD_MAY_2021_UK_BFE_V2.shp").shapeRecords()
main_boundaries = [x for x in main_shapes if x.record[2] not in removed_areas]

second_shapes = shapefile.Reader("LTLA/Local_Authority_Districts_(Dec_2017)_UK_BFE_V3/Local_Authority_Districts_(December_2017)_Boundaries_in_Great_Britain.shp").shapeRecords()
main_boundaries.extend(
    [x for x in second_shapes if x.record[2] in northampton_areas]
)

# %%
# Simplify the boundaries
for i, zone in enumerate(tqdm([x for x in second_shapes if x.record[2] in northampton_areas], desc="Simplifying boundaries")):
    simplify_boundary(data, neighbors, ltla_zone=zone, progress=False)


# %%
# Save the data
new_boundaries = pd.DataFrame.from_dict(data, orient='index')
(
    new_boundaries
    .reset_index()
    .rename({"index": "areaCode"}, axis=1)
    .to_feather('LTLA/Transformed/ltla_uk_islands_low_points_connected_mixed.feather')
)

# %%
# Plot the new boundaries
s = []
for i, (ltla_code, (x, y, ltla_name)) in enumerate(new_boundaries.iterrows()):
    s.append(
        go.Scatter(
            x=x,
            y=y,
            mode='lines',
            marker=dict(
                size=10
            ),
            fill='toself',
            name=ltla_name,
            text=ltla_name,
        )
    )

fig = go.Figure(s)
fig.update_layout(
    title=dict(text="Boundaries of the UK Islands", x=0.5),
    xaxis_title="Longitude",
    yaxis_title="Latitude",
    yaxis_scaleanchor="x",
    yaxis_scaleratio=1
)
fig.show()

# %%
# Plot the Northampton swapped boundaries
s = []
for zone in [x for x in main_shapes if x.record[2] in removed_areas]:
    x, y = np.array(zone.shape.points).T
    s.append(
        go.Scatter(
            x=x,
            y=y,
            mode='lines',
            marker=dict(
                size=10
            ),
            fill='toself',
        )
    )

fig = go.Figure(s)
fig.show()

s = []
for zone in [x for x in second_shapes if x.record[2] in northampton_areas or "Bucks" in x.record[2]]:
    x, y = np.array(zone.shape.points).T
    s.append(
        go.Scatter(
            x=x,
            y=y,
            mode='lines',
            marker=dict(
                size=10
            ),
            fill='toself',
            name=zone.record[2],
        )
    )

fig = go.Figure(s)
fig.update_layout(
    title=dict(text="Bits", x=0.5),
    xaxis_title="Longitude",
    yaxis_title="Latitude",
    yaxis_scaleanchor="x",
    yaxis_scaleratio=1
)
fig.layout.yaxis.scaleratio = 1
fig.show()