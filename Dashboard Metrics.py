# ---- Imports -------
import csv
import pickle
import os
from functools import partial
import json
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import time
from itertools import groupby
from statistics import mean

import pandas as pd
plt.style.use('seaborn-whitegrid')

from gerrychain import (
    Election,
    Graph,
    MarkovChain,
    Partition,
    GeographicPartition,
    accept,
    constraints,
    updaters,
)
from gerrychain.metrics import efficiency_gap, polsby_popper
from gerrychain.proposals import recom, propose_random_flip
from gerrychain.updaters import (
    Tally,
    boundary_nodes,
    cut_edges,
    cut_edges_by_part,
    exterior_boundaries,
    interior_boundaries,
    perimeter,
    county_splits
)
from gerrychain.tree import recursive_tree_part
from enum import Enum
import gerrymetrics as gm

from smallestenclosingcircle import *

import matplotlib.collections as mplc
from shapely import geometry as sgeom
import descartes as des
import libpysal as ps
from pointpats import *


import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import rpy2.robjects as ro
from rpy2.robjects.conversion import localconverter

class CountySplit(Enum):
    NOT_SPLIT = 0
    NEW_SPLIT = 1
    OLD_SPLIT = 2
    
start_time = time.time()


os.environ['R_HOME'] = '/Library/Frameworks/R.framework/Resources'

r = robjects.r
r['source']('calc_reock.R')

calc_Reock_function_r = robjects.globalenv['calc_Reock']


path = r'/Users/ari/Documents/PGP/Dashboard'
os.chdir(path)


# ----- Code to change -------
path_to_shapefile = r"/Users/ari/Documents/PGP/GA analysis/Data/GA_final/GA_gerrychain_input.shp"
pop_col = "tot"
num_dist = 56
district_assignment_col = "SenDist"
county_assignment_col = "FIPS2"
BVAP_col = "BVAP"
HVAP_col = "HVAP"
VAP_col = "totVAP"
cong = False
cong_seat_pop = 710767 #https://www.census.gov/prod/cen2010/briefs/c2010br-08.pdf
cong_factor = 1.01
leg_factor = 1.05

election_names = [
    "USSEN16",
    "PRES16",
    "GOV18",
    "PRES20"
]
election_columns = [
    ["G16USSDBAR", "G16USSRISA"],
    ["G16PREDCLI", "G16PRERTRU"],
    ["G18GOVDABR", "G18GOVRKEM"],
    ["C20PREDBID", "C20PRERTRU"]
]



# --- Internal Functions ---

def calc_splits(dictionary):
    count_splits = 0
    for name, data in dictionary.items():
        if data[0].value != 0:
            count_splits = count_splits + 1
    return count_splits

def num_vra_districts(partition, pop_col=VAP_col, b_pop_col=BVAP_col, h_pop_col=HVAP_col,  vra_relative_threshold = 0.5):
    # we'll count districts that satisfy the VRA threshold.
    total = 0

    # For each district,...
    for district in partition.parts:
        nodes = partition.parts[district]

        # compute total population...
        pop = sum(partition.graph.nodes[n][pop_col] for n in nodes)

        # and VRA-relevant populations...
        b_pop = sum(partition.graph.nodes[n][b_pop_col] for n in nodes)
        h_pop = sum(partition.graph.nodes[n][h_pop_col] for n in nodes)
        vra_pop = b_pop + h_pop

        # to see if their ratio is greater than the threshold.
        if vra_pop / pop >= vra_relative_threshold:
            total += 1

    return total

def num_opp_districts(partition, pop_col=VAP_col, b_pop_col=BVAP_col, h_pop_col=HVAP_col,  opp_lower = 0.37, opp_upper = 0.5):
    # we'll count districts that satisfy the opportunity-to-elect threshold.
    total = 0

    # For each district,...
    for district in partition.parts:
        nodes = partition.parts[district]

        # compute total population...
        pop = sum(partition.graph.nodes[n][pop_col] for n in nodes)

        # and VRA-relevant populations...
        b_pop = sum(partition.graph.nodes[n][b_pop_col] for n in nodes)
        h_pop = sum(partition.graph.nodes[n][h_pop_col] for n in nodes)
        vra_pop = b_pop + h_pop

        # to see if their ratio is within the threshold.
        if (vra_pop / pop >= opp_lower) and (vra_pop / pop < opp_upper):
            total += 1

    return total

def compute_reock(partition, part, area):
    points = []
    for n in partition.parts[part]:
        loc = graph.nodes[n]["geometry"].centroid
        points.append((loc.x, loc.y))
    circum_circle = make_circle(points)
    radius = circum_circle[2]
    #circum_circle = pointpats.skyum(points)
    #radius = circum_circle[1]
    try:
        return area / (math.pi * radius ** 2)
    except ZeroDivisionError:
        return math.nan

def reock(partition):
    """Computes Reock compactness scores for each district in the partition.
    """
    return {
        part: compute_reock(partition, part, partition["area"][part])
        for part in partition.parts
    }



# ---- Load in graph ------
#graph = Graph.from_file(path_to_shapefile)
#graph = Graph.from_json(path_to_shapefile)
#graph = gpd.read_file(path_to_shapefile)


# ---- Updaters & Elections -----

updaters = {
    "population": updaters.Tally(pop_col, alias="population"),
    "cut_edges": cut_edges,
    "BVAP": updaters.Tally(BVAP_col, alias = "BVAP"),
    "HVAP": updaters.Tally(HVAP_col, alias = "HVAP"),
    "VAP": updaters.Tally(VAP_col, alias = "VAP"),
    "num_vra_districts": num_vra_districts,
    "num_opp_districts": num_opp_districts,
    "county_splits": county_splits("county_splits", county_assignment_col)
}

num_elections = len(election_names)

elections = [
    Election(
        election_names[i],
        {"Dem": election_columns[i][0], "Rep": election_columns[i][1]},
    )
    for i in range(num_elections)
]

election_updaters = {election.name: election for election in elections}

updaters.update(election_updaters)

# ---- Proposed Partition -----

prop_partition = GeographicPartition(graph, district_assignment_col, updaters)

# ---- Geographic Metrics ------


# County Splits
num_county_splits = calc_splits(prop_partition["county_splits"])
upper_county = len(prop_partition["county_splits"])

county_partition = GeographicPartition(graph, county_assignment_col, updaters)

if cong:
    threshold_pop = cong_factor * cong_seat_pop
else:
    total_pop = 0
    for n in graph.nodes:
        total_pop = total_pop + graph.nodes[n][pop_col]
    threshold_pop = leg_factor * total_pop / num_dist

county_populations = county_partition["population"]
county_pops = np.fromiter(county_populations.values(), dtype=float)
county_greater_threshold = county_pops[county_pops >= threshold_pop]
lower_county = len(county_greater_threshold)
    
# --- Compactness -----

# Polsby-Popper
pp_vals = np.fromiter(polsby_popper(prop_partition).values(), dtype=float)
min_pp = min(pp_vals)
avg_pp = np.mean(pp_vals)

# Reock with smallest enclosing circle
reock_vals = np.fromiter(reock(prop_partition).values(), dtype=float)
min_reock = min(reock_vals)
avg_reock = np.mean(reock_vals)


# Reock with min bounding circle in Python
#pointpats.skyum(points)

# Reock with st_minimum_bounding_circle in R
reock_scores = []

for part in prop_partition.parts:
    l = list(prop_partition.parts[part])
    df = pd.DataFrame(l)
    with localconverter(ro.default_converter + pandas2ri.converter):
        df_r = ro.conversion.py2rpy(df)
    #Invoking the R function and getting the result
    result_r = calc_Reock_function_r(df_r, path_to_shapefile)
    #Converting it back to a pandas dataframe.
    reock_score = result_r[0]
    reock_scores.append(reock_score)
    
min_reock = min(reock_scores)
avg_reock = np.mean(reock_scores)


# non-trivial problem to find circumscribing circle
# could use this algorithm: https://people.inf.ethz.ch/gaertner/subdir/software/miniball.html
# contact Justin Solomon to see if he has the algorithm
# export to QGIS


# ---- Racial Fairness Metrics -----
num_vra_districts = prop_partition["num_vra_districts"]
num_opp_districts = prop_partition["num_opp_districts"]

# AVG Election vote shares
vote_shares = [
    np.array(sorted(prop_partition[election_names[i]].percents("Dem")))
    for i in range(num_elections)]
avg_election = np.average(vote_shares, axis = 0)

# ---- Partisan Fairness Metrics ----

# EG assumes equal turnout -- gets a different result than gerrychain
EG = gm.EG(avg_election)

partisan_bias = gm.partisan_bias(avg_election)

mean_median = gm.mean_median(avg_election)


# ---- Competitiveness -----
competitive_vote_shares = avg_election


# ---- Export to JSON ------

chain_obj = {
      "numDists": num_dist,
      "countySplits": num_county_splits,
      "lowerBoundCountySplits": lower_county,
      "upperBoundCountySplits": upper_county,
      "minPolsby-Popper": min_pp,
      "avgPolsby-Popper": avg_pp,
      "minReock": min_reock,
      "avgReock": avg_reock,
      "VRADistricts": num_vra_districts,
      "OpportunityDistricts": num_opp_districts,
      "CompetitiveElections": np.array(competitive_vote_shares).tolist(),
      "EG": EG,
      "partisanBias": partisan_bias,
      "meanMedian": mean_median
  }
#chain_json = json.dumps(chain_obj)
with open("Dashboard Metrics.json", "w") as data_file:
    json.dump(chain_obj, data_file, indent=4)
    


print("end")
time_elapsed = (time.time() - start_time)/60
print("--- %s minutes ---" % time_elapsed)

