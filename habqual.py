import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
import inp
import time


# [0] define auxiliar window functions
def lower_radius(n_position, n_radius_max=3):
    """
    :param n_position: int cell position
    :param n_radius_max: int window radius
    :return: lower radius
    """
    if n_position < n_radius_max:
        return n_position
    else:
        return n_radius_max


def upper_radius(n_position, n_max, n_radius_max=3):
    """
    :param n_position: int cell position
    :param n_max: int cell upper bound
    :param n_radius_max: int window radius
    :return: lower radius
    """
    if n_position <= n_max - 1 - n_radius_max:
        return n_radius_max
    else:
        return n_max - 1 - n_position


# [1] define the main processing function
def main(n_row_start, n_row_end, n_col_start, n_col_end):
    global grd_deg, grd_lulc, grd_radius, n_window_radius
    # [6] -- varrer a matriz
    for i in range(n_row_start, n_row_end):
        # get general row bounds
        n_row_low = lower_radius(n_position=i, n_radius_max=n_window_radius)
        n_row_upp = upper_radius(n_position=i, n_max=n_rows, n_radius_max=n_window_radius)
        # get row bounds for main grid
        n_row_low_main = i - n_row_low
        n_row_upp_main = i + 1 + n_row_upp
        # get row bounds for radius grid
        n_row_low_rad = n_window_radius - n_row_low
        n_row_upp_rad = n_window_radius + 1 + n_row_upp
        for j in range(n_col_start, n_col_end):
            # get general column bounds
            n_col_low = lower_radius(n_position=j, n_radius_max=n_window_radius)
            n_col_upp = upper_radius(n_position=j, n_max=n_cols, n_radius_max=n_window_radius)

            # get column bounds for main grid
            n_col_low_main = j - n_col_low
            n_col_upp_main = j + 1 + n_col_upp

            # get column bounds for radius grid
            n_col_low_rad = n_window_radius - n_col_low
            n_col_upp_rad = n_window_radius + 1 + n_col_upp

            # get lulc id
            n_lulc_id = grd_lulc[i][j]

            # main window grid:
            grd_window_main = grd_lulc[n_row_low_main:n_row_upp_main, n_col_low_main:n_col_upp_main]
            #plt.imshow(grd_window_main)
            #plt.show()

            for k in range(len(df_threats)):
                s_threat_name = df_threats['THREAT'].values[k]
                #print(s_threat_name)
                n_threat_w = df_threats['W'].values[k]
                n_threat_id = df_threats['Id'].values[k]
                # get lulc sensitivity
                n_lulc_sensi = df_lulc[s_threat_name].values[n_lulc_id]
                # slice impact grid
                grd_window_impact = dct_grd_impact[s_threat_name][n_row_low_rad:n_row_upp_rad, n_col_low_rad:n_col_upp_rad]
                # compute threat component
                grd_window_threat = 1 * (grd_window_main == n_threat_id) * grd_window_impact
                #plt.imshow(grd_window_threat)
                #plt.show()
                n_threat_sum = np.sum(grd_window_threat)
                # accumulate into degratation grid
                grd_deg[i][j] = grd_deg[i][j] + (n_lulc_sensi * n_threat_sum)  # todo include Beta




# import lulc map
s = 'C:/bin/teia/itabuna/lulc_2021.asc'
meta, grd_lulc = inp.asc_raster(s, dtype='byte')
n_rows = len(grd_lulc)
n_cols = len(grd_lulc[0])
print(n_rows)
print(n_cols)

'''plt.imshow(grd_lulc)
plt.show()
'''
# import lulc sensitivity table
s = 'C:/bin/teia/itabuna/lulc.csv'
df_lulc = pd.read_csv(s, sep=',')
print(df_lulc.to_string())

# import lulc threat table
s = 'C:/bin/teia/itabuna/threats_alt.csv'
df_threats = pd.read_csv(s, sep=',')
# compute effective weight
df_threats['W'] = df_threats['WEIGHT'].values / df_threats['WEIGHT'].sum()
print(df_threats.to_string())

n_max_dist_max = df_threats['MAX_DIST'].max()
print(n_max_dist_max)
# define fixed parameters
n_pixel_res = 30 # m
n_resolution = n_pixel_res / 1000 # km
print(n_resolution)
n_exp_factor = 2.3
n_window_radius = int( n_exp_factor * (int(n_max_dist_max / n_resolution) + 1))
print(n_window_radius)

# get grid for window radius
n_window_size = (n_window_radius * 2) + 1
grd_distance = np.ones(shape=(n_window_size, n_window_size), dtype='float32')
grd_distance[n_window_radius][n_window_radius] = 0.0
# distance grid
grd_distance = ndimage.distance_transform_edt(grd_distance)

# compute impact grids
dct_grd_impact = {}
for i in range(len(df_threats)):
    s_name = df_threats['THREAT'].values[i]
    n_dmax = df_threats['MAX_DIST'].values[i] / n_resolution
    s_type = df_threats['DECAY'].values[i]
    grd_impact = np.exp(-2.99 * grd_distance / n_dmax)
    '''plt.imshow(grd_impact)
    plt.title(s_name)
    plt.show()'''
    dct_grd_impact[s_name] = grd_impact

# deploy degradation grid
grd_deg = np.zeros(shape=np.shape(grd_lulc), dtype='float32')
time_start = time.perf_counter()
main(
    n_row_start=0,
    n_row_end=n_rows,
    n_col_start=0,
    n_col_end=n_cols
)
time_end = time.perf_counter()
print('Finished in {:.4f} seconds'.format(time_end - time_start))
plt.imshow(grd_deg)
plt.show()

