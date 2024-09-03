import os
import json
import pathlib
from io import BufferedReader

import numpy as np
import matplotlib.pyplot as plt

from basic_map.map_occupancy import OccupancyMap
from basic_map.map_geometric import GeometricMap


def read_pgm(pgmf: BufferedReader, bit_depth:int=16, one_line_head:bool=False, skip_second_line:bool=True) -> list[list]:
    """Return a raster of integers from a PGM file as a list of lists (The head is normally [P5 Width Height Depth])."""
    header = pgmf.readline()  # the 1st line
    if one_line_head:
        magic_num = header[:2]
        (width, height) = [int(i) for i in header.split()[1:3]]
        depth = int(header.split()[3])
    else:
        magic_num = header
        if skip_second_line:
            comment = pgmf.readline() # the 2nd line if there is
            print(f'Comment: [{str(comment)}]')
        (width, height) = [int(i) for i in pgmf.readline().split()]
        depth = int(pgmf.readline())

    if bit_depth == 8:
        assert magic_num[:2] == 'P5'
        assert depth <= 255
    elif bit_depth == 16:
        assert magic_num[:2] == b'P5'
        assert depth <= 65535

    raster = []
    for _ in range(height):
        row = []
        for _ in range(width):
            row.append(ord(pgmf.read(1)))
        raster.append(row)
    return raster

def read_pgm_and_process(pgmf: BufferedReader, inversed_pixel:bool, bit_depth:int=16, one_line_head:bool=False, skip_second_line:bool=True) -> np.ndarray:
    raw_map = read_pgm(pgmf, bit_depth, one_line_head, skip_second_line)
    the_map = np.array(raw_map)
    if inversed_pixel:
        the_map = 255 - the_map
    the_map[the_map>10] = 255
    the_map[the_map<=10] = 0
    the_map[:,[0,-1]] = 0
    the_map[[0,-1],:] = 0
    return the_map


project_root = pathlib.Path(__file__).resolve().parents[1]
map_png_path = os.path.join(project_root, 'data', 'warehouse_sim_original', 'background.png')
map_pgm_path = os.path.join(project_root, 'data', 'warehouse_sim_original', 'map.pgm')

# occ_map = OccupancyMap.from_image(map_png_path)
with open(map_pgm_path, 'rb') as f:
    occ_map = OccupancyMap(map_image=read_pgm_and_process(f, inversed_pixel=True))
boundary_coords, obstacle_list = occ_map.get_geometric_map()
geo_map = GeometricMap.from_raw(boundary_coords, obstacle_list)

json_dict = {'boundary_coords': boundary_coords, 'obstacle_list': obstacle_list}
json_path = os.path.join(project_root, 'data', 'warehouse_sim_original', 'map_gen.json')
# with open(json_path, 'w') as jf:
#     json.dump(json_dict, jf)

fig, ax = plt.subplots()

occ_map.plot(ax, cmap='gray')
geo_map.plot(ax, original_plot_args={'color': 'r'}, obstacle_filled=False)


plt.show()

