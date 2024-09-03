import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np


methods = ["MPC-nCVM", "MPC-SGAN", "RPP", "TEB-MPC", "MPC-ENLL"]
s1_cnts_with_sf = np.array([68, 67, 31, 43, 97])
s1_cnts_without_sf = np.array([8, 11, 0, 5, 96])
s2_cnts_with_sf = np.array([100, 100, 100, 100, 100])
s2_cnts_without_sf = np.array([84, 63, 16, 56, 91])
s3_cnts_with_sf = np.array([78, 75, 0, 73, 85])
s3_cnts_without_sf = np.array([5, 11, 0, 6, 77])

s1_weight_counts = {
    "Without SF": s1_cnts_without_sf,
    "With SF": s1_cnts_with_sf - s1_cnts_without_sf,
}
s2_weight_counts = {
    "Without SF": s2_cnts_without_sf,
    "With SF": s2_cnts_with_sf - s2_cnts_without_sf,
}


bar_width = 0.4
s1_index = np.arange(len(methods))
s2_index = np.arange(len(methods)) + bar_width
method_index = (s1_index + s2_index) / 2
all_index = s1_index.tolist() + s2_index.tolist()


# color_list = ['#2878b5', '#9ac9db', '#6AB829', '#CADB9E']
# color_list = ['#2878b5', '#9ac9db', '#2AB89B', '#96D9BD']
color_list = ['#2878b5', '#9ac9db', '#72b063', '#b8dbb3']
fig, ax = plt.subplots()
ax: Axes # type: ignore

bottom = np.zeros(5)
for boolean, weight_count in s1_weight_counts.items():
    p = ax.bar(s1_index, weight_count, bar_width, label=boolean, bottom=bottom, color=color_list.pop(0))
    bottom += weight_count

bottom = np.zeros(5)
for boolean, weight_count in s2_weight_counts.items():
    p = ax.bar(s2_index, weight_count, bar_width, label=boolean, bottom=bottom, color=color_list.pop(0))
    bottom += weight_count

# ax.set_title("...")
ax.set_xticks(all_index)
ax.set_xticklabels(['S1']*len(s1_index)+['S2']*len(s2_index), fontsize=14, rotation=0)
ax.set_xticks(method_index, minor=True)
ax.set_xticklabels(methods, rotation=0, fontsize=16, minor=True)
ax.tick_params(axis='x', which='minor', pad=20)
ax.tick_params(axis='y', labelsize=14)
ax.set_ylabel("Success Rate (%)", fontsize=16)
ax.set_ylim(0, 120)
ax.legend(loc="upper left", prop={"size": 16})

plt.show()