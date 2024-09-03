import os
import pathlib

import matplotlib.pyplot as plt # type: ignore
from matplotlib.axes import Axes # type: ignore

from pkg_mp_ebm.motion_predict import MotionPredictor

# Resolution = 1/40 m/pixel
input_traj = [[201, 309], [200, 302], [200, 294], [200, 286], [201, 277]]
ground_truth = [[201, 269], [201, 260], [201, 253], [202, 243], [202, 235], 
                [202, 226], [201, 218], [199, 212], [198, 205], [198, 197], 
                [200, 187], [200, 179], [199, 173], [200, 164], [199, 156], 
                [199, 148], [199, 140], [198, 134], [198, 126], [199, 116]]

### Config
root_dir = pathlib.Path(__file__).resolve().parents[1]
config_file_path = os.path.join(root_dir, 'config', 'scud_1t20_poselu_enll_test.yaml')
ref_image_path = os.path.join(root_dir, 'data', 'inference_demo_scud', 'background.png')

### Predict
motion_predictor = MotionPredictor(config_file_path=config_file_path, model_suffix='1', ref_image_path=ref_image_path)
pred, e_grid, prob_map= motion_predictor.get_network_output(input_traj=input_traj) # type: ignore

### Run
fig, axes = plt.subplots(2, 2)
ax1: Axes = axes[0,0]
ax2: Axes = axes[0,1]
ax3: Axes = axes[1,0]
ax4: Axes = axes[1,1]

### CGF
# mu_list_list = []
# std_list_list = []
# conf_list_list = []
# gmm_list:List[GaussianMixture] = []
# traj_samples = network_loader.net_manager.gen_samples(prob_map, num_samples=500, replacement=True)
# for i in range(prob_map.shape[1]):
#     clusters = network_loader.net_manager.fit_DBSCAN(traj_samples[0,i,:].numpy(), eps=10, min_sample=5)
#     if len(clusters) > 0:
#         gmm = GaussianMixture(n_components=len(clusters)).fit(traj_samples[0,i,:])
#         gmm_list.append(gmm)
#     mu_list, std_list = network_loader.net_manager.fit_cluster2gaussian(clusters)

#     conf_list = []
#     for mu in mu_list:
#         conf_list.append(prob_map[0, i, int(mu[1]), int(mu[0])].item() + 1e-9)
#     conf_list = [round(x/sum(conf_list),2) for x in conf_list]

#     mu_list_list.append(mu_list)
#     std_list_list.append(std_list)
#     conf_list_list.append(conf_list)

# Visualize
ax1.imshow(motion_predictor.ref_image, cmap='gray')
ax1.plot([x[0] for x in input_traj], [x[1] for x in input_traj], 'g.')
ax1.plot([x[0] for x in ground_truth], [x[1] for x in ground_truth], 'r.')
ax1.set_title('Input')

ax2.imshow(e_grid[-1,:,:].detach().numpy(), cmap='gray')
ax2.plot([ground_truth[-1][0]], [ground_truth[-1][1]], 'r.')
ax2.set_title('Energy grid')

ax3.imshow(prob_map[-1,:,:].detach().numpy(), cmap='gray')
ax3.plot([ground_truth[-1][0]], [ground_truth[-1][1]], 'r.')
ax3.set_title('Probability map')

# plt.draw()
# if pause_time==0:
#     plt.pause(0.01)
#     while not plt.waitforbuttonpress():
#         pass
# else:
#     plt.pause(pause_time)

plt.show()
