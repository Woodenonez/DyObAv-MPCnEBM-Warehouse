import os
import pathlib
from typing import Optional

import numpy as np
from PIL import Image # type: ignore
import torch

from .network_loader import NetworkLoader


PathNode = tuple[float, float]


### Config
root_dir = pathlib.Path(__file__).resolve().parents[2]
# config_file_path = os.path.join(root_dir, 'config', 'scbd_1t20_poselu_enll_test.yaml')
# ref_image_path = os.path.join(root_dir, 'data', 'inference_demo_scud', 'background.png')

class MotionPredictor:
    def __init__(self, config_file_path: str, model_suffix: str, ref_image_path:Optional[str]=None) -> None:
        """
        Args:
            config_file_path: The path to the configuration file.
            model_suffix: The suffix of the model to load, used to distinguish different models.
            ref_image_path: The path to the reference image. Defaults to None.
        """
        self.network_loader = NetworkLoader.from_config(config_file_path=config_file_path, project_dir=str(root_dir), verbose=True)
        self.network_loader.quick_setup_inference(model_suffix=model_suffix)
        if ref_image_path is not None:
            self.load_ref_image(ref_img_path=ref_image_path)

    def _inference(self, input_traj: list[PathNode], rescale:Optional[float]=1.0):
        if rescale is not None:
            input_traj = [(x[0]*rescale, x[1]*rescale) for x in input_traj]
        pred = self.network_loader.inference(input_traj=input_traj, ref_image=self.ref_image)
        return pred

    def load_ref_image(self, ref_img_path: str) -> None:
        self.ref_image = torch.tensor(np.array(Image.open(ref_img_path).convert('L')))

    def get_network_output(self, input_traj: list[PathNode], rescale:Optional[float]=1.0):
        """Get network output

        Args:
            input_traj: Input trajectory.
            rescale: Scale from real world to image world. Defaults to 1.0.

        Returns:
            pred: The logits prediction, [CxHxW].
            e_grid: The energy grid, [CxHxW].
            prob_map: The probability map, [CxHxW]
        """
        pred = self._inference(input_traj=input_traj, rescale=rescale)
        e_grid = self.network_loader.net_manager.to_energy_grid(pred.unsqueeze(0))[0, :]
        prob_map = self.network_loader.net_manager.to_prob_map(pred.unsqueeze(0))[0, :]
        return pred, e_grid, prob_map

    def get_motion_prediction(self, input_traj: list[PathNode], rescale:Optional[float]=1.0, debug:bool=False):
        """Get motion prediction (final clustering and fitting results)

        Args:
            input_traj: Input trajectory.
            rescale: Scale from real world to image world. Defaults to 1.0.

        Returns:
            clusters_list: A list of clusters, each cluster is a list of points.
            mu_list_list: A list of means of the clusters.
            std_list_list: A list of standard deviations of the clusters.
            conf_list_list: A list of confidence of the clusters.
            pred (if debug): The logits prediction, [CxHxW].
            prob_map (if debug): The probability map, [CxHxW]
        """
        pred = self._inference(input_traj=input_traj, rescale=rescale)
        prob_map = self.network_loader.net_manager.to_prob_map(pred.unsqueeze(0))[0, :]
        clusters_list, mu_list_list, std_list_list, conf_list_list = self.network_loader.clustering_and_fitting(prob_map)
        if debug:
            return clusters_list, mu_list_list, std_list_list, conf_list_list, pred, prob_map
        return clusters_list, mu_list_list, std_list_list, conf_list_list
    
    def get_motion_prediction_samples(self, input_traj: list[PathNode], rescale:Optional[float]=1.0, num_samples:int=100, replacement=True):
        """Get motion prediction (samples from probability map)

        Args:
            input_traj: Input trajectory.
            rescale: Scale from real world to image world. Defaults to 1.0.
            num_samples: The number of samples to generate on each probability map. Defaults to 100.
            replacement: Whether to sample with replacement (if False, each sample index appears only once). Defaults to True.

        Returns:
            prediction_samples: numpy array [T*num_samples*2]
        """
        pred = self._inference(input_traj=input_traj, rescale=rescale)
        prob_map = self.network_loader.net_manager.to_prob_map(pred.unsqueeze(0))
        prediction_samples = self.network_loader.net_manager.gen_samples(prob_map, num_samples=num_samples, replacement=replacement)[0, :].numpy()
        return prediction_samples
    
    def clustering_and_fitting_from_samples(self, traj_samples: np.ndarray, eps=10, min_sample=5, enlarge=1.0, extra_margin=0.0):
        """Inference the network and then do clustering.

        Args:
            traj_samples: numpy array [T*num_samples*2]
            eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other. Defaults to 10.
            min_sample: The number of samples in a neighborhood for a point to be considered as a core point. Defaults to 5.

        Raises:
            ValueError: If the input probability maps are not [CxHxW].

        Returns:
            clusters_list: A list of clusters, each cluster is a list of points.
            mu_list_list: A list of means of the clusters.
            std_list_list: A list of standard deviations of the clusters.
            conf_list_list: A list of confidence of the clusters.
        """
        clusters_list, mu_list_list, std_list_list, conf_list_list = self.network_loader.clustering_and_fitting_from_samples(traj_samples, eps=eps, min_sample=min_sample, enlarge=enlarge, extra_margin=extra_margin)
        return clusters_list, mu_list_list, std_list_list, conf_list_list

