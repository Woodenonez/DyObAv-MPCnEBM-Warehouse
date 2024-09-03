import math
from typing import Optional

import torch
import numpy as np
from PIL import Image # type: ignore
from sklearn.cluster import DBSCAN # type: ignore


PathNode = tuple[float, float]


class MotionPredictor:
    def __init__(self, angular_noise_sigma=math.radians(10), ref_image_path:Optional[str]=None) -> None:
        """
        Args:
            angular_noise_sigma: The standard deviation of the angular noise. Defaults to 10 degrees.
        
        Notes:
            The angular noise is assumed to be Gaussian distributed as in the reference paper.

        References:
            Paper - What the Constant Velocity Model Can Teach Us About Pedestrian Motion Prediction
        """
        self.sigma = angular_noise_sigma

        if ref_image_path is not None:
            self.load_ref_image(ref_image_path)

    def load_ref_image(self, ref_img_path: str) -> None:
        self.ref_image = torch.tensor(np.array(Image.open(ref_img_path).convert('L')))


    def get_motion_prediction_samples(self, input_traj: list[PathNode], num_samples:int=100, pred_len=20):
        """Get motion prediction (samples from the CVM with angular noise)

        Args:
            input_traj: List of tuples. Each tuple is a 2D point (x, y).
            num_samples: Number of samples to generate. Defaults to 100.
            pred_len: Prediction length. Defaults to 20.

        Returns:
            prediction_samples: [T*num_samples*2]

        Notes:
            The sampling time is obtained from the input trajectory.
        """
        # [T*num_samples*2]
        if len(input_traj) < 2:
            prediction_samples = np.tile(input_traj[0][:2], (num_samples, 1))
            prediction_samples = np.tile(prediction_samples, (pred_len, 1, 1))
            return prediction_samples

        last_position = np.array(input_traj[-1][:2])
        llast_position = np.array(input_traj[-2][:2])
        pred_direction = last_position - llast_position
        pred_direction = pred_direction[:2]
        # pred_step_size = np.linalg.norm(pred_direction)
        # pred_direction_unit = pred_direction / pred_step_size

        multiplier = np.arange(1, pred_len+1)
        prediction_sample_pure = np.vstack((last_position[0] + pred_direction[0]*multiplier,
                                            last_position[1] + pred_direction[1]*multiplier)).T
        prediction_samples = np.zeros((pred_len, num_samples, 2))
        prediction_samples[:, 0, :] = prediction_sample_pure
        
        noise = np.random.normal(0, self.sigma, num_samples-1).reshape(-1, 1, 1)
        rotation_matrix = np.concatenate((np.cos(noise), -np.sin(noise), np.sin(noise), np.cos(noise)), axis=2).reshape(-1, 2, 2)
        pred_direction_noisy:np.ndarray = np.dot(rotation_matrix, pred_direction) # (n, 2)
        pred_direction_noisy = pred_direction_noisy[:, :, np.newaxis] * multiplier[np.newaxis, np.newaxis, :]
        prediction_samples_noisy = last_position[np.newaxis, np.newaxis, :] + pred_direction_noisy.transpose(2, 0, 1)
        prediction_samples[:, 1:, :] = prediction_samples_noisy

        return prediction_samples
    
    def clustering_and_fitting_from_samples(self, traj_samples: np.ndarray, eps=10, min_sample=5, enlarge=1.0, extra_margin=0.0):
        """Inference the network and then do clustering.

        Args:
            traj_samples: numpy array [T*x*y], meaning (x, y) at time T
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
        if len(traj_samples.shape) != 3:
            raise ValueError('The input trajectory samples should be [T*(x,y)].')

        clusters_list = []
        mu_list_list = []
        std_list_list = []
        conf_list_list = []
        for i in range(traj_samples.shape[0]):
            clusters = self.fit_DBSCAN(traj_samples[i,:], eps=eps, min_sample=min_sample)
            clusters_list.append(clusters)
            mu_list, std_list = self.fit_cluster2gaussian(clusters, enlarge, extra_margin)

            conf_list = []
            for cluster in clusters:
                conf_list.append(float(cluster.shape[0]))
            conf_list = [round(x/sum(conf_list), 2) for x in conf_list]

            mu_list_list.append(mu_list)
            std_list_list.append(std_list)
            conf_list_list.append(conf_list)
        return clusters_list, mu_list_list, std_list_list, conf_list_list
        
    @staticmethod
    def fit_DBSCAN(data, eps: float, min_sample: int) -> list[np.ndarray]:
        """Generate clusters using DBSCAN.

        Args:
            data: Should be a 2D array, each row is a sample.
            eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
            min_sample: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.

        Returns:
            clusters: A list of clusters, each cluster is a 2D array.
        """
        clustering = DBSCAN(eps=eps, min_samples=min_sample).fit(data)
        nclusters = len(list(set(clustering.labels_)))
        if -1 in clustering.labels_:
            nclusters -= 1
        clusters = []
        for i in range(nclusters):
            cluster = data[clustering.labels_==i, :]
            clusters.append(cluster)
        return clusters
    
    @staticmethod
    def fit_cluster2gaussian(clusters: list[np.ndarray], enlarge=1.0, extra_margin=0.0) -> tuple[list, list]:
        """Generate Gaussian distributions from clusters.

        Args:
            clusters: A list of clusters, each cluster is a 2D array.

        Returns:
            mu_list: A list of means, each mean is a pair of coordinates.
            std_list: A list of standard deviations, each std is a pair of numbers.
        """
        mu_list  = []
        std_list = []
        for cluster in clusters:
            mu_list.append(np.mean(cluster, axis=0))
            std_list.append(np.std(cluster, axis=0)*enlarge+extra_margin)
        return mu_list, std_list