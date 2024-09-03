import warnings
from typing import Optional

import torch
import numpy as np
from PIL import Image # type: ignore
from sklearn.cluster import DBSCAN # type: ignore

from configs import SGANConfig
from .sgan.models import TrajectoryGenerator


# This code is modified from the original code in the Social GAN repository (https://github.com/agrimgupta92/sgan).


PathNode = tuple[float, float]

class MotionPredictor:
    def __init__(self, config_file_path: str, model_suffix=None, ref_image_path=None) -> None:
        if model_suffix is not None:
            warnings.warn(f'[{self.__class__.__name__}] model_suffix is not used in the current implementation.')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(config_file_path)
        config = SGANConfig.from_torch(config_file_path)
        self.generator = self._get_generator(config, checkpoint) # trained on ETH data, dt=0.4s

        if ref_image_path is not None:
            self.load_ref_image(ref_img_path=ref_image_path)

    def _get_generator(self, config: SGANConfig, checkpoint: dict):
        generator = TrajectoryGenerator(
            obs_len=config.obs_len,
            pred_len=config.pred_len,
            embedding_dim=config.embedding_dim,
            encoder_h_dim=config.encoder_h_dim,
            decoder_h_dim=config.decoder_h_dim,
            mlp_dim=config.mlp_dim,
            num_layers=config.num_layers,
            noise_dim=config.noise_dim,
            noise_type=config.noise_type,
            noise_mix_type=config.noise_mix_type,
            pooling_type=config.pooling_type,
            pool_every_timestep=config.pool_every_timestep,
            dropout=config.dropout,
            bottleneck_dim=config.bottleneck_dim,
            neighborhood_size=config.neighborhood_size,
            grid_size=config.grid_size,
            batch_norm=config.batch_norm)

        generator.load_state_dict(checkpoint['g_state'])
        generator.cuda()
        generator.train()
        return generator
    
    def _trajs_to_input_tensor(self, human_traj_list: list[list[PathNode]], obsv_len=8):
        n_ped = len(human_traj_list)
        cur_seq     = np.zeros((n_ped, 2, obsv_len))
        cur_seq_rel = np.zeros((n_ped, 2, obsv_len))
        for i, traj_tuple in enumerate(human_traj_list):
            traj = np.array(traj_tuple).T
            traj_rel = np.zeros_like(traj)
            traj_rel[:, 1:] = traj[:, 1:] - traj[:, :-1]
            cur_seq[i, :, :obsv_len] = traj
            cur_seq_rel[i, :, :obsv_len] = traj_rel

        seq_start_end = torch.tensor([(0, n_ped)]).to(self.device)

        obsv_traj_ts = torch.from_numpy(cur_seq).type(torch.float).permute(2, 0, 1).to(self.device)
        obsv_traj_rel_ts = torch.from_numpy(cur_seq_rel).type(torch.float).permute(2, 0, 1).to(self.device)
        return obsv_traj_ts, obsv_traj_rel_ts, seq_start_end

    def _relative_to_abs(self, rel_traj: torch.Tensor, start_pos):
        """
        Args:
            rel_traj: tensor of shape (seq_len, batch, 2)
            start_pos: tensor of shape (batch, 2)
        
        Returns:
            abs_traj: tensor of shape (seq_len, batch, 2)
        """
        # batch, seq_len, 2
        rel_traj = rel_traj.permute(1, 0, 2)
        displacement = torch.cumsum(rel_traj, dim=1)
        start_pos = torch.unsqueeze(start_pos, dim=1)
        abs_traj = displacement + start_pos
        return abs_traj.permute(1, 0, 2)

    def _inferece_original(self, input_trajs: list[list[PathNode]], num_samples=20):
        """Standard SGAN inference function.

        Args:
            input_trajs: A list of input trajectories. Each trajectory is from an object.
            num_samples: The number of desired modes. Defaults to 20.

        Returns:
            pred_traj_list: A list of predicted trajectories. Each element is a sample.
                Each sample is an array of shape (num_time_steps, num_objects, 2).

        Notes:
            Input: 8 steps. Output: 12 steps. Sampling time: 0.4s (trained on the ETH dataset)
        """
        obsv_traj_ts, obsv_traj_rel_ts, seq_start_end = self._trajs_to_input_tensor(input_trajs)
        pred_traj_list:list[np.ndarray] = []
        with torch.no_grad():
            for _ in range(num_samples):
                pred_traj_rel = self.generator(obsv_traj_ts, obsv_traj_rel_ts, seq_start_end)
                pred_traj = self._relative_to_abs(pred_traj_rel, obsv_traj_ts[-1])
                pred_traj_list.append(pred_traj.cpu().numpy())
        return pred_traj_list
    
    def _inference(
            self, input_trajs: list[list[PathNode]], rescale:Optional[float]=1.0, 
            num_samples=20, sampling_time=0.2, pred_len=20) -> list[np.ndarray]:
        """Generate motion prediction from standard SGAN model.

        Args:
            input_trajs: A list of input trajectories. Each trajectory is from an object.
            rescale: Rescale factor. Defaults to 1.0.
            num_samples: The number of desired modes. Defaults to 20.
            sampling_time: The sampling time of the input trajectories. Defaults to 0.2.
            pred_len: The desired length of the predicted trajectories. Defaults to 20.

        Returns:
            pred_traj_list: A list of predicted trajectories. Each element is a sample.
                Each sample is an array of shape (num_objects, pred_len, 2).

        Notes:
            The input trajectory is resampled to 0.4s before feeding to the SGAN model.
            The output trajectory is resampled back to the original sampling time.
        """
        input_trajs_sgan = []
        for traj in input_trajs:
            if rescale is not None:
                traj = [(x[0]*rescale, x[1]*rescale) for x in traj]
            resampled_traj = self.resample_traj(traj, dt_original=sampling_time, dt_new=0.4) # resample for SGAN
            if len(resampled_traj) < 8:
                resampled_traj = [resampled_traj[0]]*(8-len(resampled_traj)) + resampled_traj
            else:
                resampled_traj = resampled_traj[-8:]
            input_trajs_sgan.append(resampled_traj)

        pred_traj_list_sgan = self._inferece_original(input_trajs_sgan, num_samples=num_samples)
        
        pred_traj_list = []
        for pred_traj_sgan in pred_traj_list_sgan:
            pred_traj_sgan = pred_traj_sgan.transpose(1, 0, 2)          # (num_objects, num_time_steps, 2)
            pred_traj = np.empty((pred_traj_sgan.shape[0], pred_len, 2))# (num_objects, num_time_steps, 2)
            for object_i in range(pred_traj_sgan.shape[0]):
                pred_traj_for_i = self.resample_traj(pred_traj_sgan[object_i,:,:].tolist(), dt_original=0.4, dt_new=sampling_time)
                if len(pred_traj_for_i) < pred_len:
                    pred_traj_for_i += [pred_traj_for_i[-1]]*(pred_len-len(pred_traj_for_i))
                else:
                    pred_traj_for_i = pred_traj_for_i[:pred_len]
                pred_traj[object_i] = np.array(pred_traj_for_i)
            pred_traj_list.append(pred_traj)
        return pred_traj_list
    
    def load_ref_image(self, ref_img_path: str) -> None:
        self.ref_image = torch.tensor(np.array(Image.open(ref_img_path).convert('L')))

    def get_motion_prediction_samples(
            self, input_trajs: list[list[PathNode]], rescale:Optional[float]=1.0, 
            num_samples:int=100, sampling_time=0.2, pred_len=20):
        """Get motion prediction (samples from SGAN)

        Args:
            input_trajs: A list of input trajectories. Each trajectory is from an object.
            rescale: Rescale factor. Defaults to 1.0.
            num_samples: The number of desired modes. Defaults to 20.
            sampling_time: The sampling time of the input trajectories. Defaults to 0.2.
            pred_len (T): The desired length of the predicted trajectories. Defaults to 20.

        Returns:
            prediction_samples_list: Each element/object is a numpy array [T*num_samples*2]
        """
        pred_traj_list = self._inference(input_trajs, rescale=rescale, num_samples=num_samples, sampling_time=sampling_time, pred_len=pred_len)
        pred_traj_np = np.array(pred_traj_list) # (num_samples, num_objects, T, 2)
        prediction_samples_np = pred_traj_np.transpose(1, 2, 0, 3)
        prediction_samples_list:list[np.ndarray] = [prediction_samples_np[i] for i in range(prediction_samples_np.shape[0])]
        return prediction_samples_list
    
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
    def resample_traj(traj: list[PathNode], dt_original: float, dt_new: float):
        x_coords = [x[0] for x in traj]
        y_coords = [x[1] for x in traj]

        
        t_original = np.arange(0, len(traj)*dt_original, dt_original)[:len(traj)]
        t_new      = np.arange(0, len(traj)*dt_original, dt_new)[:len(traj)]

        x_new = np.interp(t_new, t_original, x_coords)
        y_new = np.interp(t_new, t_original, y_coords)

        return list(zip(x_new, y_new))
        
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




