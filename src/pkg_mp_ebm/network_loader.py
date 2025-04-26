import os
import sys
import time
import math
import pickle
from datetime import datetime
from typing import Tuple, Optional, List

import torch
import numpy as np
import matplotlib.pyplot as plt # type: ignore

from .net_module.net import UNetExtra
from .network_manager import NetworkManager
from ._data_handle_mmp import data_handler as dh
from ._data_handle_mmp import dataset as ds

from .util.utils_yaml import QuickYaml


class NetworkLoader:
    def __init__(self, dataset_name_abbr: str, pred_range: Tuple[int, int], mode: str, out_layer: Optional[str], loss_type: str, project_dir: str, config_dir: str, verbose=False) -> None:
        """
        Args:
            dataset_name: Name of the dataset, such as 'sdd'.
            pred_range: The range of the prediction length, such as (1,10).
            mode: Should be 'train' or 'test'.
            out_layer: The output layer of the network, should be 'softplus', 'poselu', or None.
            loss_type: Should be 'nll', 'enll', 'bce', or 'kld'
            config_dir: The directory containing the config file.
            verbose: Whether to print out more information. Defaults to False.
        """
        self.check_device()

        self.root_dir = project_dir
        self.config_dir = config_dir
        self.vb = verbose
        self.ready = False
        self._canvas = None

        self.dataset_name_abbr = dataset_name_abbr.lower()
        self.pred_range = pred_range
        self.mode = mode.lower()
        if isinstance(out_layer, str):
            self.out_layer = out_layer
        else:
            self.out_layer = 'none'
        self.loss_type = loss_type.lower()

        self.config_fname = f'{self.dataset_name_abbr}_{pred_range[0]}t{pred_range[1]}_{self.out_layer}_{self.loss_type}_{self.mode}.yaml'

        print(f'[{self.__class__.__name__}] Initialized. Use "quick_setup" to load all parts.')

    @property
    def model(self):
        return self.net_manager.model

    @classmethod
    def from_config(cls, config_file_path: str, project_dir: str, verbose=False):
        """Initialize from a config file.

        Args:
            config_file_path: The path to the config file.
        """
        config_fname = os.path.basename(config_file_path)
        dataset_name_abbr = config_fname.split('_')[0]
        pred_range_str = config_fname.split('_')[1]
        pred_range = (int(pred_range_str.split('t')[0]), int(pred_range_str.split('t')[1]))
        out_layer = config_fname.split('_')[2]
        loss_type = config_fname.split('_')[3]
        mode = config_fname.split('_')[4].split('.')[0]
        config_dir = os.path.dirname(config_file_path)
        return cls(dataset_name_abbr, pred_range, mode, out_layer, loss_type, project_dir, config_dir, verbose)

    def check_device(self):
        """Check if GPUs are available. Record the current time.
        """
        if torch.cuda.is_available():
            print('GPU count:', torch.cuda.device_count(),
                  'First GPU:', torch.cuda.current_device(), torch.cuda.get_device_name(0))
        else:
            print(f'CUDA not working! Pytorch: {torch.__version__}.')
            sys.exit(0)
        torch.cuda.empty_cache()
        print(f'[{self.__class__.__name__}] Pre-check at {datetime.now().strftime("%H:%M:%S, %D")}')

    def quick_setup(self, transform, loss: Optional[dict], model_suffix: str, 
                    num_workers:int=0, ref_image_name:Optional[str]=None, image_ext='png'):
        """Quick setup for training or testing. Load all parts.

        Args:
            transform: Should be torchvision.transforms related.
            Net: The network class to initialize an object.
            loss: Dictionary with loss functions, should have "loss" and "metric (can be None)".
            num_workers: The number of subprocesses to use for data loading. If 0, the main process only.
            ref_image_name: The name of the background reference image. Defaults to None.
            image_ext: The extension of the image file. Defaults to 'png'.

        Notes:
            Load parameters, relevant paths, dataset, data handler, and network manager.
            The loss argument can be None if the mode is 'test'.
        """
        self.load_param()
        self.load_path(model_suffix=model_suffix)
        self.load_data(transform, num_workers, ref_image_name, image_ext)
        self.load_manager(loss)
        self.ready = True

    def quick_setup_inference(self, model_suffix: str):
        self.load_param()
        self.load_path(model_suffix=model_suffix)
        self.load_manager(loss=None)
        assert self.save_path is not None
        self.model.load_state_dict(torch.load(self.save_path, weights_only=True))
        self.model.eval()


    def load_param(self, param_in_list=True) -> dict:
        """Read all parameters from the config yaml file."""
        coonfig_file_path = os.path.join(self.config_dir, self.config_fname)
        if param_in_list:
            param_list = QuickYaml.from_yaml_all(coonfig_file_path, vb=self.vb)
            self.param = {k:v for x in param_list for k,v in x.items()}
        else:
            self.param = QuickYaml.from_yaml(coonfig_file_path, vb=self.vb)
        return self.param

    def load_path(self, model_suffix:str='X'):
        """Load relevant paths from parameters (to self also to return).

        Args:
            param: A dictionary of parameters.

        Returns:
            save_path: The path to save the model.
            csv_path: The path to the csv (label) file.
            data_dir: The path to the data directory.
        """
        self.save_path = None # to save the model
        if self.param['model_path'] is not None:
            self.save_path = os.path.join(self.root_dir, self.param['model_path']+'_'+model_suffix)
        self.csv_path  = os.path.join(self.root_dir, self.param['dataset_label_path'])
        self.data_dir  = os.path.join(self.root_dir, self.param['dataset_path'])
        return self.save_path, self.csv_path, self.data_dir

    def load_data(self, transform, num_workers=0, ref_image_name:Optional[str]=None, image_ext='png'):
        """Load the dataset and data handler.

        Args:
            transform: Should be torchvision.transforms related.
            num_workers: The number of subprocesses to use for data loading. If 0, the main process only. Defaults to 0.
            ref_image_name: The name of the background reference image. Defaults to None.
            image_ext: The extension of the image file. Defaults to 'png'.

        Returns:
            myDS: The dataset.
            myDH: The data handler.
        """
        self.dataset = ds.ImageStackDataset(csv_path=self.csv_path, 
                                            root_dir=self.data_dir, 
                                            transform=transform,
                                            pred_offset_range=self.pred_range, 
                                            ref_image_name=ref_image_name, 
                                            image_ext=image_ext)
        self.data_handler = dh.DataHandler(self.dataset, 
                                           batch_size=self.param['batch_size'], 
                                           num_workers=num_workers)
        print(f'[{self.__class__.__name__}] Data prepared. #Samples(training, val):{self.data_handler.get_num_data()}, #Batches:{self.data_handler.get_num_batch()}')
        print(f'[{self.__class__.__name__}] Sample (shape): image:', self.dataset[0]['input'].shape,', label:', self.dataset[0]['target'].shape)
        return self.dataset, self.data_handler

    def load_manager(self, loss: Optional[dict]):
        self.net = UNetExtra(self.param['input_channel'], num_classes=self.param['pred_len'], out_layer=self.out_layer)
        self.net_manager = NetworkManager(self.net, loss, loss_type=self.loss_type, training_parameter=self.param, device=self.param['device'], verbose=self.vb)
        self.net_manager.build_Network()
        return self.net_manager

    def save_profile(self, save_path:str='./'):
        """Save the training profile to pickle and figure."""
        if not self.ready:
            raise ValueError('The NetManager is not ready.')
        dt = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
        self.net_manager.plot_history_loss()
        plt.savefig(os.path.join(save_path, dt+'.png'), bbox_inches='tight')
        plt.close()
        loss_dict = {'loss':self.net_manager.loss_list, 'val_loss':self.net_manager.val_loss_list}
        with open(os.path.join(save_path, dt+'.pickle'), 'wb') as pf:
            pickle.dump(loss_dict, pf)

    def create_ref_canvas(self, ref_image: torch.Tensor):
        center = (ref_image.shape[1], ref_image.shape[0])
        self._canvas = self.ts_dist_map(center, torch.zeros([x*2 for x in ref_image.shape]), normalize=False)


    def train(self, batch_size:Optional[int]=None, epoch:Optional[int]=None, runon:str='LOCAL', save_profile_path:Optional[str]=None):
        """Train the network.

        Args:
            batch_size: The batch size. If None, read from the config file. Defaults to None.
            epoch: The number of epochs. Defaults to None.
            runon: The running mode. 'LOCAL' or 'REMOTE'. Defaults to 'LOCAL'.
        """
        if not self.ready:
            raise ValueError('Please run "quick_setup" to load all parts first.')

        if epoch is None:
            epoch = self.param['epoch']
        if batch_size is None:
            batch_size = self.param['batch_size']

        print(f'[{self.__class__.__name__}] Model load: {self.param["model_path"]}')

        ### Training
        start_time = time.time()
        self.net_manager.train(self.data_handler, batch_size, epoch, runon=runon, data_handler_val=self.data_handler)
        total_time = round((time.time()-start_time)/3600, 4)
        if (self.save_path is not None) & self.net_manager.complete:
            assert self.save_path is not None
            torch.save(self.model.state_dict(), self.save_path)
        nparams = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'\n[{self.__class__.__name__}] Training done: {nparams} parameters. Cost time: {total_time}h.')

        if save_profile_path is not None:
            self.save_profile(save_profile_path)

    def test(self, idx:int):
        if not self.ready:
            raise ValueError('Please run "quick_setup" to load all parts first.')
        
        img:torch.Tensor   = self.dataset[idx]['input']  # originally np.ndarray
        label:torch.Tensor = self.dataset[idx]['target'] # originally np.ndarray
        traj  = self.dataset[idx]['traj']
        index = self.dataset[idx]['index']
        time = self.dataset[idx]['time']
        pred = self.net_manager.inference(img.unsqueeze(0))
        traj = torch.tensor(traj)
        try:
            ref = self.dataset[idx]['ref']
        except:
            ref = img[-1,:,:]
        return img, label, traj, index, time, pred.cpu(), ref
    
    def inference(self, input_traj: List[Tuple[float, float]], ref_image: torch.Tensor, rescale:float=1.0, device:str='cuda') -> torch.Tensor:
        """Inference the network.

        Args:
            input_traj: The input trajectory.
            ref_image: The background image representing the map/surroundings.
            rescale: Rescale the input trajectory to match the image frame. Defaults to 1.0.

        Returns:
            The raw output of the network with size (C*H*W).
        """
        if (device != 'cpu') and (not torch.cuda.is_available()):
            device = 'cpu'
            print(f'[{self.__class__.__name__}] CUDA not working. Switch to CPU.')
        input_traj_rescale = [(x[0]*rescale, x[1]*rescale) for x in input_traj] # from the world coordinate to the image coordinate
        input_img = self.traj_to_input(input_traj_rescale, ref_image) # C*H*W
        output = self.net_manager.inference(input_img.unsqueeze(0).to(device), device=device)
        return output[0].cpu()
    
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
            clusters = self.net_manager.fit_DBSCAN(traj_samples[i,:], eps=eps, min_sample=min_sample)
            clusters_list.append(clusters)
            mu_list, std_list = self.net_manager.fit_cluster2gaussian(clusters, enlarge, extra_margin)

            conf_list = []
            for cluster in clusters:
                conf_list.append(float(cluster.shape[0]))
            conf_list = [round(x/sum(conf_list), 2) for x in conf_list]

            mu_list_list.append(mu_list)
            std_list_list.append(std_list)
            conf_list_list.append(conf_list)
        return clusters_list, mu_list_list, std_list_list, conf_list_list
    
    def clustering_and_fitting(self, prob_maps: torch.Tensor, num_samples=500, replacement=True, eps=10, min_sample=5, enlarge=1.0, extra_margin=0.0):
        """Inference the network and then do clustering.

        Args:
            prob_maps: [CxHxW] The probability maps along the time axis.
            num_samples: The number of samples to generate on each probability map. Defaults to 500.
            replacement: Whether to sample with replacement (if False, each sample index appears only once). Defaults to True.
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
        if len(prob_maps.shape) != 3:
            raise ValueError('The input probability maps should be [CxHxW].')
        traj_samples = self.net_manager.gen_samples(prob_maps.unsqueeze(0), num_samples=num_samples, replacement=replacement)
        clusters_list, mu_list_list, std_list_list, conf_list_list = self.clustering_and_fitting_from_samples(traj_samples[0, :].numpy(), eps, min_sample, enlarge, extra_margin)
        return clusters_list, mu_list_list, std_list_list, conf_list_list
    

    def traj_to_input(self, input_traj: List[Tuple[float, float]], ref_image: torch.Tensor, normalize=True):
        """From the trajectory to the network input image."""
        assert self._canvas is not None
        if normalize:
            ref_image = ref_image/255.0

        ### XXX For zospital
        ref_image[ref_image>0.5] = 0.68
        ref_image[ref_image<=0.5] = 0.2

        obsv_len = self.param['obsv_len']
        if len(input_traj)<obsv_len:
            input_traj = input_traj + [input_traj[-1]]*(obsv_len-len(input_traj))
        input_traj = input_traj[-obsv_len:]

        input_img = self.get_patches(self._canvas, input_traj, ref_image.shape)
        input_img = torch.concat((input_img, ref_image.unsqueeze(0)), dim=0)
        # input_img = torch.empty(size=[ref_image.shape[0],ref_image.shape[1],0])
        # for position in input_traj:
            # obj_map = self.ts_gaudist_map(position[:2], torch.zeros_like(ref_image), sigmas=(10, 10))
            # obj_map = self.ts_dist_map((int(position[0]), int(position[1])), torch.zeros_like(ref_image))
            # obj_map = self.crop_canvas(self._canvas, (int(position[0]), int(position[1])), ref_image.shape)
            # obj_map = obj_map / obj_map.max()
        #     input_img = torch.concat((input_img, obj_map.unsqueeze(2)), dim=2)
        # input_img = torch.concat((input_img, ref_image.unsqueeze(2)), dim=2)
        # input_img = input_img.permute(2,0,1) # H*W*C -> C*H*W
        return input_img
    
    @staticmethod
    def ts_dist_map(centre, base_matrix: torch.Tensor, normalize=True):
        '''Create a normalized Euclidean distance map given the centre and map size.'''
        base_matrix = torch.zeros(base_matrix.shape)
        x = torch.arange(0, base_matrix.shape[1])
        y = torch.arange(0, base_matrix.shape[0])
        x, y = torch.meshgrid(x, y, indexing='xy')
        base_matrix = torch.norm(torch.stack((x-centre[0], y-centre[1])).float(), dim=0)
        if normalize:
            return base_matrix/base_matrix.max()
        return base_matrix

    @staticmethod
    def ts_gaudist_map(centre, base_matrix: torch.Tensor, sigmas=(100.0, 100.0), rho=0.0, flip=False):
        '''Create a normalized Gaussian distance map given the centre and map size.'''
        sigma_x, sigma_y = sigmas[0], sigmas[1]
        x = torch.arange(0, base_matrix.shape[1])
        y = torch.arange(0, base_matrix.shape[0])
        x, y = torch.meshgrid(x, y, indexing='xy')
        in_exp = -1/(2*(1-rho**2)) * ((x-centre[0])**2/(sigma_x**2) 
                                    + (y-centre[1])**2/(sigma_y**2) 
                                    - 2*rho*(x-centre[0])/(sigma_x)*(y-centre[1])/(sigma_y))
        z:torch.Tensor = 1/(2*torch.pi*sigma_x*sigma_y*math.sqrt(1-rho**2)) * torch.exp(in_exp)
        if flip:
            return 1 - z/z.max()
        else:
            return z/z.max()
        
    @staticmethod
    def crop_canvas(canvas: torch.Tensor, center: Tuple[int, int], size: Tuple):
        '''Crop the canvas given the center and size.'''
        x, y = center
        h, w = size
        start_y = h - y
        start_x = w - x
        end_y = start_y + h
        end_x = start_x + w
        return canvas[start_y:end_y, start_x:end_x]
    
    @staticmethod
    def get_patches(canvas: torch.Tensor, traj, size: Tuple):
        x = np.clip(np.array(traj)[:,0].astype('int'), a_min=0, a_max=canvas.shape[1]-1)
        y = np.clip(np.array(traj)[:,1].astype('int'), a_min=0, a_max=canvas.shape[0]-1)
        h, w = size

        start_y = h - y
        start_x = w - x
        end_y = start_y + h
        end_x = start_x + w

        patches = [canvas[s_y:e_y, s_x:e_x] for s_y, e_y, s_x, e_x in zip(start_y, end_y, start_x, end_x)]
        patches = [patch/patch.max() for patch in patches]
        return torch.stack(patches)