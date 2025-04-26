from timeit import default_timer as timer
from datetime import timedelta
from typing import Tuple, List, Optional

import numpy as np
import matplotlib.pyplot as plt # type: ignore
from sklearn.cluster import DBSCAN # type: ignore

import torch
import torch.nn as nn
import torch.optim as optim

# from util import utils_kmean
from .net_module import loss_functions
from .net_module.net import UNetExtra
from ._data_handle_mmp.data_handler import DataHandler


class NetworkManager:
    """Manage training the given network and inference/test."""
    def __init__(self, network: UNetExtra, loss_dict: Optional[dict], loss_type: str, training_parameter: dict, device:str='cuda', verbose:bool=False):
        """
        Args:
            network: The neural network.
            loss_dict: Dictionary with loss functions, should have "loss" and "metric (can be None)".
            loss_type: Should be 'nll', 'enll', 'bce', or 'kld'
            training_parameter: Dictionary with all training parameters, such as the learning rate.
            device: The device for running the program, should be "cpu", "cuda", or "multi (multiple GPUs)".
        """
        self.vb = verbose
        self.device = device
        self.complete = False
        self.__load_parameters(training_parameter)
        self.__training_time(None, None, None, init=True)

        self.loss_list: List[float] = []     # track the loss
        self.val_loss_list: List[float] = [] # track the validation loss

        self.network = network
        self.loss_type = loss_type
        if (loss_dict is not None):
            self.loss_func = loss_dict['loss']
            self.metric = loss_dict['metric']
        else:
            print(f'[{self.__class__.__name__}] >>> No loss function detected <<<')

    def __load_parameters(self, training_parameter:dict):
        self.lr = training_parameter['learning_rate']
        self.wr = training_parameter['weight_regularization'] # L2 regularization
        self.es = training_parameter['early_stopping']
        self.cp = training_parameter['checkpoint_dir']

    def __training_time(self, remaining_epoch, remaining_batch, batch_per_epoch, init=False):
        if init:
            self.batch_time = []
            self.epoch_time = []
        else:
            batch_time_average = sum(self.batch_time)/max(len(self.batch_time),1)
            if len(self.epoch_time) == 0:
                epoch_time_average = batch_time_average * batch_per_epoch
            else:
                epoch_time_average = sum(self.epoch_time)/max(len(self.epoch_time),1)
            eta = round(epoch_time_average * remaining_epoch + batch_time_average * remaining_batch, 1)
            return timedelta(seconds=batch_time_average), timedelta(seconds=epoch_time_average), timedelta(seconds=eta)

    def plot_history_loss(self):
        plt.figure()
        if len(self.val_loss_list):
            plt.plot(np.array(self.val_loss_list)[:,0], np.array(self.val_loss_list)[:,1], '.', label='val_loss')
        plt.plot(np.linspace(1,len(self.loss_list),len(self.loss_list)), self.loss_list, '.', label='loss')
        plt.xlabel('#batch')
        plt.ylabel('Loss')
        plt.legend()


    def build_Network(self):
        self.gen_model()
        self.gen_optimizer(self.model.parameters())

    def gen_model(self):
        self.model = nn.Sequential()
        self.model.add_module('Net', self.network)
        if self.device == 'multi':
            self.model = nn.DataParallel(self.model.to(torch.device("cuda:0")))
        elif self.device == 'cuda':
            self.model = self.model.to(torch.device("cuda:0"))
        elif self.device != 'cpu':
            raise ModuleNotFoundError(f'[{self.__class__.__name__}] No such device as {self.device} (should be "multi", "cuda", or "cpu").')
        return self.model

    def gen_optimizer(self, parameters):
        self.optimizer = optim.Adam(parameters, lr=self.lr, weight_decay=self.wr, betas=(0.99, 0.999))
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.99)
        return self.optimizer


    def inference(self, data: torch.Tensor, device='cpu') -> torch.Tensor:
        if self.device in ['multi', 'cuda']:
            device = torch.device("cuda:0")
        # with torch.no_grad():
        start_time = timer()
        with torch.inference_mode():
            output = self.model(data.float().to(device))
        # print(f'Inference time: {timer()-start_time}')
        return output

    def validate(self, data, labels, loss_function=None) -> torch.Tensor:
        """Calculate the loss given data and labels."""
        if loss_function is None:
            loss_function = self.loss_func
        outputs = self.model(data)
        if isinstance(self.loss_func, torch.nn.BCEWithLogitsLoss): # XXX to compare with BCE
            labels = loss_functions.get_weight(outputs, labels, sigmas=(10, 10)) # default sigma is 10
        loss = loss_function(outputs, labels)
        return loss

    def train_batch(self, batch, label, loss_function=None):
        self.model.zero_grad()
        loss = self.validate(batch, label, loss_function)
        loss.backward()
        self.optimizer.step()
        return loss

    def train(self, data_handler_train: DataHandler, batch_size:int, epochs:int, 
              current_epoch:int=0, runon:str='LOCAL', data_handler_val:Optional[DataHandler]=None):
        print(f'\n[{self.__class__.__name__}] Training...')
        if runon == 'LOCAL':
            report_after_batch = 10
        elif runon == 'REMOTE':
            report_after_batch = 1000
        else:
            raise ModuleNotFoundError('RUNON ERROR.')

        if self.device in ['multi', 'cuda']:
            device = 'cuda'
        else:
            device = 'cpu'

        if data_handler_val is None:
            data_handler_val = data_handler_train

        ### Training
        num_batch_per_epoch = data_handler_train.get_num_batch()
        cnt = 0 # counter for batches over all epochs
        for ep in range(epochs):
            if ep<current_epoch:
                continue

            epoch_time_start = timer() ### TIMER

            cnt_per_epoch = 0 # counter for batches within the epoch
            while (cnt_per_epoch<num_batch_per_epoch):
                cnt += 1
                cnt_per_epoch += 1

                batch_time_start = timer() ### TIMER
                batch, label = data_handler_train.return_batch()
                batch, label = batch.float().to(device), label.float().to(device)

                loss = self.train_batch(batch, label, loss_function=self.loss_func) # train here
                self.loss_list.append(loss.item())
                self.batch_time.append(timer()-batch_time_start)  ### TIMER

                if np.isnan(loss.item()):
                    print(f'[{self.__class__.__name__}] Loss goes to NaN! Fail after {cnt} batches.')
                    self.complete = False
                    return -1

                if (cnt_per_epoch%report_after_batch==0 or cnt_per_epoch==num_batch_per_epoch) & (self.vb):
                    _, _, eta = self.__training_time(epochs-ep-1, num_batch_per_epoch-cnt_per_epoch, num_batch_per_epoch) # TIMER
                    prt_loss = f'Training loss: {round(loss.item(),4)}'
                    prt_num_samples = f'{cnt_per_epoch*batch_size/1000}k/{num_batch_per_epoch*batch_size/1000}k'
                    prt_num_epoch = f'Epoch {ep+1}/{epochs}'
                    prt_eta = f'ETA {eta}'
                    print(f'\r[{self.__class__.__name__}] {prt_loss}, {prt_num_samples}, {prt_num_epoch}, {prt_eta}       ', end='')
            ### Epoch training done

            ### Testing
            # cnt_per_epoch = 0 # counter for batches within the epoch
            # test_ADE_list = []
            # while (cnt_per_epoch<data_handler_val.get_num_batch()):
            #     cnt_per_epoch += 1

            #     if cnt_per_epoch>5:
            #         break

            #     test_data, test_label = data_handler_val.return_batch()
            #     test_data, test_label = test_data.float().to(device), test_label.float().to(device)
            #     if isinstance(self.loss_func, torch.nn.BCEWithLogitsLoss):
            #         test_label = loss_functions.get_weight(test_data, test_label, sigma=20) # default sigma is 20
            #     test_ADE = self.test(test_data, test_label, input_prob=isinstance(self.loss_func, torch.nn.BCEWithLogitsLoss)) # test here
            #     test_ADE_list.append(test_ADE)
            # print(f'Test ADE: {np.mean(test_ADE_list)}')

            # if np.mean(test_ADE_list) < min_test_loss:
            #     min_test_loss = np.mean(test_ADE_list)
            #     num_epochs_no_improve = 0
            #     if self.cp is not None:
            #         self.save_checkpoint(self.model, self.optimizer, save_path=os.path.join(self.cp, f'ckp.pt'),
            #             epoch=ep, loss=self.Loss[-1])
            # else:
            #     num_epochs_no_improve += 1

            # if (self.es > 0) & (num_epochs_no_improve >= self.es):
            #     print(f'\n[{self.__class__.__name__}] Early stopping after {self.es} epochs with no improvement.')
            #     break
            ### Test end

            self.epoch_time.append(timer()-epoch_time_start)  ### TIMER
            self.lr_scheduler.step() # end while
        self.complete = True
        print(f'\n[{self.__class__.__name__}] Training Complete!')

    def test(self, data: torch.Tensor, label: torch.Tensor):
        outputs:torch.Tensor = self.model(data)
        prob_map = self.to_prob_map(outputs)
        
        traj_samples = self.gen_samples(prob_map, num_samples=100, replacement=True)

        traj_error = []
        for bh in range(outputs.shape[0]):     # batch dimension
            for ti in range(outputs.shape[1]): # time dimention
                clusters = self.fit_DBSCAN(traj_samples[bh,ti,:].cpu().numpy(), eps=10, min_sample=5)
                mu_list, _ = self.fit_cluster2gaussian(clusters)
                if not mu_list:
                    mu_list = [(0,0)]
                this_error = np.min(np.sum((np.array(mu_list) - label[bh, ti].cpu().numpy())**2, axis=1)) # best prediction at this time step
                traj_error.append(np.square(this_error))
        return np.mean(traj_error)


    def to_energy_grid(self, output:torch.Tensor) -> torch.Tensor:
        return self.network.to_energy_grid(output)

    def to_prob_map(self, output:torch.Tensor, threshold=0.99, temperature=1.0) -> torch.Tensor:
        """If ebm is True, output is treated as a processed energy grid, otherwise it is treated as logits."""
        if self.loss_type == 'bce':
            ebm = False
        else:
            ebm = True
        if self.loss_type == 'nll':
            smaller_grid = True
        else:
            smaller_grid = False
        return self.network.to_prob_map(output, threshold, temperature, ebm, smaller_grid)


    @staticmethod
    def save_checkpoint(model:nn.Module, optimizer:optim.Optimizer, save_path:str, epoch, loss):
        save_info = {'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': loss}
        torch.save(save_info, save_path)

    @staticmethod
    def load_checkpoint(model:nn.Module, optimizer:optim.Optimizer, load_path:str):
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss  = checkpoint['loss']
        return model, optimizer, epoch, loss


    @staticmethod
    def fit_DBSCAN(data, eps: float, min_sample: int) -> List[np.ndarray]:
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
    def fit_cluster2gaussian(clusters: List[np.ndarray], enlarge=1.0, extra_margin=0.0) -> Tuple[list, list]:
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

    @staticmethod
    def gen_samples(prob_map: torch.Tensor, num_samples: int, replacement=False):
        prob_map_flat = prob_map.view(prob_map.size(0) * prob_map.size(1), -1)
        # samples.shape: [batch*timestep, num_samples] -> [batch, timestep, num_samples]
        # or torch.distributions.categorical.Categorical()
        samples = torch.multinomial(prob_map_flat, num_samples=num_samples, replacement=replacement)
        # unravel sampled idx into coordinates of shape [batch, time, sample, 2]
        samples = samples.view(prob_map.size(0), prob_map.size(1), -1)
        idx = samples.unsqueeze(3)
        preds = idx.repeat(1, 1, 1, 2).float()
        preds[:, :, :, 0] = (preds[:, :, :, 0]) % prob_map.size(3)
        preds[:, :, :, 1] = torch.floor((preds[:, :, :, 1]) / prob_map.size(3))
        return preds

    @staticmethod
    def softargmax_torch(X: torch.Tensor):
        assert(torch.is_tensor(X) & len(X.shape) == 4), ('Softargmax input error.')
        x = X.view(X.shape[0], X.shape[1], -1)

        # create coordinates grid
        xs = torch.linspace(0, X.shape[3]-1, X.shape[3], device=X.device, dtype=X.dtype)
        ys = torch.linspace(0, X.shape[2]-1, X.shape[2], device=X.device, dtype=X.dtype)
        try:
            pos_y, pos_x = torch.meshgrid(ys, xs, indexing='ij')
        except:
            pos_y, pos_x = torch.meshgrid(ys, xs)
        pos_x = pos_x.reshape(-1)
        pos_y = pos_y.reshape(-1)

        # compute the expected coordinates
        expected_x = torch.sum((pos_x * x), dim=-1, keepdim=True)
        expected_y = torch.sum((pos_y * x), dim=-1, keepdim=True)
        output = torch.cat([expected_x, expected_y], dim=-1).view(X.shape[0], X.shape[1], 2)
        return output


    


