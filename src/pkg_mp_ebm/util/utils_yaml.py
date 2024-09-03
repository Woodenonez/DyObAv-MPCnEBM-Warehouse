import yaml # type: ignore
from typing import List

'''
This is used to load and dump parameters in the form of YAML
'''

class QuickYaml:

    @staticmethod
    def to_yaml(data: dict, save_path: str, style=None):
        with open(save_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, default_style=style)
        print(f'Save to {save_path}.')
        return 1

    @staticmethod
    def to_yaml_all(data_list: List[dict], save_path: str, style=None):
        with open(save_path, 'w') as f:
            yaml.dump_all(data_list, f, explicit_start=True, default_flow_style=False, default_style=style)
        print(f'Save to {save_path}.')
        return 1

    @staticmethod
    def from_yaml(load_path: str, vb=True):
        with open(load_path, 'r') as stream:
            try:
                parsed_yaml = yaml.safe_load(stream)
                if vb:
                    print(f'Load from {load_path}.')
            except yaml.YAMLError as exc:
                print(exc)
        return parsed_yaml

    @staticmethod
    def from_yaml_all(load_path: str, vb=True) -> List[dict]:
        with open(load_path, 'r') as stream:
            parsed_yaml_list = []
            try:
                for data in yaml.load_all(stream, Loader=yaml.FullLoader):
                    parsed_yaml_list.append(data)
                if vb:
                    print(f'Load from {load_path}.')
            except yaml.YAMLError as exc:
                print(exc)
        return parsed_yaml_list

if __name__ == '__main__':
    import os
    import pathlib

    # out_layer_list = ['none', 'none', 'none', 'poselu', 'softplus']
    # loss_type_list = ['bce',  'kld',  'nll',  'enll',   'enll']
    out_layer_list = ['none', 'none', 'poselu']
    loss_type_list = ['bce',  'kld',  'enll']

    ### Parameters
    pred_start = 1
    pred_len = 20
    obsv_len = 5
    device = 'multi'
    # out_layer = 'poselu'

    epoch = 20
    batch_size = 20
    early_stopping = 0
    learning_rate = 1e-3
    weight_regularization = 1e-4
    # loss_type = 'enll'

    mode = 'train' # 'train' or 'test'
    dataset_name = 'warehouse_sim_dataset'
    dataset_name_abbr = 'wsd'

    for out_layer, loss_type in zip(out_layer_list, loss_type_list):

        checkpoint_dir = f'Model/{dataset_name_abbr}_{out_layer}_{loss_type}'
        model_path = f'Model/{dataset_name_abbr}_{out_layer}_{loss_type}'

        file_name = f'{dataset_name_abbr}_{pred_start}t{pred_len}_{out_layer}_{loss_type}_{mode}.yaml'
        sl_path = os.path.join(pathlib.Path(__file__).resolve().parents[3], 'config/', file_name)

        ### Dictionary
        general_param = {
            'device' : 'multi',
            'pred_len': pred_len,
            'obsv_len': obsv_len,
            'input_channel': obsv_len + 1,
            'out_layer': out_layer,
            }
        training_param = {
            'epoch': epoch,
            'batch_size': batch_size,
            'early_stopping': early_stopping,
            'learning_rate': learning_rate,
            'weight_regularization': weight_regularization,
            'checkpoint_dir': checkpoint_dir,
            }
        path_param = {
            'model_path': model_path,
            'dataset_name': dataset_name,
            'dataset_path': os.path.join('data/', dataset_name, mode),
            'dataset_label_path': os.path.join('data/', dataset_name, mode, 'all_data.csv'),
            }

        ### Yaml
        qy = QuickYaml()

        # param = {**general_param, **training_param, **converting_param, **path_param}
        # qy.to_yaml(param, sl_path, style=None)
        # test_dict = qy.from_yaml(sl_path) # ensure the yaml file is saved

        param_list: List[dict] = [general_param, training_param, path_param]
        qy.to_yaml_all(param_list, sl_path, style=None)
        test_dict_list = qy.from_yaml_all(sl_path) # ensure the yaml file is saved