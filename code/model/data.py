import numpy as np
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader

def angle_to_range(yaw):
    yaw = (yaw - np.pi) % (2 * np.pi) - np.pi
    return yaw

def normalize(data, config):
    features = tuple(config["train"]["data_config"]["dataset_config"]["lstm_input_data"])
    if features == ("xy", "yaw", "speed", "valid"):
        normalizarion_means = {
            "target/history/lstm_data": np.array([-2.9633209705352783,0.005308846477419138,-0.0032201323192566633,6.059162616729736,0.0,0.0,0.0,0.0,0.0,0.0,0.0], dtype=np.float32),
            "target/history/lstm_data_diff": np.array([0.5990191698074341,-0.001871750457212329,0.0006287908181548119,0.0017820476787164807,0.0,0.0,0.0,0.0,0.0,0.0,0.0], dtype=np.float32),
            "other/history/lstm_data": np.array([5.600991249084473,1.4952889680862427,-0.013044122606515884,1.4446792602539062,0.0,0.0,0.0,0.0,0.0,0.0,0.0], dtype=np.float32),
            "other/history/lstm_data_diff": np.array([0.02598918415606022,-0.0008670506067574024,9.520053572487086e-05,0.0014659279258921742,0.0,0.0,0.0,0.0,0.0,0.0,0.0], dtype=np.float32),
            "target/history/mcg_input_data": np.array([-2.9633209705352783,0.005308846477419138,-0.0032201323192566633,6.059162616729736,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], dtype=np.float32),
            "other/history/mcg_input_data": np.array([5.600991249084473,1.4952889680862427,-0.013044122606515884,1.4446792602539062,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], dtype=np.float32),
            "road_network_embeddings": np.array([77.35614776611328,0.12081649899482727,0.054874029010534286,0.0041899788193404675,-0.0015182862989604473,2.011556386947632,0.9601882696151733,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], dtype=np.float32)}
        normalizarion_stds = {
            "target/history/lstm_data": np.array([3.73840594291687,0.11283700168132782,0.10155521333217621,5.5526909828186035,1.0,1.0,1.0,1.0,1.0,1.0,1.0], dtype=np.float32),
            "target/history/lstm_data_diff": np.array([0.5629123449325562,0.03494573011994362,0.045531339943408966,0.5765337347984314,1.0,1.0,1.0,1.0,1.0,1.0,1.0], dtype=np.float32),
            "other/history/lstm_data": np.array([33.89802932739258,25.64965057373047,1.3623442649841309,3.841723680496216,1.0,1.0,1.0,1.0,1.0,1.0,1.0], dtype=np.float32),
            "other/history/lstm_data_diff": np.array([0.360639750957489,0.18855567276477814,0.08697637170553207,0.43654000759124756,1.0,1.0,1.0,1.0,1.0,1.0,1.0], dtype=np.float32),
            "target/history/mcg_input_data": np.array([3.73840594291687,0.11283700168132782,0.10155521333217621,5.5526909828186035,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0], dtype=np.float32),
            "other/history/mcg_input_data": np.array([33.89802932739258,25.64965057373047,1.3623442649841309,3.841723680496216,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0], dtype=np.float32),
            "road_network_embeddings": np.array([36.71141052246094,0.7614905834197998,0.6328929662704468,0.7438844442367554,0.6675090193748474,0.9678531289100647,1.1907329559326172,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0], dtype=np.float32)}
    elif features == ("xy", "yaw", "speed", "width", "length", "valid"):
        normalizarion_means = {
            "target/history/lstm_data": np.array([-2.9633283615112305,0.005309064872562885,-0.003220283193513751,6.059159278869629,1.9252972602844238,4.271720886230469,0.0,0.0,0.0,0.0,0.0,0.0,0.0], dtype=np.float32),
            "target/history/lstm_data_diff": np.array([0.5990215539932251,-0.0018718164646998048,0.0006288147415034473,0.0017819292843341827,0.0,0.0,0.0,0.0,0.0,0.0,0.0], dtype=np.float32),
            "other/history/lstm_data": np.array([5.601348876953125,1.4943491220474243,-0.013019951991736889,1.44475519657135,1.072572946548462,2.4158480167388916,0.0,0.0,0.0,0.0,0.0,0.0,0.0], dtype=np.float32),
            "other/history/lstm_data_diff": np.array([0.025991378352046013,-0.0008657555445097387,9.549396054353565e-05,0.001465122913941741,0.0,0.0,0.0,0.0,0.0,0.0,0.0], dtype=np.float32),
            "target/history/mcg_input_data": np.array([-2.9633283615112305,0.005309064872562885,-0.003220283193513751,6.059159278869629,1.9252972602844238,4.271720886230469,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], dtype=np.float32),
            "other/history/mcg_input_data": np.array([5.601348876953125,1.4943491220474243,-0.013019951991736889,1.44475519657135,1.072572946548462,2.4158480167388916,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], dtype=np.float32),
            "road_network_embeddings": np.array([77.35582733154297,0.12082172930240631,0.05486442521214485,0.004187341313809156,-0.0015162595082074404,2.011558771133423,0.9601883888244629,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], dtype=np.float32)
        }
        normalizarion_stds = {
            "target/history/lstm_data": np.array([3.738459825515747,0.11283490061759949,0.10153655707836151,5.553133487701416,0.5482628345489502,1.6044323444366455,1.0,1.0,1.0,1.0,1.0,1.0,1.0], dtype=np.float32),
            "target/history/lstm_data_diff": np.array([0.5629324316978455,0.03495170176029205,0.04547161981463432,0.5762772560119629,1.0,1.0,1.0,1.0,1.0,1.0,1.0], dtype=np.float32),
            "other/history/lstm_data": np.array([33.899658203125,25.64937973022461,1.3623465299606323,3.8417460918426514,1.0777146816253662,2.4492409229278564,1.0,1.0,1.0,1.0,1.0,1.0,1.0], dtype=np.float32),
            "other/history/lstm_data_diff": np.array([0.36061710119247437,0.1885228455066681,0.08698483556509018,0.43648791313171387,1.0,1.0,1.0,1.0,1.0,1.0,1.0], dtype=np.float32),
            "target/history/mcg_input_data": np.array([3.738459825515747,0.11283490061759949,0.10153655707836151,5.553133487701416,0.5482628345489502,1.6044323444366455,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0], dtype=np.float32),
            "other/history/mcg_input_data": np.array([33.899658203125,25.64937973022461,1.3623465299606323,3.8417460918426514,1.0777146816253662,2.4492409229278564,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0], dtype=np.float32),
            "road_network_embeddings": np.array([36.71162414550781,0.761500358581543,0.6328969597816467,0.7438802719116211,0.6675100326538086,0.9678668975830078,1.1907216310501099,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0], dtype=np.float32)
        }
    elif features == ("xy", "yaw_sin", "yaw_cos", "speed", "valid"):
        normalizarion_means = {
            "target/history/lstm_data": np.array([-2.963352680206299,0.005309187341481447,-0.0031980471685528755,0.9703145623207092,6.059169292449951,0.0,0.0,0.0,0.0,0.0,0.0,0.0], dtype=np.float32),
            "target/history/lstm_data_diff": np.array([0.5990246534347534,-0.0018718652427196503,0.0006487328791990876,0.9678786993026733,0.0017822050722315907,0.0,0.0,0.0,0.0,0.0,0.0,0.0], dtype=np.float32),
            "other/history/lstm_data": np.array([5.601662635803223,1.494566798210144,0.0031144910026341677,0.0433664545416832,1.4448018074035645,0.0,0.0,0.0,0.0,0.0,0.0,0.0], dtype=np.float32),
            "other/history/lstm_data_diff": np.array([0.025993138551712036,-0.0008638832368887961,9.389788465341553e-05,0.5554189085960388,0.0014658357249572873,0.0,0.0,0.0,0.0,0.0,0.0,0.0], dtype=np.float32),
            "target/history/mcg_input_data": np.array([-2.963352680206299,0.005309187341481447,-0.0031980471685528755,0.9703145623207092,6.059169292449951,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], dtype=np.float32),
            "other/history/mcg_input_data": np.array([5.601662635803223,1.494566798210144,0.0031144910026341677,0.0433664545416832,1.4448018074035645,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], dtype=np.float32),
            "road_network_embeddings": np.array([77.35610961914062,0.12084171921014786,0.054869141429662704,0.004183736629784107,-0.0015176727902144194,2.0115585327148438,0.960183322429657,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], dtype=np.float32)
        }
        normalizarion_stds = {
            "target/history/lstm_data": np.array([3.738659143447876,0.11287359893321991,0.07889654487371445,0.15939009189605713,5.5531511306762695,1.0,1.0,1.0,1.0,1.0,1.0,1.0], dtype=np.float32),
            "target/history/lstm_data_diff": np.array([0.5629534721374512,0.03495863825082779,0.024357756599783897,0.1730499267578125,0.5766724944114685,1.0,1.0,1.0,1.0,1.0,1.0,1.0], dtype=np.float32),
            "other/history/lstm_data": np.array([33.8993034362793,25.648475646972656,0.49251335859298706,0.5677053928375244,3.8419690132141113,1.0,1.0,1.0,1.0,1.0,1.0,1.0], dtype=np.float32),
            "other/history/lstm_data_diff": np.array([0.36064836382865906,0.1885451078414917,0.03633992373943329,0.4971870481967926,0.43645790219306946,1.0,1.0,1.0,1.0,1.0,1.0,1.0], dtype=np.float32),
            "target/history/mcg_input_data": np.array([3.738659143447876,0.11287359893321991,0.07889654487371445,0.15939009189605713,5.5531511306762695,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0], dtype=np.float32),
            "other/history/mcg_input_data": np.array([33.8993034362793,25.648475646972656,0.49251335859298706,0.5677053928375244,3.8419690132141113,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0], dtype=np.float32),
            "road_network_embeddings": np.array([36.711769104003906,0.7614925503730774,0.6328906416893005,0.7438828349113464,0.667508602142334,0.9677839279174805,1.19071626663208,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0], dtype=np.float32)
        }
    else:
        raise Exception("Wrong features set")
    keys = [
        'target/history/lstm_data', 'target/history/lstm_data_diff',
        'other/history/lstm_data', 'other/history/lstm_data_diff',
        'target/history/mcg_input_data', 'other/history/mcg_input_data',
        'road_network_embeddings']
    for k in keys:
        data[k] = (data[k] - normalizarion_means[k]) / (normalizarion_stds[k] + 1e-6)
        data[k].clamp_(-15, 15)
    data[f"target/history/lstm_data_diff"] *= data[f"target/history/valid_diff"]
    data[f"other/history/lstm_data_diff"] *= data[f"other/history/valid_diff"]
    data[f"target/history/lstm_data"] *= data[f"target/history/valid"]
    data[f"other/history/lstm_data"] *= data[f"other/history/valid"]
    return data

def dict_to_cuda(d):
    passing_keys = set([
                    'target/history/lstm_data', 'target/history/lstm_data_diff',
                    'other/history/lstm_data', 'other/history/lstm_data_diff',
                    'target/history/mcg_input_data', 'other/history/mcg_input_data',
                    'other_agent_history_scatter_idx', 'road_network_scatter_idx',
                    'other_agent_history_scatter_numbers', 'road_network_scatter_numbers',
                    'batch_size',
                    'road_network_embeddings',
                    'target/future/xy', 'target/future/valid'])
    for k in d.keys():
        if k not in passing_keys:
            continue
        v = d[k]
        if not isinstance(v, torch.Tensor):
            continue
        d[k] = d[k].cuda()

class MultiPathPPDataset(Dataset):
    def __init__(self, config):
        self._data_path = config["data_path"]
        self._config = config
        files = os.listdir(self._data_path)
        self._files = [os.path.join(self._data_path, f) for f in files]
        random.shuffle(self._files)
        if "max_length" in config:
            self._files = self._files[:config["max_length"]]
        assert len(self._files) > 0
    
    def __len__(self):
        return len(self._files)
    
    def _generate_sin_cos(self, data):
        data["target/history/yaw_sin"] = np.sin(data["target/history/yaw"])
        data["target/history/yaw_cos"] = np.cos(data["target/history/yaw"])
        data["other/history/yaw_sin"] = np.sin(data["other/history/yaw"])
        data["other/history/yaw_cos"] = np.cos(data["other/history/yaw"])
        return data
    
    def _add_length_width(self, data):
        data["target/history/length"] = \
            data["target/length"].reshape(-1, 1, 1) * np.ones_like(data["target/history/yaw"])
        data["target/history/width"] = \
            data["target/width"].reshape(-1, 1, 1) * np.ones_like(data["target/history/yaw"])

        data["other/history/length"] = \
            data["other/length"].reshape(-1, 1, 1) * np.ones_like(data["other/history/yaw"])
        data["other/history/width"] = \
            data["other/width"].reshape(-1, 1, 1) * np.ones_like(data["other/history/yaw"])
        return data
    
    def _compute_agent_diff_features(self, data):
        diff_keys = ["target/history/xy", "target/history/yaw", "target/history/speed",
            "other/history/xy", "other/history/yaw", "other/history/speed"]
        for key in diff_keys:
            if key.endswith("yaw"):
                data[f"{key}_diff"] = angle_to_range(np.diff(data[key], axis=1))
            else:
                data[f"{key}_diff"] = np.diff(data[key], axis=1)
        data["target/history/yaw_sin_diff"] = np.sin(data["target/history/yaw_diff"])
        data["target/history/yaw_cos_diff"] = np.cos(data["target/history/yaw_diff"])
        data["other/history/yaw_sin_diff"] = np.sin(data["other/history/yaw_diff"])
        data["other/history/yaw_cos_diff"] = np.cos(data["other/history/yaw_diff"])
        data["target/history/valid_diff"] = (data["target/history/valid"] * \
            np.concatenate([
                data["target/history/valid"][:, 1:, :],
                np.zeros((data["target/history/valid"].shape[0], 1, 1))
            ], axis=1))[:, :-1, :]
        data["other/history/valid_diff"] = (data["other/history/valid"] * \
            np.concatenate([data["other/history/valid"][:, 1:, :],
            np.zeros((data["other/history/valid"].shape[0], 1, 1))], axis=1))[:, :-1, :]
        return data
    
    def _compute_agent_type_and_is_sdc_ohe(self, data, subject):
        I = np.eye(5)
        agent_type_ohe = I[np.array(data[f"{subject}/agent_type"])]
        is_sdc = np.array(data[f"{subject}/is_sdc"]).reshape(-1, 1)
        ohe_data = np.concatenate([agent_type_ohe, is_sdc], axis=-1)[:, None, :]
        ohe_data = np.repeat(ohe_data, data["target/history/xy"].shape[1], axis=1)
        return ohe_data
    
    def _mask_history(self, ndarray, fraction):
        assert fraction >=0 and fraction < 1
        ndarray = ndarray * (np.random.uniform(size=ndarray.shape) > fraction)
        return ndarray
    
    def _compute_lstm_input_data(self, data):
        keys_to_stack = self._config["lstm_input_data"]
        keys_to_stack_diff = self._config["lstm_input_data_diff"]
        for subject in ["target", "other"]:
            agent_type_ohe = self._compute_agent_type_and_is_sdc_ohe(data, subject)
            data[f"{subject}/history/lstm_data"] = np.concatenate(
                [data[f"{subject}/history/{k}"] for k in keys_to_stack] + [agent_type_ohe], axis=-1)
            data[f"{subject}/history/lstm_data"] *= data[f"{subject}/history/valid"]
            data[f"{subject}/history/lstm_data_diff"] = np.concatenate(
                [data[f"{subject}/history/{k}_diff"] for k in keys_to_stack_diff] + \
                    [agent_type_ohe[:, 1:, :]], axis=-1)
            data[f"{subject}/history/lstm_data_diff"] *= data[f"{subject}/history/valid_diff"]
        return data

    def _compute_mcg_input_data(self, data):
        for subject in ["target", "other"]:
            agent_type_ohe = self._compute_agent_type_and_is_sdc_ohe(data, subject)
            lstm_input_data = data[f"{subject}/history/lstm_data"]
            I = np.eye(lstm_input_data.shape[1])[None, ...]
            timestamp_ohe = np.repeat(I, lstm_input_data.shape[0], axis=0)
            data[f"{subject}/history/mcg_input_data"] = np.concatenate(
                [lstm_input_data, timestamp_ohe], axis=-1)
        return data
    
    def __getitem__(self, idx):
        try:
            np_data = dict(np.load(self._files[idx], allow_pickle=True))
        except:
            print("Error reading", self._files[idx])
            idx = 0
            np_data = dict(np.load(self._files[0], allow_pickle=True))
        np_data["scenario_id"] = np_data["scenario_id"].item()
        np_data["filename"] = self._files[idx]
        np_data["target/history/yaw"] = angle_to_range(np_data["target/history/yaw"])
        np_data["other/history/yaw"] = angle_to_range(np_data["other/history/yaw"])
        np_data = self._generate_sin_cos(np_data)
        np_data = self._add_length_width(np_data)
        if self._config["mask_history"]:
            for subject in ["target", "other"]:
                np_data[f"{subject}/history/valid"] = self._mask_history(
                        np_data[f"{subject}/history/valid"], self._config["mask_history_fraction"])
        np_data = self._compute_agent_diff_features(np_data)
        np_data = self._compute_lstm_input_data(np_data)
        np_data = self._compute_mcg_input_data(np_data)
        return np_data

    @staticmethod
    def collate_fn(batch):
        batch_keys = batch[0].keys()
        result_dict = {k: [] for k in batch_keys}
        other_agent_history_scatter_idx = []
        road_network_scatter_idx = []
        other_agent_history_scatter_numbers = []
        road_network_scatter_numbers = []
        for sample_num, sample in enumerate(batch):
            for k in batch_keys:
                if not isinstance(sample[k], str) and len(sample[k].shape) == 0:
                    result_dict[k].append(sample[k].item())
                else:
                    result_dict[k].append(sample[k])
                if k == "road_network_embeddings":
                    road_network_scatter_idx.extend([sample_num] * sample[k].shape[0])
                    road_network_scatter_numbers.append(sample[k].shape[0])
                if k == "other/history/xy":
                    other_agent_history_scatter_idx.extend([sample_num] * sample[k].shape[0])
                    other_agent_history_scatter_numbers.append(sample[k].shape[0])
        for k, v in result_dict.items():
            if not isinstance(v[0], np.ndarray):
                continue
            result_dict[k] = torch.Tensor(np.concatenate(v, axis=0))
        result_dict["other_agent_history_scatter_idx"] = torch.Tensor(
            other_agent_history_scatter_idx).type(torch.long)
        result_dict["road_network_scatter_idx"] = torch.Tensor(
            road_network_scatter_idx).type(torch.long)
        result_dict["other_agent_history_scatter_numbers"] = torch.Tensor(
            other_agent_history_scatter_numbers).type(torch.long)
        result_dict["road_network_scatter_numbers"] = torch.Tensor(
            road_network_scatter_numbers).type(torch.long)
        result_dict["batch_size"] = len(batch)
        return result_dict


def get_dataloader(config):
    dataset = MultiPathPPDataset(config["dataset_config"])
    dataloader = DataLoader(
        dataset, collate_fn=MultiPathPPDataset.collate_fn, **config["dataloader_config"])
    return dataloader