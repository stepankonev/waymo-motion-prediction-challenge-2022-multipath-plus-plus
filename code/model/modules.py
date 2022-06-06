import math
import numpy as np
import torch
from torch import nn
from torch_scatter import scatter_max


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        modules = []
        assert config["n_layers"] > 0
        for i in range(config["n_layers"]):
            modules.append(nn.Linear(config["n_in"], config["n_out"]))
            if i < config["n_layers"] - 1:
                if config["batchnorm"]:
                    modules.append(nn.BatchNorm1d(config["n_out"]))
                if config["dropout"]:
                    modules.append(nn.Dropout(p=0.1))
                modules.append(nn.ReLU())
        self._mlp = nn.Sequential(*modules)
        self.n_in = config["n_in"]
        self.n_out = config["n_out"]
    
    def forward(self, x):
        output = self._mlp(x)
        return output


class NormalMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        modules = []
        layers = config["layers"]
        assert len(layers) > 0
        if config["pre_batchnorm"]:
            modules.append(nn.BatchNorm1d(layers[0]))
        if config["pre_activation"]:
            modules.append(nn.ReLU())
        for i in range(1, len(layers)):
            modules.append(nn.Linear(layers[i - 1], layers[i]))
            if i < len(layers) - 1:
                if config["batchnorm"]:
                    modules.append(nn.BatchNorm1d(layers[i]))
                modules.append(nn.ReLU())
        self._mlp = nn.ModuleList(modules)

    def forward(self, x):
        tmp = []
        prev_x_shape = x.shape
        assert torch.isfinite(x).all()
        tmp.append(x)
        for l in self._mlp:
            x = l(x)
            tmp.append(x)
            assert torch.isfinite(x).all()
        return x
        output = self._mlp(x)
        return output
            

class CGBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self.s_mlp = MLP(config["mlp"])
        self.c_mlp = nn.Identity() if config["identity_c_mlp"] else MLP(config["mlp"])
        self.n_in = self.s_mlp.n_in
        self.n_out = self.s_mlp.n_out

    def forward(self, scatter_numbers, s, c):
        prev_s_shape, prev_c_shape = s.shape, c.shape
        s = self.s_mlp(s.view(-1, s.shape[-1])).view(prev_s_shape)
        c = self.c_mlp(c.view(-1, c.shape[-1])).view(prev_c_shape)
        s = s * c
        if self._config["agg_mode"] == "max":
            aggregated_c = torch.max(s, dim=1, keepdim=True)[0]
        elif self._config["agg_mode"] in ["mean", "avg"]:
            aggregated_c = torch.mean(s, dim=1, keepdim=True)
        else:
            raise Exception("Unknown agg mode for MCG")
        return s, aggregated_c

    
class MCGBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self._blocks = []
        for i in range(config["n_blocks"]):
            current_block_config = config["block"].copy()
            if i == 0 and config["identity_c_mlp"]:
                current_block_config["identity_c_mlp"] = True
            else:
                current_block_config["identity_c_mlp"] = False
            current_block_config["agg_mode"] = config["agg_mode"]
            self._blocks.append(CGBlock(current_block_config))
        self._blocks = nn.ModuleList(self._blocks)
        self.n_in = self._blocks[0].n_in
        self.n_out = self._blocks[-1].n_out
    
    def _repeat_tensor(self, tensor, scatter_numbers, axis=0):
        result = []
        for i in range(len(scatter_numbers)):
            result.append(tensor[[i]].expand((int(scatter_numbers[i]), -1, -1)))
        result = torch.cat(result, axis=0)
        return result

    def _compute_running_mean(self, prevoius_mean, new_value, i):
        if self._config["running_mean_mode"] == "real":
            result = (prevoius_mean * i + new_value) / i
        elif self._config["running_mean_mode"] == "sliding":
            assert self._config["alpha"] + self._config["beta"] == 1
            result = self._config["alpha"] * prevoius_mean + self._config["beta"] * new_value
        return result

    def forward(
            self, scatter_numbers, scatter_idx, s, c=None, aggregate_batch=True, return_s=False):
        if c is None:
            assert self._config["identity_c_mlp"], self._config["identity_c_mlp"]
            c = torch.ones(s.shape[0], 1, self.n_in, requires_grad=True).cuda()
        else:
            assert not self._config["identity_c_mlp"]
        c = self._repeat_tensor(c, scatter_numbers)
        assert torch.isfinite(s).all()
        assert torch.isfinite(c).all()
        running_mean_s, running_mean_c = s, c
        for i, cg_block in enumerate(self._blocks, start=1):
            s, c = cg_block(scatter_numbers, running_mean_s, running_mean_c)
            assert torch.isfinite(s).all()
            assert torch.isfinite(c).all()
            running_mean_s = self._compute_running_mean(running_mean_s, s, i)
            running_mean_c = self._compute_running_mean(running_mean_c, c, i)
            assert torch.isfinite(running_mean_s).all()
            assert torch.isfinite(running_mean_c).all()
        if return_s:
            return running_mean_s 
        if aggregate_batch:
            return scatter_max(running_mean_c, scatter_idx, dim=0)[0]
        return running_mean_c


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self._return_embedding = config["return_embedding"]
        self._learned_anchor_embeddings = torch.empty(
            (1, config["n_trajectories"], config["size"]))
        stdv = 1. / math.sqrt(config["size"])
        # stdv = 1. / config["size"]
        # nn.init.xavier_normal_(self._learned_anchor_embeddings)
        self._learned_anchor_embeddings.uniform_(-stdv, stdv)
        self._learned_anchor_embeddings.requires_grad_(True)
        self._learned_anchor_embeddings = nn.Parameter(self._learned_anchor_embeddings)
        self._mcg_predictor = MCGBlock(config["mcg_predictor"])
        if not self._return_embedding:
            self._mlp_decoder = NormalMLP(config["DECODER"])
    
    def forward(self, target_scatter_numbers, target_scatter_idx, final_embedding, batch_size):
        # assert torch.isfinite(self._learned_anchor_embeddings).all()
        assert torch.isfinite(final_embedding).all()
        trajectories_embeddings = self._mcg_predictor(
            target_scatter_numbers, target_scatter_idx, self._learned_anchor_embeddings,
            final_embedding, return_s=True)
        assert torch.isfinite(trajectories_embeddings).all()
        if self._return_embedding:
            return trajectories_embeddings
        # 
        res = self._mlp_decoder(trajectories_embeddings)
        coordinates = res[:, :, :80 * 2].reshape(
            batch_size, self._config["n_trajectories"], 80, 2)
        assert torch.isfinite(coordinates).all()
        a = res[:, :, 80 * 2: 80 * 3].reshape(
            batch_size, self._config["n_trajectories"], 80, 1)
        assert torch.isfinite(a).all()
        b = res[:, :, 80 * 3: 80 * 4].reshape(
            batch_size, self._config["n_trajectories"], 80, 1)
        assert torch.isfinite(b).all()
        c = res[:, :, 80 * 4: 80 * 5].reshape(
            batch_size, self._config["n_trajectories"], 80, 1)
        assert torch.isfinite(c).all()
        probas = res[:, :, -1]
        assert torch.isfinite(probas).all()
        if self._config["trainable_cov"]:
            # http://www.inference.org.uk/mackay/covariance.pdf
            covariance_matrices = (torch.cat([
                torch.exp(a) * torch.cosh(b), torch.sinh(b),
                torch.sinh(b), torch.exp(-a) * torch.cosh(b)
            ], axis=-1) * torch.exp(c)).reshape(
                coordinates.shape[0], coordinates.shape[1], coordinates.shape[2], 2, 2)
        else:
            _zeros, _ones = torch.zeros_like(a), torch.ones_like(a)
            covariance_matrices = torch.cat([_ones, _zeros, _zeros, _ones], axis=-1).reshape(
                coordinates.shape[0], coordinates.shape[1], coordinates.shape[2], 2, 2)
        return probas, coordinates, covariance_matrices


class DecoderHandler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._return_embedding = config["return_embedding"]
        config["decoder_config"]["return_embedding"] = self._return_embedding
        self._n_decoders = int(config["n_decoders"])
        self._decoders = nn.ModuleList([
            Decoder(config["decoder_config"]) for _ in range(self._n_decoders)])
    
    def forward(self, target_scatter_numbers, target_scatter_idx, final_embedding, batch_size):
        stacked_probas, stacked_coordinates, stacked_covariance_matrices = [], [], []
        stacked_embeddings = []
        random_head_selector = np.random.uniform(low=0.0, high=1.0, size=self._n_decoders)
        random_head_selector = np.ones_like(random_head_selector) * (random_head_selector > 0.5)
        if self._n_decoders == 1:
            random_head_selector = np.array([1.0])
        while random_head_selector.sum() == 0:
            random_head_selector = np.random.uniform(low=0.0, high=1.0, size=self._n_decoders)
            random_head_selector = np.ones_like(random_head_selector) * (random_head_selector > 0.5)
        if self._return_embedding:
            for coeff, decoder in zip(random_head_selector, self._decoders):
                embeddings = decoder(
                    target_scatter_numbers, target_scatter_idx, final_embedding, batch_size)
                stacked_embeddings.append(embeddings)
            stacked_embeddings = torch.cat(stacked_embeddings, dim=1)
            return stacked_embeddings, self._n_decoders / random_head_selector.sum()
        else:
            for coeff, decoder in zip(random_head_selector, self._decoders):
                probas, coordinates, covariance_matrices = decoder(
                    target_scatter_numbers, target_scatter_idx, final_embedding, batch_size)
                probas, coordinates, covariance_matrices = [
                    coeff * x + (1 - coeff) * x.detach() for x in [
                        probas, coordinates, covariance_matrices]]
                stacked_probas.append(probas)
                stacked_coordinates.append(coordinates)
                stacked_covariance_matrices.append(covariance_matrices)
            stacked_probas, stacked_coordinates, stacked_covariance_matrices = [
                torch.cat(x, dim=1) for x in [
                    stacked_probas, stacked_coordinates, stacked_covariance_matrices]]
            return (stacked_probas, stacked_coordinates, stacked_covariance_matrices,
                max(self._n_decoders / random_head_selector.sum(), 1))


class EM(nn.Module):
    def __init__(self):
        super().__init__()
        self._selector = torch.LongTensor([29, 49, 79])
        self._n_final_trajs = 6
    
    @torch.no_grad()
    def _compute_initial_state(self, probas, trajectories):
        result_idx = torch.zeros((trajectories.shape[0], self._n_final_trajs), dtype=torch.long).cuda()
        worked = torch.ones((trajectories.shape[0], trajectories.shape[1])).cuda()
        pairwise_distances = torch.einsum('bmtd,bMtd->btmM', trajectories, trajectories)
        pairwise_distances = pairwise_distances[:, self._selector, :, :]
        d1 = torch.diagonal(pairwise_distances, dim1=-2, dim2=-1)
        R = torch.sqrt(
            torch.ones_like(pairwise_distances) * d1.unsqueeze(2) + \
            torch.ones_like(pairwise_distances) * d1.unsqueeze(3) - \
            2 * pairwise_distances)
        thresholds = R[:, -1, :, :].reshape(R.shape[0], -1).quantile(
            q=0.15, dim=-1, keepdim=True)[..., None, None]
        adj_matrix = (torch.ones_like(R) * (R <= thresholds)).prod(dim=1)
        for i in range(self._n_final_trajs):
            amax = torch.argmax((adj_matrix * probas.unsqueeze(1)).sum(axis=2) * worked, dim=-1)
            worked = worked * (1 - adj_matrix[torch.arange(worked.shape[0]), amax])
            result_idx[:, i] = amax
        return result_idx
    
    @torch.no_grad()
    def _compute_coefficients(self, _covariance_matrices6, trajectories, trajectories6, probas6):
        covariance_matrices6 = _covariance_matrices6# * 500.0
        precision_matrices6 = torch.inverse(covariance_matrices6)
        diff = trajectories6[:, :, self._selector, :].unsqueeze(2) - \
            trajectories[:, :, self._selector, :].unsqueeze(1)
        A = diff.unsqueeze(-2)
        B = diff.unsqueeze(-1)
        C = precision_matrices6[:, :, self._selector, :, :].unsqueeze(2)
        qform = (A @ C @ B)[..., 0, 0]
        logdetCovM = torch.logdet(covariance_matrices6[:, :, self._selector, :, :].unsqueeze(2))
        assert torch.isfinite(logdetCovM).all()
        pMatrix = torch.exp((
            -np.log(2 * np.pi) - 0.5 * logdetCovM - 0.5 * qform).sum(dim=-1)) + 1e-8
        pMatrix = (pMatrix * probas6.unsqueeze(2)) / ((
            pMatrix * probas6.unsqueeze(2)).sum(dim=1, keepdims=True) + 1e-8)
        return pMatrix
    
    def forward(self, non_normalized_probas, trajectories, covariance_matrices):
        probas = nn.functional.softmax(non_normalized_probas, dim=-1)
        result_idx = self._compute_initial_state(probas, trajectories)
        trajectories6 = trajectories.gather(
            1, result_idx[:, :, None, None].expand(
                -1, -1, trajectories.shape[2], trajectories.shape[3]))
        probas6 = nn.functional.softmax(non_normalized_probas.gather(1, result_idx), dim=-1)
        covariance_matrices6 = covariance_matrices.gather(
            1, result_idx[:, :, None, None, None].expand(
                -1, -1, covariance_matrices.shape[2], covariance_matrices.shape[3],
                covariance_matrices.shape[4]))

        for _ in range(10):
            pMatrix = self._compute_coefficients(
                covariance_matrices6, trajectories, trajectories6, probas6)
            P = probas.unsqueeze(1) * pMatrix
            probas6 = (P).sum(dim=-1)
            trajectories6 = ((P)[..., None, None] * \
                trajectories.unsqueeze(1)).sum(axis=2) / probas6[..., None, None]
            diff = trajectories6.unsqueeze(2) - trajectories.unsqueeze(1)
            covariance_matrices6 = ((P)[..., None, None, None] * \
                (covariance_matrices.unsqueeze(1) + (diff.unsqueeze(-1) @ diff.unsqueeze(-2)))
                ).sum(axis=2)
            covariance_matrices6 = covariance_matrices6 / probas6[..., None, None, None]
            with torch.no_grad():
                assert torch.isfinite(torch.logdet(covariance_matrices6)).all()
        return probas6, trajectories6, covariance_matrices6


class HistoryEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self._position_lstm = nn.LSTM(batch_first=True, **config["position_lstm_config"])
        self._position_diff_lstm = nn.LSTM(batch_first=True, **config["position_diff_lstm_config"])
        self._position_mcg = MCGBlock(config["position_mcg_config"])

    def forward(self, scatter_numbers, scatter_idx, lstm_data, lstm_data_diff, mcg_data):
        position_lstm_embedding = self._position_lstm(lstm_data)[0][:, -1:, :]
        position_diff_lstm_embedding = self._position_diff_lstm(lstm_data_diff)[0][:, -1:, :]
        position_mcg_embedding = self._position_mcg(
            scatter_numbers, scatter_idx, mcg_data, aggregate_batch=False)
        return torch.cat([
            position_lstm_embedding, position_diff_lstm_embedding, position_mcg_embedding], axis=-1)


class MHA(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_in = 640 #config["n_in"]
        n_out = 640 #config["n_out"]
        self._config = config
        self._q = nn.Linear(n_in, n_out)
        self._k = nn.Linear(n_in, n_out)
        self._v = nn.Linear(n_in, n_out)
        self._mha = nn.MultiheadAttention(n_out, 4, batch_first=True)
    
    def forward(self, traj_embeddings):
        batch_size = traj_embeddings.shape[0]
        target_scatter_numbers = torch.ones(batch_size, dtype=torch.long).cuda()
        target_scatter_idx = torch.arange(batch_size, dtype=torch.long).cuda()
        Q = self._q(traj_embeddings)
        K = self._k(traj_embeddings)
        V = self._v(traj_embeddings)
        trajectories_embeddings, _ = self._mha(Q, K, V)
        return trajectories_embeddings