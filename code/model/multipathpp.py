import torch
from torch import nn
from .modules import MCGBlock, HistoryEncoder, MLP, NormalMLP, Decoder, DecoderHandler, EM, MHA

class MultiPathPP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self._agent_history_encoder = HistoryEncoder(config["agent_history_encoder"])
        self._agent_mcg_linear = NormalMLP(config["agent_mcg_linear"])
        self._interaction_mcg_linear = NormalMLP(config["interaction_mcg_linear"])
        self._interaction_history_encoder = HistoryEncoder(config["interaction_history_encoder"])
        self._polyline_encoder = NormalMLP(config["polyline_encoder"])
        self._history_mcg_encoder = MCGBlock(config["history_mcg_encoder"])
        self._interaction_mcg_encoder = MCGBlock(config["interaction_mcg_encoder"])
        self._agent_and_interaction_linear = NormalMLP(config["agent_and_interaction_linear"])
        self._roadgraph_mcg_encoder = MCGBlock(config["roadgraph_mcg_encoder"])
        self._decoder_handler = DecoderHandler(config["decoder_handler_config"])
        if config["multiple_predictions"]:
            self._decoder = Decoder(config["final_decoder"])
        if config["make_em"]:
            self._em = EM()
        if self._config["mha_decoder"]:
            self._mha_decoder = MHA(config["mha_decoder"])
    
    def forward(self, data, num_steps):
        target_scatter_numbers = torch.ones(data["batch_size"], dtype=torch.long).cuda()
        target_scatter_idx = torch.arange(data["batch_size"], dtype=torch.long).cuda()
        target_mcg_input_data_linear = self._agent_mcg_linear(data["target/history/mcg_input_data"])
        assert torch.isfinite(target_mcg_input_data_linear).all()
        target_agents_embeddings = self._agent_history_encoder(
            target_scatter_numbers, target_scatter_idx, data["target/history/lstm_data"],
            data["target/history/lstm_data_diff"], target_mcg_input_data_linear)
        assert torch.isfinite(target_agents_embeddings).all()
        other_mcg_input_data_linear = self._interaction_mcg_linear(
            data["other/history/mcg_input_data"])
        assert torch.isfinite(other_mcg_input_data_linear).all()

        interaction_agents_embeddings = self._interaction_history_encoder(
            data["other_agent_history_scatter_numbers"], data["other_agent_history_scatter_idx"],
            data["other/history/lstm_data"], data["other/history/lstm_data_diff"],
            other_mcg_input_data_linear)
        assert torch.isfinite(interaction_agents_embeddings).all()
        target_mcg_embedding = self._history_mcg_encoder(
            target_scatter_numbers, target_scatter_idx, target_agents_embeddings)
        assert torch.isfinite(target_mcg_embedding).all()
        interaction_mcg_embedding = self._interaction_mcg_encoder(
            data["other_agent_history_scatter_numbers"], data["other_agent_history_scatter_idx"],
            interaction_agents_embeddings, target_agents_embeddings)
        assert torch.isfinite(interaction_mcg_embedding).all()
        segment_embeddings = self._polyline_encoder(data["road_network_embeddings"])
        assert torch.isfinite(segment_embeddings).all()
        target_and_interaction_embedding = torch.cat(
            [target_mcg_embedding, interaction_mcg_embedding], axis=-1)
        assert torch.isfinite(target_and_interaction_embedding).all()
        target_and_interaction_embedding_linear = self._agent_and_interaction_linear(
            target_and_interaction_embedding)
        assert torch.isfinite(target_and_interaction_embedding_linear).all()
        roadgraph_mcg_embedding = self._roadgraph_mcg_encoder(
            data["road_network_scatter_numbers"], data["road_network_scatter_idx"],
            segment_embeddings, target_and_interaction_embedding_linear)
        assert torch.isfinite(roadgraph_mcg_embedding).all()
        final_embedding = torch.cat(
            [target_mcg_embedding, interaction_mcg_embedding, roadgraph_mcg_embedding], dim=-1)
        assert torch.isfinite(final_embedding).all()
        if self._config["multiple_predictions"]:
            if self._config["make_em"]:
                probas, coordinates, covariance_matrices, loss_coeff = self._decoder_handler(
                    target_scatter_numbers, target_scatter_idx, final_embedding, data["batch_size"])
                if num_steps > 1000:
                    probas, coordinates, covariance_matrices = self._em(
                        probas, coordinates, covariance_matrices)
                return probas, coordinates, covariance_matrices, loss_coeff
            trajectories_embeddings, loss_coeff = self._decoder_handler(
                target_scatter_numbers, target_scatter_idx, final_embedding, data["batch_size"])
            if self._config["mha_decoder"]:
                trajectories_embeddings = self._mha_decoder(trajectories_embeddings)
            trajectories_embeddings, _ = trajectories_embeddings.max(dim=1)
            probas, coordinates, covariance_matrices = self._decoder(
                target_scatter_numbers, target_scatter_idx, trajectories_embeddings,
                data["batch_size"])
        else:
            probas, coordinates, covariance_matrices, loss_coeff = self._decoder_handler(
                target_scatter_numbers, target_scatter_idx, final_embedding, data["batch_size"])
            assert probas.shape[1] == coordinates.shape[1] == covariance_matrices.shape[1] == 6
        assert torch.isfinite(probas).all()
        assert torch.isfinite(coordinates).all()
        assert torch.isfinite(covariance_matrices).all()

        return probas, coordinates, covariance_matrices, loss_coeff
