import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn, Tensor
from transformers import AutoModel
from transformers.file_utils import ModelOutput
from huggingface_hub import PyTorchModelHubMixin

logger = logging.getLogger(__name__)


@dataclass
class EncoderOutput(ModelOutput):
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None


class BiEncoderModel(nn.Module, PyTorchModelHubMixin):
    TRANSFORMER_CLS = AutoModel

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        normlized: bool = True,
        sentence_pooling_method: str = "cls",
        temperature: float = 1.0,
        use_independent: bool = False,
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        hidden_size = self.model.config.hidden_size

        # self.compute_sim_module = nn.Linear(self.model.config.hidden_size * 2, 1)
        classifier_dropout = (
            self.model.config.classifier_dropout
            if self.model.config.classifier_dropout is not None
            else self.model.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.model.config.hidden_size, 1)

        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")
        self.mse = nn.MSELoss()

        self.normlized = normlized
        self.sentence_pooling_method = sentence_pooling_method
        self.temp = nn.Parameter(torch.tensor(temperature, requires_grad=True))
        self.alpha = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.use_independent = use_independent
        self.config = self.model.config

    def encode(self, features):
        if features is None:
            return None
        psg_out = self.model(**features, return_dict=True)
        p_reps = psg_out.last_hidden_state[:, 0]
        if self.normlized:
            p_reps = torch.nn.functional.normalize(p_reps, dim=-1)
        # p_reps = p_reps.reshape(-1, p_reps.size(1) * 4)
        p_reps = p_reps.reshape(-1, 4, p_reps.size(1)).mean(dim=1)
        return p_reps.contiguous()

    def encode_avg(self, features):
        if features is None:
            return None
        # B (rB * 4), S (512)
        bs = features["input_ids"].size(0)
        attention_mask = features["attention_mask"]
        attention_mask = attention_mask.reshape(int(bs / 4), -1)
        psg_out = self.model(**features, return_dict=True)
        last_hidden = psg_out.last_hidden_state
        last_hidden = last_hidden.reshape(int(bs / 4), -1, last_hidden.size(-1))
        last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def encode_sim(self, features):
        if features is None:
            return None
        psg_out = self.sim_model(**features, return_dict=True)
        p_reps = psg_out.last_hidden_state[:, 0]
        if self.normlized:
            p_reps = torch.nn.functional.normalize(p_reps, dim=-1)
        return p_reps.contiguous()

    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    def forward(
        self,
        query: Dict[str, Tensor] = None,
        passage: Dict[str, Tensor] = None,
        label: Tensor = None,
    ):
        # query size (bs, seq_len)
        # passage size (bs, seq_len)
        q_reps = self.encode(query)
        p_reps = self.encode(passage)

        pooled_output_q = self.dropout(q_reps)
        independent_scores_1 = self.classifier(pooled_output_q).squeeze()
        pooled_output_p = self.dropout(p_reps)
        independent_scores_2 = self.classifier(pooled_output_p).squeeze()

        dependent_scores = self.compute_similarity(q_reps, p_reps) / self.temp
        dependent_scores = dependent_scores.diagonal()

        # new dependent_scores
        # dependent_scores = self.compute_sim_module(torch.cat([q_reps, p_reps], dim=-1)).reshape(-1)

        if self.use_independent:
            scores = independent_scores_2
        else:
            scores = (
                independent_scores_2
                - self.alpha * (dependent_scores - 1) * independent_scores_1
            )
        loss = self.compute_loss(scores, label)
        return EncoderOutput(
            loss=loss,
            scores=scores,
            # q_reps=q_reps,
            # p_reps=p_reps,
        )

    def compute_loss(self, scores, target):
        return self.mse(scores, target)


class RolloutModel(nn.Module, PyTorchModelHubMixin):
    TRANSFORMER_CLS = AutoModel

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        normlized: bool = True,
        sentence_pooling_method: str = "cls",
        temperature: float = 1.0,
        use_independent: bool = False,
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        hidden_size = self.model.config.hidden_size

        # self.compute_sim_module = nn.Linear(self.model.config.hidden_size * 2, 1)
        classifier_dropout = (
            self.model.config.classifier_dropout
            if self.model.config.classifier_dropout is not None
            else self.model.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.model.config.hidden_size, 1)

        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")
        self.mse = nn.MSELoss()

        self.normlized = normlized
        self.sentence_pooling_method = sentence_pooling_method
        self.temp = nn.Parameter(torch.tensor(temperature, requires_grad=True))
        self.alpha = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.use_independent = use_independent
        self.config = self.model.config

    def encode(self, features):
        if features is None:
            return None
        psg_out = self.model(**features, return_dict=True)
        p_reps = psg_out.last_hidden_state[:, 0]
        if self.normlized:
            p_reps = torch.nn.functional.normalize(p_reps, dim=-1)
        # p_reps = p_reps.reshape(-1, p_reps.size(1) * 4)
        p_reps = p_reps.reshape(-1, 4, p_reps.size(1)).mean(dim=1)
        return p_reps.contiguous()

    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    def forward(
        self,
        passage: Dict[str, Tensor] = None,
        label: Tensor = None,
    ):
        # (bs * num_rollouts, hidden_size)
        p_reps = self.encode(passage)

        pooled_output_p = self.dropout(p_reps)
        scores = self.classifier(pooled_output_p).squeeze()
        # (bs * num_rollouts, bs * num_rollouts)
        # dependent_scores = self.compute_similarity(p_reps, p_reps) / self.temp

        # N = scores.shape[0]
        # mask = torch.tril(torch.ones(N, N, device=scores.device), diagonal=-1)
        # penalties = (dependent_scores - 1) * mask

        # penalty_sum = penalties.sum(dim=1)
        # divisors = torch.arange(1, N, device=scores.device)
        # scores[1:] = -scores[1:] * self.alpha * penalty_sum[1:] / divisors

        # penalty_max = penalties.max(dim=1).values
        # scores[1:] = -scores[1:] * self.alpha * penalty_max[1:]

        loss = self.compute_loss(scores, label)
        return EncoderOutput(
            loss=loss,
            scores=scores,
        )

    def compute_loss(self, scores, target):
        return self.mse(scores, target)
