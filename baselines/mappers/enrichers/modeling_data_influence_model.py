from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import BertPreTrainedModel, BertModel
from typing import List, Optional, Tuple, Union
from torch import nn
import torch


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        # self.mlp = nn.Linear(config.hidden_size, config.hidden_size * 10)
        # self.gelu = nn.GELU()
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def average_pool(
        self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        bs = input_ids.size(0)
        outputs = self.bert(
            input_ids.reshape(bs * 4, -1),
            attention_mask=attention_mask.reshape(bs * 4, -1),
            token_type_ids=token_type_ids.reshape(bs * 4, -1),
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = self.average_pool(
            outputs.last_hidden_state.reshape(
                bs, -1, outputs.last_hidden_state.shape[-1]
            ),
            attention_mask,
        )

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="/data/users/zichunyu")
    parser.add_argument("--model_name", type=str, default="pythia-1b")
    parser.add_argument("--ckpt", type=int, default=10000)
    parser.add_argument("--base", type=int, default=0)
    parser.add_argument("-S", "--shard", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--map_batch_size", type=int, default=1024)
    parser.add_argument("-b", "--device_batch_size", type=int, default=128)

    args = parser.parse_args()
    print(args)

    model_dir = f"{args.base_dir}/out/{args.model_name}/fineweb/sample-100BT/{args.ckpt}-data_influence_model-flan-bs-1"
    output_dir = f"{args.base_dir}/out/{args.model_name}/fineweb/sample-350BT/train/1/{args.ckpt}-data_influence_model-flan-bs-1-prediction"

    num_proc = 8
    # 50M examples in total, 100B Tokens
    dataset = StreamingDataset(
        input_dir=f"{args.base_dir}/data/fineweb/sample-350BT/train/1",
        item_loader=TokensLoader(block_size=2048 + 1),
    )
    # 3M examples/GPU for 10k steps
    shard_size = int(1e6)
    # shard_size = len(dataset) // args.shard[1]
    dataset = dataset[
        args.base
        + args.shard[0]
        * shard_size : (
            args.base + (args.shard[0] + 1) * shard_size
            if args.shard[0] + 1 < args.shard[1]
            else len(dataset)
        )
    ]

    os.makedirs(output_dir + f"/{args.shard[0]}", exist_ok=True)
    torch.save(
        dataset[:5] + dataset[-5:],
        output_dir + f"/{args.shard[0]}/sanity_check.pt",
    )
    dataset = Dataset.from_list([{"ori_input_ids": d[:2048]} for d in dataset])

    print("Total number of examples:", len(dataset))

    # Load pythia tokenizer
    pythia_tokenizer = AutoTokenizer.from_pretrained("checkpoints/EleutherAI/pythia-1b")
    tokenizer = AutoTokenizer.from_pretrained(
        "checkpoints/bert-base-uncased",
        max_length=2048,
        padding="max_length",
    )

    def preprocess_data(examples):
        texts = pythia_tokenizer.batch_decode(
            examples["ori_input_ids"],
            skip_special_tokens=True,
        )
        encoding = tokenizer.batch_encode_plus(
            texts,
            max_length=2048,
            padding="max_length",
            truncation=True,
        )
        return encoding

    dataset = dataset.map(
        preprocess_data,
        batched=True,
        batch_size=args.map_batch_size,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
    )
    print("After tokenization: Total number of examples:", len(dataset))

    dataset = dataset.map(
        ModelAnnotator(model_dir, args.device_batch_size),
        batched=True,
        with_indices=True,
        batch_size=args.device_batch_size,
        remove_columns=dataset.column_names,
    )
    print("After annotation: Total number of examples:", len(dataset))

    print(f"Saving to {output_dir}")
    dataset.save_to_disk(output_dir + f"/{args.shard[0]}")
