from transformers.models.bert.modeling_bert import *
from torch.nn import CrossEntropyLoss

logger = logging.get_logger(__name__)
def soft_logits(input : torch.Tensor ,target : torch.Tensor ,mode:str="average",tmp: float = 1):
    input=torch.div(input,tmp)
    denominator = torch.log(torch.sum(torch.exp(input), dim=1))
    log_pro = -input + denominator.view(-1, 1)
    soft_scores = torch.sum(torch.mul(target,log_pro),dim=1)
    if mode=="average":
        return torch.mean(soft_scores)
    elif mode=="sum":
        return torch.sum(soft_scores)
    elif mode=="no reduction":
        return soft_scores
    else:
        raise ValueError("loss mode error")

class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()


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
        mode:Optional[str] = "train",
        use_soft:Optional[bool]=False,
        tmp: Optional[float] = 1,
    ) :
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            loss_fct = CrossEntropyLoss()
            if mode == "train":
                if use_soft == False:
                    loss = loss_fct(torch.div(logits.view(-1, self.num_labels),tmp), labels.view(-1))
                else:
                    loss =soft_logits(logits.view(-1, self.num_labels),labels.view(-1, self.num_labels),tmp=tmp)
                    # loss = loss_fct(torch.div(logits.view(-1, self.num_labels),tmp), labels.view(-1, self.num_labels))
            elif mode == "val" or mode == "test":
                # loss = loss_fct(torch.div(logits.view(-1, self.num_labels),tmp), labels.view(-1))
                loss = -1
            else:
                raise ValueError("loss value error")
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
