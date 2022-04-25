from transformers.models.bert.modeling_bert import *
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
logger = logging.get_logger(__name__)
@dataclass
class ClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    binary_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

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

class classification_moudle(nn.Module):
    def __init__(self,hidden_size):
        super(classification_moudle,self).__init__()
        self.hidden_size=hidden_size
        self.linear1=nn.Linear(self.hidden_size,2*self.hidden_size)
        self.linear2=nn.Linear(2*self.hidden_size,self.hidden_size)
    def forward(self,x):
        y=x
        y=self.linear1(y)
        y=F.leaky_relu(y)
        y=F.normalize(y)
        y=F.dropout(y,p=0.7)
        y=self.linear2(y)
        y=F.leaky_relu(y)
        y=F.normalize(y)
        y = F.dropout(y, p=0.7)
        x=x+y#残差链接
        return x

class classification_model(nn.Module):
    def __init__(self,hidden_size,output_size):
        super(classification_model,self).__init__()
        self.hidden_size,self.output_size=hidden_size,output_size
        self.moudle1=classification_moudle(self.hidden_size)
        self.moudle2=classification_moudle(self.hidden_size)
        self.linear1=nn.Linear(self.hidden_size,int(self.hidden_size/2))
        self.linear2=nn.Linear(int(self.hidden_size/2),self.output_size)
    def forward(self,x):
        y=self.moudle1(x)
        y=self.moudle2(y)
        y=self.linear1(y)
        y=F.leaky_relu(y)
        y=F.normalize(y)
        y = F.dropout(y, p=0.5)
        y=self.linear2(y)
        return y

class binary_model(nn.Module):
    def __init__(self, hidden_size, output_size, softmax_size):
        super(binary_model,self).__init__()
        self.hidden_size, self.output_size,self.softmax_size = hidden_size, output_size,softmax_size
        self.moudle1 = classification_moudle(self.hidden_size)
        self.moudle2 = classification_moudle(self.hidden_size)
        self.moudle3 = classification_moudle(self.hidden_size)
        self.moudle4 = classification_moudle(self.hidden_size)
        self.linear1 = nn.Linear(self.hidden_size+self.softmax_size, int(self.hidden_size / 2))
        self.linear2 = nn.Linear(int(self.hidden_size / 2)+self.softmax_size, int(self.hidden_size / 2)+self.softmax_size)
        self.linear3 = nn.Linear(int(self.hidden_size / 2)+self.softmax_size,self.output_size)

    def forward(self, x , soft_scores):
        y = self.moudle1(x)
        y = self.moudle2(y)
        y = self.moudle3(y)
        y = self.moudle4(y)
        y = torch.cat([y,soft_scores],dim=1)
        y = self.linear1(y)
        y = F.leaky_relu(y)
        y = F.normalize(y)
        y = F.dropout(y, p=0.8)
        y = torch.cat([y,soft_scores],dim=1)
        y = self.linear2(y)
        y = F.leaky_relu(y)
        y = F.normalize(y)
        y = F.dropout(y, p=0.8)
        y= self.linear3(y)
        return y

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
        # self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.classifier = classification_model(config.hidden_size, config.num_labels)
        self.binary_classifier=binary_model(2*config.hidden_size,2,config.num_labels)
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
        binary_labels: Optional[torch.Tensor]=None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        beta:Optional[float]=1.0,
        alpha:Optional[float]=1.0,
        tmp: Optional[float] = 1,
        ood_label: Optional[int]=-1,
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

        hidden_states = outputs.hidden_states
        last_layer_hidden_states=hidden_states[-1]
        cls_hidden_state = last_layer_hidden_states[:, 0, :]
        sep_hidden_state=last_layer_hidden_states[:, -1, :]
        cls_hidden_state = self.dropout(cls_hidden_state)
        sep_hidden_state = self.dropout(sep_hidden_state)
        classify_logits=self.classifier(cls_hidden_state)
        classify_logits_id=classify_logits[(1 - binary_labels).bool()]
        labels_id=labels[(1 - binary_labels).bool()]
        cls_copy=cls_hidden_state.clone().detach()
        cls_copy.requires_grad=True
        copy_classify_logits=classify_logits.clone().detach()
        copy_classify_logits.requires_grad=True
        binary_logits=self.binary_classifier(torch.cat([cls_copy,sep_hidden_state],dim=1),copy_classify_logits)
        loss = None
        if labels is not None and binary_labels is not None:
            classify_loss_fct = CrossEntropyLoss(reduction='none')
            binary_loss_fct = CrossEntropyLoss()
            pos_loss_seq = classify_loss_fct(torch.div(classify_logits_id.view(-1, self.num_labels), tmp),
                                                  labels_id.view(-1))
            if len(pos_loss_seq)>0:
                classify_loss = torch.div(torch.sum(pos_loss_seq), len(pos_loss_seq))
            else:
                classify_loss = 0
            binary_loss = binary_loss_fct(binary_logits.view(-1, 2), binary_labels.view(-1))
            print(f"classify_loss:{classify_loss},binary:{binary_loss}")
            loss = alpha * classify_loss + beta * binary_loss

        return ClassifierOutput(
            loss=loss,
            logits=classify_logits,
            binary_logits=binary_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
