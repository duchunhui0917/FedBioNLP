from torch import nn
import torch
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.modeling_outputs import MaskedLMOutput
from transformers import AutoModel, AutoConfig
from transformers import DistilBertModel, DistilBertPreTrainedModel, PretrainedConfig
from transformers.activations import get_activation
from transformers import AutoModelForMaskedLM
from typing import Optional, Tuple, Union, Any
import math
import torch
from torch import nn
from scipy.special import binom


class MMDLoss(nn.Module):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
    source: 源域数据（n * len(x))
    target: 目标域数据（m * len(y))
    kernel_mul:
    kernel_num: 取不同高斯核的数量
    fix_sigma: 不同高斯核的sigma值
    Return:
    loss: MMD loss
    '''

    def __init__(self, kernel_type='linear', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss


class GRL(torch.autograd.Function):
    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        pass

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_output *= -ctx.lambda_
        return grad_output, None


class GRLFunc(torch.autograd.Function):
    def __init__(self):
        super(GRLFunc, self).__init__()

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # print('gradient reverse')
        lambda_, = ctx.saved_variables
        grad_input = grad_output.clone()
        return - lambda_ * grad_input, None


class GRLModule(nn.Module):
    def __init__(self, lambda_=0.):
        super(GRLModule, self).__init__()
        self.lambda_ = torch.tensor(lambda_)

    def set_lambda(self, lambda_):
        self.lambda_ = torch.tensor(lambda_)

    def forward(self, x):
        return GRLFunc.apply(x, self.lambda_)


# focal_loss func, L = -α(1-yi)**γ *ce_loss(xi,yi)
class focal_loss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, num_classes=2, size_average=True):
        super(focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)

        self.gamma = gamma

    def forward(self, preds, labels):
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds, dim=1)
        preds_logsoft = torch.log(preds_softmax)

        # focal_loss func, Loss = -α(1-yi)**γ *ce_loss(xi,yi)
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.view(-1))
        # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)

        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class DistilBertForMaskedLM(DistilBertPreTrainedModel):
    def __init__(self, config: PretrainedConfig):
        super(DistilBertForMaskedLM, self).__init__(config)

        self.activation = get_activation(config.activation)

        self.distilbert = DistilBertModel(config)
        self.vocab_transform = nn.Linear(config.dim, config.dim)
        self.vocab_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)
        self.vocab_projector = nn.Linear(config.dim, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

        self.mlm_loss_fct = nn.CrossEntropyLoss()

        for param in self.distilbert.parameters():
            param.requires_grad = True

    def get_position_embeddings(self) -> nn.Embedding:
        """
        Returns the position embeddings
        """
        return self.distilbert.get_position_embeddings()

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.
        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end. If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the
                size will add correct vectors at the end following the position encoding algorithm, whereas reducing
                the size will remove vectors from the end.
        """
        self.distilbert.resize_position_embeddings(new_num_position_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.vocab_projector

    def set_output_embeddings(self, new_embeddings: nn.Module):
        self.vocab_projector = new_embeddings

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[MaskedLMOutput, Tuple[torch.Tensor, ...]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        dlbrt_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = dlbrt_output.last_hidden_state  # (bs, seq_length, dim)

        prediction_logits = self.vocab_transform(last_hidden_state)  # (bs, seq_length, dim)
        prediction_logits = self.activation(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)

        mlm_loss = None
        if labels is not None:
            mlm_loss = self.mlm_loss_fct(prediction_logits.view(-1, prediction_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (prediction_logits,) + dlbrt_output[1:]
            return ((mlm_loss,) + output) if mlm_loss is not None else output

        return MaskedLMOutput(
            loss=mlm_loss,
            logits=prediction_logits,
            hidden_states=dlbrt_output.hidden_states,
            attentions=dlbrt_output.attentions,
        )


class MaskedLMClassifier(nn.Module):
    def __init__(self, config):
        super(MaskedLMClassifier, self).__init__()
        model = AutoModelForMaskedLM.from_pretrained('distilbert-base-cased')
        self.activation = model.activation
        self.vocab_transform = model.vocab_transform
        self.vocab_layer_norm = model.vocab_layer_norm
        self.vocab_projector = model.vocab_projector
        self.mlm_loss_fct = model.mlm_loss_fct

        # self.activation = get_activation(config.activation)
        # self.vocab_transform = nn.Linear(config.dim, config.dim)
        # self.vocab_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)
        # self.vocab_projector = nn.Linear(config.dim, config.vocab_size)
        # self.mlm_loss_fct = nn.CrossEntropyLoss()

    def forward(self, labels, hidden_state):
        logits = self.vocab_transform(hidden_state)  # (bs, seq_length, dim)
        logits = self.activation(logits)  # (bs, seq_length, dim)
        logits = self.vocab_layer_norm(logits)  # (bs, seq_length, dim)
        logits = self.vocab_projector(logits)  # (bs, seq_length, vocab_size)

        loss = self.mlm_loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return logits, loss


# class SupConLoss(nn.Module):
#     def __init__(self, temperature=0.07):
#         super(SupConLoss, self).__init__()
#         self.temperature = temperature
#
#     def forward(self, features, labels):
#         features = F.normalize(features, dim=1)
#
#         batch_size = features.shape[0]
#
#         labels = labels.contiguous().view(-1, 1)
#         mask = torch.eq(labels, labels.T).float()
#
#         # compute logits
#         anchor_dot_contrast = torch.div(
#             torch.matmul(features, features.T),
#             self.temperature)
#         # for numerical stability
#         logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
#         logits = anchor_dot_contrast - logits_max.detach()
#
#         # mask-out self-contrast cases
#         logits_mask = torch.scatter(
#             torch.ones_like(mask),
#             1,
#             torch.arange(batch_size).view(-1, 1).cuda(),
#             0
#         )
#         mask = mask * logits_mask
#
#         # compute log_prob
#         exp_logits = torch.exp(logits) * logits_mask
#         log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
#
#         # loss
#         loss = - (mask * log_prob).sum(1) / mask.sum(1)
#         loss = loss.mean()
#
#         return loss


def SupConLoss(features, labels):
    temperature = 0.07
    ent_norm = F.normalize(features, dim=1)
    ent_pie = torch.transpose(ent_norm, 0, 1)

    cos = torch.matmul(ent_norm, ent_pie)
    exp_cos = torch.exp(cos / temperature)
    log_exp_cos = - torch.log(exp_cos / (torch.sum(exp_cos, dim=1) - torch.diag(exp_cos)))

    equal = torch.zeros_like(log_exp_cos)

    for i in range(len(equal)):
        idx = torch.where(labels == labels[i])[0]
        if len(idx) > 1:
            equal[i][idx] = 1 / (len(idx) - 1)
            equal[i][i] = 0

    cls = log_exp_cos * equal
    cls = cls.sum(1).mean()
    return cls


class LSoftmaxLinear(nn.Module):

    def __init__(self, input_features, output_features, margin):
        super(LSoftmaxLinear, self).__init__()
        self.input_dim = input_features  # number of input feature i.e. output of the last fc layer
        self.output_dim = output_features  # number of output = class numbers
        self.margin = margin  # m
        self.beta = 100
        self.beta_min = 0
        self.scale = 0.99

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Initialize L-Softmax parameters
        self.weight = nn.Parameter(torch.FloatTensor(input_features, output_features))
        self.divisor = math.pi / self.margin  # pi/m
        self.C_m_2n = torch.Tensor(binom(margin, range(0, margin + 1, 2))).to(device)  # C_m{2n}
        self.cos_powers = torch.Tensor(range(self.margin, -1, -2)).to(device)  # m - 2n
        self.sin2_powers = torch.Tensor(range(len(self.cos_powers))).to(device)  # n
        self.signs = torch.ones(margin // 2 + 1).to(device)  # 1, -1, 1, -1, ...
        self.signs[1::2] = -1

    def calculate_cos_m_theta(self, cos_theta):
        sin2_theta = 1 - cos_theta ** 2
        cos_terms = cos_theta.unsqueeze(1) ** self.cos_powers.unsqueeze(0)  # cos^{m - 2n}
        sin2_terms = (sin2_theta.unsqueeze(1)  # sin2^{n}
                      ** self.sin2_powers.unsqueeze(0))

        cos_m_theta = (self.signs.unsqueeze(0) *  # -1^{n} * C_m{2n} * cos^{m - 2n} * sin2^{n}
                       self.C_m_2n.unsqueeze(0) *
                       cos_terms *
                       sin2_terms).sum(1)  # summation of all terms

        return cos_m_theta

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight.data.t())

    def find_k(self, cos):
        # to account for acos numerical errors
        eps = 1e-7
        cos = torch.clamp(cos, -1 + eps, 1 - eps)
        acos = cos.acos()
        k = (acos / self.divisor).floor().detach()
        return k

    def forward(self, input, target=None):
        if self.training:
            assert target is not None
            x, w = input, self.weight
            beta = max(self.beta, self.beta_min)
            logit = x.mm(w)
            indexes = range(logit.size(0))
            logit_target = logit[indexes, target]

            # cos(theta) = w * x / ||w||*||x||
            w_target_norm = w[:, target].norm(p=2, dim=0)
            x_norm = x.norm(p=2, dim=1)
            cos_theta_target = logit_target / (w_target_norm * x_norm + 1e-10)

            # equation 7
            cos_m_theta_target = self.calculate_cos_m_theta(cos_theta_target)

            # find k in equation 6
            k = self.find_k(cos_theta_target)

            # f_y_i
            logit_target_updated = (w_target_norm *
                                    x_norm *
                                    (((-1) ** k * cos_m_theta_target) - 2 * k))
            logit_target_updated_beta = (logit_target_updated + beta * logit[indexes, target]) / (1 + beta)

            logit[indexes, target] = logit_target_updated_beta
            self.beta *= self.scale
            return logit
        else:
            return input.mm(self.weight)


def valid_filter(sequence, valid_ids):
    """

    Args:
        sequence: (B, L, H)
        valid_ids: (B, L, H)

    Returns:

    """
    batch_size, max_len, hidden_dim = sequence.shape
    valid_output = torch.zeros(batch_size, max_len, hidden_dim,
                               dtype=sequence.dtype,
                               device=sequence.device)
    for i in range(batch_size):
        tmp = sequence[i][valid_ids[i] == 1]  # (L, H)
        valid_output[i][:tmp.size(0)] = tmp
    return valid_output


def max_pooling(sequence, e_mask):
    """

    Args:
        sequence: (B, L, H)
        e_mask: (B, L, H)

    Returns:

    """
    entity_output = sequence * torch.stack([e_mask] * sequence.shape[-1], 2) + torch.stack(
        [(1.0 - e_mask) * -1000.0] * sequence.shape[-1], 2)
    entity_output = torch.max(entity_output, -2)[0]
    return entity_output.type_as(sequence)


class InfoNCE(nn.Module):
    def __init__(self, temperature=0.05):
        super(InfoNCE, self).__init__()
        self.temperature = temperature

    def forward(self, h1, h2):
        pos_sim = F.cosine_similarity(h1, h2, dim=-1)
        pos_sim = torch.exp(pos_sim / self.temperature)
        return pos_sim


class MyLSTM(nn.Module):
    def __init__(self):
        super(MyLSTM, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=10002, embedding_dim=100)
        bidirectional = True
        self.lstm = nn.LSTM(input_size=100, hidden_size=670, num_layers=2, batch_first=True,
                            bidirectional=bidirectional)
        self.l1 = nn.Linear(670, 100)

    def forward(self, x, output_hidden_states=False):
        try:
            x = self.embedding(x)
        except:
            pass
        output, (h_n, c_n) = self.lstm(x)
        x = torch.mean(h_n, dim=0)
        x = F.relu(self.l1(x))
        if output_hidden_states:
            return x, output
        else:
            return x


class NERLSTM(nn.Module):
    def __init__(self):
        super(NERLSTM, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=10002, embedding_dim=100)
        bidirectional = True
        self.lstm = nn.LSTM(input_size=100, hidden_size=670, num_layers=2, batch_first=True,
                            bidirectional=bidirectional)
        b = 2 if bidirectional else 1
        self.l1 = nn.Linear(670 * b, 100)

    def forward(self, x):
        try:
            x = self.embedding(x)
        except:
            pass
        output, (h_n, c_n) = self.lstm(x)
        output = F.relu(self.l1(output))
        return output
