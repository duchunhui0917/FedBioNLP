import torch.nn
import torch.nn.functional as F
from torchvision import models
from transformers import AutoModelForTokenClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch import nn
from transformers import AutoModel, AutoModelForMaskedLM
from utils.GCN_utils import GraphConvolution, LSR
import logging
import os
from .tokenizers import *
from .datasets import *

logger = logging.getLogger(os.path.basename(__file__))


class ImgRegLR(nn.Module):
    def __init__(self, input_channel, input_dim, output_dim):
        super(ImgRegLR, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.input_channel = input_channel
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer = nn.Linear(input_channel * input_dim * input_dim, output_dim)

    def forward(self, inputs, labels):
        x = inputs.squeeze(dim=1).view(-1, self.input_channel * self.input_dim * self.input_dim)
        logits = self.layer(x)
        loss = self.criterion(logits, labels)
        return logits, loss


class ConvModule(nn.Module):
    def __init__(self):
        super(ConvModule, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)

    def forward(self, x):
        # (b, 3, 32, 32)
        x = self.pool(self.relu(self.conv1(x)))  # (b, 6, 14, 14)
        x = self.pool(self.relu(self.conv2(x)))  # (b, 16, 5, 5)
        x = x.view(-1, 16 * 5 * 5)  # (b, 16 * 5 * 5)
        x = self.relu(self.fc1(x))  # (b, 120)
        x = self.relu(self.fc2(x))  # (b, 84)
        return x


class ImageConvNet(nn.Module):
    def __init__(self):
        super(ImageConvNet, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.features = ConvModule()
        self.l1 = nn.Linear(84, 84)
        self.l2 = nn.Linear(84, 256)
        self.l3 = nn.Linear(256, 10)

    def forward(self, inputs, labels):
        inputs, labels = inputs[0], labels[0]
        features = self.features(inputs)
        x = F.relu(self.l1(features))
        x = F.relu(self.l2(x))
        logits = self.l3(x)

        loss = self.criterion(logits, labels)
        return [features], logits, [loss]


class ImageConvCCSANet(nn.Module):
    def __init__(self):
        super(ImageConvCCSANet, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.extractor = ConvModule()
        self.classifier = nn.Linear(84, 10)
        self.alpha = 0.25

    @staticmethod
    # Contrastive Semantic Alignment Loss
    def csa_loss(x, y, class_eq):
        margin = 1
        dist = F.pairwise_distance(x, y)
        loss = class_eq * dist.pow(2)
        loss += (1 - class_eq) * (margin - dist).clamp(min=0).pow(2)
        return loss.mean()

    def forward(self, inputs, labels):
        if self.training:
            src_inputs, tgt_inputs = inputs
            src_labels, tgt_labels = labels
            src_features = self.extractor(src_inputs)
            tgt_features = self.extractor(tgt_inputs)

            src_logits = self.classifier(src_features)
            tgt_logits = self.classifier(tgt_features)
            csa = self.csa_loss(src_features, tgt_features, (src_labels == tgt_labels).float())

            src_label_loss = self.criterion(src_logits, src_labels)
            tgt_label_loss = self.criterion(tgt_logits, tgt_labels)
            loss = (1 - self.alpha) * src_label_loss + self.alpha * csa

            return None, [src_logits, tgt_logits], [loss, src_label_loss, tgt_label_loss, csa]
        else:
            inputs, labels = inputs[0], labels[0]
            features = self.extractor(inputs)
            logits = self.classifier(features)
            loss = self.criterion(logits, labels)

            return [features], logits, [loss]


class ImgRegResNet(nn.Module):
    def __init__(self, model_name, output_dim, pretrained=True):
        super(ImgRegResNet, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        if model_name == 'ResNet18':
            self.resnet = models.resnet18(pretrained=pretrained)
        elif model_name == 'ResNet50':
            self.resent = models.resnet50(pretrained=pretrained)
        elif model_name == 'ResNet152':
            self.resent = models.resnet152(pretrained=pretrained)
        else:
            raise Exception("Invalid model name. Must be 'ResNet18' or 'ResNet50' or 'ResNet152'.")
        in_features = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Linear(in_features, output_dim)

    def forward(self, inputs, labels):
        logits = self.resnet(inputs)
        loss = self.criterion(logits, labels)
        return None, logits, loss


class SeqClsCNN(nn.Module):
    pass


class SeqClsRNN(nn.Module):
    pass


class MaskedLMBERT(nn.Module):
    def __init__(self, model_name):
        super(MaskedLMBERT, self).__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)

    def forward(self, inputs, labels):
        inputs, mask = inputs
        outputs = self.model(inputs, labels=labels, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        logits = outputs.logits
        loss = outputs.loss

        score, pred_labels = logits.max(-1)
        acc = (pred_labels == labels).sum() / mask.size(0)
        mask_acc = ((pred_labels == labels) * mask).sum() / mask.sum()

        return hidden_states, logits, [loss]


class SequenceClassificationBert(nn.Module):
    def __init__(self, model_name, output_dim):
        super(SequenceClassificationBert, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Linear(768, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(768, output_dim)
        )

        self.criterion = nn.CrossEntropyLoss()

        # for param in self.model.base_model.parameters():
        #     param.requires_grad = False

    def forward(self, inputs, labels):
        inputs = inputs[0]
        outputs = self.model(inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        last_hidden_state = outputs.last_hidden_state
        pooled_output = last_hidden_state[:, 0]
        logits = self.classifier(pooled_output)

        loss = self.criterion(logits, labels)
        return hidden_states, logits, [loss]


class REModel(nn.Module):
    def __init__(self, model_name, output_dim):
        super(REModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout()
        self.classifier = nn.Sequential(
            nn.Linear(768 * 2, 768 * 2),
            nn.ReLU(),
            nn.Linear(768 * 2, output_dim)
        )

        self.criterion = nn.CrossEntropyLoss()
        self.tokenizer = re_tokenizer
        self.dataset = REDataset

        # for param in self.encoder.parameters():
        #     param.requires_grad = False

    @staticmethod
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

    def forward(self, inputs, labels):
        input_ids, input_mask, e1_mask, e2_mask = inputs
        labels = labels[0]
        outputs = self.encoder(input_ids, attention_mask=input_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        encoder_output = outputs.last_hidden_state  # (B, L, H)

        # extract entity
        e1_h = self.max_pooling(encoder_output, e1_mask)  # (B, H)
        e2_h = self.max_pooling(encoder_output, e2_mask)  # (B, H)
        ent = torch.cat([e1_h, e2_h], dim=-1)  # (B, 2H)
        ent = self.dropout(ent)

        # classifier
        logits = self.classifier(ent)  # (B, C)
        loss = self.criterion(logits, labels)

        return hidden_states, logits, [loss]


class REGCNModel(REModel):
    def __int__(self, model_name, output_dim, num_gcn_layers=1):
        super(REGCNModel, self).__int__(model_name, output_dim)
        gcn_layer = GraphConvolution(768, 768)
        self.gcn_layers = nn.ModuleList([copy.deepcopy(gcn_layer) for _ in range(num_gcn_layers)])
        self.tokenizer = re_dep_tokenizer
        self.dataset = REGCNDataset

    @staticmethod
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

    def forward(self, inputs, labels):
        input_ids, input_mask, valid_ids, e1_mask, e2_mask, dep_matrix = inputs
        labels = labels[0]
        outputs = self.encoder(input_ids, attention_mask=input_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        encoder_output = outputs.last_hidden_state  # (B, L, H)

        # filter valid output
        valid_output = self.valid_filter(encoder_output, valid_ids)  # (B, L, H)
        valid_output = self.dropout(valid_output)

        # gcn
        gcn_output = valid_output
        for gcn_layer in self.gcn_layers:
            gcn_output = gcn_layer(gcn_output, dep_matrix)  # (B, L, H)
        gcn_output = self.dropout(gcn_output)

        # extract entity
        e1_h = self.max_pooling(gcn_output, e1_mask)  # (B, H)
        e2_h = self.max_pooling(gcn_output, e2_mask)  # (B, H)
        ent = torch.cat([e1_h, e2_h], dim=-1)  # (B, 2H)
        ent = self.dropout(ent)

        # classifier
        logits = self.classifier(ent)  # (B, C)
        loss = self.criterion(logits, labels)

        return hidden_states, logits, [loss]


class RELatentGCNModel(REModel):
    def __init__(self, model_name, output_dim):
        super(RELatentGCNModel, self).__init__(model_name, output_dim)
        self.lsr = LSR(768)

    def forward(self, inputs, labels):
        input_ids, input_mask, e1_mask, e2_mask = inputs
        labels = labels[0]
        outputs = self.encoder(input_ids, attention_mask=input_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        encoder_output = outputs.last_hidden_state
        encoder_output = self.dropout(encoder_output)  # (B, L, H)

        # process by LF-GCN
        lsr_output, _ = self.lsr(encoder_output, input_mask)  # (B, L, H)

        # extract entity
        e1_h = self.max_pooling(lsr_output, e1_mask)  # (B, H)
        e2_h = self.max_pooling(lsr_output, e2_mask)  # (B, H)
        ent = torch.cat([e1_h, e2_h], dim=-1)  # (B, 2H)
        ent = self.dropout(ent)

        # classifier
        logits = self.classifier(ent)
        loss = self.criterion(logits, labels)

        return hidden_states, logits, [loss]


class REHorizonModel(REModel):
    def __init__(self, model_name, output_dim):
        super(REHorizonModel, self).__init__(model_name, output_dim)
        self.model = AutoModel.from_pretrained(model_name)
        self.patch = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 768),
            nn.ReLU()
        )

    def forward(self, inputs, labels):
        input_ids, input_mask, valid_ids, e1_mask, e2_mask, dep_matrix = inputs
        labels = labels[0]
        outputs = self.encoder(input_ids, attention_mask=input_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        encoder_output = outputs.last_hidden_state  # (B, L, H)

        # add patch
        path_outputs = self.patch(hidden_states[0])  # (B, L, H)
        encoder_output += path_outputs

        # filter valid output
        valid_output = self.valid_filter(encoder_output, valid_ids)  # (B, L, H)
        valid_output = self.dropout(valid_output)

        # gcn
        gcn_output = valid_output
        for gcn_layer in self.gcn_layers:
            gcn_output = gcn_layer(gcn_output, dep_matrix)  # (B, L, H)

        # extract entity
        e1_h = self.max_pooling(gcn_output, e1_mask)  # (B, H)
        e2_h = self.max_pooling(gcn_output, e2_mask)  # (B, H)
        ent = torch.cat([e1_h, e2_h], dim=-1)  # (B, 2H)
        ent = self.dropout(ent)

        # classifier
        logits = self.classifier(ent)  # (B, C)
        loss = self.criterion(logits, labels)

        return hidden_states, logits, [loss]


class GRLFunc(torch.autograd.Function):
    def __init__(self):
        super(GRLFunc, self).__init__()

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_variables
        grad_input = grad_output.clone()
        return - lambda_ * grad_input, None


class GRL(nn.Module):
    def __init__(self, lambda_=0.):
        super(GRL, self).__init__()
        self.lambda_ = torch.tensor(lambda_)

    def set_lambda(self, lambda_):
        self.lambda_ = torch.tensor(lambda_)

    def forward(self, x):
        return GRLFunc.apply(x, self.lambda_)


class REGRLModel(REModel):
    def __init__(self, model_name, output_dim):
        super(REGRLModel, self).__init__(model_name, output_dim)

        self.domain_classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(768 * 2, output_dim)
        )
        # self.count = 0
        # self.Lambda = 2 / (1 + math.exp(- 0.01 * self.count)) - 1
        self.grl = GRL(0.1)
        self.tokenizer = re_tokenizer
        self.dataset = REGRLDataset

    def forward(self, inputs, labels):
        input_ids, input_mask, e1_mask, e2_mask, doc = inputs
        labels = labels[0]
        outputs = self.encoder(input_ids, attention_mask=input_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        encoder_output = outputs.last_hidden_state  # (B, L, H)

        # extract entity
        e1_h = self.max_pooling(encoder_output, e1_mask)  # (B, H)
        e2_h = self.max_pooling(encoder_output, e2_mask)  # (B, H)
        ent = torch.cat([e1_h, e2_h], dim=-1)  # (B, 2H)
        ent = self.dropout(ent)

        label_logits = self.classifier(ent)
        re_ent = self.grl(ent)
        domain_logits = self.domain_classifier(re_ent)

        if self.training:
            source_doc = torch.where(doc == 0)
            label_loss = self.criterion(label_logits[source_doc], labels[source_doc])
            domain_loss = self.criterion(domain_logits, doc)
            loss = label_loss + 0 * domain_loss
            return hidden_states, label_logits, [loss, label_loss, domain_loss]
        else:
            loss = self.criterion(label_logits, labels)
            return hidden_states, label_logits, [loss]


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

    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
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


class RESCLModel(REModel):
    def __init__(self, model_name, output_dim):
        super(RESCLModel, self).__init__(model_name, output_dim)
        self.Lambda = 0.5
        self.tau = 0.3

    def scl(self, ent, labels):
        ent_norm = F.normalize(ent, dim=1)
        ent_pie = torch.transpose(ent_norm, 0, 1)

        cos = torch.matmul(ent_norm, ent_pie)
        exp_cos = torch.exp(cos / self.tau)
        log_exp_cos = - torch.log(exp_cos / (torch.sum(exp_cos, dim=1) - torch.diag(exp_cos)))

        equal = torch.zeros_like(log_exp_cos)

        for i in range(len(equal)):
            idx = torch.where(labels == labels[i])[0]
            if len(idx) > 1:
                equal[i][idx] = 1 / (len(idx) - 1)
                equal[i][i] = 0

        cls = log_exp_cos * equal
        return cls

    def forward(self, inputs, labels):
        input_ids, input_mask, e1_mask, e2_mask = inputs
        labels = labels[0]
        outputs = self.encoder(input_ids, attention_mask=input_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        encoder_output = outputs.last_hidden_state  # (B, L, H)

        # extract entity
        e1_h = self.max_pooling(encoder_output, e1_mask)  # (B, H)
        e2_h = self.max_pooling(encoder_output, e2_mask)  # (B, H)
        ent = torch.cat([e1_h, e2_h], dim=-1)  # (B, 2H)
        ent = self.dropout(ent)

        logits = self.classifier(ent)
        ce = self.criterion(logits, labels)

        if self.training:
            scl = self.scl(ent, labels)
            loss = (1 - self.Lambda) * ce + self.Lambda * scl

            return None, logits, [loss, ce, scl]
        else:
            return hidden_states, logits, [ce]


class REMMDModel(REModel):
    def __init__(self, model_name, output_dim):
        super(REMMDModel, self).__init__(model_name, output_dim)
        self.mmd = MMDLoss()
        self.Lambda = 0.1

    def forward(self, inputs, labels):
        if self.training:
            src_idx_tokens, src_pos0, src_pos1, tgt_idx_tokens, tgt_pos0, tgt_pos1 = inputs
            src_labels, tgt_labels = labels

            src_hidden_states = self.extract_feature(src_idx_tokens, src_pos0, src_pos1)
            tgt_hidden_states = self.extract_feature(tgt_idx_tokens, tgt_pos0, tgt_pos1)
            src_ent, tgt_ent = src_hidden_states[-1], tgt_hidden_states[-1]

            src_logits = self.classifier(src_ent)
            tgt_logits = self.classifier(tgt_ent)

            src_label_loss = self.criterion(src_logits, src_labels)
            tgt_label_loss = self.criterion(tgt_logits, tgt_labels)

            # domain_loss = self.mmd(src_ent, tgt_ent)
            domain_loss = self.mmd(src_hidden_states[-2].mean(dim=1), tgt_hidden_states[-2].mean(dim=1))
            loss = 0 * src_label_loss + 0 * tgt_label_loss + self.Lambda * domain_loss

            return None, [src_logits, tgt_logits], [loss, src_label_loss, tgt_label_loss, domain_loss]
        else:
            labels = labels[0]
            idx_tokens, pos0, pos1, docs = inputs
            hidden_states = self.extract_feature(idx_tokens, pos0, pos1)
            ent = hidden_states[-1]
            logits = self.classifier(ent)
            loss = self.criterion(logits, labels)
            return hidden_states, logits, [loss]


class RECCSAModel(REModel):
    def __init__(self, model_name, output_dim):
        super(RECCSAModel, self).__init__(model_name, output_dim)
        self.alpha = 0.25

    @staticmethod
    # Contrastive Semantic Alignment Loss
    def csa_loss(x, y, class_eq):
        margin = 1
        dist = F.pairwise_distance(x, y)
        loss = class_eq * dist.pow(2)
        loss += (1 - class_eq) * (margin - dist).clamp(min=0).pow(2)
        return loss.mean()

    def forward(self, inputs, labels):
        if self.training:
            src_idx_tokens, src_pos0, src_pos1, tgt_idx_tokens, tgt_pos0, tgt_pos1 = inputs
            src_labels, tgt_labels = labels

            src_hidden_states = self.extract_feature(src_idx_tokens, src_pos0, src_pos1)
            tgt_hidden_states = self.extract_feature(tgt_idx_tokens, tgt_pos0, tgt_pos1)
            src_ent, tgt_ent = src_hidden_states[-1], tgt_hidden_states[-1]

            src_logits = self.classifier(src_ent)
            tgt_logits = self.classifier(tgt_ent)

            src_label_loss = self.criterion(src_logits, src_labels)
            tgt_label_loss = self.criterion(tgt_logits, tgt_labels)
            csa = self.csa_loss(src_ent, tgt_ent, (src_labels == tgt_labels).float())

            loss = (1 - self.alpha) * src_label_loss + self.alpha * csa

            return None, [src_logits, tgt_logits], [loss, src_label_loss, tgt_label_loss, csa]
        else:
            idx_tokens, pos0, pos1, docs = inputs
            labels = labels[0]
            hidden_states = self.extract_feature(idx_tokens, pos0, pos1)
            ent = hidden_states[-1]
            logits = self.classifier(ent)
            loss = self.criterion(logits, labels)
            return hidden_states, logits, [loss]


class PLONERBERT(nn.Module):
    def __init__(self, bert_name):
        super(PLONERBERT, self).__init__()
        self.model = AutoModelForTokenClassification.from_pretrained(bert_name, num_labels=7)

    def forward(self, inputs, labels):
        x = self.model(inputs, labels=labels)
        logits = x.logits
        loss = x.loss
        return logits, loss


class WikiNERBERT(nn.Module):
    def __init__(self, bert_name):
        super(WikiNERBERT, self).__init__()
        self.model = AutoModelForTokenClassification.from_pretrained(bert_name, num_labels=9)

    def forward(self, x, labels):
        x = self.model(x, labels=labels)
        logits = x.logits
        loss = x.loss
        # outputs = self.model(x)
        # logits = outputs.logits
        return logits, loss

    def test(self, data_loaders):
        n_dataloader = len(data_loaders)
        batch_size = data_loaders.batch_size
        y_true, y_pred = (np.zeros((n_dataloader, batch_size)) for _ in range(2))

        for i, data in enumerate(data_loaders):
            inputs, labels = data
            logits, loss = self.forward(inputs, labels)

            labels = labels.cpu()
            predict_labels = np.argmax(logits.cpu().detach().numpy(), axis=-1)
            acc = 0
            for i in range(len(labels)):
                acc = (acc * i + accuracy_score(labels[i], predict_labels[i])) / (i + 1)

            y_true[i] = labels
            y_pred[i] = predict_labels
        acc = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        cfm = confusion_matrix(y_true, y_pred)
        metric = {'Acc': acc, 'Rec': recall, 'Pre': precision, 'F1': f1, 'CFM': cfm}
        logger.info(metric)
        return metric
