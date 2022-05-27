import torch.nn
import torch.nn.functional as F
from torchvision import models
from transformers import AutoModelForTokenClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch import nn
from transformers import AutoModel, AutoModelForMaskedLM, AutoConfig
from .utils.GCN_utils import GraphConvolution, LSR
import logging
import os
import copy
from .tokenizers import *
from .datasets import *
from .modules import DistilBertForMaskedLM, MMDLoss, GRL, SupConLoss, LSoftmaxLinear, valid_filter, max_pooling

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


class SCModel(nn.Module):
    def __init__(self, model_name, output_dim):
        super(SCModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(768, output_dim)
        # self.classifier = LSoftmaxLinear(768, output_dim, 0)

        self.criterion = nn.CrossEntropyLoss()
        self.tokenizer = sc_tokenizer
        self.dataset = SCDataset

        for param in self.encoder.parameters():
            param.requires_grad = True

        # for param in self.model.base_model.parameters():
        #     param.requires_grad = False

    def forward(self, inputs, labels):
        input_ids, attention_mask = inputs
        labels = labels[0]
        outputs = self.encoder(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        last_hidden_state = outputs.last_hidden_state

        cls = last_hidden_state[:, 0]
        logits = self.classifier(cls, labels)

        loss = self.criterion(logits, labels)
        return hidden_states, logits, [loss]


class SCSCLModel(SCModel):
    def __init__(self, model_name, output_dim):
        super(SCSCLModel, self).__init__(model_name, output_dim)

        self.Lambda = 0.9
        self.tau = 0.3
        self.projection = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 384)
        )
        self.scl = SupConLoss
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(self, inputs, labels, ite=None):
        # if ite is not None:
        #     self.Lambda = max(1 - ite / 10, 0)

        input_ids, attention_mask = inputs
        labels = labels[0]
        outputs = self.encoder(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        last_hidden_state = outputs.last_hidden_state

        pooled_output = last_hidden_state[:, 0]
        logits = self.classifier(pooled_output)

        ce = self.criterion(logits, labels)
        if self.training:
            scl = self.scl(pooled_output, labels)
            loss = (1 - self.Lambda) * ce + self.Lambda * scl

            return None, logits, [loss, ce, scl]
        else:
            return hidden_states, logits, [ce]


class REModel(nn.Module):
    def __init__(self, model_name, output_dim, num_gcn_layers):
        super(REModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        gcn_layer = GraphConvolution(768, 768)
        self.gcn_layers = nn.ModuleList([copy.deepcopy(gcn_layer) for _ in range(num_gcn_layers)])
        self.dropout = nn.Dropout()
        self.classifier = nn.Linear(768 * 2, output_dim)

        config = AutoConfig.from_pretrained(model_name)
        self.mlm = DistilBertForMaskedLM(config)

        self.criterion = nn.CrossEntropyLoss()
        self.dataset = REDataset

        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(self, inputs, labels):
        input_ids, input_mask, e1_mask, e2_mask, mlm_input_ids, valid_ids, dep_matrix = inputs
        labels = labels[0]
        outputs = self.encoder(input_ids, attention_mask=input_mask, output_hidden_states=True)
        encoder_hidden_states = outputs.hidden_states
        encoder_output = outputs.last_hidden_state  # (B, L, H)
        encoder_output = self.dropout(encoder_output)

        if torch.max(valid_ids) > 0 and torch.max(dep_matrix) > 0:
            # filter valid output
            valid_output = valid_filter(encoder_output, valid_ids)  # (B, L, H)

            # gcn
            gcn_output = valid_output
            for gcn_layer in self.gcn_layers:
                gcn_output = gcn_layer(gcn_output, dep_matrix)  # (B, L, H)
            encoder_output = self.dropout(gcn_output)

        # extract entity
        e1_h = max_pooling(encoder_output, e1_mask)  # (B, H)
        e2_h = max_pooling(encoder_output, e2_mask)  # (B, H)
        ent = torch.cat([e1_h, e2_h], dim=-1)  # (B, 2H)
        ent = self.dropout(ent)

        # classifier
        logits = self.classifier(ent)  # (B, C)
        ce_loss = self.criterion(logits, labels)

        hidden_states = encoder_hidden_states + (ent, logits)

        if self.training and torch.max(mlm_input_ids) > 0:
            mlm_outputs = self.mlm(input_ids, labels=mlm_input_ids, attention_mask=input_mask,
                                   output_hidden_states=True)
            mlm_loss = mlm_outputs.loss
            loss = ce_loss + 0.01 * mlm_loss
            return hidden_states, logits, [loss, ce_loss, mlm_loss]
        else:
            return hidden_states, logits, [ce_loss]



class REGSNModel(REModel):
    def __init__(self, model_name, output_dim, num_gcn_layers):
        super(REGSNModel, self).__init__(model_name, output_dim, num_gcn_layers)

    def forward(self, inputs, labels):
        input_ids, input_mask, e1_mask, e2_mask, mlm_input_ids, valid_ids, dep_matrix = inputs
        labels = labels[0]

        # relation extraction
        outputs = self.encoder(input_ids, attention_mask=input_mask, output_hidden_states=True)
        encoder_hidden_states = outputs.hidden_states
        encoder_output = outputs.last_hidden_state  # (B, L, H)

        if torch.max(valid_ids) > 0 and torch.max(dep_matrix) > 0:
            # filter valid output
            valid_output = valid_filter(encoder_output, valid_ids)  # (B, L, H)
            valid_output = self.dropout(valid_output)

            # gcn
            gcn_output = valid_output
            for gcn_layer in self.gcn_layers:
                gcn_output = gcn_layer(gcn_output, dep_matrix)  # (B, L, H)
            encoder_output = self.dropout(gcn_output)

        # extract entity
        e1_h = max_pooling(encoder_output, e1_mask)  # (B, H)
        e2_h = max_pooling(encoder_output, e2_mask)  # (B, H)
        ent = torch.cat([e1_h, e2_h], dim=-1)  # (B, 2H)
        ent = self.dropout(ent)

        # classifier
        logits = self.classifier(ent)  # (B, C)
        ce_loss = self.criterion(logits, labels)

        hidden_states = encoder_hidden_states + (ent, logits)

        if self.training:
            # encoder_rep = encoder_hidden_states[-1][:, 0]  # (B, H)
            # encoder_rep = torch.max(encoder_hidden_states[-1], -2)[0]  # (B, H)
            encoder_rep = torch.mean(encoder_hidden_states[-1], -2)  # (B, H)

            # masked language model
            mlm_outputs = self.mlm(input_ids, labels=mlm_input_ids, attention_mask=input_mask,
                                   output_hidden_states=True)
            mlm_hidden_states = mlm_outputs.hidden_states
            # mlm_rep = mlm_hidden_states[-1][:, 0]  # (B, H)
            # mlm_rep = torch.max(mlm_hidden_states[-1], -2)[0]  # (B, H)
            mlm_rep = torch.mean(mlm_hidden_states[-1], -2)  # (B, H)

            mlm_loss = mlm_outputs.loss

            dif_loss = torch.norm(torch.matmul(encoder_rep, mlm_rep.T)) ** 2

            loss = ce_loss + 0.01 * mlm_loss + 0.01 * dif_loss
            return hidden_states, logits, [loss, ce_loss, mlm_loss, dif_loss]
        else:
            return hidden_states, logits, [ce_loss]


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


class RESCLModel(REModel):
    def __init__(self, model_name, output_dim):
        super(RESCLModel, self).__init__(model_name, output_dim)
        self.Lambda = 0.9
        self.projection = nn.Sequential(
            nn.Linear(768 * 3, 768),
            nn.ReLU(),
            nn.Linear(768, 384)
        )
        self.scl = SupConLoss
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(self, inputs, labels, ite=None):
        # if ite is not None:
        #     self.Lambda = max(1 - ite / 10, 0)
        input_ids, input_mask, e1_mask, e2_mask = inputs
        labels = labels[0]
        outputs = self.encoder(input_ids, attention_mask=input_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        encoder_output = outputs.last_hidden_state  # (B, L, H)

        # outputs_aug = self.encoder(input_ids, attention_mask=input_mask, output_hidden_states=True)
        # hidden_states_aug = outputs_aug.hidden_states
        # encoder_output_aug = outputs_aug.last_hidden_state  # (B, L, H)

        # extract entity
        e1_h = self.max_pooling(encoder_output, e1_mask)  # (B, H)
        e2_h = self.max_pooling(encoder_output, e2_mask)  # (B, H)
        ent = torch.cat([e1_h, e2_h], dim=-1)  # (B, 2H)
        ent = self.dropout(ent)

        # # extract entity
        # e1_h_aug = self.max_pooling(encoder_output_aug, e1_mask)  # (B, H)
        # e2_h_aug = self.max_pooling(encoder_output, e2_mask)  # (B, H)
        # ent_aug = torch.cat([e1_h_aug, e2_h_aug], dim=-1)  # (B, 2H)
        # ent_aug = self.dropout(ent_aug)

        logits = self.classifier(ent)
        ce = self.criterion(logits, labels)

        hidden_states += (ent, logits)
        if self.training:
            # ent = torch.cat([ent, ent_aug], dim=0)
            # labels = torch.cat([labels, labels], dim=0)
            # features = self.projection(ent)
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
