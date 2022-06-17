import math

import torch.nn
import torch.nn.functional as F
from torchvision import models
from transformers import AutoModelForTokenClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch import nn
from transformers import AutoModel, AutoModelForMaskedLM
from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification
from transformers import AutoConfig, DistilBertModel
from .utils.GCN_utils import GraphConvolution, LSR
import logging
import os
import copy
from .tokenizers import *
from .datasets import *
from .modules import *

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


class SequenceClassificationBERT(nn.Module):
    def __init__(self, model_name, output_dim, load_pretrain):
        super(SequenceClassificationBERT, self).__init__()

        if load_pretrain:
            self.encoder = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=output_dim)
        else:
            config = AutoConfig.from_pretrained(model_name)
            config.num_labels = output_dim
            self.encoder = AutoModelForSequenceClassification.from_config(config)

        # self.encoder = AutoModel.from_pretrained(model_name)
        # self.classifier = nn.Linear(768, output_dim)

        self.criterion = nn.CrossEntropyLoss()
        self.tokenizer = nlp_tokenizer
        self.dataset = NLPDataset

        # for param in self.model.base_model.parameters():
        #     param.requires_grad = False

    def forward(self, inputs, labels, ite=None):
        input_ids, attention_mask = inputs
        labels = labels[0]

        outputs = self.encoder(input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        loss = outputs.loss
        hidden_states = [logits]

        # outputs = self.encoder(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # last_hidden_state = outputs.last_hidden_state
        # cls = last_hidden_state[:, 0]
        # logits = self.classifier(cls)
        # loss = self.criterion(logits, labels)
        # hidden_states = (cls, logits)
        return hidden_states, [logits], [loss]


class SequenceClassificationLSTM(nn.Module):
    def __init__(self, output_dim):
        super(SequenceClassificationLSTM, self).__init__()
        self.encoder = MyLSTM()
        self.classifier = nn.Linear(100, output_dim)
        self.criterion = nn.CrossEntropyLoss()
        self.tokenizer = nlp_tokenizer
        self.dataset = NLPDataset

    def forward(self, inputs, labels, ite=None):
        input_ids = inputs[0]
        labels = labels[0]
        x, output = self.encoder(input_ids, output_hidden_states=True)
        logits = self.classifier(x)
        loss = self.criterion(logits, labels)
        return [output], [logits], [loss]


class TokenClassificationBERT(nn.Module):
    def __init__(self, model_name, output_dim, load_pretrain):
        super(TokenClassificationBERT, self).__init__()
        if load_pretrain:
            self.model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=output_dim)
        else:
            config = AutoConfig.from_pretrained(model_name)
            config.num_labels = output_dim
            self.encoder = AutoModelForTokenClassification.from_config(config)

        self.tokenizer = token_classification_tokenizer
        self.dataset = NERDataset

    def forward(self, inputs, labels, ite=None):
        input_ids, attention_mask = inputs
        labels = labels[0]
        output = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        logits = output.logits
        loss = output.loss
        return [logits], [logits], [loss]


class TokenClassificationLSTM(nn.Module):
    def __init__(self, output_dim):
        super(TokenClassificationLSTM, self).__init__()
        self.encoder = NERLSTM()
        self.classifier = nn.Linear(100, output_dim)
        self.criterion = nn.CrossEntropyLoss()
        self.tokenizer = token_classification_tokenizer
        self.dataset = NERDataset

    def forward(self, inputs, labels, ite=None):
        input_ids = inputs[0]
        labels = labels[0]
        output = self.encoder(input_ids)
        logits = self.classifier(output)

        logits = logits.reshape(-1, logits.size(-1))
        labels = labels.reshape(labels.size(0) * labels.size(1))
        loss = self.criterion(logits, labels)
        return [output], [logits], [loss]


class REModel(nn.Module):
    def __init__(self, model_name, output_dim, num_gcn_layers):
        super(REModel, self).__init__()
        self.re_encoder = AutoModel.from_pretrained(model_name)
        self.unmask_encoder = AutoModel.from_pretrained(model_name)
        gcn_layer = GraphConvolution(768, 768)
        self.gcn_layers = nn.ModuleList([copy.deepcopy(gcn_layer) for _ in range(num_gcn_layers)])
        self.dropout = nn.Dropout()
        self.classifier = nn.Sequential(
            nn.Linear(768 * 2, 768),
            nn.ReLU(),
            nn.Linear(768, output_dim),
        )
        self.dataset = REDataset
        self.criterion = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def extract_feature(self, output, valid_ids, dep_matrix, e1_mask, e2_mask):
        if torch.max(valid_ids) > 0:
            # filter valid output
            output = valid_filter(output, valid_ids)  # (B, L, H)

            # gcn
            for gcn_layer in self.gcn_layers:
                output = gcn_layer(output, dep_matrix)  # (B, L, H)
            output = self.dropout(output)
        else:
            output = self.dropout(output)

        # extract entity
        e1_h = max_pooling(output, e1_mask)  # (B, H)
        e2_h = max_pooling(output, e2_mask)  # (B, H)
        ent = torch.cat([e1_h, e2_h], dim=-1)  # (B, 2H)
        return ent

    def forward(self, inputs, labels, ite=None):
        input_ids, attention_mask, e1_mask, e2_mask, valid_ids, mlm_input_ids, dep_matrix = inputs
        re_labels, mlm_labels = labels

        # unmask_input_ids = input_ids.masked_fill((mlm_input_ids != 103) & (mlm_input_ids != 0), 103)
        # unmask_labels = input_ids.masked_fill((mlm_input_ids == 103) | (input_ids == 0), -100)

        re_outputs = self.re_encoder(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        re_hidden_states = re_outputs.hidden_states
        re_output = re_outputs.last_hidden_state  # (B, L, H)

        # unmask_outputs = self.unmask_encoder(unmask_input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # unmask_hidden_states = unmask_outputs.hidden_states
        # unmask_output = unmask_outputs.last_hidden_state  # (B, L, H)

        re_ent = self.extract_feature(re_output, valid_ids, dep_matrix, e1_mask, e2_mask)
        # unmask_ent = self.extract_feature(unmask_output, valid_ids, dep_matrix, e1_mask, e2_mask)

        # re_classifier
        logits = self.classifier(re_ent)  # (B, C)
        ce = self.criterion(logits, re_labels)

        # unmask_logits = self.classifier(unmask_ent)  # (B, C)
        # kl = self.kl_loss(
        #     F.log_softmax(logits, dim=-1),
        #     F.softmax(unmask_logits, dim=-1)
        # )
        # kl = self.criterion(unmask_logits, re_labels)
        # hidden_states = re_hidden_states + (ent, logits)
        hidden_states = (re_ent, logits)

        # loss = ce + 0 * kl
        return hidden_states, [logits], [ce]

        # if torch.max(mlm_labels) > 0 and self.training:
        if torch.max(mlm_labels) > 0:
            mlm_outputs = self.re_encoder(mlm_input_ids, attention_mask=attention_mask)
            mlm_output = mlm_outputs.last_hidden_state  # (B, L, H)

            unmask_outputs = self.re_encoder(unmask_input_ids, attention_mask=attention_mask)
            unmask_output = unmask_outputs.last_hidden_state  # (B, L, H)

            mlm_logits, mlm_loss = self.mlm_classifier(labels=mlm_labels, hidden_state=mlm_output)
            unmask_logits, unmask_loss = self.mlm_classifier(labels=unmask_labels, hidden_state=unmask_output)

            # if self.lambda_ <= 0 and self.training:
            #     self.mlm_optimizer.zero_grad()
            #     mlm_classifier_loss = mlm_loss + unmask_loss
            #     mlm_classifier_loss.backward(retain_graph=True)
            #     self.mlm_optimizer.step()
            #     mlm_logits, mlm_loss = self.mlm_classifier(labels=mlm_labels, hidden_state=mlm_output)
            #     unmask_logits, unmask_loss = self.mlm_classifier(labels=unmask_labels, hidden_state=unmask_output)

            loss = re_loss + self.lambda_ * unmask_loss
            return hidden_states, [re_logits, mlm_logits, unmask_logits], [loss, re_loss, mlm_loss, unmask_loss]
        else:
            return hidden_states, [re_logits], [re_loss]


class REMTLModel(REModel):
    def __init__(self, model_name, output_dim, num_gcn_layers, num_docs):
        super(REMTLModel, self).__init__(model_name, output_dim, num_gcn_layers)
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(768 * 2, 768),
                nn.ReLU(),
                nn.Linear(768, output_dim),
            ) for _ in range(num_docs)
        ])

    def forward(self, inputs, labels, ite=None):
        input_ids, attention_mask, e1_mask, e2_mask, valid_ids, mlm_input_ids, dep_matrix, doc = inputs
        re_labels, mlm_labels = labels

        re_outputs = self.re_encoder(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        re_hidden_states = re_outputs.hidden_states
        re_output = re_outputs.last_hidden_state  # (B, L, H)

        re_ent = self.extract_feature(re_output, valid_ids, dep_matrix, e1_mask, e2_mask)

        # re_classifier
        logits = self.classifiers[doc](re_ent)  # (B, C)
        # logits = self.classifier(re_ent)
        ce = self.criterion(logits, re_labels)

        hidden_states = (re_ent, logits)

        return hidden_states, [logits], [ce]


# class REModel(nn.Module):
#     def __init__(self, model_name, output_dim, num_gcn_layers, lambda_):
#         super(REModel, self).__init__()
#         self.re_encoder = AutoModel.from_pretrained(model_name)
#         gcn_layer = GraphConvolution(768, 768)
#         self.lambda_ = lambda_
#         self.gcn_layers = nn.ModuleList([copy.deepcopy(gcn_layer) for _ in range(num_gcn_layers)])
#         self.dropout = nn.Dropout()
#         self.re_classifier = nn.Sequential(
#             nn.Linear(768 * 2, 768),
#             nn.ReLU(),
#             nn.Linear(768, output_dim),
#         )
#         config = AutoConfig.from_pretrained(model_name)
#         self.mlm_classifier = MaskedLMClassifier(config)
#
#         self.criterion = nn.CrossEntropyLoss()
#         self.mlm_optimizer = torch.optim.Adam(self.mlm_classifier.parameters(), lr=1e-3)
#         self.dataset = REDataset
#
#     def forward(self, inputs, labels, ite=None):
#         input_ids, attention_mask, e1_mask, e2_mask, valid_ids, mlm_input_ids, dep_matrix = inputs
#         re_labels, mlm_labels = labels
#
#         unmask_input_ids = input_ids.masked_fill((mlm_input_ids != 103) & (mlm_input_ids != 0), 103)
#         unmask_labels = input_ids.masked_fill((mlm_input_ids == 103) | (input_ids == 0), -100)
#
#         if self.lambda_ > 0 and self.training:
#             input_ids = torch.cat([input_ids, unmask_input_ids])
#             attention_mask = torch.cat([attention_mask, attention_mask])
#             valid_ids = torch.cat([valid_ids, valid_ids])
#             e1_mask = torch.cat([e1_mask, e1_mask])
#             e2_mask = torch.cat([e2_mask, e2_mask])
#             re_labels = torch.cat([re_labels, re_labels])
#
#         re_outputs = self.re_encoder(input_ids, attention_mask=attention_mask, output_hidden_states=True)
#         re_hidden_states = re_outputs.hidden_states
#         re_output = re_outputs.last_hidden_state  # (B, L, H)
#
#         if torch.max(valid_ids) > 0:
#             # filter valid output
#             re_output = valid_filter(re_output, valid_ids)  # (B, L, H)
#
#             # gcn
#             for gcn_layer in self.gcn_layers:
#                 re_output = gcn_layer(re_output, dep_matrix)  # (B, L, H)
#             re_output = self.dropout(re_output)
#         else:
#             re_output = self.dropout(re_output)
#
#         # extract entity
#         e1_h = max_pooling(re_output, e1_mask)  # (B, H)
#         e2_h = max_pooling(re_output, e2_mask)  # (B, H)
#         ent = torch.cat([e1_h, e2_h], dim=-1)  # (B, 2H)
#
#         # re_classifier
#         re_logits = self.re_classifier(ent)  # (B, C)
#         re_loss = self.criterion(re_logits, re_labels)
#
#         hidden_states = re_hidden_states + (ent, re_logits)
#
#         # if torch.max(mlm_labels) > 0 and self.training:
#         return hidden_states, [re_logits], [re_loss]


class REGSNModel(REModel):
    def __init__(self, model_name, output_dim, num_gcn_layers, gradient_reverse):
        super(REGSNModel, self).__init__(model_name, output_dim, num_gcn_layers, gradient_reverse)

    def forward(self, inputs, labels):
        input_ids, input_mask, e1_mask, e2_mask, mlm_input_ids, valid_ids, dep_matrix = inputs
        labels = labels[0]

        # relation extraction
        outputs = self.re_encoder(input_ids, attention_mask=input_mask, output_hidden_states=True)
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

        # re_classifier
        logits = self.re_classifier(ent)  # (B, C)
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
        outputs = self.re_encoder(input_ids, attention_mask=input_mask, output_hidden_states=True)
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

        # re_classifier
        logits = self.re_classifier(ent)
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
        outputs = self.re_encoder(input_ids, attention_mask=input_mask, output_hidden_states=True)
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

        # re_classifier
        logits = self.re_classifier(ent)  # (B, C)
        loss = self.criterion(logits, labels)

        return hidden_states, logits, [loss]


class REDomainMMDModel(REModel):
    def __init__(self, model_name, output_dim, num_gcn_layers, lambda_):
        super(REDomainMMDModel, self).__init__(model_name, output_dim, num_gcn_layers, lambda_)
        self.domain_classifier = nn.Sequential(
            nn.Linear(768 * 2, 384),
            nn.ReLU(),
            nn.Linear(384, 2)
        )
        self.domain_optimizer = torch.optim.Adam(self.domain_classifier.parameters(), lr=1e-5)
        self.dataset = REMDomainAdaptationDataset
        self.mmd = MMDLoss()

    def forward(self, inputs, labels, ite=None):
        input_ids, attention_mask, e1_mask, e2_mask, valid_ids, dep_matrix = inputs
        labels, doc = labels
        outputs = self.re_encoder(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        encoder_hidden_states = outputs.hidden_states
        encoder_output = outputs.last_hidden_state  # (B, L, H)

        if torch.max(valid_ids) > 0:
            # filter valid output
            valid_output = valid_filter(encoder_output, valid_ids)  # (B, L, H)

            # gcn
            gcn_output = valid_output
            for gcn_layer in self.gcn_layers:
                gcn_output = gcn_layer(gcn_output, dep_matrix)  # (B, L, H)
            output = self.dropout(gcn_output)
        else:
            output = self.dropout(encoder_output)

        # extract entity
        e1_h = max_pooling(output, e1_mask)  # (B, H)
        e2_h = max_pooling(output, e2_mask)  # (B, H)
        ent = torch.cat([e1_h, e2_h], dim=-1)  # (B, 2H)
        ent = self.dropout(ent)

        source_labels = labels[torch.where(doc == 0)]
        target_labels = labels[torch.where(doc == 1)]

        source_ent = ent[torch.where(doc == 0)]
        target_ent = ent[torch.where(doc == 1)]
        mmd_loss = self.mmd(source_ent, target_ent)

        logits = self.re_classifier(ent)
        ce_loss = self.criterion(logits, labels)

        # source_logits = self.re_classifier(source_ent)
        # ce_loss = self.criterion(source_logits, source_labels)

        self.domain_optimizer.zero_grad()
        domain_logits = self.domain_classifier(ent)

        hidden_states = encoder_hidden_states + (ent, logits)
        if self.training:
            domain_loss = self.criterion(domain_logits, doc)
            domain_loss.backward(retain_graph=True)
            self.domain_optimizer.step()

            loss = ce_loss + self.lambda_ * mmd_loss
            return hidden_states, [logits, domain_logits], [loss, ce_loss, mmd_loss]
        else:
            return hidden_states, [logits, domain_logits], [ce_loss]


class REDomainGRLModel(REModel):
    def __init__(self, model_name, output_dim, num_gcn_layers, lambda_):
        super(REDomainGRLModel, self).__init__(model_name, output_dim, num_gcn_layers)

        self.domain_classifier = nn.Sequential(
            nn.Linear(768 * 2, 384),
            nn.ReLU(),
            nn.Linear(384, 2)
        )
        self.domain_optimizer = torch.optim.Adam(self.domain_classifier.parameters(), lr=0.01)
        self.dataset = REMDomainAdaptationDataset
        self.lambda_ = lambda_

    def forward(self, inputs, labels, ite=None):
        input_ids, attention_mask, e1_mask, e2_mask, valid_ids, dep_matrix = inputs
        labels, doc = labels
        outputs = self.re_encoder(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        encoder_hidden_states = outputs.hidden_states
        encoder_output = outputs.last_hidden_state  # (B, L, H)

        if torch.max(valid_ids) > 0:
            # filter valid output
            valid_output = valid_filter(encoder_output, valid_ids)  # (B, L, H)

            # gcn
            gcn_output = valid_output
            for gcn_layer in self.gcn_layers:
                gcn_output = gcn_layer(gcn_output, dep_matrix)  # (B, L, H)
            output = self.dropout(gcn_output)
        else:
            output = self.dropout(encoder_output)

        # extract entity
        e1_h = max_pooling(output, e1_mask)  # (B, H)
        e2_h = max_pooling(output, e2_mask)  # (B, H)
        ent = torch.cat([e1_h, e2_h], dim=-1)  # (B, 2H)
        ent = self.dropout(ent)

        logits = self.re_classifier(ent)
        ce_loss = self.criterion(logits, labels)

        if self.lambda_ is not None:
            grl_output = self.grl(ent)
        else:
            lambda_ = 2 / (1 + math.exp(-0.1 * ite)) - 1 if ite is not None else 0.1
            grl_output = GRL.apply(ent, lambda_)

        domain_logits = self.domain_classifier(grl_output)
        domain_loss = self.criterion(domain_logits, doc)

        hidden_states = encoder_hidden_states + (ent, logits)
        if self.training:
            loss = ce_loss + self.lambda_ * domain_loss
            return hidden_states, [logits, domain_logits], [loss, ce_loss, domain_loss]
        else:
            return hidden_states, [logits, domain_logits], [ce_loss]


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
        for param in self.re_encoder.parameters():
            param.requires_grad = True

    def forward(self, inputs, labels, ite=None):
        # if ite is not None:
        #     self.Lambda = max(1 - ite / 10, 0)
        input_ids, input_mask, e1_mask, e2_mask = inputs
        labels = labels[0]
        outputs = self.re_encoder(input_ids, attention_mask=input_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        encoder_output = outputs.last_hidden_state  # (B, L, H)

        # outputs_aug = self.re_encoder(input_ids, attention_mask=input_mask, output_hidden_states=True)
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

        logits = self.re_classifier(ent)
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

            src_logits = self.re_classifier(src_ent)
            tgt_logits = self.re_classifier(tgt_ent)

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
            logits = self.re_classifier(ent)
            loss = self.criterion(logits, labels)
            return hidden_states, logits, [loss]
