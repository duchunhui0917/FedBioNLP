import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
from entmax import entmax_bisect


class LSR(nn.Module):
    def __init__(self, dim_hidden, num_layers=2, num_first_gcn_layers=2, num_second_gcn_layers=4, num_trees=2,
                 alpha=1.5):
        super(LSR, self).__init__()
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.num_first_gcn_layers = num_first_gcn_layers
        self.num_second_gcn_layers = num_second_gcn_layers
        self.num_trees = num_trees
        self.alpha = alpha

        self.dep_inducer = nn.ModuleList([
            StructuredAttention(self.dim_hidden, self.num_trees) for _ in range(self.num_layers)
        ])
        self.multi_gcn_backbone = nn.ModuleList([
            nn.ModuleList([
                MultiGraphConvolution(self.dim_hidden, self.num_first_gcn_layers, self.num_trees),
                MultiGraphConvolution(self.dim_hidden, self.num_second_gcn_layers, self.num_trees)
            ])
            for _ in range(self.num_layers)
        ])
        self.agg_gcn_backbone = nn.Linear(2 * self.num_layers * self.dim_hidden, self.dim_hidden)

        self.dropout = nn.Dropout()

    def forward(self, encoder_outputs, mask):
        adj_list = None
        outputs = encoder_outputs
        gcn_outputs = []

        for i in range(self.num_layers):
            # induce dependency matrix
            dep_inducer = self.dep_inducer[i]
            adj_list = dep_inducer(outputs, mask)
            # adaptive pruning by entmax
            for j in range(len(adj_list)):
                adj_list[j] = entmax_bisect(adj_list[j], self.alpha)

            gcn = self.multi_gcn_backbone[i]
            outputs0 = gcn[0](adj_list, outputs)
            gcn_outputs.append(outputs0)
            outputs1 = gcn[1](adj_list, outputs)
            gcn_outputs.append(outputs1)

        aggregate_output = torch.cat(gcn_outputs, dim=2)
        gcn_output = self.agg_gcn_backbone(aggregate_output)
        adj = torch.stack(adj_list, dim=1).sum(dim=1)

        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)

        return gcn_output, mask


class MultiGraphConvolution(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, dim_hidden, num_layers, num_heads):
        super(MultiGraphConvolution, self).__init__()
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.head_dim = self.dim_hidden // self.num_layers
        self.num_heads = num_heads
        self.gcn_drop = nn.Dropout()

        # gcn layer
        self.Linear = nn.Linear(self.dim_hidden * self.num_heads, self.dim_hidden)
        self.weight_list = nn.ModuleList()

        for i in range(self.num_heads):
            for j in range(self.num_layers):
                self.weight_list.append(nn.Linear(self.dim_hidden + self.head_dim * j, self.head_dim))

    def forward(self, adj_list, gcn_inputs):

        multi_head_list = []
        for i in range(self.num_heads):
            adj = adj_list[i]
            denom = adj.sum(2).unsqueeze(2) + 1
            outputs = gcn_inputs
            cache_list = [outputs]
            output_list = []
            for l in range(self.num_layers):
                index = i * self.num_layers + l
                Ax = adj.bmm(outputs)
                AxW = self.weight_list[index](Ax)
                AxW = AxW + self.weight_list[index](outputs)  # self loop
                AxW = AxW / denom
                gAxW = F.relu(AxW)
                cache_list.append(gAxW)
                outputs = torch.cat(cache_list, dim=2)
                output_list.append(self.gcn_drop(gAxW))

            gcn_outputs = torch.cat(output_list, dim=2)
            gcn_outputs = gcn_outputs + gcn_inputs

            multi_head_list.append(gcn_outputs)

        final_output = torch.cat(multi_head_list, dim=2)
        out = self.Linear(final_output)

        return out


class GraphConvolution(nn.Module):
    """
    Simple GCN layer
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, text, adj):
        hidden = torch.matmul(text.float(), self.weight.float())
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj.float(), hidden) / denom
        if self.bias is not None:
            output = output + self.bias

        return F.relu(output.type_as(text))


class TypeGraphConvolution(nn.Module):
    """
    Type GCN layer
    """

    def __init__(self, in_features, out_features, bias=True):
        super(TypeGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, text, adj, dep_embed):
        batch_size, max_len, feat_dim = text.shape
        val_us = text.unsqueeze(dim=2)
        val_us = val_us.repeat(1, 1, max_len, 1)
        val_sum = val_us + dep_embed
        adj_us = adj.unsqueeze(dim=-1)
        adj_us = adj_us.repeat(1, 1, 1, feat_dim)
        hidden = torch.matmul(val_sum.float(), self.weight.float())
        output = hidden.transpose(1, 2) * adj_us.float()
        output = torch.sum(output, dim=2)

        if self.bias is not None:
            output = output + self.bias

        return F.relu(output.type_as(text))


class StructuredAttention(nn.Module):
    def __init__(self, dim_hidden, num_trees):
        super(StructuredAttention, self).__init__()
        self.str_dim_size = dim_hidden
        self.dim_hidden = dim_hidden
        self.num_trees = num_trees

        self.linear_roots = nn.ModuleList(
            [nn.Linear(self.dim_hidden, 1) for _ in range(self.num_trees)]
        )

        self.attn = MultiHeadAttention(self.num_trees, self.dim_hidden)

    def forward(self, input, src_mask):
        """

        Args:
            input: (B, L, H)
            src_mask:

        Returns:

        """
        batch_size, token_size, dim_size = input.size()

        attn_tensor = self.attn(input, input, src_mask)  # (B, L, H, H)
        attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]

        adj_list = list()

        for i in range(self.num_trees):
            f_i = self.linear_roots[i](input).squeeze(-1)  # (B, L, H)
            f_ij = attn_adj_list[i]

            mask = torch.ones(f_ij.size(1), f_ij.size(1)) - torch.eye(f_ij.size(1), f_ij.size(1))
            mask = mask.unsqueeze(0).expand(f_ij.size(0), mask.size(0), mask.size(1)).cuda()
            A_ij = torch.exp(f_ij) * mask

            tmp = torch.sum(A_ij, dim=1)
            res = torch.zeros(batch_size, token_size, token_size).cuda()

            res.as_strided(tmp.size(), [res.stride(0), res.size(2) + 1]).copy_(tmp)
            L_ij = -A_ij + res  # A_ij has 0s as diagonals

            L_ij_bar = L_ij
            L_ij_bar[:, 0, :] = f_i

            LLinv = torch.inverse(L_ij_bar)

            d0 = f_i * LLinv[:, :, 0]

            LLinv_diag = torch.diagonal(LLinv, dim1=-2, dim2=-1).unsqueeze(2)

            tmp1 = (A_ij.transpose(1, 2) * LLinv_diag).transpose(1, 2)
            tmp2 = A_ij * LLinv.transpose(1, 2)

            temp11 = torch.zeros(batch_size, token_size, 1)
            temp21 = torch.zeros(batch_size, 1, token_size)

            temp12 = torch.ones(batch_size, token_size, token_size - 1)
            temp22 = torch.ones(batch_size, token_size - 1, token_size)

            mask1 = torch.cat([temp11, temp12], 2).cuda()
            mask2 = torch.cat([temp21, temp22], 1).cuda()

            dx = mask1 * tmp1 - mask2 * tmp2

            d = torch.cat([d0.unsqueeze(1), dx], dim=1)
            df = d.transpose(1, 2)

            att = df[:, :, 1:]

            adj_list.append(att)

        return adj_list


def pool(h, mask, type='max'):
    if type == 'max':
        h = h.masked_fill(mask, -constant.INFINITY_NUMBER)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)


def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    return h0.cuda(), c0.cuda()


def attention(query, key, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)

        n_batches = query.size(0)

        query, key = [l(x).view(n_batches, -1, self.h, self.d_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]
        attn = attention(query, key, mask=mask, dropout=self.dropout)

        return attn
