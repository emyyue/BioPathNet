import collections.abc as container_abcs

import torch
from torch import nn
from torch.nn import functional as F

from torch_scatter import scatter_add, scatter_mean, scatter_max, scatter_min

from torchdrug import layers, utils, models
from torchdrug.layers import functional

from torchdrug.core import Registry as R
from torchdrug.core import core


from nbfnet.extension import sparse


class GeneralizedRelationalConv(layers.MessagePassingBase):

    eps = 1e-6

    message2mul = {
        "transe": "add",
        "distmult": "mul",
    }

    def __init__(self, input_dim, output_dim, num_relation, query_input_dim, message_func="distmult",
                 aggregate_func="pna", layer_norm=False, activation="relu", dependent=True):
        super(GeneralizedRelationalConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.query_input_dim = query_input_dim
        self.message_func = message_func
        self.aggregate_func = aggregate_func
        self.dependent = dependent

        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        if self.aggregate_func == "pna":
            self.linear = nn.Linear(input_dim * 13, output_dim)
        else:
            self.linear = nn.Linear(input_dim * 2, output_dim)
        if dependent:
            self.relation_linear = nn.Linear(query_input_dim, num_relation * input_dim)
        else:
            self.relation = nn.Embedding(num_relation, input_dim)

    def message(self, graph, input):
        assert graph.num_relation == self.num_relation

        batch_size = len(graph.query)
        node_in, node_out, relation = graph.edge_list.t()
        if self.dependent:
            relation_input = self.relation_linear(graph.query).view(batch_size, self.num_relation, self.input_dim)
        else:
            relation_input = self.relation.weight.expand(batch_size, -1, -1)
        relation_input = relation_input.transpose(0, 1)
        node_input = input[node_in]
        edge_input = relation_input[relation]

        if self.message_func == "transe":
            message = edge_input + node_input
        elif self.message_func == "distmult":
            message = edge_input * node_input
        elif self.message_func == "rotate":
            node_re, node_im = node_input.chunk(2, dim=-1)
            edge_re, edge_im = edge_input.chunk(2, dim=-1)
            message_re = node_re * edge_re - node_im * edge_im
            message_im = node_re * edge_im + node_im * edge_re
            message = torch.cat([message_re, message_im], dim=-1)
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)
        message = torch.cat([message, graph.boundary])

        return message

    def aggregate(self, graph, message):
        node_out = graph.edge_list[:, 1]
        node_out = torch.cat([node_out, torch.arange(graph.num_node, device=graph.device)])
        edge_weight = torch.cat([graph.edge_weight, torch.ones(graph.num_node, device=graph.device)])
        edge_weight = edge_weight.unsqueeze(-1).unsqueeze(-1)
        degree_out = graph.degree_out.unsqueeze(-1).unsqueeze(-1) + 1

        if self.aggregate_func == "sum":
            update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
        elif self.aggregate_func == "mean":
            update = scatter_mean(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
        elif self.aggregate_func == "max":
            update = scatter_max(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)[0]
        elif self.aggregate_func == "pna":
            mean = scatter_mean(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
            sq_mean = scatter_mean(message ** 2 * edge_weight, node_out, dim=0, dim_size=graph.num_node)
            max = scatter_max(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)[0]
            min = scatter_min(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)[0]
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2)
            scale = degree_out.log()
            scale = scale / scale.mean()
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
            update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        else:
            raise ValueError("Unknown aggregation function `%s`" % self.aggregate_func)

        return update

    def message_and_aggregate(self, graph, input):
        if graph.requires_grad or self.message_func == "rotate":
            return super(GeneralizedRelationalConv, self).message_and_aggregate(graph, input)

        assert graph.num_relation == self.num_relation

        batch_size = len(graph.query)
        input = input.flatten(1)
        boundary = graph.boundary.flatten(1)

        degree_out = graph.degree_out.unsqueeze(-1) + 1
        if self.dependent:
            relation_input = self.relation_linear(graph.query).view(batch_size, self.num_relation, self.input_dim)
            relation_input = relation_input.transpose(0, 1).flatten(1)
        else:
            relation_input = self.relation.weight.repeat(1, batch_size)
        adjacency = graph.adjacency.transpose(0, 1)

        if self.message_func in self.message2mul:
            mul = self.message2mul[self.message_func]
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)
        if self.aggregate_func == "sum":
            update = functional.generalized_rspmm(adjacency, relation_input, input, sum="add", mul=mul)
            update = update + boundary
        elif self.aggregate_func == "mean":
            update = functional.generalized_rspmm(adjacency, relation_input, input, sum="add", mul=mul)
            update = (update + boundary) / degree_out
        elif self.aggregate_func == "max":
            update = functional.generalized_rspmm(adjacency, relation_input, input, sum="max", mul=mul)
            update = torch.max(update, boundary)
        elif self.aggregate_func == "pna":
            sum = functional.generalized_rspmm(adjacency, relation_input, input, sum="add", mul=mul)
            sq_sum = functional.generalized_rspmm(adjacency, relation_input ** 2, input ** 2, sum="add", mul=mul)
            max = functional.generalized_rspmm(adjacency, relation_input, input, sum="max", mul=mul)
            min = functional.generalized_rspmm(adjacency, relation_input, input, sum="min", mul=mul)
            mean = (sum + boundary) / degree_out
            sq_mean = (sq_sum + boundary ** 2) / degree_out
            max = torch.max(max, boundary)
            min = torch.min(min, boundary)
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2)
            scale = degree_out.log()
            scale = scale / scale.mean()
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
            update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        else:
            raise ValueError("Unknown aggregation function `%s`" % self.aggregate_func)

        return update.view(len(update), batch_size, -1)

    def combine(self, input, update):
        output = self.linear(torch.cat([input, update], dim=-1))
        if self.layer_norm:
            output = self.layer_norm(output)
        if self.activation:
            output = self.activation(output)
        return output

class RelationalGraphConv(layers.RelationalGraphConv):

    def __init__(self, input_dim, output_dim, num_relation, inner_dim=None, edge_input_dim=None,
                 aggregation="sum", batch_norm=False, pair_norm=False, layer_norm=False, activation="relu",
                 second_order=False, diagonal=False, diagonal_init=False):
        nn.Module.__init__(self)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.edge_input_dim = edge_input_dim
        self.aggregation = aggregation
        self.second_order = second_order
        self.diagonal = diagonal
        self.diagonal_init = diagonal_init

        assert not (batch_norm and pair_norm)
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None
        if pair_norm:
            self.pair_norm = layers.PairNorm()
        else:
            self.pair_norm = None
        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        if inner_dim:
            self.pre_linear = nn.Linear(input_dim, inner_dim)
        else:
            self.pre_linear = None
        inner_dim = inner_dim or input_dim
        self.inner_dim = inner_dim
        self.self_loop = nn.Linear(input_dim, output_dim)
        self.linear = nn.Linear(num_relation * inner_dim, output_dim)
        if edge_input_dim:
            self.edge_linear = nn.Linear(edge_input_dim, inner_dim)
        else:
            self.edge_linear = None

        assert not (diagonal and diagonal_init)

        if diagonal:
            assert input_dim == inner_dim == output_dim
            self.linear_diag = nn.Parameter(torch.zeros(num_relation * input_dim))
            self.self_loop_diag = nn.Parameter(torch.zeros(input_dim))
            nn.init.uniform_(self.linear_diag, 1 / (num_relation + 1) - 1e-3, 1 / (num_relation + 1) + 1e-3)
            nn.init.uniform_(self.self_loop_diag, 1 / (num_relation + 1) - 1e-3, 1 / (num_relation + 1) + 1e-3)
        if self.diagonal_init:
            assert input_dim == inner_dim == output_dim
            diag = torch.diag(torch.ones(input_dim) / (num_relation + 1))
            with torch.no_grad():
                self.linear.weight += diag.repeat(1, num_relation)
                self.self_loop.weight += diag

        assert aggregation in ["sum", "mean", "max", "min"]

    def message_and_aggregate(self, graph, input):
        # torch.spmm doesn't support gradient for sparse tensor
        if graph.requires_grad:
            raise NotImplementedError
        if self.pre_linear:
            input = self.pre_linear(input)
            input = self.activation(input)

        assert graph.num_relation == self.num_relation

        node_in, node_out, relation = graph.edge_list.t()
        node_out = node_out * self.num_relation + relation
        if self.aggregation == "mean":
            degree_out = scatter_add(graph.edge_weight, node_out, dim_size=graph.num_node * graph.num_relation)
            edge_weight = graph.edge_weight / degree_out[node_out]
            aggregation = "sum"
        else:
            edge_weight = graph.edge_weight
            aggregation = self.aggregation

        assert (node_in < graph.num_node).all()
        assert (node_out < graph.num_node * graph.num_relation).all()
        adjacency = utils.sparse_coo_tensor(torch.stack([node_in, node_out]), edge_weight,
                                            (graph.num_node, graph.num_node * graph.num_relation))

        update = sparse.generalized_spmm(adjacency.t(), input.flatten(1), addition=aggregation)
        if self.second_order:
            node_in = node_in * self.num_relation + relation
            adjacency2 = utils.sparse_coo_tensor(torch.stack([node_in, node_out]), edge_weight,
                                                 (graph.num_node * graph.num_relation,
                                                  graph.num_node * graph.num_relation))
            update = sparse.generalized_spmm(adjacency2.t(), update, addition=aggregation)

        if self.edge_linear:
            edge_input = graph.edge_feature.float()
            if self.edge_linear.in_features > self.edge_linear.out_features:
                edge_input = self.edge_linear(edge_input)
            edge_update = scatter_add(edge_input * edge_weight, node_out, dim=0,
                                      dim_size=graph.num_node * graph.num_relation)
            if self.edge_linear.in_features <= self.edge_linear.out_features:
                edge_update = self.edge_linear(edge_update)
            update += edge_update

        update = update.view(graph.num_node.item(), self.num_relation, -1, self.inner_dim)

        if input.ndim == 2:
            return update.flatten(1)
        else:
            return update.transpose(1, 2).flatten(2)

    def forward(self, graph, input):
        if not self.diagonal or graph.num_node * graph.num_relation < graph.num_edge:
            return super(RelationalGraphConv, self).forward(graph, input)

        input_ = input
        if self.pre_linear:
            input = self.pre_linear(input)
            input = self.activation(input)

        assert graph.num_relation == self.num_relation

        node_in, node_out, relation = graph.edge_list.t()
        linear_diag = self.linear_diag.view(self.num_relation, -1)
        if input.ndim == 3:
            linear_diag = linear_diag.unsqueeze(1)
        message = input[node_in] * linear_diag[relation]
        if self.edge_linear:
            message += self.edge_linear(graph.edge_feature.float())

        if self.aggregation == "mean":
            node_out_ = node_out * self.num_relation + relation
            degree_out = scatter_add(graph.edge_weight, node_out_, dim_size=graph.num_node * graph.num_relation)
            edge_weight = graph.edge_weight / degree_out[node_out_]
        else:
            edge_weight = graph.edge_weight
        if input.ndim == 2:
            edge_weight = edge_weight.unsqueeze(-1)
        else:
            edge_weight = edge_weight.unsqueeze(-1).unsqueeze(-1)

        import pdb; pdb.set_trace()
        if self.aggregation in ["mean", "sum"]:
            update = scatter_add(message * edge_weight, node_out*(relation+1), dim=0, dim_size=graph.num_node*graph.num_relation)
        elif self.aggregation == "max":
            update = scatter_max(message * edge_weight, node_out*(relation+1), dim=0, dim_size=graph.num_node*graph.num_relation)[0]
        elif self.aggregation == "min":
            update = scatter_min(message * edge_weight, node_out*(relation+1), dim=0, dim_size=graph.num_node*graph.num_relation)[0]
        relation_x_node = torch.repeat_interleave(
            torch.arange(graph.num_node, device=self.device), graph.num_relation, dim=0
        ) # alternative do this with reshape, but not sure of memory layout guarantees at this stage in the code
        update = scatter_add(update, relation_x_node, dim=0, dim_size=graph.num_node)

        assert not self.second_order

        x = update + self.self_loop(input_)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.pair_norm:
            x = self.pair_norm(x)
        if self.layer_norm:
            x = self.layer_norm(x)
        if self.activation:
            output = self.activation(x)
        else:
            output = x
        return output

    def combine(self, input, update):
        if self.diagonal:
            x = self.linear_diag * update
            x = x.view(*x.shape[:-1], self.num_relation, self.output_dim).sum(dim=-2)
            x = x + self.self_loop_diag * input
        else:
            x = self.linear(update) + self.self_loop(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.pair_norm:
            x = self.pair_norm(x)
        if self.layer_norm:
            x = self.layer_norm(x)
        if self.activation:
            output = self.activation(x)
        else:
            output = x
        return output


@R.register("model.BatchedRGCN")
class RelationalGraphConvolutionalNetwork(models.RGCN):

    def __init__(self, input_dim, hidden_dims, num_relation, inner_dims=None, edge_input_dim=None, short_cut=False,
                 batch_norm=False, learnable_short_cut=False, pair_norm=False, layer_norm=False, activation="relu",
                 concat_hidden=False, readout="sum", aggregation="sum", second_order=False, diagonal=False,
                 diagonal_init=False, dropout_edge=0):
        nn.Module.__init__(self)

        if not isinstance(hidden_dims, container_abcs.Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1)
        self.dims = [input_dim] + list(hidden_dims)
        self.num_relation = num_relation
        self.short_cut = short_cut
        self.learnable_short_cut = learnable_short_cut
        self.concat_hidden = concat_hidden
        self.dropout_edge = dropout_edge

        self.layers = nn.ModuleList()
        inner_dims = inner_dims or [None] * len(hidden_dims)
        for i in range(len(self.dims) - 1):
            self.layers.append(RelationalGraphConv(self.dims[i], self.dims[i + 1], num_relation, inner_dims[i],
                                                   edge_input_dim, aggregation, batch_norm, pair_norm, layer_norm,
                                                   activation, second_order, diagonal, diagonal_init))
        if self.learnable_short_cut:
            weight = torch.empty(len(self.dims) - 1)
            nn.init.normal_(weight, 1, 1.0e-3)
            self.weight = nn.Parameter(weight)

        if readout == "sum":
            self.readout = layers.SumReadout()
        elif readout == "mean":
            self.readout = layers.MeanReadout()
        else:
            raise ValueError("Unknown readout `%s`" % readout)

    def forward(self, graph, input, all_loss=None, metric=None):
        """"""
        if self.dropout_edge > 0:
            graph = edge_dropout(graph, self.dropout_edge)

        hiddens = []
        layer_input = input

        for i, layer in enumerate(self.layers):
            hidden = layer(graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                if self.learnable_short_cut:
                    hidden = hidden + layer_input * self.weight[i]
                else:
                    hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]
        graph_feature = self.readout(graph, node_feature)

        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature
        }

@R.register("model.RotatEScore")
class RotatEScore(models.RotatE, core.Configurable):

    def __init__(self, num_entity, num_relation, embedding_dim, max_score=12):
        super(RotatEScore, self).__init__(num_entity, num_relation, embedding_dim, max_score)
        self.entity_norm = nn.LayerNorm(embedding_dim)
        self.relation_norm = nn.LayerNorm(embedding_dim // 2)

        nn.init._no_grad_uniform_(self.entity_norm.weight, self.max_score * 2 / embedding_dim * 0.5,
                                  self.max_score * 2 / embedding_dim * 1.5)
        nn.init._no_grad_uniform_(self.relation_norm.weight, self.max_score * 2 / embedding_dim * 0.5,
                                  self.max_score * 2 / embedding_dim * 1.5)

        # nn.init.kaiming_uniform_(self.entity.weight, a=math.sqrt(5), mode="fan_in")
        # nn.init.kaiming_uniform_(self.relation.weight, a=math.sqrt(5), mode="fan_in")

    def forward(self, head, tail, relation, h_index=None, t_index=None, r_index=None, all_loss=None, metric=None):
        head = self.entity_norm(head)
        tail = self.entity_norm(tail)
        relation = self.relation_norm(relation)
        if h_index is not None:
            h = head[h_index]
            r = relation[r_index]
            t = tail[t_index]
        else:
            h = head
            r = relation
            t = tail
        r = r * self.relation_scale

        h_re, h_im = h.chunk(2, dim=-1)
        r_re, r_im = torch.cos(r), torch.sin(r)
        t_re, t_im = t.chunk(2, dim=-1)

        x_re = h_re * r_re - h_im * r_im - t_re
        x_im = h_re * r_im + h_im * r_re - t_im
        x = torch.stack([x_re, x_im], dim=-1)
        x = x.norm(p=2, dim=-1).sum(dim=-1)
        score = self.max_score - x
        return score

    def flip_relation(self, relation):
        return -relation


@R.register("model.TransEScore")
class TransEScore(models.TransE, core.Configurable):

    def __init__(self, num_entity, num_relation, embedding_dim, max_score=12, checkpoint=None, grad=True,
                 learnable_score=False):
        super(TransEScore, self).__init__(num_entity, num_relation, embedding_dim, max_score)
        self.entity_norm = nn.LayerNorm(embedding_dim)
        self.relation_norm = nn.LayerNorm(embedding_dim)
        self.grad = grad
        self.learnable_score = learnable_score

        nn.init._no_grad_uniform_(self.entity_norm.weight, self.max_score / embedding_dim * 0.5, self.max_score / embedding_dim * 1.5)
        nn.init._no_grad_uniform_(self.relation_norm.weight, self.max_score / embedding_dim * 0.5, self.max_score / embedding_dim * 1.5)

        if learnable_score:
            self.max_score = nn.Parameter(torch.tensor(self.max_score, dtype=torch.float))

        if checkpoint:
            state = torch.load(checkpoint)
            with torch.no_grad():
                self.entity.weight.copy_(state["model"]["model.entity.weight"].to(self.device))
                self.relation.weight.copy_(state["model"]["model.relation.weight"].to(self.device))
        else:
            nn.init.kaiming_uniform_(self.entity, a=math.sqrt(5), mode="fan_in")
            nn.init.kaiming_uniform_(self.relation, a=math.sqrt(5), mode="fan_in")

    def forward(self, head, tail, relation, h_index=None, t_index=None, r_index=None, all_loss=None, metric=None):
        with torch.set_grad_enabled(self.grad):
            if head is self.entity and tail is self.entity and relation is self.relation and not self.learnable_score:
                return super(TransEScore, self).forward(h_index, t_index, r_index, all_loss, metric)
            head = self.entity_norm(head)
            tail = self.entity_norm(tail)
            relation = self.relation_norm(relation)
            if h_index is not None:
                h = head[h_index]
                r = relation[r_index]
                t = tail[t_index]
            else:
                h = head
                r = relation
                t = tail
            x = (h + r - t).norm(p=1, dim=-1)
        score = self.max_score - x
        return score

    def flip_relation(self, relation):
        return -relation
