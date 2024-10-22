import os
import sys
import time
from typing import List
from collections import Counter
import networkx as nx
import numpy as np
import torch
from torch import nn

PackageDir = os.path.dirname(__file__)
sys.path.insert(1, PackageDir)

'''
xERTE codes are adapted from https://github.com/TemporalKGTeam/xERTE.
'''

class NeighborFinder:
    def __init__(self, adj, sampling=3, max_time=366, num_entities=None, weight_factor=2, time_granularity=1):
        """
        Params
        ------
        adj: list or dict, if list: adj[i] is the list of all (o,p,t) for entity i, if dict: adj[i] is the list of all (o,p,t)
        sampling: sample strategy from neighborhood: -1: whole neighbor, 0: uniform, 1: first N, 2: last N, 3: time difference weighted sampling
        node_idx_l: List[int]
        node_ts_l: List[int]
        off_set_l: List[int], such that node_idx_l[off_set_l[i]:off_set_l[i + 1]] = adjacent_list[i][:,0]
        off_set_t_l: node_idx_l[off_set_l[i]:off_set_l[i + 1]][:off_set_t_l[i][cut_time/time_granularity]] --> object of entity i that happen before cut time
        num_entities: number of entities, if adj is dict it cannot be None
        weight_factor: if sampling==3, use weight_factor to scale the time difference
        """

        self.time_granularity = time_granularity
        node_idx_l, node_ts_l, edge_idx_l, off_set_l, off_set_t_l = self.init_off_set(adj, max_time, num_entities)
        self.node_idx_l = node_idx_l
        self.node_ts_l = node_ts_l
        self.edge_idx_l = edge_idx_l

        self.off_set_l = off_set_l
        self.off_set_t_l = off_set_t_l

        self.sampling = sampling
        self.weight_factor = weight_factor

    def init_off_set(self, adj, max_time, num_entities):
        """
        for events with entity of index i being subject:
        node_idx_l[off_set_l[i]:off_set_l[i+1]] is the list of object index
        node_ts_l[off_set_l[i]:off_set_l[i+1]] is the list of timestamp
        edge_idx_l[off_set_l[i]:off_set_l[i+1]] is the list of relation
        ordered by (ts, ent, rel) ascending
        Params
        ------
        adj_dict: List[List[int]]

        """
        n_idx_l = []
        n_ts_l = []
        e_idx_l = []
        off_set_l = [0]
        off_set_t_l = []

        if isinstance(adj, list):
            for i in range(len(adj)):
                assert len(adj) == num_entities
                curr = adj[i]
                curr = sorted(curr, key=lambda x: (int(x[2]), int(x[0]), int(x[1])))
                n_idx_l.extend([x[0] for x in curr])
                e_idx_l.extend([x[1] for x in curr])
                curr_ts = [x[2] for x in curr]
                n_ts_l.extend(curr_ts)

                off_set_l.append(len(n_idx_l))
                off_set_t_l.append([np.searchsorted(curr_ts, cut_time, 'left') for cut_time in range(0, max_time+1, self.time_granularity)])# max_time+1 so we have max_time
        elif isinstance(adj, dict):
            for i in range(num_entities):
                curr = adj.get(i, [])
                curr = sorted(curr, key=lambda x: (int(x[2]), int(x[0]), int(x[1])))
                n_idx_l.extend([x[0] for x in curr])
                e_idx_l.extend([x[1] for x in curr])
                curr_ts = [x[2] for x in curr]
                n_ts_l.extend(curr_ts)

                off_set_l.append(len(n_idx_l))
                off_set_t_l.append([np.searchsorted(curr_ts, cut_time, 'left') for cut_time in range(0, max_time+1, self.time_granularity)])# max_time+1 so we have max_time

        n_idx_l = np.array(n_idx_l)
        n_ts_l = np.array(n_ts_l)
        e_idx_l = np.array(e_idx_l)
        off_set_l = np.array(off_set_l)

        assert (len(n_idx_l) == len(n_ts_l))
        assert (off_set_l[-1] == len(n_ts_l))

        return n_idx_l, n_ts_l, e_idx_l, off_set_l, off_set_t_l

    def get_temporal_degree(self, src_idx_l, cut_time_l):
        """
        return how many neighbros exist for each (src, ts)
        :param src_idx_l:
        :param cut_time_l:
        :return:
        """
        assert (len(src_idx_l) == len(cut_time_l))

        temp_degree = []
        for i, (src_idx, cut_time) in enumerate(zip(src_idx_l, cut_time_l)):
            temp_degree.append(self.off_set_t_l[src_idx][int(cut_time / self.time_granularity)])  # every timestamp in neighbors_ts[:mid] is smaller than cut_time
        return np.array(temp_degree)

    def find_before(self, src_idx, cut_time):
        neighbors_idx = self.node_idx_l[self.off_set_l[src_idx]:self.off_set_l[src_idx + 1]]
        neighbors_ts = self.node_ts_l[self.off_set_l[src_idx]:self.off_set_l[src_idx + 1]]
        neighbors_e_idx = self.edge_idx_l[self.off_set_l[src_idx]:self.off_set_l[src_idx + 1]]
        mid = np.searchsorted(neighbors_ts, cut_time)
        ngh_idx, ngh_eidx, ngh_ts = neighbors_idx[:mid], neighbors_e_idx[:mid], neighbors_ts[:mid]
        return ngh_idx, ngh_eidx, ngh_ts

    def get_temporal_neighbor_v2(self, src_idx_l, cut_time_l, query_time_l, num_neighbors=20):
        """
        temporal neighbors are not limited to be drawn from events happen before cut_time,
        but are extended to be drawn from all events that happen before query time
        More specifically, for each query we have (sub_q, rel_q, ?, t_q). By each step, for
        every node, i.e. entity-timestamp pair (e_i, t_i), we looked for such entity-timestamp
        pair (e, t) that (e_i, some_relation, e, t) exists. By first step, (e_i, t_i) == (sub_q, t_q)
        where t < t_q is the restriction (rather than t<t_0)
        Arguments:
            src_idx_l {numpy.array, 1d} -- entity index
            cut_time_l {numpy.array, 1d} -- timestamp of events
            query_time_l {numpy.array, 1d} -- timestamp of query

        Keyword Arguments:
            num_neighbors {int} -- [number of neighbors for each node] (default: {20})
        """
        assert (len(src_idx_l) == len(cut_time_l))
        assert (len(src_idx_l) == len(query_time_l))
        assert all([cut_time <= query_time for cut_time, query_time in list(zip(cut_time_l, query_time_l))])
        assert (num_neighbors % 2 == 0)

        out_ngh_node_batch = -np.ones((len(src_idx_l), num_neighbors)).astype(np.int32)
        out_ngh_t_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.float32)
        out_ngh_eidx_batch = -np.ones((len(src_idx_l), num_neighbors)).astype(np.int32)

        for i, (src_idx, cut_time, query_time) in enumerate(zip(src_idx_l, cut_time_l, query_time_l)):
            neighbors_idx = self.node_idx_l[self.off_set_l[src_idx]:self.off_set_l[src_idx + 1]]
            neighbors_ts = self.node_ts_l[self.off_set_l[src_idx]:self.off_set_l[src_idx + 1]]
            neighbors_e_idx = self.edge_idx_l[self.off_set_l[src_idx]:self.off_set_l[src_idx + 1]]
            mid = self.off_set_t_l[src_idx][int(cut_time / self.time_granularity)]
            end = self.off_set_t_l[src_idx][int(query_time / self.time_granularity)]
            # every timestamp in neighbors_ts[:mid] is smaller than cut_time
            ngh_idx_before, ngh_eidx_before, ngh_ts_before = neighbors_idx[:mid], neighbors_e_idx[:mid], neighbors_ts[:mid]
            # every timestamp in neighbors_ts[mid:end] is bigger than cut_time and smaller than query_time
            ngh_idx_after, ngh_eidx_after, ngh_ts_after = neighbors_idx[mid:end], neighbors_e_idx[mid:end], neighbors_ts[mid:end]

            # choose events happen closest in time
            half_num_neighbors = num_neighbors//2
            ngh_ts_before = ngh_ts_before[-half_num_neighbors:]
            ngh_idx_before = ngh_idx_before[-half_num_neighbors:]
            ngh_eidx_before = ngh_eidx_before[-half_num_neighbors:]

            out_ngh_node_batch[i, half_num_neighbors - len(ngh_idx_before):half_num_neighbors] = ngh_idx_before
            out_ngh_t_batch[i, half_num_neighbors - len(ngh_ts_before):half_num_neighbors] = ngh_ts_before
            out_ngh_eidx_batch[i, half_num_neighbors - len(ngh_eidx_before):half_num_neighbors] = ngh_eidx_before

            ngh_ts_after = ngh_ts_after[:half_num_neighbors]
            ngh_idx_after = ngh_idx_after[:half_num_neighbors]
            ngh_eidx_after = ngh_eidx_after[:half_num_neighbors]

            out_ngh_node_batch[i, half_num_neighbors:len(ngh_eidx_after) + half_num_neighbors] = ngh_idx_after
            out_ngh_t_batch[i, half_num_neighbors: len(ngh_ts_after) + half_num_neighbors] = ngh_ts_after
            out_ngh_eidx_batch[i, half_num_neighbors: len(ngh_eidx_after) + half_num_neighbors] = ngh_eidx_after

        out_ngh_query_t_batch = np.repeat(np.repeat(query_time_l[:, np.newaxis], num_neighbors, axis=1))

        return out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch, out_ngh_query_t_batch

    def get_temporal_neighbor(self, src_idx_l, cut_time_l, num_neighbors=20):
        """
        each entity has exact num_neighbors neighbors, neighbors are sampled according to sample strategy
        Params
        ------
        src_idx_l: List[int]
        cut_time_l: List[float],
        num_neighbors: int, ignored if sampling==-1,
        return:
        out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch: sorted by out_ngh_t_batch
        """
        assert (len(src_idx_l) == len(cut_time_l))

        out_ngh_node_batch = -np.ones((len(src_idx_l), num_neighbors)).astype(np.int32)
        out_ngh_t_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.int32)
        out_ngh_eidx_batch = -np.ones((len(src_idx_l), num_neighbors)).astype(np.int32)

        if self.sampling == -1:
            full_ngh_node = []
            full_ngh_t = []
            full_ngh_edge = []
        for i, (src_idx, cut_time) in enumerate(zip(src_idx_l, cut_time_l)):
            neighbors_idx = self.node_idx_l[self.off_set_l[src_idx]:self.off_set_l[src_idx + 1]]
            neighbors_ts = self.node_ts_l[self.off_set_l[src_idx]:self.off_set_l[src_idx + 1]]
            neighbors_e_idx = self.edge_idx_l[self.off_set_l[src_idx]:self.off_set_l[src_idx + 1]]
            mid = self.off_set_t_l[src_idx][
                int(cut_time / self.time_granularity)]  # every timestamp in neighbors_ts[:mid] is smaller than cut_time
            ngh_idx, ngh_eidx, ngh_ts = neighbors_idx[:mid], neighbors_e_idx[:mid], neighbors_ts[:mid]

            if len(ngh_idx) > 0:
                if self.sampling == 0:
                    sampled_idx = np.random.randint(0, len(ngh_idx), num_neighbors)

                    sampled_idx = np.sort(sampled_idx)

                    out_ngh_node_batch[i, :] = ngh_idx[sampled_idx]
                    out_ngh_t_batch[i, :] = ngh_ts[sampled_idx]
                    out_ngh_eidx_batch[i, :] = ngh_eidx[sampled_idx]

                elif self.sampling == 1:
                    ngh_ts = ngh_ts[:num_neighbors]
                    ngh_idx = ngh_idx[:num_neighbors]
                    ngh_eidx = ngh_eidx[:num_neighbors]
                    out_ngh_node_batch[i, num_neighbors - len(ngh_idx):] = ngh_idx
                    out_ngh_t_batch[i, num_neighbors - len(ngh_ts):] = ngh_ts
                    out_ngh_eidx_batch[i, num_neighbors - len(ngh_eidx):] = ngh_eidx
                elif self.sampling == 2:
                    ngh_ts = ngh_ts[-num_neighbors:]
                    ngh_idx = ngh_idx[-num_neighbors:]
                    ngh_eidx = ngh_eidx[-num_neighbors:]
                    out_ngh_node_batch[i, num_neighbors - len(ngh_idx):] = ngh_idx
                    out_ngh_t_batch[i, num_neighbors - len(ngh_ts):] = ngh_ts
                    out_ngh_eidx_batch[i, num_neighbors - len(ngh_eidx):] = ngh_eidx
                elif self.sampling == 3:
                    delta_t = (ngh_ts - cut_time)/(self.time_granularity*self.weight_factor)
                    weights = np.exp(delta_t) + 1e-9
                    weights = weights / sum(weights)

                    if len(ngh_idx) >= num_neighbors:
                        sampled_idx = np.random.choice(len(ngh_idx), num_neighbors, replace=False, p=weights)
                    else:
                        sampled_idx = np.random.choice(len(ngh_idx), len(ngh_idx), replace=False, p=weights)

                    sampled_idx = np.sort(sampled_idx)
                    out_ngh_node_batch[i, num_neighbors - len(sampled_idx):] = ngh_idx[sampled_idx]
                    out_ngh_t_batch[i, num_neighbors - len(sampled_idx):] = ngh_ts[sampled_idx]
                    out_ngh_eidx_batch[i, num_neighbors - len(sampled_idx):] = ngh_eidx[sampled_idx]
                elif self.sampling == 4:
                    weights = (ngh_ts + 1) / sum(ngh_ts + 1)

                    if len(ngh_idx) >= num_neighbors:
                        sampled_idx = np.random.choice(len(ngh_idx), num_neighbors, replace=False, p=weights)
                    else:
                        sampled_idx = np.random.choice(len(ngh_idx), len(ngh_idx), replace=False, p=weights)

                    sampled_idx = np.sort(sampled_idx)
                    out_ngh_node_batch[i, num_neighbors - len(sampled_idx):] = ngh_idx[sampled_idx]
                    out_ngh_t_batch[i, num_neighbors - len(sampled_idx):] = ngh_ts[sampled_idx]
                    out_ngh_eidx_batch[i, num_neighbors - len(sampled_idx):] = ngh_eidx[sampled_idx]

                elif self.sampling == -1: # use whole neighborhood
                    full_ngh_node.append(ngh_idx[-300:])
                    full_ngh_t.append(ngh_ts[-300:])
                    full_ngh_edge.append(ngh_eidx[-300:])
                else:
                    raise ValueError("invalid input for sampling")

        if self.sampling == -1:
            max_num_neighbors = max(map(len, full_ngh_edge))
            out_ngh_node_batch = -np.ones((len(src_idx_l), max_num_neighbors)).astype(np.int32)
            out_ngh_t_batch = np.zeros((len(src_idx_l), max_num_neighbors)).astype(np.int32)
            out_ngh_eidx_batch = -np.ones((len(src_idx_l), max_num_neighbors)).astype(np.int32)
            for i in range(len(full_ngh_node)):
                out_ngh_node_batch[i, max_num_neighbors-len(full_ngh_node[i]):] = full_ngh_node[i]
                out_ngh_eidx_batch[i, max_num_neighbors-len(full_ngh_edge[i]):] = full_ngh_edge[i]
                out_ngh_t_batch[i, max_num_neighbors-len(full_ngh_t[i]):] = full_ngh_t[i]

        return out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch

    def get_neighbor_subgraph(self, src_idx_l, cut_time_l, level=2, num_neighbors=20):
        Gs = [nx.Graph() for _ in range(len(src_idx_l))]
        for i, G in enumerate(Gs):
            G.add_node((src_idx_l[i], None, cut_time_l[i]), rel=None, time=cut_time_l[i])

        def get_neighbors_recursive(graph_index_l, src_idx_l, rel_idx_l, cut_time_l, level, num_neighbors):
            if level == 0:
                return
            else:
                src_ngh_node_batch, src_ngh_eidx_batch, src_ngh_t_batch = self.get_temporal_neighbor(
                    src_idx_l,
                    cut_time_l,
                    num_neighbors=num_neighbors)

                for batch_idx in range(len(src_idx_l)):
                    ngh_nodes = src_ngh_node_batch[batch_idx]
                    ngh_edges = src_ngh_eidx_batch[batch_idx]
                    ngh_ts = src_ngh_t_batch[batch_idx]

                    Gs[graph_index_l[batch_idx]].add_nodes_from(
                        [((node, rel, t), {'rel': rel, 'time': t}) for node, rel, t in
                         list(zip(ngh_nodes, ngh_edges, ngh_ts))])
                    Gs[graph_index_l[batch_idx]].add_edges_from([((src_idx_l[batch_idx], rel_idx_l[batch_idx], cut_time_l[batch_idx]),
                                                                  (node, edge, t))
                                                                 for node, edge, t in list(zip(ngh_nodes, ngh_edges, ngh_ts))])

                    get_neighbors_recursive(np.repeat(graph_index_l[batch_idx],
                                                      len(ngh_nodes)), ngh_nodes, ngh_edges, ngh_ts, level - 1, num_neighbors)

        get_neighbors_recursive(np.arange(len(src_idx_l)), src_idx_l, [None for _ in src_idx_l], cut_time_l, level, num_neighbors)
        return Gs

class TimeEncode(torch.nn.Module):
    '''
    This class implemented the Bochner's time embedding
    expand_dim: int, dimension of temporal entity embeddings
    enitity_specific: bool, whether use entith specific freuency and phase.
    num_entities: number of entities.
    '''

    def __init__(self, expand_dim, entity_specific=False, num_entities=None, args=None):
        """
        :param expand_dim: number of samples draw from p(w), which are used to estimate kernel based on MCMC
        :param entity_specific: if use entity specific time embedding
        :param num_entities: number of entities
        refer to Self-attention with Functional Time Representation Learning for more detail
        """
        super(TimeEncode, self).__init__()
        self.time_dim = expand_dim
        self.entity_specific = entity_specific
        self.args = args

        if entity_specific:
            self.basis_freq = torch.nn.Parameter(
                torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim)).float().unsqueeze(dim=0).repeat(
                    num_entities, 1))
            self.phase = torch.nn.Parameter(torch.zeros(self.time_dim).float().unsqueeze(dim=0).repeat(num_entities, 1))
        else:
            self.basis_freq = torch.nn.Parameter(
                torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim)).float())  # shape: num_entities * time_dim
            self.phase = torch.nn.Parameter(torch.zeros(self.time_dim).float())

    def forward(self, ts, entities=None):
        '''
        :param ts: [batch_size, seq_len]
        :param entities: which entities do we extract their time embeddings.
        :return: [batch_size, seq_len, time_dim]
        '''
        batch_size = ts.size(0)
        seq_len = ts.size(1)
        ts = torch.unsqueeze(ts, dim=2).to(self.basis_freq.device)
        # print("Forward in TimeEncode: ts is on ", ts.get_device())
        if self.entity_specific:
            map_ts = ts * self.basis_freq[entities].unsqueeze(
                dim=1)  # self.basis_freq[entities]:  [batch_size, time_dim]
            map_ts += self.phase[entities].unsqueeze(dim=1)
        else:
            map_ts = ts * self.basis_freq.view(1, 1, -1)  # [batch_size, 1, time_dim]
            map_ts += self.phase.view(1, 1, -1)
        harmonic = torch.cos(map_ts)
        return harmonic

class G3(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        """[summary]
        bilinear mapping along last dimension of x and y:
        output = MLP_1(x)^T A MLP_2(y), where A is two-dimenion matrix

        Arguments:
            left_dims {[type]} -- input dims of MLP_1
            right_dims {[type]} -- input dims of MLP_2
            output_dims {[type]} -- [description]
        """
        super(G3, self).__init__()
        self.dim_out = dim_out
        self.query_proj = nn.Linear(dim_in, dim_out, bias=False)
        nn.init.normal_(self.query_proj.weight, mean=0, std=np.sqrt(2.0 / (dim_in)))
        self.key_proj = nn.Linear(dim_in, dim_out, bias=False)
        nn.init.normal_(self.key_proj.weight, mean=0, std=np.sqrt(2.0 / (dim_in)))

    def forward(self, inputs):
        """[summary]
        Arguments:
            inputs: (left, right)
            left[i] -- tensor, bs x ... x left_dims
            right[i] -- tensor, bs x ... x right_dims
        """
        vi, vj = inputs
        left_x = torch.cat(vi, dim=-1)
        right_x = torch.cat(vj, dim=-1)
        # speed of batch-wise dot production: sum over element-wise product > matmul > bmm
        # refer to https://discuss.pytorch.org/t/dot-product-batch-wise/9746/12
        return torch.sum(self.query_proj(left_x) * self.key_proj(right_x), dim=-1) #/ np.sqrt(self.dim_out)

class AttentionFlow(nn.Module):
    def __init__(self, n_dims_in, n_dims_out, ratio_update=0, node_score_aggregation='sum', device = "cuda:0"):
        """[summary]

        Arguments:
            n_dims -- int, dimension of entity and relation embedding
            n_dims_sm -- int, smaller than n_dims to reduce the compuation consumption of calculating attention score
            ratio_update -- new node representation = ratio*self+(1-ratio)\sum{aggregation of neighbors' representation}
        """
        super(AttentionFlow, self).__init__()
        self.transition_fn = G3(4 * n_dims_in, 4 * n_dims_in)

        # dense layer between steps
        self.linear_between_steps = nn.Linear(n_dims_in, n_dims_out, bias=True)
        torch.nn.init.xavier_normal_(self.linear_between_steps.weight)
        self.act_between_steps = torch.nn.LeakyReLU()

        self.node_score_aggregation = node_score_aggregation
        self.ratio_update = ratio_update

        self.query_src_ts_emb = None
        self.query_rel_emb = None
        
        self.device = device

    def set_query_emb(self, query_src_ts_emb, query_rel_emb):
        self.query_src_ts_emb, self.query_rel_emb = query_src_ts_emb, query_rel_emb

    def _topk_att_score(self, edges, logits, k: int, tc=None):
        """

        :param edges: numpy array, (eg_idx, vi, ti, vj, tj, rel, node_idx_i, node_idx_j), dtype np.int32
        :param logits: tensor, same length as edges, dtype=torch.float32
        :param k: number of nodes in attended-from horizon
        :return:
        pruned_edges, numpy.array, (eg_idx, vi, ts)
        pruned_logits, tensor, same length as pruned_edges
        origin_indices
        """
        if tc:
            t_start = time.time()
        res_edges = []
        res_logits = []
        res_indices = []
        for eg_idx in sorted(set(edges[:, 0])):
            mask = edges[:, 0] == eg_idx
            orig_indices = np.arange(len(edges))[mask]
            masked_edges = edges[mask]
            masked_edges_logits = logits[mask]
            if masked_edges.shape[0] <= k:
                res_edges.append(masked_edges)
                res_logits.append(masked_edges_logits)
                res_indices.append(orig_indices)
            else:
                topk_edges_logits, indices = torch.topk(masked_edges_logits, k)
                res_indices.append(orig_indices[indices.cpu().numpy()])
                # pdb.set_trace()
                try:
                    res_edges.append(masked_edges[indices.cpu().numpy()])
                except Exception as e:
                    print(indices.cpu().numpy())
                    print(max(indices.cpu().numpy()))
                    print(str(e))
                    raise KeyError
                res_logits.append(topk_edges_logits)
        if tc:
            tc['graph']['topk'] += time.time() - t_start

        return np.concatenate(res_edges, axis=0), torch.cat(res_logits, dim=0), np.concatenate(res_indices, axis=0)

    def _cal_attention_score(self, edges, memorized_embedding, rel_emb):
        """
        calculating node attention from memorized embedding
        """
        hidden_vi_orig = memorized_embedding[edges[:, -2]]
        hidden_vj_orig = memorized_embedding[edges[:, -1]]

        return self.cal_attention_score(edges[:, 0], hidden_vi_orig, hidden_vj_orig, rel_emb)

    def cal_attention_score(self, query_idx, hidden_vi, hidden_vj, rel_emb):
        """
        calculate attention score between two nodes of edges
        wraped as a separate method so that it can be used for calculating attention between a node and it's full
        neighborhood, attention is used to select important nodes from the neighborhood
        :param query_idx: indicating in subgraph for which query the edge lies.
        """

        # [embedding]_repeat is a new tensor which index [embedding] so that it mathes hidden_vi and hidden_vj along dim 0
        # i.e. hidden_vi[i] and hidden_vj[i] is representation of node vi, vj that lie in subgraph corresponding to the query,
        # whose src, rel, time embedding is [embedding]_repeat[i]
        # [embedding] is one of query_src, query_rel, query_time
        query_src_ts_emb_repeat = torch.index_select(self.query_src_ts_emb, dim=0,
                                                     index=torch.from_numpy(query_idx).long().to(
                                                         self.device))
        query_rel_emb_repeat = torch.index_select(self.query_rel_emb, dim=0,
                                                  index=torch.from_numpy(query_idx).long().to(
                                                      self.device))

        transition_logits = self.transition_fn(
            ((hidden_vi, rel_emb, query_src_ts_emb_repeat, query_rel_emb_repeat),
             (hidden_vj, rel_emb, query_src_ts_emb_repeat, query_rel_emb_repeat)))

        return transition_logits

    def forward(self, visited_node_score, selected_edges_l=None, visited_node_representation=None, rel_emb_l=None,
                max_edges=10, analysis=False, tc=None, device = "cuda:0"):
        """calculate attention score

        Arguments:
            node_attention {tensor, num_edges} -- src_attention of selected_edges, node_attention[i] is the attention score
            of (selected_edge[i, 1], selected_edge[i, 2]) in eg_idx==selected_edge[i, 0]

        Keyword Arguments:
            selected_edges {numpy.array, num_edges x 8} -- (eg_idx, vi, ti, vj, tj, rel, idx_eg_vi_ti, idx_eg_vj_tj) (default: {None})
            contain selfloop
            memorized_embedding torch.Tensor,
        return:
            pruned_edges, orig_indices
            updated_memorized_embedding:
            updated_node_score: Tensor, shape: n_new_node
            :param attended_nodes: 
        """
        self.device = device
        updated_edge_attention = []  # for analysis

        transition_logits = self._cal_attention_score(selected_edges_l[-1], visited_node_representation, rel_emb_l[-1])

        # prune edges whose target node score is small
        # get source attention score
        src_score = visited_node_score[selected_edges_l[-1][:, -2]]
        transition_logits_softmax = segment_softmax_op_v2(transition_logits, selected_edges_l[-1][:, -2],
                                                          tc=tc)  # TB Check
        edge_attn_before_pruning = transition_logits_softmax  # for analysis
        target_score = transition_logits_softmax * src_score
        pruned_edges, pruned_target_score, orig_indices = self._topk_att_score(selected_edges_l[-1], target_score,
                                                                               max_edges)
        pruned_src_score = src_score[orig_indices]

        # transition_logits_pruned_softmax contains information of all selected_edges
        transition_logits_pruned_softmax = transition_logits_softmax[orig_indices]
        updated_edge_attention.append(transition_logits_pruned_softmax)

        num_nodes = len(visited_node_representation)
        if self.node_score_aggregation == 'max':
            max_dict = dict()
            for i in range(len(pruned_edges)):
                score_i = pruned_target_score[i].cpu().detach().numpy()
                if score_i > max_dict.get(pruned_edges[i, -1], (0, 0))[1]:
                    max_dict[pruned_edges[i, -1]] = (i, score_i)

            # biggest score from all edges (some edges may have the same subject)
            sparse_index = torch.LongTensor(
                np.stack([np.array(list(max_dict.keys())), np.array([_[0] for _ in max_dict.values()])])).to(
                self.device)
            trans_matrix_sparse = torch.sparse.FloatTensor(sparse_index, torch.ones(len(max_dict)).to(self.device),
                                                           torch.Size([num_nodes, len(pruned_edges)])).to(self.device)
            updated_node_score = torch.squeeze(torch.sparse.mm(trans_matrix_sparse, pruned_target_score.unsqueeze(1)))
        elif self.node_score_aggregation in ['mean', 'sum']:
            sparse_index = torch.LongTensor(np.stack([pruned_edges[:, 7], np.arange(len(pruned_edges))])).to(
                self.device)

            # node score aggregation
            if self.node_score_aggregation == 'mean':
                c = Counter(pruned_edges[:, -1])
                target_node_cnt = torch.tensor([c[_] for _ in pruned_edges[:, -1]]).to(self.device)
                transition_logits_pruned_softmax = torch.div(transition_logits_pruned_softmax, target_node_cnt)

            trans_matrix_sparse = torch.sparse.FloatTensor(sparse_index, transition_logits_pruned_softmax,
                                                           torch.Size([num_nodes, len(pruned_edges)])).to(self.device)
            # ATTENTION: updated_node_score[i] must be node score of node with node_idx==i
            updated_node_score = torch.squeeze(torch.sparse.mm(trans_matrix_sparse, pruned_src_score.unsqueeze(1)))
        else:
            raise ValueError("node score aggregate can only be mean, sum or max")

        # only message passing and aggregation
        updated_visited_node_representation = self._update_node_representation_along_edges(pruned_edges,
                                                                                           visited_node_representation,
                                                                                           transition_logits_pruned_softmax,
                                                                                           linear_act=False)

        for selected_edges, rel_emb in zip(selected_edges_l[:-1][::-1], rel_emb_l[:-1][::-1]):
            transition_logits = self._cal_attention_score(selected_edges, updated_visited_node_representation,
                                                          rel_emb)
            transition_logits_softmax = segment_softmax_op_v2(transition_logits, selected_edges[:, -2], tc=tc)
            updated_edge_attention.append(transition_logits_softmax)
            updated_visited_node_representation = self._update_node_representation_along_edges(selected_edges,
                                                                                               updated_visited_node_representation,
                                                                                               transition_logits_softmax,
                                                                                               linear_act=False)

        #apply dense layer and activation on updated_memorized_embedding
        updated_visited_node_representation = self.bypass_forward(updated_visited_node_representation)

        if analysis:
            return updated_node_score, updated_visited_node_representation, pruned_edges, orig_indices, edge_attn_before_pruning, updated_edge_attention[
                                                                                                                                  ::-1]
        else:
            return updated_node_score, updated_visited_node_representation, pruned_edges, orig_indices

    def _update_node_representation_along_edges_old(self, edges, memorized_embedding, transition_logits):
        num_nodes = len(memorized_embedding)
        # update representation of nodes with neighbors
        # 1. message passing and aggregation
        sparse_index_rep = torch.from_numpy(edges[:, [-2, -1]]).to(torch.int64).to(self.device)
        sparse_value_rep = transition_logits
        trans_matrix_sparse_rep = torch.sparse.FloatTensor(sparse_index_rep.t(), sparse_value_rep,
                                                           torch.Size([num_nodes, num_nodes])).to(self.device)
        updated_memorized_embedding = torch.sparse.mm(trans_matrix_sparse_rep, memorized_embedding)
        # 2. linear
        updated_memorized_embedding = self.act_between_steps(self.linear_between_steps(updated_memorized_embedding))
        # 3. pass representation of nodes without neighbors, i.e. not updated
        sparse_index_identical = torch.from_numpy(np.setdiff1d(np.arange(num_nodes), edges[:, -2])).unsqueeze(
            1).repeat(1, 2).to(self.device)
        sparse_value_identical = torch.ones(len(sparse_index_identical)).to(self.device)
        trans_matrix_sparse_identical = torch.sparse.FloatTensor(sparse_index_identical.t(), sparse_value_identical,
                                                                 torch.Size([num_nodes, num_nodes])).to(self.device)
        identical_memorized_embedding = torch.sparse.mm(trans_matrix_sparse_identical, memorized_embedding)
        updated_memorized_embedding = updated_memorized_embedding + identical_memorized_embedding
        return updated_memorized_embedding

    def _update_node_representation_along_edges(self, edges, node_representation, transition_logits, linear_act=True):
        """

        :param edges:
        :param memorized_embedding:
        :param transition_logits:
        :param linear_act: whether apply linear and activation layer after message aggregation
        :return:
        """
        num_nodes = len(node_representation)
        sparse_index_rep = torch.from_numpy(edges[:, [-2, -1]]).to(torch.int64).to(self.device)
        sparse_value_rep = (1 - self.ratio_update) * transition_logits
        sparse_index_identical = torch.from_numpy(np.setdiff1d(np.arange(num_nodes), edges[:, -2])).unsqueeze(
            1).repeat(1, 2).to(self.device)
        sparse_value_identical = torch.ones(len(sparse_index_identical)).to(self.device)
        sparse_index_self = torch.from_numpy(np.unique(edges[:, -2])).unsqueeze(1).repeat(1, 2).to(self.device)
        sparse_value_self = self.ratio_update * torch.ones(len(sparse_index_self)).to(self.device)
        sparse_index = torch.cat([sparse_index_rep, sparse_index_identical, sparse_index_self], axis=0)
        sparse_value = torch.cat([sparse_value_rep, sparse_value_identical, sparse_value_self])
        trans_matrix_sparse = torch.sparse.FloatTensor(sparse_index.t(), sparse_value,
                                                       torch.Size([num_nodes, num_nodes])).to(self.device)
        updated_node_representation = torch.sparse.mm(trans_matrix_sparse, node_representation)
        if linear_act:
            updated_node_representation = self.act_between_steps(self.linear_between_steps(updated_node_representation))

        return updated_node_representation

    def bypass_forward(self, embedding):
        return self.act_between_steps(self.linear_between_steps(embedding))


class xERTE(torch.nn.Module):
    # DP_num_edges=15
    def __init__(self, args, base_graph,
                 emb_dim: List[int] = [256, 128, 64, 32],
                 DP_num_edges=15, DP_steps=3,
                 emb_static_ratio=1, diac_embed=False,
                 node_score_aggregation='sum', ent_score_aggregation='sum', max_attended_edges=40, ratio_update=0,
                 analysis=False, use_time_embedding=True, **kwargs):
        """[summary]

        Arguments:
            ngh_finder {[type]} -- an instance of NeighborFinder, find neighbors of a node from temporal KG
            according to TGAN scheme

        Keyword Arguments:
            num_entity {[type]} -- [description] (default: {None})
            num_rel {[type]} -- [description] (default: {None})
            embed_dim {[type]} -- [dimension of ERTKG embedding] (default: {None})
            attn_mode {str} -- [currently only prod is supported] (default: {'prod'})
            use_time {str} -- [use time embedding] (default: {'time'})
            agg_method {str} -- [description] (default: {'attn'})
            tgan_num_layers {int} -- [description] (default: {2})
            tgan_n_head {int} -- [description] (default: {4})
            null_idx {int} -- [description] (default: {0})
            drop_out {float} -- [description] (default: {0.1})
            seq_len {[type]} -- [description] (default: {None})
            max_attended_nodes {int} -- [max number of nodes in attending-from horizon] (default: {20})
            ratio_update: new node representation = ratio*self+(1-ratio)\sum{aggregation of neighbors' representation}
            device {str} -- [description] (default: {'cpu'})
        """
        super(xERTE, self).__init__()
        assert len(emb_dim) == DP_steps + 1

        self.DP_num_edges = DP_num_edges
        self.DP_steps = DP_steps
        self.use_time_embedding = use_time_embedding
        self.args = args
        self.device = self.args.DEVICE
        self.ngh_finder = NeighborFinder(base_graph, num_entities = self.args.ent_num)


        self.temporal_embed_dim = [int(emb_dim[_] * 2 / (1 + emb_static_ratio)) for _ in range(DP_steps)]
        self.static_embed_dim = [emb_dim[_] * 2 - self.temporal_embed_dim[_] for _ in range(DP_steps)]

        '''
        self.entity_raw_embed = torch.nn.Embedding(self.args.ent_num, self.static_embed_dim[0]).cpu()
        nn.init.xavier_normal_(self.entity_raw_embed.weight)
        self.relation_raw_embed = torch.nn.Embedding(self.args.rel_num + 1, emb_dim[0]).cpu()
        nn.init.xavier_normal_(self.relation_raw_embed.weight)
        '''
        
        self.ent_proj = nn.Linear(self.args.llm_dim, self.static_embed_dim[0])
        self.rel_proj = nn.Linear(self.args.llm_dim, emb_dim[0])
        
        
        self.selfloop = self.args.rel_num  # index of relation "selfloop"
        self.att_flow_list = nn.ModuleList([AttentionFlow(emb_dim[_], emb_dim[_ + 1],
                                                          node_score_aggregation=node_score_aggregation,
                                                          ratio_update=self.args.ratio_update,
                                                          device = self.device)
                                            for _ in range(DP_steps)])
        if use_time_embedding:
            self.node_emb_proj = nn.Linear(2 * emb_dim[0], emb_dim[0])
        else:
            self.node_emb_proj = nn.Linear(emb_dim[0], emb_dim[0])

        nn.init.xavier_normal_(self.node_emb_proj.weight)
        self.max_attended_edges = self.args.max_attended_edges

        if use_time_embedding:
            self.time_encoder = TimeEncode(expand_dim=self.temporal_embed_dim[0], entity_specific=diac_embed,
                                           num_entities=self.args.ent_num, args = self.args)
        self.ent_spec_time_embed = diac_embed
        self.analysis = analysis
        self.ent_score_aggregation = ent_score_aggregation
        
        #weight with sigmoid
        self.weight_MLP = nn.Sequential(
            nn.Linear(emb_dim[-2], 1),
            nn.Sigmoid()
        )

    def set_init(self, src_idx_l, rel_idx_l, cut_time_l):
        # save query information
        self.src_idx_l = src_idx_l
        self.rel_idx_l = rel_idx_l
        self.cut_time_l = cut_time_l
        self.sampled_edges_l = []
        self.rel_emb_l = []
        # for each quard in a batch, construct an edge and a node
        # for input queries/nodes, node_idx == eg_idx
        self.node2index = {(i, src, ts): i for i, (src, rel, ts) in
                           enumerate(zip(src_idx_l, rel_idx_l, cut_time_l))}  # (eg_idx, ent, ts) -> node_idx
        self.num_existing_nodes = len(src_idx_l)

        query_src_emb = self.get_ent_emb(self.src_idx_l, self.device)
        query_rel_emb = self.get_rel_emb(self.rel_idx_l, self.device)
        if self.use_time_embedding:
            if self.ent_spec_time_embed:
                query_ts_emb = self.time_encoder(
                    torch.zeros(len(self.cut_time_l), 1).to(torch.float32).to(self.device),
                    entities=self.src_idx_l)
            else:
                query_ts_emb = self.time_encoder(
                    torch.zeros(len(self.cut_time_l), 1).to(torch.float32).to(self.device))
            query_ts_emb = torch.squeeze(query_ts_emb, 1)
            query_src_ts_emb = self.node_emb_proj(torch.cat([query_src_emb, query_ts_emb], axis=1))
        else:
            query_src_ts_emb = self.node_emb_proj(query_src_emb)

        # init query_src_ts_emb and query_rel_emb for each AttentionFlow layer
        for i, att_flow in enumerate(self.att_flow_list):
            if i > 0:
                query_src_ts_emb = self.att_flow_list[i - 1].bypass_forward(query_src_ts_emb)
                query_rel_emb = self.att_flow_list[i - 1].bypass_forward(query_rel_emb)
            att_flow.set_query_emb(query_src_ts_emb, query_rel_emb)

    def initialize(self):
        """get initial node (entity+time) embedding and initial node score

        Returns:
            attending_nodes, np.array -- n_attending_nodes x 4, (eg_idx, entity_id, ts, node_idx)
            edge_idx and node_idx are both initialized as batch_ids
            attending_node_attention, np,array -- n_attending_nodes, (1,)
            memorized_embedding, dict ((entity_id, ts): TGAN_embedding)
        """
        
        
        eg_idx_l = np.arange(len(self.src_idx_l), dtype=np.int32)
        att_score = np.ones_like(self.src_idx_l, dtype=np.float32) * (1 - 1e-8)

        attended_nodes = np.stack([eg_idx_l, self.src_idx_l, self.cut_time_l, np.arange(len(self.src_idx_l))], axis=1)
        visited_nodes_score = torch.from_numpy(att_score).to(self.device)
        visited_nodes = attended_nodes

        visited_node_representation = self.att_flow_list[0].query_src_ts_emb
        return attended_nodes, visited_nodes, visited_nodes_score, visited_node_representation

    def forward(self, batch_data, device):
        self.device = device
        #sample: batch of queries
        batch_quires = batch_data["batch_queries"] # (ts, head_id, rel_id)
        src_idx_l, rel_idx_l, cut_time_l = batch_quires[:, 1], batch_quires[:, 2], batch_quires[:, 0]
        src_idx_l = src_idx_l.numpy()
        rel_idx_l = rel_idx_l.numpy()
        cut_time_l = cut_time_l.numpy()
        self.set_init(src_idx_l, rel_idx_l, cut_time_l)
        attended_nodes, visited_nodes, visited_node_score, visited_node_representation = self.initialize()
        for step in range(self.DP_steps):
            #            print("{}-th DP step".format(step))
            # * attended_nodes: selected topk nodes
            # * visited_nodes: all nodes visited during the expansion, num_nodes_visited x 4 (batch_idx, entity_id, ts, node_idx)
            #! nodes in each step kept or not, can we replicate the paths?
            attended_nodes, visited_nodes, visited_node_score, visited_node_representation = \
                self._flow(attended_nodes, visited_nodes, visited_node_score, visited_node_representation, step, device = device)
            # pdb.set_trace()
            visited_node_score = segment_norm_l1(visited_node_score, visited_nodes[:, 0])
        
        entity_att_score, entities = self.get_entity_attn_score(visited_node_score[attended_nodes[:, -1]],
                                                                attended_nodes)
        
        # entity_att_score: ent_id's attention score of entities, tensor
        # entities: np.array, each one as [bsz_id, ent_id]
        # get ent distribution [bsz, num_ent]
        entities = torch.from_numpy(entities)
        bsz = src_idx_l.size
        ent_distribution = torch.zeros(bsz, self.args.ent_num).to(self.device)
        for i in range(bsz):
            ent_distribution[i, entities[:, 1][entities[:, 0] == i]] = \
            entity_att_score[entities[:, 0] == i]
            
        #turn self.sampled_edges_l: [batch_idx, vi, ti, vj, tj, rel] to a batch_list of [vi, vj, tj, rel]
        batch_quads = []
        sampled_edges_l = self.sampled_edges_l[0]
        #sampled_edges_l = torch.from_numpy(sampled_edges_l)
        for i in range(bsz):
            sampled_edges_i = sampled_edges_l[sampled_edges_l[:, 0] == i]
            sampled_edges_i = sampled_edges_i[:, [1, 3, 4, 5]]
            batch_quads.append(sampled_edges_i)
            
        # visited_node_representation:[node_size, dim]
        # mean visited_node_representation where attended_nodes[:, 0] is the same
        #mean_node_representation = scatter_mean(visited_node_representation[attended_nodes[:, -1]], torch.from_numpy(attended_nodes[:, 0]).long().to(self.device), dim = 0)
        #lamda = self.weight_MLP(mean_node_representation)
        
        
        lamda = self.weight_MLP(self.att_flow_list[-1].query_rel_emb)
                
        return entity_att_score, entities, ent_distribution, batch_quads, lamda

    def _flow(self, attended_nodes, visited_nodes, visited_node_score, visited_node_representation, step, tc=None, device = "cuda:0"):
        """[summary]
        Arguments:
            visited_nodes {numpy.array} -- num_nodes_visited x 4 (eg_idx, entity_id, ts, node_idx), dtype: numpy.int32, sort (eg_idx, ts, entity_id)
            all nodes visited during the expansion
            visited_node_score {Tensor} -- num_nodes_visited, dtype: torch.float32
            visited_node_representation {Tensor} -- num_nodes_visited x emb_dim_l[step]
            visited_node_score[node_idx] is the prediction score of node_idx
            visited_node_representation[node_idx] is the hidden representation of node_idx
        return:
            pruned_node {numpy.array} -- num_nodes_ x 4 (eg_idx, entity_id, ts, node_idx) sorted by (eg_idx, ts, entity_id)
            new_node_score {Tensor} -- new num_nodes_visited
            so that new_node_score[i] is the node prediction score of??
            updated_visited_node_representation: Tensor -- num_nodes_visited x emb_dim_l[step+1]
        """
        
        #! self.sampled_edges_l: all sampled_edges in this batch, each as: [batch_idx, vi, ti, vj, tj]

        # Sampling Horizon
        # sampled_edges: (eg_idx, vi, ti, vj, tj, rel, idx_eg_vi_ti, idx_eg_vj_tj)
        # selfloop is added
        sampled_edges, new_sampled_nodes, new_attended_nodes = self._get_sampled_edges(attended_nodes,
                                                                                       num_neighbors=self.DP_num_edges,
                                                                                       step=step,
                                                                                       add_self_loop=True, tc=tc)
        if len(new_sampled_nodes):
            new_sampled_nodes_emb = self.get_node_emb(new_sampled_nodes[:, 1], new_sampled_nodes[:, 2],
                                                      eg_idx=new_sampled_nodes[:, 0])
            for i in range(step):
                new_sampled_nodes_emb = self.att_flow_list[i].bypass_forward(new_sampled_nodes_emb)
            visited_node_representation = torch.cat([visited_node_representation, new_sampled_nodes_emb], axis=0)
            visited_nodes = np.concatenate([visited_nodes, new_sampled_nodes], axis=0)

            assert len(visited_node_representation) == self.num_existing_nodes
            assert max(new_sampled_nodes[:, -1]) + 1 == self.num_existing_nodes
            assert max(sampled_edges[:, -1]) < self.num_existing_nodes

        self.sampled_edges_l.append(sampled_edges)

        rel_emb = self.get_rel_emb(sampled_edges[:, 5], self.device)
        for i in range(step):
            rel_emb = self.att_flow_list[i].bypass_forward(rel_emb)
        # update relation representation of edges sampled from previous steps
        for j in range(step):
            self.rel_emb_l[j] = self.att_flow_list[step - 1].bypass_forward(self.rel_emb_l[j])
        self.rel_emb_l.append(rel_emb)

        new_visited_node_score, updated_visited_node_representation, pruned_edges, orig_indices = \
            self.att_flow_list[step](visited_node_score,
                                     selected_edges_l=self.sampled_edges_l,
                                     visited_node_representation=visited_node_representation,
                                     rel_emb_l=self.rel_emb_l,
                                     max_edges=self.max_attended_edges, tc=tc, device = device)

        assert len(pruned_edges) == len(orig_indices)

        self.sampled_edges_l[-1] = pruned_edges
        self.rel_emb_l[-1] = self.rel_emb_l[-1][orig_indices]

        # get pruned nodes
        _, indices = np.unique(pruned_edges[:, [0, 4, 3]], return_index=True, axis=0)
        updated_attended_nodes = pruned_edges[:, [0, 3, 4, 7]][indices]

        return updated_attended_nodes, visited_nodes, new_visited_node_score, updated_visited_node_representation

    def get_node_emb(self, src_idx_l, cut_time_l, eg_idx):

        hidden_node = self.get_ent_emb(src_idx_l, self.device)
        if self.use_time_embedding:
            cut_time_l = cut_time_l - self.cut_time_l[eg_idx]
            if self.ent_spec_time_embed:
                hidden_time = self.time_encoder(torch.from_numpy(cut_time_l[:, np.newaxis]).to(self.device),
                                                entities=src_idx_l)
            else:
                hidden_time = self.time_encoder(torch.from_numpy(cut_time_l[:, np.newaxis]).to(self.device))
            return self.node_emb_proj(torch.cat([hidden_node, torch.squeeze(hidden_time, 1)], axis=1))
        else:
            return self.node_emb_proj(hidden_node)

    def get_entity_attn_score(self, logits, nodes, tc=None):
        if tc:
            t_start = time.time()
        entity_attn_score, entities = self._aggregate_op_entity(logits, nodes, self.ent_score_aggregation)
        #        # normalize entity prediction score
        #        entity_attn_score = segment_norm_l1(entity_attn_score, entities[:, 0])
        if tc:
            tc['model']['entity_attn'] = time.time() - t_start
        return entity_attn_score, entities

    def _aggregate_op_entity(self, logits, nodes, aggr='sum'):
        """aggregate attention score of same entity, i.e. same (eg_idx, v)

        Arguments:
            logits {Tensor} -- attention score
            nodes {numpy.array} -- shape len(logits) x 3, (eg_idx, v, t), sorted by eg_idx, v, t
        return:
            entity_att_score {Tensor}: shape num_entity
            entities: numpy.array -- shape num_entity x 2, (eg_idx, v)
            att_score[i] if the attention score of entities[i]
        """

        num_nodes = len(nodes)
        entities, entities_idx = np.unique(nodes[:, :2], axis=0, return_inverse=True)
        sparse_index = torch.LongTensor(np.stack([entities_idx, np.arange(num_nodes)]))
        sparse_value = torch.ones(num_nodes, dtype=torch.float)
        if aggr == 'mean':
            c = Counter([(node[0], node[1]) for node in nodes[:, :2]])
            target_node_cnt = torch.tensor([c[(_[0], _[1])] for _ in nodes[:, :2]])
            sparse_value = torch.div(sparse_value, target_node_cnt)

        trans_matrix_sparse = torch.sparse.FloatTensor(sparse_index, sparse_value,
                                                       torch.Size([len(entities), num_nodes])).to(self.device)
        entity_att_score = torch.squeeze(torch.sparse.mm(trans_matrix_sparse, logits.unsqueeze(1)))

        return entity_att_score, entities

    def _get_sampled_edges(self, attended_nodes, num_neighbors: int = 20, step=None, add_self_loop=True, tc=None):
        """[summary]
        sample neighbors for attended_nodes from all events happen before attended_nodes
        with strategy specified by ngh_finder, selfloop is added
        attended nodes: nodes in the current subgraph
        Arguments:
            attended_nodes {numpy.array} shape: num_attended_nodes x 4 (eg_idx, vi, ti, node_idx), dtype int32
            -- [nodes (with time) in attended from horizon, for detail refer to ERTKG paper]

        Returns:
            sampled_edges: {numpy.array, num_edges x 8} -- (eg_idx, vi, ti, vj, tj, rel, idx_eg_vi_ti, idx_eg_vj_tj) (default: {None}), sorted ascending by eg_idx, ti, vi, tj, vj, rel dtype int32
            new_sampled_nodes: {Tensor} shape: new_sampled_nodes
        """
        if tc:
            t_start = time.time()
        src_idx_l = attended_nodes[:, 1]
        cut_time_l = attended_nodes[:, 2]
        node_idx_l = attended_nodes[:, 3]

        # src_ngh_x_batch: len(src_idx_l) x num_neighbors
        # concat(src_ngh_node_batch, src_ngh_eidx_batch, src_ngh_t_batch) is ordered by (t, node, edge) ascending
        src_ngh_node_batch, src_ngh_eidx_batch, src_ngh_t_batch = self.ngh_finder.get_temporal_neighbor(
            src_idx_l,
            cut_time_l,
            num_neighbors=num_neighbors)

        if self.ngh_finder.sampling == -1:  # full neighborhood, select neighbors with largest attention score
            assert step is not None

            selected_src_ngh_node_batch = []
            selected_src_ngh_eidx_batch = []
            selected_src_ngh_t_batch = []
            with torch.no_grad():
                for i in range(len(src_ngh_eidx_batch)):
                    src_ngh_nodes = src_ngh_eidx_batch[i]
                    if sum(src_ngh_nodes != -1) > num_neighbors:

                        mask = (src_ngh_nodes != -1)
                        src_ngh_nodes = src_ngh_nodes[mask]
                        src_ngh_eidx = src_ngh_eidx_batch[i][mask]
                        src_ngh_t = src_ngh_t_batch[i][mask]
                        src_node_embed = self.get_node_emb(np.array([src_idx_l[i]] * len(src_ngh_nodes)),
                                                           np.array([cut_time_l[i]] * len(src_ngh_nodes)),
                                                           np.array([attended_nodes[i, 0] * len(src_ngh_nodes)]))
                        ngh_node_embed = self.get_node_emb(src_ngh_nodes, src_ngh_t,
                                                           np.array([attended_nodes[i, 0] * len(src_ngh_nodes)]))
                        rel_emb = self.get_rel_emb(src_ngh_eidx, self.device)

                        att_scores = self.att_flow_list[step].cal_attention_score(
                            np.ones(len(src_ngh_nodes)) * attended_nodes[i, 0], src_node_embed, ngh_node_embed, rel_emb)
                        _, indices = torch.topk(att_scores, num_neighbors)
                        indices = indices.cpu().numpy()
                        indices_sorted_by_timestamp = sorted(indices, key=lambda x: (
                            src_ngh_t[x], src_ngh_nodes[x], src_ngh_eidx[x]))
                        selected_src_ngh_node_batch.append(src_ngh_nodes[indices_sorted_by_timestamp])
                        selected_src_ngh_eidx_batch.append(src_ngh_eidx[indices_sorted_by_timestamp])
                        selected_src_ngh_t_batch.append(src_ngh_t[indices_sorted_by_timestamp])
                    else:
                        selected_src_ngh_node_batch.append(src_ngh_nodes[-num_neighbors:])
                        selected_src_ngh_eidx_batch.append(src_ngh_eidx_batch[i][-num_neighbors:])
                        selected_src_ngh_t_batch.append(src_ngh_t_batch[i][-num_neighbors:])
                src_ngh_node_batch = np.stack(selected_src_ngh_node_batch)
                src_ngh_eidx_batch = np.stack(selected_src_ngh_eidx_batch)
                src_ngh_t_batch = np.stack(selected_src_ngh_t_batch)

        # add selfloop
        if add_self_loop:
            src_ngh_node_batch = np.concatenate([src_ngh_node_batch, src_idx_l[:, np.newaxis]], axis=1)
            src_ngh_eidx_batch = np.concatenate(
                [src_ngh_eidx_batch, np.array([[self.selfloop] for _ in range(len(attended_nodes))], dtype=np.int32)],
                axis=1)
            src_ngh_t_batch = np.concatenate([src_ngh_t_batch, cut_time_l[:, np.newaxis]], axis=1)
        # removed padded neighbors, with node idx == rel idx == -1
        src_ngh_node_batch_flatten = src_ngh_node_batch.flatten()
        src_ngh_eidx_batch_flatten = src_ngh_eidx_batch.flatten()
        src_ngh_t_batch_faltten = src_ngh_t_batch.flatten()
        eg_idx = np.repeat(attended_nodes[:, 0], num_neighbors + int(add_self_loop))
        mask = src_ngh_node_batch_flatten != -1

        # (eg_idx, src_ent, src_ts, tar_ent, tar_ts, rel, src_node_idx)
        sampled_edges = np.stack([eg_idx,
                                  np.repeat(src_idx_l, num_neighbors + int(add_self_loop)),
                                  np.repeat(cut_time_l, num_neighbors + int(add_self_loop)), \
                                  src_ngh_node_batch_flatten, src_ngh_t_batch_faltten, \
                                  src_ngh_eidx_batch_flatten,
                                  np.repeat(node_idx_l, num_neighbors + int(add_self_loop))], axis=1)[mask]

        # index new selected nodes
        target_nodes_index = []
        new_sampled_nodes = []
        for eg, tar_node, tar_ts in sampled_edges[:, [0, 3, 4]]:
            if (eg, tar_node, tar_ts) in self.node2index.keys():
                target_nodes_index.append(self.node2index[(eg, tar_node, tar_ts)])
            else:
                self.node2index[(eg, tar_node, tar_ts)] = self.num_existing_nodes
                target_nodes_index.append(self.num_existing_nodes)
                new_sampled_nodes.append([eg, tar_node, tar_ts, self.num_existing_nodes])
                self.num_existing_nodes += 1

        sampled_edges = np.concatenate([sampled_edges, np.array(target_nodes_index)[:, np.newaxis]], axis=1)
        # new_sampled_nodes sorted by node_idx so that visited_node_representation[i] is the node representation of node i
        new_sampled_nodes = sorted(new_sampled_nodes, key=lambda x: x[-1])
        new_sampled_nodes = np.array(new_sampled_nodes)

        # new_attended_nodes sorted by (eg_idx, t, ent)
        _, new_attended_nodes_idx = np.unique(sampled_edges[:, [0, 4, 3]], return_index=True, axis=0)
        new_attended_nodes = sampled_edges[:, [0, 3, 4]][new_attended_nodes_idx]

        if tc:
            tc['graph']['sample'] += time.time() - t_start
        return sampled_edges, new_sampled_nodes, new_attended_nodes

    def _topk_att_score(self, attending_nodes, attending_node_attention, k: int, tc=None):
        """

        :param attending_nodes: numpy array, N_visited_nodes x 4 (eg_idx, vi, ts, node_idx), dtype np.int32
        :param attending_node_attention: tensor, N_all_visited_nodes, dtype=torch.float32
        :param k: number of nodes in attended-from horizon
        :return:
        attended_nodes, numpy.array, (eg_idx, vi, ts)
        attended_node_attention, tensor, attention_score, same length as attended_nodes
        attended_node_emb, tensor, same length as attended_nodes
        """
        if tc:
            t_start = time.time()
        res_nodes = []
        res_att = []
        attending_node_attention = attending_node_attention[
            torch.from_numpy(attending_nodes[:, 3]).to(torch.int64).to(self.device)]
        for eg_idx in sorted(set(attending_nodes[:, 0])):
            mask = attending_nodes[:, 0] == eg_idx
            masked_nodes = attending_nodes[mask]
            masked_node_attention = attending_node_attention[mask]
            if masked_nodes.shape[0] <= k:
                res_nodes.append(masked_nodes)
                res_att.append(masked_node_attention)
            else:
                topk_node_attention, indices = torch.topk(masked_node_attention, k)
                try:
                    res_nodes.append(masked_nodes[indices.cpu().numpy()])
                except Exception as e:
                    print(indices.cpu().numpy())
                    print(max(indices.cpu().numpy()))
                    print(str(e))
                    raise KeyError
                res_att.append(topk_node_attention)
        if tc:
            tc['graph']['topk'] += time.time() - t_start

        return np.concatenate(res_nodes, axis=0), torch.cat(res_att, dim=0)

    def get_ent_emb(self, ent_idx_l, device):
        """
        help function to get node embedding
        self.entity_raw_embed[0] is the embedding for dummy node, i.e. node non-existing

        Arguments:
            node_idx_l {np.array} -- indices of nodes
        """
        
        '''
        embed_device = next(self.entity_raw_embed.parameters()).get_device()
        if embed_device == -1:
            embed_device = torch.device('cpu')
        else:
            embed_device = torch.device('cuda:{}'.format(embed_device))
        #return self.entity_raw_embed(ent_idx_l.to(embed_device)).to(device)
        return self.entity_raw_embed(torch.from_numpy(ent_idx_l).long().to(embed_device)).to(device)
        '''
        return self.ent_proj(self.entity_raw_embed[torch.from_numpy(ent_idx_l).long()])

    def get_rel_emb(self, rel_idx_l, device):
        """
        help function to get relation embedding
        self.edge_raw_embed[0] is the embedding for dummy relation, i.e. relation non-existing
        Arguments:
            rel_idx_l {[type]} -- [description]
        """
        
        '''
        embed_device = next(self.relation_raw_embed.parameters()).get_device()
        if embed_device == -1:
            embed_device = torch.device('cpu')
        else:
            embed_device = torch.device('cuda:{}'.format(embed_device))
        #return self.relation_raw_embed(rel_idx_l.to(embed_device)).to(device)
        return self.relation_raw_embed(torch.from_numpy(rel_idx_l).long().to(embed_device)).to(device)
        '''
    
        return self.rel_proj(self.relation_raw_embed[torch.from_numpy(rel_idx_l).long()])
    
    
    

def segment_max(logits, segment_ids, keep_length=True):
    """

    :param logits:
    :param segment_ids:
    :param keep_length:
    if True, return a Tensor with the same length as logits
    out[i] is the sum of segments of segment_ids[i]
    else, return a Tensor with the length of segment_ids
    :return:
    1d Tensor
    """
    n_logits = len(segment_ids)
    mask = segment_ids[1:] != segment_ids[:-1]
    seg_head_ids = np.concatenate([np.array([0]),
                                   np.arange(1, n_logits)[mask],
                                   np.array([n_logits])]).astype(np.int64)
    if keep_length:
        seg_max_ind = torch.cat([(torch.argmax(logits[torch.arange(head, tail).to(torch.int64).to(logits.device)]) + torch.tensor([head]).to(torch.int64).to(logits.device)).repeat(tail - head) for head, tail in zip(seg_head_ids[:-1], seg_head_ids[1:])])
    else:
        seg_max_ind = torch.cat([torch.argmax(logits[torch.arange(head, tail).to(torch.int64).to(logits.device)]) + torch.tensor([head]).to(torch.int64).to(logits.device) for head, tail in zip(seg_head_ids[:-1], seg_head_ids[1:])])
    return logits[seg_max_ind]

def segment_softmax_op_v2(logits, segment_ids, tc=None):
    """

    :param logits:
    :param segment_ids: numpy array, same length as logits, logits[i] belongs to segment segment_ids[i]
    logits in the same segment should aranged in a continuous block
    :param tc:
    :return:
    """

    if tc:
        t_start = time.time()

    logits_len = len(segment_ids)
    num_segments = max(segment_ids) + 1
    # numerical stable softmax
    logits = logits - segment_max(logits, segment_ids, keep_length=True)
    logits_exp = torch.exp(logits).unsqueeze(1)  # e^{logit} N x 1

    # calculate summation of exponential of logits value for each group
    sparse_index = torch.LongTensor(np.stack([segment_ids, np.arange(logits_len)]))
    sparse_value = torch.ones(logits_len, dtype=torch.float)
    trans_matrix_sparse = torch.sparse.FloatTensor(sparse_index, sparse_value,
                                                   torch.Size([num_segments, logits_len])).to(logits.device)
    softmax_den = torch.sparse.mm(trans_matrix_sparse, logits_exp)

    # repeat softmax denominator to have the same length as logits
    sparse_index2 = torch.LongTensor(np.stack([np.arange(logits_len), segment_ids]))
    sparse_value2 = torch.ones(logits_len, dtype=torch.float)
    trans_matrix_sparse2 = torch.sparse.FloatTensor(sparse_index2, sparse_value2,
                                                    torch.Size([logits_len, num_segments])).to(logits.device)
    softmax_den_repeat = torch.sparse.mm(trans_matrix_sparse2, softmax_den)

    out = torch.squeeze(logits_exp / softmax_den_repeat)
    if tc:
        tc['model']['DP_attn_softmax_v2'] += time.time() - t_start
    return out

def segment_norm_l1(logits, segment_ids):
    """
    segment_ids doesn't have to be ordered
    :param logits: Tensor
    :param segment_ids: 1-d numpy array
    :return:
    """

    N_segment = max(segment_ids) + 1
    # get denominator by multiplication logits with a matrix
    # get a 1-d tensor with a length of N_segment
    sparse_index = torch.LongTensor(np.vstack([segment_ids, np.arange(len(segment_ids))]))
    sparse_value = torch.ones(len(segment_ids), dtype=torch.float)
    trans_matrix_sparse_th = torch.sparse.FloatTensor(sparse_index, sparse_value,
                                                      torch.Size([N_segment, len(segment_ids)])).to(logits.device)
    norm_den = torch.sparse.mm(trans_matrix_sparse_th, logits.unsqueeze(1))

    # copy denominator so that it has the same lenghth as the logits and the dominator lies in the same position
    # ie den[i] is the denominator for segment_ids[i]
    sparse_index = torch.LongTensor(np.vstack([np.arange(len(segment_ids)), segment_ids]))
    sparse_value = torch.ones(len(segment_ids), dtype=torch.float)
    trans_matrix_sparse_th = torch.sparse.FloatTensor(sparse_index, sparse_value,
                                                      torch.Size([len(segment_ids), N_segment])).to(logits.device)
    den = torch.squeeze(torch.sparse.mm(trans_matrix_sparse_th, norm_den))
    res = logits / den
    res[res != res] = 0  # res != res inidcates where NaNs (0/0) are
    return res