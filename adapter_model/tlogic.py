import torch
import torch.nn as nn
import os
import json
#from line_profiler import LineProfiler

class TLogic(nn.Module):
    def __init__(self, args, rules, **kwargs):
        '''
        1. Preload all extracted rules and construct the rule matrix, use neural network and time delta to calculate each rule score.
        2. Use ent_rule_sparse to construct the sparse tensor (ent_id, its satified rule id) for each batch.
        3. ent_rule_sparse * rule matrix → ent_score 
        '''
        super(TLogic, self).__init__()
        self.args = args
        self.rules = rules
        self.max_rule_len = 3
        

        with torch.no_grad():
            self.rule_matrix, self.rule_conf = self.construct_rule_matrix()
            
        self.MLP = nn.Sequential(
            nn.Linear(self.max_rule_len * args.EMBEDDING_DIM, args.EMBEDDING_DIM),
            nn.ReLU(),
            nn.Linear(args.EMBEDDING_DIM, args.EMBEDDING_DIM)
        )
        self.rel_MLP = nn.Sequential(
            nn.Linear(args.EMBEDDING_DIM, args.EMBEDDING_DIM),
            nn.ReLU(),
            nn.Linear(args.EMBEDDING_DIM, 1),
            nn.Sigmoid()
        )
        self.time_MLP = nn.Sequential(
            nn.Linear(1, args.EMBEDDING_DIM),
            nn.ReLU(),
            nn.Linear(args.EMBEDDING_DIM, 1),
            nn.Sigmoid()
        )
        
        self.rel_proj = nn.Linear(args.llm_dim, args.EMBEDDING_DIM)
    
        #self.gru = torch.nn.GRU(args.EMBEDDING_DIM, args.EMBEDDING_DIM, batch_first=True, bidirectional=True).to(args.DEVICE)
        #self.rnn = torch.nn.RNN(args.EMBEDDING_DIM, args.EMBEDDING_DIM,batch_first=True).to(args.DEVICE)
        self.lstm = torch.nn.LSTM(args.EMBEDDING_DIM, args.EMBEDDING_DIM, batch_first=True).to(args.DEVICE)
        
        
    def turn_ent_rule_sparse(self, batch_ent_rules):
        '''
        Turn:
        batch_ent_rules: each batch as {ent_id: [rule_id in its head_rel_list]}
        
        To:
        construct spare matrix, [batch, ent_nums, rule_nums]
        '''
        rule_bool_result = []
        rule_time_result = []
        ent_nums = self.args.ent_num
        rule_nums = self.args.rule_num
        for index, ent_rules in enumerate(batch_ent_rules):
            indices = []
            time_delta_values = []
            for ent_id, rule_info in ent_rules.items():
                for rule_id, time_delta in rule_info:
                    indices.append([int(ent_id), rule_id])
                    time_delta_values.append(time_delta)
            if (len(indices)) != 0:
                values = [1] * len(indices)
                indices = torch.LongTensor(indices).t().to(self.args.DEVICE)
                values = torch.FloatTensor(values).to(self.args.DEVICE)
                time_delta_values = torch.FloatTensor(time_delta_values).to(self.args.DEVICE).requires_grad_(False)
                time_delta_values = torch.exp(-0.1 * time_delta_values)
                #time_delta_values = self.time_MLP(time_delta_values.unsqueeze(1)).squeeze(1)
            else:
                indices = torch.LongTensor([[0, 0]]).t().to(self.args.DEVICE)
                values = torch.FloatTensor([0]).to(self.args.DEVICE)
                time_delta_values = torch.FloatTensor([0]).to(self.args.DEVICE).requires_grad_(False)
            bool_sparse_tensor = torch.sparse_coo_tensor(indices, values, (ent_nums, rule_nums))
            time_sparse_tensor = torch.sparse_coo_tensor(indices, time_delta_values, (ent_nums, rule_nums))
            rule_bool_result.append(bool_sparse_tensor)
            rule_time_result.append(time_sparse_tensor)
            
        return rule_bool_result, rule_time_result

    def construct_rule_matrix(self):
        '''
        Construct the rule matrix, each rule is a row, each column is a rel id
        # row_num: num_rules
        # col_num: max rule body len
        '''
        result = []
        rule_conf = []
        for query_rel, ass_rules in self.rules.items():
            for i, rule_dict in enumerate(ass_rules):
                body_rels = rule_dict["body_rels"]
                # add pad, set to max rel_id
                pad_id = self.args.rel_num
                for i in range(len(body_rels), self.max_rule_len):
                    body_rels.append(pad_id)
                each_row = [int(query_rel)] + body_rels
                result.append(each_row)
                rule_conf.append(rule_dict["conf"])
        #turn to LongTensor
        result = torch.LongTensor(result).to(self.args.DEVICE)
        rule_conf = torch.FloatTensor(rule_conf).to(self.args.DEVICE)
        return result, rule_conf
    
    def get_rule_score(self):
        '''
        Use neural network to calculate score for each rule.
        # Input: [N, max_rule_len + 1 (query_rule, body_rule1, body_rule2, ...)],
        # query_rule * LSTM (body_rule1, body_rule2, ...)
        # output: [N,1]
        '''
        with torch.no_grad():
            query_rules = self.rule_matrix[:, 0].clone() #[N]
            body_rules = self.rule_matrix[:, 1:].clone() #[N, max_rule_len]
        query_embs = self.rel_embedding[query_rules] #[N, emb_dim]
        body_embs = self.rel_embedding[body_rules] #[N, max_rule_len, emb_dim]
        
        
        # Method1: MLP
        '''
        body_embs_view = body_embs.reshape(body_embs.size(0),  body_embs.size(1)* body_embs.size(-1)) #[N, max_rule_len*emb_dim]
        body_out = self.MLP(body_embs_view) #[N, emb_dim]
        '''
        
        # Method2: LSTM to compute the last hidden state
        body_out, _ = self.lstm(body_embs)
        body_out = body_out[:, -1, :] #[N, emb_dim]

        scores = torch.nn.functional.cosine_similarity(query_embs, body_out, dim=1, eps=1e-8).unsqueeze(1)
        return scores

 
    def forward(self, batch_data):
        '''
        Calculate the score for each entity in the batch by adding the NN score and time score.
        '''

        with torch.no_grad():
            batch_ent_rules = batch_data["batch_ent_rules"]
            batch_query_rels = batch_data["batch_queries"][:, -1].to(self.args.DEVICE)
            batch_rule_index, batch_rule_time = self.turn_ent_rule_sparse(batch_ent_rules)
            
        # use query_rels and MLP to calculate the weight for time and rule
        self.rel_embedding = self.rel_proj(self.relation_raw_embed)
        self.rel_embedding = torch.cat((self.rel_embedding, torch.zeros(1, self.args.EMBEDDING_DIM).to(self.args.DEVICE)), dim=0)
        batch_query_rel_embs = self.rel_embedding[batch_query_rels] # [batch, emb_dim]
        b_time_weight = self.rel_MLP(batch_query_rel_embs)
        
        # use NN(eg. LSTM) to calculate the score for each rule
        rule_scores = self.get_rule_score() # [rule_nums, 1]

        # add time score and learned rule score
        rule_time_scores = []
        ent_scores = []
        for batch_id, sparse_rule_matrix in enumerate(batch_rule_index):
            # sum the row of batch_rule_time[batch_id]
            # [ent_nums, rule_nums] → [ent_nums]
            rule_time_score = torch.sparse.sum(batch_rule_time[batch_id], dim=1).to_dense()
            rule_time_scores.append(rule_time_score)
            # [ent_nums, rule_nums] * [rule_nums, 1] → [ent_nums]
            ent_score = torch.sparse.mm(sparse_rule_matrix, rule_scores).squeeze(1)
            ent_scores.append(ent_score)

        rule_time_scores = torch.stack(rule_time_scores, dim=0) # [batch, ent_nums]
        rule_time_scores = torch.nn.functional.softmax(rule_time_scores)

        ent_scores = torch.stack(ent_scores, dim=0) # [batch, ent_nums]
        ent_scores = torch.nn.functional.softmax(ent_scores)
        
        combine_score = (1-b_time_weight)*ent_scores + b_time_weight * rule_time_scores
        #combine_score = ent_scores
        
        return combine_score, rule_scores