import json
import os
import torch
import re
import numpy as np
from torch.utils.data import Dataset
import collections
from tqdm import tqdm
from collections import defaultdict

class BatchData(Dataset):
    """
    Dataset class for batch data
    """
    def __init__(self, data, tokenizer, ent2id, rel2id, args, rules = None, graph = None):
        '''
        Data: a list of {"context": str of his, the last one is query, "target": ent text of golden answer}, maybe with all_targets
        '''
        self.tokenizer = tokenizer
        self.ent2id = ent2id
        self.rel2id = rel2id
        self.args = args 
        self.data = data
        self.rules = rules
        self.graph = graph

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        instance = self.data[idx]
        return instance

    def collate_fn(self, data):
        '''
        Turn bs * instances in to bs * batch_input_ids, bs * targets, ...
        '''
        batch_prompts = [instance['context'] for instance in data]
        batch_targets = [instance['target'] for instance in data]
        batch_input_ids = self.tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).input_ids
        batch_at = [instance['all_targets'] for instance in data]
        # turn list of bsz tensors to [bsz, ent_nums]
        batch_at_distribution = torch.stack([instance['at_distribution'] for instance in data])
        #batch_at_distribution.requires_grad = True
        
        batch_data = {
            "batch_input_ids": batch_input_ids, 
            "batch_targets": batch_targets, 
            "batch_prompts": batch_prompts,
            "batch_at": batch_at,
            "batch_at_distribution": batch_at_distribution}
        
        if 'query_tuple' in data[0]:
            batch_queries = [instance['query_tuple'] for instance in data] # (ts, head_id, rel_id)
            batch_queries = torch.LongTensor(batch_queries)
            batch_data["batch_queries"] = batch_queries
        
        if 'his_quads' in data[0]:
            batch_his_quads = [instance['his_quads'] for instance in data]
            batch_data["batch_his_quads"] = batch_his_quads
        
        if "LLM_result" in data[0]:
            batch_LLM_result = [instance["LLM_result"] for instance in data]
            batch_LLM_result = torch.stack(batch_LLM_result) # [bsz, ent_nums]
            batch_data["batch_LLM_result"] = batch_LLM_result
            
        if "ent_rules" in data[0]:
            batch_ent_rules = [instance['ent_rules'] for instance in data]
            batch_data["batch_ent_rules"] = batch_ent_rules
            
        return batch_data


def prepare_data(args):
    '''
    Load dataset and return each item as:
    {"context": str of his, the last one is query,
     "target": ent text of golden answer}
    '''
    #load train dataset
    train_data_path = os.path.join(args.DATA_PATH, "train", args.DATASET, args.DATASET + ".json")
    train_data = json.load(open(train_data_path))
    test_context_path = os.path.join(args.DATA_PATH, "eval/history_facts/history_facts_" + args.DATASET + ".txt")
    test_answer_path = os.path.join(args.DATA_PATH, "eval/test_answers/test_ans_" + args.DATASET + ".txt")
    with open(test_context_path, 'r', encoding='utf-8') as f:
        test_context = f.read()
        test_context = test_context.split('\n\n')
    with open(test_answer_path, "r", encoding='utf-8') as f:
        lines = f.readlines()
        test_ans = [lines[i].split('\t')[2] for i in range(len(lines))]
    test_data = [{"context": context, "target": target} for context, target in zip(test_context, test_ans)]
    #split train data to train and valid
    datalen = int(args.TRAIN_PROP * (int(len(train_data)*0.8) - int(len(train_data)*0.2)))
    train_split = train_data[int(len(train_data)*0.8) - datalen:int(len(train_data)*0.8)]
    valid_split = train_data[int(len(train_data)*0.8):]
    return {"train": train_split, "valid":valid_split, "test": test_data}

def prepare_graph(args, rel_num):
    '''
    Load facts of each split and construct the graph,
    return adj_dict as {head_id : {rel_id: [(tail_id, ts), ...]}}
    '''
    splits = ["train", "valid", "test"]
    facts = {}
    adj_dict = collections.defaultdict(lambda: collections.defaultdict(list))
    for split in splits:
        with open (os.path.join(args.ORI_DATA_PATH, args.DATASET, split + ".txt"), 'r') as f:
            lines = f.readlines()
            facts[split] = [line.strip() for line in lines] #head, rel, tail, ts
    # Create adjacency list for the graph, each item as {head_id : {rel_id: [tail_id, ts]}}
        for i, fact in enumerate(facts[split]):
            if args.DATASET == "GDELT":
                head, rel, tail, ts, _ = fact.split("\t")
                ts = int(ts)//15
            else:
                head, rel, tail, ts = fact.split("\t")
            if split != "test":
                adj_dict[int(head)][int(rel)].append((int(tail), int(ts)))
                # add reverse
                adj_dict[int(tail)][int(rel) + rel_num].append((int(head), int(ts)))
            facts[split][i] = (int(head), int(rel), int(tail), int(ts))
    #convert adj_dict to normal dict
    adj_dict = {k: dict(v) for k, v in adj_dict.items()}
    #sort the tail entities by ts
    for value in adj_dict.values():
        for rel_id, rel_value in value.items():
            rel_value.sort(key=lambda
                            x: (x[1], x[0]))
            
    return facts, adj_dict

def prepare_graph_xERTE(args, rel_num):
    '''
    Load facts of each split and construct the graph (for xERTE),
    return adj_dict as {head_id : [(tail_id, rel_id, ts), ...]}
    '''
    splits = ["train", "valid", "test"]
    facts = {}
    adj_dict = defaultdict(list)
    for split in splits:
        with open (os.path.join(args.ORI_DATA_PATH, args.DATASET, split + ".txt"), 'r') as f:
            lines = f.readlines()
            facts[split] = [line.strip() for line in lines] #head, rel, tail, ts
    # Create adjacency list for the graph, each item as {head_id : {rel_id: [tail_id, ts]}}
        for i, fact in enumerate(facts[split]):
            if args.DATASET == "GDELT":
                head, rel, tail, ts, _ = fact.split("\t")
                ts = int(ts)//15
            else:
                head, rel, tail, ts = fact.split("\t")
            if split != "test":
                adj_dict[int(head)].append((int(tail), int(rel), int(ts)))
                # add reverse
                adj_dict[int(tail)].append((int(head), int(rel) + rel_num, int(ts)))
            facts[split][i] = (int(head), int(rel), int(tail), int(ts))
            
    adj_dict = dict(adj_dict)        
    for value in adj_dict.values():
        value.sort(key=lambda x: (x[2], x[0], x[1]))
    return facts, adj_dict


def load_vocab(ori_dataset_path):
    '''
    Load relation2id and entity2id from ori_dataset_path.
    '''
    rel2id_file = os.path.join(ori_dataset_path, "relation2id.json")
    ent2id_file = os.path.join(ori_dataset_path, "entity2id.json")
    relation2id_old = json.load(open(rel2id_file))
    #lower key
    relation2id_old = dict((k.lower(), v) for k, v in relation2id_old.items())
    relation2id = relation2id_old.copy()
    counter = len(relation2id_old)
    for relation in relation2id_old:
        relation2id["Inv_" + relation] = counter  # Inverse relation
        counter += 1    
    id2relation = dict([(v, k) for k, v in relation2id.items()])

    ent2id = json.load(open(ent2id_file))
    ent2id = dict((k.lower(), v) for k, v in ent2id.items())
    id2ent = dict([(v, k) for k, v in ent2id.items()])
    return {"rel2id": relation2id, "id2rel": id2relation, "ent2id": ent2id, "id2ent": id2ent}


def llm_transform(data, ori_data, ent2id, rel2id, args):
    '''
    Transform the context and target format according to args.LABEL_TYPE as llm input
    instance: dict with keys 'context' and 'target'
    '''
    new_data = []
    #avg_len = 0
    for i, instance in enumerate(data):
        his_query = instance['context']
        if "\n\n" in instance['context']:
            instruct, his_query = instance['context'].split("\n\n")
            instruct = instruct.replace('<s>', "").replace('[INST]', "").replace('<<SYS>>', "")
            his_query = his_query.replace("<</SYS>>", "").replace("[/INST]", "")
        
        if args.LABEL_TYPE == "id":
            if args.WITH_INSTRUCT:
                instruct = instruct.replace('{object_label}.{object}', '{object_label}')
            his_query = re.sub(r'\.[a-zA-Z0-9_.\-()/,]+]', '', his_query)
            # substitute ent_name to id in his_query and query
            li_each_hist = his_query.split('\n')
            #print('context length', len(li_each_hist)-1)
            new_his_query = ""
            for j in range(len(li_each_hist)-1):
                head_time, rel_name, tail_id = li_each_hist[j].split(', ')
                ts, head_name = head_time.split(':')
                if args.DATASET == "GDELT":
                    ts = str(int(ts)//15)
                head_name = head_name.replace("[", "").strip()
                rel_name = rel_name.strip()
                head_id = ent2id[head_name.lower().replace(" ", "_")]
                tail_id = tail_id.strip() +"]"
                new_each_hist = f"[{head_id}, {rel_name},{tail_id}"
                new_his_query += ts + ": " + new_each_hist + "\n"
                
            #deal with the last line
            try:
                head_time, rel_name = li_each_hist[-1].split(', ')
            except:
                head_id, rel_id, tail_id, ts = ori_data[i]
                if args.DATASET == "GDELT":
                    ts = str(int(ts)//15)
                instance['query'] = f"{ts}: [{head_id}, {rel_id}"
                instance['context'] = instance['query']
                instance['query_tuple'] = (ts, head_id, rel_id)
                instance['target'] = str(instance['target'])
                continue
                
            ts, head_name = head_time.split(':')
            if args.DATASET == "GDELT":
                ts = str(int(ts)//15)
            head_name = head_name.replace("[", "").strip()
            rel_name = rel_name.strip()
            head_id = ent2id[head_name.lower().replace(" ", "_")]
            instance['query'] = f"{ts}: [{head_id}, {rel_name}"
            instance['query_tuple'] = (int(ts), head_id, rel2id[rel_name[:-1].lower().replace(" ", "_")])
            
            new_his_query += instance['query']
            if re.match(r'\d+\.', instance['target']):
                instance['target'] = instance['target'].split(".")[0]
            else:
                instance['target'] = str(ent2id[instance['target'].lower().replace(" ", "_")])
            instance['context'] = new_his_query
            
        elif args.LABEL_TYPE == "text":
            if args.WITH_INSTRUCT:
                instruct = instruct.replace('{object_label}.{object}', '{object}')
                instruct = instruct.replace('{object_label}', '{object}')
            his_query = re.sub(r'\d+\.', '', his_query)
            
            instance['target'] = re.sub(r'\d+\.', '', instance['target'])
            instance['query'] = his_query.split('\n')[-1]
            instance['context'] = his_query
            try:
                head_ts, rel_name = instance['query'].split(", ")
            except:
                head_id, rel_id, tail_id, ts = ori_data[i]
                if args.DATASET == "GDELT":
                    ts = str(int(ts)//15)
                instance['query_tuple'] = (int(ts), head_id, rel_id)
                continue
                
            ts, head_name = head_ts.split(":")
            if args.DATASET == "GDELT":
                ts = str(int(ts)//15)
            head_name = head_name.replace("[", "").strip()
            rel_name = rel_name[:-1].strip()
            head_id = ent2id[head_name.lower().replace(" ", "_")]
            rel_id = rel2id[rel_name.lower().replace(" ", "_")]
            instance['query_tuple'] = (int(ts),head_id, rel_id)
            
        elif args.LABEL_TYPE == "both":
            if re.match(r'\d+\.', instance['target']) is not None:
                #instance['target'] = str(ent2id[instance['target'].lower()]) + "." + instance['target']
                #instance['target'] = str(ent2id[instance['target'].lower()])
                instance['target'] = instance['target'].split(".")[0]
            instance['query'] = his_query.split('\n')[-1]
            instance['context'] = his_query

        new_data.append(instance)

        #avg_len += len(instance['context'])
    
    #print("avg_len: ", avg_len/len(new_data))
    return new_data

def load_rules(rule_file_path, args):
    '''
    Load rules from rule_file_path,
    return rules as {head_rel_id: list of dict {"head_rel": head_rel_id, "body_rels":[rel_id1, rel_id2, ...], "conf":0.05}}
    '''
    # get all file names in the path
    all_files = os.listdir(rule_file_path)
    # get first file
    rule_file = os.path.join(rule_file_path, all_files[0])
    with open(rule_file, 'r') as f:
        rules = json.load(f)
    # add rule idx to each rule
    base = 0
    filter_rules = {}
    for rel_id, ass_rules in rules.items():
        filter_ass_rules = []
        for j, each_rule in enumerate(ass_rules):
            if each_rule["conf"] > 0.01:
                filter_ass_rules.append(each_rule)
                each_rule["idx"] = base
                base += 1
        filter_rules[rel_id] = filter_ass_rules
    args.rule_num = base
    return filter_rules

def add_history(instances, graphs, p_rel_num):
    '''
    Add "his_quads" to instance based on graph information.
    '''
    for instance in instances:
        q_ts, q_head_id, q_rel_id = instance['query_tuple']
        his_quad_reuslt = [] #result: [vi, vj, tj, rel]
        if q_head_id not in graphs:
            head_his = [] if isinstance(next(iter(graphs.values())), list) else {}
        else:
            head_his = graphs[q_head_id]
            
        # not add inverse relations to his
        if isinstance(head_his, list):
            # {head_id : [(tail_id, rel_id, ts), ...]}
            for quad_i in head_his:
                if quad_i[1] < p_rel_num and quad_i[2] < q_ts:
                    his_quad_reuslt.append([q_head_id, quad_i[0], quad_i[2], quad_i[1]])
                    
        if isinstance(head_his, dict):
            # {head_id : {rel_id: [(tail_id, ts), ...]}}
            for rel_id, tail_ts_list in head_his.items():
                if rel_id < p_rel_num:
                    for tail_id, ts in tail_ts_list:
                        if ts < q_ts:
                            his_quad_reuslt.append([q_head_id, tail_id, ts, rel_id])
                
        #turn to numpy
        his_quad_reuslt = np.asarray(his_quad_reuslt)
        instance["his_quads"] = his_quad_reuslt
    return instances


def apply_rule_to_data(instances, rules, graph, split):
    '''
    Apply rules to instance, find the target ent_id that can be reached by the rule:
    rules: {"head_rel_id": list of dict {"head_rel": head_rel_id, "body_rels":[rel_id1, rel_id2, ...]}}
    For each rule, find the target ent_id,
    return: {ent_id: [rule_id in its head_rel_list]}
    '''
    def get_graph_relations(graph, ent, rel):
        return graph.get(ent, {}).get(rel, [])
    
    def apply_rule_to_instance(instance, rules, graph):
        ts, head_id, rel_id = instance["query_tuple"]
        ass_rules = rules.get(str(rel_id), [])
        result = defaultdict(list)
        
        for rule_dict in ass_rules:
            body_rels = rule_dict["body_rels"]
            for i, rel in enumerate(body_rels):
                if i == 0:
                    current_entities = {head_id: 0}
                next_entities = {}
                for ent, min_ts in current_entities.items():
                    #for ent in current_entities:
                    for item in get_graph_relations(graph, ent, rel):
                        tail_id, edge_ts = item
                        if edge_ts < ts:
                            if i == 0: # mark the first added edge 
                                min_ts = edge_ts
                            next_entities[tail_id] = min_ts
                current_entities = next_entities
                
            #for ent in next_entities:
                #result[ent].append(rule_dict["idx"])
            for ent, min_ts in next_entities.items():
                result[ent].append([rule_dict["idx"], ts - min_ts])
        instance['ent_rules'] = result
        return result
    
    all_results = []
    for i in tqdm(range(len(instances)), desc=f"Apply rules to {split}"):
        all_results.append(apply_rule_to_instance(instances[i], rules, graph))
    return all_results


        
def gen_ent_distribution(data, ent2id):
    '''
    Turn golden targets to entity distribution.
    '''
    for splits, instances in data.items():
        #generate the all_targets_dict, key is the query, value is the set of all targets
        all_targets_dict = {}
        for instance in instances:
            query = instance['query']
            target = instance['target']
            if query not in all_targets_dict:
                all_targets_dict[query] = set()
            all_targets_dict[query].add(target)
                
        for instance in instances:
            instance['all_targets'] = all_targets_dict[instance['query']]
            #generate the distribution of all targets, 1 for true target, 0 for others
            target_distribution = torch.zeros(len(ent2id))
            for target in instance['all_targets']:
                #preprocess the target to ent id
                if re.match(r'\d+\.', target) is not None:
                    target_id = re.match(r'(\d+)\.', target).group(1)
                elif target.isdigit():
                    target_id = target
                else:
                    target_id = ent2id[target.lower()]
                target_distribution[int(target_id)] = 1.0
            instance['at_distribution'] = target_distribution
    return data

def turn_topk2distribution(topk, ent2id):
    '''
    Turn the topk list to distribution.
    '''
    distribution = torch.zeros(len(ent2id))
    for ent_id_str, score in topk:
        try:
            ent_id = int(ent_id_str)
            distribution[ent_id] = score
        except:
            pass
            #print(f"ent_id_str: {ent_id_str} is not int")
    return distribution