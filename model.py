import torch.nn as nn
import torch
import numpy as np

class MainModel(nn.Module):
    def __init__(self, adapter_model, llm_gen, vocab, args):
        super(MainModel, self).__init__()
        self.args = args
        self.adapter_model = adapter_model
        self.llm_gen = llm_gen
        self.vocab = vocab
        
        if self.args.ADAPTER_NAME is not None:
            # Init embedding
            self.ent_emb, self.rel_emb = llm_gen.get_ent_rel_emb(vocab["rel2id"], vocab["ent2id"])
            self.adapter_model.entity_raw_embed = self.ent_emb
            self.adapter_model.relation_raw_embed = self.rel_emb
            # del llm_gen and empty cache
            del self.llm_gen
            torch.cuda.empty_cache()
            
        # BCE LOSS
        if self.args.LOSS_TYPE == "target_loss":
            self.loss = self.target_loss
        elif self.args.LOSS_TYPE == "all_loss":
            self.loss = nn.BCELoss()
            
        
    def forward(self, batch_data):
        '''
        Use adapter model to adapt the LLM output.
        '''
        batch_answers = None
        batch_scores = None
        
        # adapter
        if self.args.SAVE_LLM:
            adpter_distribution = None
        elif self.args.ADAPTER_NAME == "xERTE":
            entity_att_score, entities, adpter_distribution, batch_quads, lamda = self.adapter_model.forward(batch_data, device = self.args.DEVICE)
        elif self.args.ADAPTER_NAME is not None:
            adpter_distribution, rule_scores = self.adapter_model.forward(batch_data)
        else:
            adpter_distribution = None

            
        # llm
        llm_ent_distribution = None
        if "batch_LLM_result" in batch_data:
            llm_ent_distribution = batch_data["batch_LLM_result"]
        elif self.llm_gen.model is not None:
            batch_answers, batch_scores, llm_ent_distribution = self.llm_gen.forward(batch_data)

        
        # fusion
        if llm_ent_distribution is not None and adpter_distribution is not None:
            if self.args.ADAPTER_NAME != "xERTE":
                lamda = 0.5
            llm_ent_distribution = llm_ent_distribution.to(self.args.DEVICE)
            ent_distribution = lamda*adpter_distribution + (1-lamda)*llm_ent_distribution
            # ent_distribution = torch.mul(adpter_distribution + 1e-4, llm_ent_distribution + 1e-4)
            # ent_distribution = torch.nn.functional.softmax(ent_distribution)
        elif adpter_distribution is not None:
            ent_distribution = adpter_distribution
        elif llm_ent_distribution is not None:
            ent_distribution = llm_ent_distribution
        else:
            raise ValueError("No llm and adapter model")
        if self.args.ADAPTER_NAME != "xERTE":
            entity_att_score, entities = self.turn_dis_target(ent_distribution)
            
        #! For Explain
        '''
        batch_target = batch_data["batch_targets"]
        max_ent_id = ent_distribution.argmax(dim=1)
        ori_pre_id = adpter_distribution.argmax(dim=1)
        rule_matrix = self.adapter_model.rule_matrix #[num_rules, max_rule_len]
        rule_ori_conf = self.adapter_model.rule_conf #[num_rules]
        
        #save to file
        f = open("result.txt", "a")
        out = ""
        for i in range (len(batch_target)):
            right_entity_id = int(self.vocab["ent2id"][batch_target[i].lower()])
            if  right_entity_id != ori_pre_id[i].item() and right_entity_id == max_ent_id[i].item():
                print("______________________________________________________________")
                out += "______________________________________________________________\n"
                print(batch_data["batch_prompts"][i])
                out+= batch_data["batch_prompts"][i] + "\n"
                print("************************************")
                out += "************************************\n"
                print("LLM distribution:" )
                out += "LLM distribution:\n"
                
                llm_result = llm_ent_distribution[i]
                top_k = torch.topk(llm_result, 10)
                for j in range(10):
                    print(self.vocab["id2ent"][top_k.indices[j].item()], top_k.values[j].item())
                    out += self.vocab["id2ent"][top_k.indices[j].item()] + " " + str(top_k.values[j].item()) + "\n"
                
                ent_rules = batch_data["batch_ent_rules"][i]
                # list of rules, each as [rule_id ,delta_time]
                try:
                    max_ent_rules = [i[0] for i in ent_rules[str(right_entity_id)]]
                    for rule_id in max_ent_rules:
                        print("RULE:", [self.vocab["id2rel"][rel_id.item()] if rel_id.item() <len(self.vocab["id2rel"]) else 'PAD' for rel_id in rule_matrix[rule_id]], 
                            'ori_score:', rule_ori_conf[rule_id].item(),
                            'now_score:', rule_scores.squeeze(1)[rule_id].item(), '\n')
                        out += "RULE: " + str([self.vocab["id2rel"][rel_id.item()] if rel_id.item() <len(self.vocab["id2rel"]) else 'PAD' for rel_id in rule_matrix[rule_id]]) + " ori_score: " + str(rule_ori_conf[rule_id].item()) + " now_score: " + str(rule_scores.squeeze(1)[rule_id].item()) + "\n"

                    print("Adapter distribution:")
                    out += "Adapter distribution:\n"
                    #print top 10 entity id and their score
                    top_k = torch.topk(adpter_distribution[i], 10)
                    for j in range(10):
                        print(self.vocab["id2ent"][top_k.indices[j].item()], top_k.values[j].item())
                        out += self.vocab["id2ent"][top_k.indices[j].item()] + " " + str(top_k.values[j].item()) + "\n"
                except:
                    out+= "ERROR!\n"
        f.write(out)
        '''
        # entity_att_scores: [canditate ent in a batch], entities: canditate_ent_num*(bsz_id, ent_id)
        # ent_distribution: [bsz, num_ent]
        # batch_answers： K length list, each element is idx
        # batch_scores: K length list, each element is score
        if self.args.SAVE_LLM:
            return batch_answers, batch_scores, ent_distribution
        elif self.args.LOSS_TYPE == "target_loss":
            return entity_att_score, entities, ent_distribution
        else:
            return batch_answers, batch_scores, ent_distribution

    def target_loss(self, entity_att_score, entities, target_idx_l):
        # only set true ent in predicted to 1
        one_hot_label = torch.from_numpy(
            np.array([int(v == int(target_idx_l[eg_idx])) for eg_idx, v in entities], dtype=np.float32)).to(self.args.DEVICE)
        entity_att_score = entity_att_score*0.999+0.0009
        # 
        loss = torch.nn.BCELoss()(entity_att_score, one_hot_label)
        return loss
    
    def turn_dis_target(self, ent_distribution):
        '''
        Turn entity distribution to entity_att_scores: [canditate ent in a batch], entities: canditate_ent_num*(bsz_id, ent_id)
        Input: ent_distribution: [bsz, num_ent]
        Return: entity_att_scores: ent_id's attention score of entities in a batch, tensor
                entities: np.array, each one as [bsz_id, ent_id]
        '''
        
        entity_att_scores = []
        entities = []
        for i in range(ent_distribution.size(0)):
            #select ent_distribution[i] where is not 0
            ent_idx = torch.nonzero(ent_distribution[i]).squeeze(1)
            entity_att_score = ent_distribution[i][ent_idx]
            entity_att_score = torch.nn.functional.softmax(entity_att_score)
            entity_att_scores.append(entity_att_score)
            entities += [[i, int(v)] for v in ent_idx]
        entity_att_scores = torch.cat(entity_att_scores, dim=0)
        entities = np.array(entities)
        return entity_att_scores, entities
    
    
            


