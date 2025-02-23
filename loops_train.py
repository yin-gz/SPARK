import torch
from tqdm import tqdm
import os
import time
import wandb
from torch.cuda import amp
import numpy as np
from load_data import BatchData, llm_transform, load_rules, apply_rule_to_data, gen_ent_distribution, turn_topk2distribution, add_history
from torch.utils.data import DataLoader
from llm import LLMGenerator
from typing import List
from model import MainModel
from adapter_model.tlogic import TLogic
from adapter_model.xERTE import xERTE
import json
import os


class Trainer:
    def __init__(self, args, tlr_data, facts, graphs, vocab_dict):
        self.args = args
        self.tlr_data = tlr_data
        self.ori_facts = facts
        self.graphs = graphs
        self.rel2id, self.id2rel, self.ent2id, self.id2ent = \
        vocab_dict["rel2id"], vocab_dict["id2rel"], vocab_dict["ent2id"], vocab_dict["id2ent"]
        self.args.rel_num = len(vocab_dict["rel2id"])
        self.args.ent_num = len(vocab_dict["ent2id"])
        self.rules = load_rules("./data/rule_output/" + self.args.DATASET, args)
        
        #!1. Complete instance information
                
        #1.1 Format data for LLM input
        self.data = {}
        for split, split_data in self.tlr_data.items():
            self.data[split] = llm_transform(split_data, self.ori_facts[split], self.ent2id, self.rel2id, self.args)
            # add extracted historical information to each instance
            if self.args.RAG == "ICL":
                self.data[split] = add_history(self.data[split], self.graphs, p_rel_num = self.args.rel_num/2)
                
        #1.2 Add rule information to each instance (only for TLogic)
        # find the target ent_id that can be reached by the rule {ent_id: [rule_id in its head_rel_list]}
        if args.LOAD_RULE:
            rule_data_path = os.path.join(args.RULE_DATA_PATH, args.DATASET)
            if not os.path.exists(rule_data_path):
                os.makedirs(rule_data_path)
            for split in ["train", "valid", "test"]:
                split_rule_file_path = f"{rule_data_path}/{split}.json"
                if os.path.exists(split_rule_file_path):
                    with open(split_rule_file_path, "r") as f:
                        print("Load rules of ", split)
                        split_ent_rules = json.load(f)
                        for i, instance in enumerate(self.data[split]):
                            instance['ent_rules'] = split_ent_rules[i]
                else:
                    # save the preprocessed data as json files
                    split_data = self.data[split]
                    split_ent_rules = apply_rule_to_data(self.data[split], self.rules, self.graphs, split)
                    with open(split_rule_file_path, "w") as f:
                        json.dump(split_ent_rules, f)
                    
        #1.3 add precomputed LLM result to each instance if exists
        self.llm_result_path = os.path.join("./data/llm_result")
        if self.args.LOAD_LLM:
            for split, v in self.data.items():
                self.data[split] = self.load_LLM_output(self.data[split], split)
                
        #!2. Set LLM and Adapter Models
        # load llm model
        self.llm_gen = LLMGenerator(self.ent2id, self.id2rel, self.args)
        self.llm_model, self.tokenizer = self.llm_gen.model, self.llm_gen.tokenizer
        
        # laod adapter model
        self.optimizer = None
        self.adapter_model = None
        if self.args.ADAPTER_NAME is not None:
            self.apth = "model_checkpoints/" + self.args.DATASET + "-" + self.args.ADAPTER_NAME + ".pt"
            if self.args.LOAD_ADAPTER and os.path.exists(self.apth):
                self.adapter_model = torch.load(self.apth, map_location = self.args.DEVICE) 
            else:
                self.adapter_model = globals()[args.ADAPTER_NAME](self.args, 
                                                              rules = self.rules, 
                                                              base_graph = self.graphs)
                                                              #DP_steps=4,
                                                              #emb_dim=[256, 128, 64, 32, 16])
            self.adapter_model.to(self.args.DEVICE)
            if self.adapter_model.parameters() is not None:
                self.optimizer = torch.optim.AdamW(self.adapter_model.parameters(), lr=self.args.LEARNING_RATE)
                if self.args.DR == "plateau":
                    self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", patience=1, factor=0.8, min_lr=1e-7, verbose=False)
                elif self.args.DR == "onecycle":
                    self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.args.LEARNING_RATE, steps_per_epoch=50, epochs=self.args.EPOCHS)
                elif self.args.DR == "cosine":
                    self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50, eta_min=0)
                elif self.args.DR == "stepLR":
                    #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.EPOCHS*len(data['train'])//args.TRAIN_BS, gamma=0.1)
                    self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.9)
                self.scaler = amp.GradScaler()
                
        # init model
        self.model = MainModel(self.adapter_model, self.llm_gen, vocab_dict, args)
        self.model.to(self.args.DEVICE)
        
        # print trainable parameters size
        trainable_params_size = sum(p.numel() * p.element_size() for p in self.model.parameters() if p.requires_grad)
        print(f"Total trainable parameters size: {trainable_params_size / 1024 / 1024:.2f} MB")
        
        #!3. Construct data loader
        self.data = gen_ent_distribution(self.data, self.ent2id) # add key "all_targets"
        self.data_iter = self.get_data_loader(self.data)
        
        #clear exist results
        if self.args.SAVE_LLM:
            self.clear_LLM_output()

    def get_data_loader(self, data_splits):
        self.data_iter = {}
        if self.args.SAVE_LLM or self.args.ONLY_TEST:
            is_shuffle = False
        else:
            is_shuffle = True
        for split, split_data in data_splits.items():
            dataset_class = BatchData(split_data, self.tokenizer, self.ent2id, 
                                      self.rel2id, self.args, self.rules, self.graphs)
            if split == "train":
                bsz = self.args.TRAIN_BS
            else:
                bsz = self.args.EVAL_BS
            self.data_iter[split] = DataLoader(
                dataset_class,
                batch_size=bsz,
                shuffle=is_shuffle,
                num_workers=max(0, self.args.NUM_WORKERS),
                collate_fn=dataset_class.collate_fn
            )
        return self.data_iter
        
    def train(self, resume_from_checkpoint=None):
        torch.autograd.set_detect_anomaly(True)
        if resume_from_checkpoint is not None:
            checkpoint = torch.load(resume_from_checkpoint)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = 0
        else:
            start_epoch = 0
        
        if self.args.ONLY_TEST:
            test_results = self.evaluate(start_epoch, 'test')
            print(f"Test Results: {test_results}")
            return
        
        results = None
        best_mrr = 0
        for epoch in range(start_epoch, self.args.EPOCHS):
            #self.model.train()
            per_epoch_loss = [] # store loss of each step in one epoch
            
            for i, batch in enumerate(tqdm(self.data_iter['train'], desc=f"Epoch {epoch}")):
                batch = self.batch_to_device(batch, self.args.DEVICE)
                if self.args.LOSS_TYPE == "target_loss" and self.args.SAVE_LLM is False:
                    entity_att_score, entities, ent_distribution = self.model.forward(batch)
                else:
                    batch_answers, batch_scores, ent_distribution = self.model.forward(batch)
                
                if self.args.SAVE_LLM:
                    self.save_LLM_output(batch_answers, batch_scores, 'train')
                        
                # calculate loss
                if self.optimizer is not None:
                    if self.args.LOSS_TYPE == "target_loss":
                        loss = self.model.loss(entity_att_score, entities, batch["batch_targets"])
                    else:
                        loss = self.model.loss(ent_distribution, batch["batch_at_distribution"])
                    per_epoch_loss.append(loss.detach().item())
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.adapter_model.parameters(), 1.)
                    self.optimizer.step()
                    if self.args.DR != "plateau":
                        self.scheduler.step()
                        
                    if i % self.args.LOGGING_STEPS == 0:
                        print(f"Step Loss: {per_epoch_loss[-1]}")
                    
                results = self.cal_metrics_all(ent_distribution, batch["batch_targets"], batch["batch_at_distribution"], results)
                # print loss and results
                if i % self.args.LOGGING_STEPS == 0:
                    print(f"Train Results: {results}")

                # report train result
                if self.args.WANDB:
                    #train_time = time.time() - start_train
                    if hasattr(self.scheduler, '_last_lr'):
                        wandb.log({"epoch": epoch, "loss": float(per_epoch_loss[-1]),
                                   'lr': float(self.scheduler._last_lr[0])})
                    else:
                        wandb.log({"epoch": epoch, "loss": float(per_epoch_loss[-1]), 
                                   'lr': float(self.optimizer.param_groups[0]['lr'])})  
                    for key, value in results.items():
                        if "c" in key:
                            continue
                        else:
                            wandb.log({f"train/{key}":value ,"epoch": epoch})             
            
            # scheduler
            if self.optimizer is not None:
                if self.args.DR == "plateau":
                    self.scheduler.step(np.mean(per_epoch_loss))
                    
            # evaluate on the valid set
            start_valid = time.time()
            valid_results = self.evaluate(epoch, 'valid')
            valid_time = time.time() - start_valid
            print(f"Valid Results: {valid_results}")
            if self.args.WANDB:
                wandb.log({f"valid/time": valid_time, "epoch": epoch})
                for key, value in valid_results.items():
                    if "c" in key:
                        continue
                    else:
                        wandb.log({f"valid/{key}":value ,"epoch": epoch})
            
            # evaluate on the test set
            start_test = time.time()
            test_results = self.evaluate(epoch, 'test')
            test_time = time.time() - start_test
            print(f"Test Results: {test_results}")
            if self.args.WANDB:
                wandb.log({f"test/time": test_time, "epoch": epoch})
                for key, value in test_results.items():
                    if "c" in key:
                        continue
                    else:
                        wandb.log({f"test/{key}":value ,"epoch": epoch})

            if valid_results['mrr'] > best_mrr:
                best_mrr = valid_results['mrr']
                print("Get best MRR: ", best_mrr)
                if self.args.TEST_ADAPTER:
                    # save model as pt
                    torch.save(self.adapter_model, self.apth)
                elif self.optimizer is not None:
                    self.save_model(epoch)
            else:
                print("Mrr not improved")
                
            
            
    def evaluate(self, epoch, split):
        self.model.eval()
        results = None
        print(f"Start evaluating on {split} set for epoch {epoch}")
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.data_iter[split], desc=f"Epoch {epoch}")):
                batch = self.batch_to_device(batch, self.args.DEVICE)
                if self.args.LOSS_TYPE == "target_loss" and self.args.SAVE_LLM is False:
                    entity_att_score, entities, ent_distribution = self.model.forward(batch)
                else:
                    batch_answers, batch_scores, ent_distribution = self.model.forward(batch)
                    
                if self.args.SAVE_LLM:
                    self.save_LLM_output(batch_answers, batch_scores, split)
                results = self.cal_metrics_all(ent_distribution, batch["batch_targets"], batch["batch_at_distribution"], results)
                print(results)
        return results
    
    def save_model(self, epoch):
        if not os.path.exists(self.args.OUTPUT_DIR):
            os.makedirs(self.args.OUTPUT_DIR)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'epoch': epoch
        }, f"{self.args.OUTPUT_DIR}/"+self.args.RUN_NAME+".pt")
        
    def clear_LLM_output(self):
        if not os.path.exists(self.llm_result_path):
            os.makedirs(self.llm_result_path)

        suffix = ".txt"
        model_name = self.args.MODEL_NAME.split("/")[-1]
        path = f"{self.llm_result_path}/{model_name}/{self.args.RAG}"
    
        if not os.path.exists(path):
            os.makedirs(path)
        
        # if exist file_name, delete it
        for split in ["train", "valid", "test"]:
            file_name = split+"_ans_"+self.args.DATASET+suffix
            if os.path.exists(path + "/" + file_name):
                os.remove(path + "/" + file_name)
            
        
    def save_LLM_output(self, batch_answers, batch_scores, split):
        suffix = ".txt"
        model_name = self.args.MODEL_NAME.split("/")[-1]
        path = f"{self.llm_result_path}/{model_name}/{self.args.RAG}"
        file_name = split+"_ans_"+self.args.DATASET+suffix
        
        with open(path + "/" + file_name, "a") as f:
            for i in range(len(batch_answers)):
                #format to tuple list
                tuple_list = [(ent, score) for ent, score in zip(batch_answers[i], batch_scores[i])]
                f.write(json.dumps(tuple_list) + "\n")
                
    def load_LLM_output(self, instances, split):
        '''
        Add "LLM_result" to instances.
        '''
        suffix = ".txt"
        model_name = self.args.MODEL_NAME.split("/")[-1]
        ent_distribution_all = []
        path = f"{self.llm_result_path}/{model_name}/{self.args.RAG}/"+split+"_ans_"+self.args.DATASET+suffix
            
        with open(path, "r") as f:
            for i, line in enumerate(f):
                topk_ent_scores = json.loads(line)
                # filter and turn to ent distribution
                ent_distribution = turn_topk2distribution(topk_ent_scores, self.ent2id)
                ent_distribution_all.append(ent_distribution)
        for j in range(len(instances)):
            instances[-1-j]["LLM_result"] = ent_distribution_all[-1-j]
        return instances

    def batch_to_device(self, batch, device):
        '''
        If batch is pytorch tensor, move to device.
        '''
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                if key != "batch_queries":
                    batch[key] = value.to(device)
            if isinstance(value, List):
                for i, item in enumerate(value):
                    if isinstance(item, torch.Tensor):
                        batch[key][i] = item.to(device)
        return batch
    
    def cal_metrics_all(self, pred_scores, gt, ent_dis, results = None):
        """
        pred_scores: [batch_size, num_entities]
        gt: [batch_size]
        ent_dis: [batch_size, num_entities], where 1 means the entity is a target
        return: results
        """
        
        if results is None:
            results = {'c1': 0, 'c3': 0, 'c5':0, 'c10': 0, 'count': 0, 'mrr':0.0,
                       'hits@1': 0.0, 'hits@3': 0.0, 'hits@5': 0.0, 'hits@10': 0.0}

        
        with torch.no_grad():
            b_range = torch.arange(pred_scores.size()[0], device=pred_scores.device)                
            #turn list gt to tensor
            gt_tensor = []
            for answer in gt:
                if answer.isdigit():
                    answer_id = int(answer)
                elif "." in answer:
                    answer_id = int(answer.split(".")[0])
                else:
                    answer_id = self.ent2id[answer.lower()]
                gt_tensor.append(answer_id)
            gt_tensor = torch.tensor(gt_tensor, device=pred_scores.device)
                    

            irrelevant = ent_dis.clone()
            irrelevant[b_range, gt_tensor] = 0  #irrelevant true target is 1
            pred_scores[irrelevant.bool()] = -1000000 #chage other true target to -1000

            ranks = 1 + torch.argsort(torch.argsort(pred_scores, dim=1, descending=True), dim=1, descending=False)[
                                        b_range, gt_tensor]

            ranks = ranks.float()
            results['c1'] = torch.sum(ranks <= 1).item() + results.get('c1', 0.0)
            results['c3'] = torch.sum(ranks <= 3).item() + results.get('c3', 0.0)
            results['c5'] = torch.sum(ranks <= 5).item() + results.get('c5', 0.0)
            results['c10'] = torch.sum(ranks <= 10).item() + results.get('c10', 0.0)
            results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('count', 0.0)*results.get('mrr', 0.0)
            results['count'] = torch.numel(ranks) + results.get('count', 0.0)
            results['hits@1'] = round(results['c1'] / results['count'], 3)
            results['hits@3'] = round(results['c3'] / results['count'], 3)
            results['hits@5'] = round(results['c5'] / results['count'], 3)
            results['hits@10'] = round(results['c10'] / results['count'], 3)
            results['mrr'] = round(results['mrr'] / results['count'], 3)
            
            return results

        