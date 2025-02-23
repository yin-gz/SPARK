import re
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaModel, BitsAndBytesConfig
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch.nn as nn
import numpy as np
from transformers import AutoConfig
 

class LLMGenerator:
    def __init__(self, ent2id, id2rel, args):
        self.args = args
        self.ent2id = ent2id
        self.id2rel = id2rel
        self.model, self.tokenizer, self.config = self.load_llm_model()
        if self.config is not None:
            self.args.llm_dim = self.config.hidden_size
        else:
            self.args.llm_dim = 1024
        self.return_token_id = self.tokenizer.convert_tokens_to_ids(']')
        if args.LABEL_TYPE == "both":
            self.pattern = re.compile(r'<s>.*?[\d:@][._](.*?)[\]\[]?([< ].*?)?$')
        elif args.LABEL_TYPE == "text":
            self.pattern = re.compile(r'<s>\s+(.+)\]')
        elif args.LABEL_TYPE == "id":
            self.pattern = re.compile(r'<s>\s+(\d+)\]')
            
    def load_llm_model(self):
        llm_model, tokenizer, config = self.get_model_and_tokenizer()
        tokenizer.pad_token = tokenizer.eos_token
        #tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("]")
        #! Freeze the llm model
        if llm_model is not None:
            if hasattr(llm_model, "parameters"):
                for param in llm_model.parameters():
                    param.requires_grad = False
        return llm_model, tokenizer, config
            
    def get_model_and_tokenizer(self):
        try:
            llm_config = AutoConfig.from_pretrained(self.args.MODEL_NAME)
        except:
            llm_config = None
        ori_device = self.args.DEVICE
        
        if self.args.DEVICE == -1 or self.args.TEST_ADAPTER or self.args.LOAD_LLM:
            # only load the tokenizer
            model = None
            tokenizer = LlamaTokenizer.from_pretrained("../LLM/lmsys/llama-2-7b-hf", trust_remote_code=True)
            llm_config = AutoConfig.from_pretrained("../LLM/lmsys/llama-2-7b-hf")
            return model, tokenizer, llm_config
        
        else:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            model = LLM(self.args.MODEL_NAME, trust_remote_code=True, tensor_parallel_size = torch.cuda.device_count())
            if self.args.GEN_MODE == "iterative":
                self.sampling_params = SamplingParams(
                    #temperature=0,
                    max_tokens = self.args.TARGET_LEN,
                    stop = "]",
                    include_stop_str_in_output = True
                )
            else:
                self.sampling_params = SamplingParams(
                    n=self.args.NUM_SEQS,
                    temperature=0,
                    top_p=1,
                    #top_k = 1,
                    max_tokens = self.args.TARGET_LEN,
                    stop = "]",
                    include_stop_str_in_output = False,
                    logprobs = 1,
                    use_beam_search = True
                )

        if torch.cuda.device_count() > 1:
            print(f"Let's use {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

        tokenizer = AutoTokenizer.from_pretrained(self.args.MODEL_NAME, trust_remote_code=True)
        self.args.DEVICE = ori_device
        return model, tokenizer, llm_config
    
    def get_ent_rel_emb(self, rel2id, ent2id):
        # Load llama-2-7b-hf for initial entity and relation embeddings
        all_ent = list(ent2id.keys()) + ['PAD']
        all_rel = list(rel2id.keys()) + ['PAD']
        if self.model is None:
            emb_model = LlamaModel.from_pretrained("../LLM/lmsys/llama-2-7b-hf", trust_remote_code=True).to(self.args.DEVICE)            
            with torch.no_grad():
                input_ents = self.tokenizer(all_ent, return_tensors='pt', padding=True, truncation=True, max_length=256).input_ids.to(self.args.DEVICE)
                input_rels = self.tokenizer(all_rel, return_tensors='pt', padding=True, truncation=True, max_length=256).input_ids.to(self.args.DEVICE)
                embedding_layer = emb_model.embed_tokens
                ent_embedding = embedding_layer(input_ents).mean(dim=1) # [num_ent, dim]
                rel_embedding = embedding_layer(input_rels).mean(dim=1) # [num_rel, dim]
                torch.save(torch.cat([ent_embedding, rel_embedding], dim=0), './data/original/'+ self.args.DATASET +'/ent_rel_emb.pt')
            del emb_model
        else:
            embedding = torch.load('./data/original/'+ self.args.DATASET +'/ent_rel_emb.pt')
            ent_embedding = embedding[:len(all_ent)]
            rel_embedding = embedding[len(all_ent):]
        return ent_embedding, rel_embedding
            
            
            
    def forward(self, batch_data):
        return self.vllm_generate(batch_data)

        
    def gen_his_prompt(self, batch_data, his_len = 100):
        '''
        Turn a list of batch quads to a list of batch prompts
        '''
        batch_prompts = []
        batch_his_quads = batch_data["batch_his_quads"]
        bsz = len(batch_his_quads)

        batch_query = batch_data["batch_queries"] # {ts}: [{head_id}, {rel_id}
        for i in range(bsz):
            # sort by time
            ori_quads_i = batch_his_quads[i]
            quads_i = np.unique(ori_quads_i, axis=0)
            # sort by sampled_edges_i[2]
            if len(quads_i) != 0:
                quads_i = quads_i[quads_i[:, 2].argsort()]
            max_his_len = min(len(quads_i), his_len)
            if max_his_len!=0:
                quads_i = quads_i[-max_his_len:]
            else:
                quads_i = []
            
            query_i = batch_query[i]
            query_prompt_i = f"{query_i[0].item()}: [{query_i[1].item()}, {self.id2rel[query_i[2].item()]}," 
            prompts_i = ""
            context_prompt_i = ""
            for j in range(len(quads_i)):
                if quads_i[j][3].item() < len(self.id2rel):
                    rel_text = self.id2rel[quads_i[j][3].item()]
                else:
                    continue
                context_prompt_i += f"{quads_i[j][2].item()}: [{quads_i[j][0].item()}, {rel_text}, {quads_i[j][1].item()}]\n" 
            prompts_i += context_prompt_i + query_prompt_i
            batch_prompts.append(prompts_i)
        return batch_prompts
        

    def vllm_generate(self, batch_data):
        '''
        Generate the answers and scores using vllm (BSL generation)
        '''
        if self.args.RAG == "ICL":
            batch_prompts = self.gen_his_prompt(batch_data)
        else:
            batch_prompts = batch_data["batch_prompts"]
            
        bsz = len(batch_prompts)
        
        #calculate the average length of the prompts
        #avg_len = sum([len(prompt) for prompt in batch_prompts])/bsz
        #print("avg_len: ", avg_len)
        
        batch_answers = []
        batch_scores = []
        if self.args.GEN_MODE == "iterative":
            all_answers = [] # List of k_length, [batch_ans_1, batch_ans_2, ...]
            for i in range(0,10):
                if torch.cuda.device_count() > 1:
                    outputs = self.model.module.generate(batch_prompts, self.sampling_params)
                else:
                    outputs = self.model.generate(batch_prompts, self.sampling_params)
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                if self.args.LABEL_TYPE == "both":
                    all_answers.append([self.decode_reg_answer_id(output.outputs[0].text.strip()) for output in outputs])
                else:
                    all_answers.append([output.outputs[0].text.split("]")[0].strip() for output in outputs])
            #turn all ansert to list of batch size, each batch size is a list of k answers
            for i in range(bsz):
                batch_answers.append([all_answers[j][i] for j in range(10)])
            #set batch_scores all to 0.1
            batch_scores = [[1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1] for _ in range(bsz)] 
        else:
            if torch.cuda.device_count() > 1:
                outputs = self.model.module.generate(batch_prompts, self.sampling_params)
            else:
                outputs = self.model.generate(batch_prompts, self.sampling_params)
            for output in outputs:
                generated_results = output.outputs
                k_answers = []
                k_scores = []
                for generated_k in generated_results:
                    score = generated_k.cumulative_logprob
                    answer = generated_k.text.strip()
                    k_answers.append(answer)
                    k_scores.append(score)
                sort_answers, sort_scores = self.sort_answers(k_answers, k_scores)
                sort_scores = torch.nn.functional.softmax(torch.tensor(k_scores), dim=-1).tolist()
                batch_answers.append(sort_answers)
                batch_scores.append(sort_scores)
            torch.cuda.empty_cache()
        del outputs

        ent_distribution = self.turn_to_distribution(batch_answers, batch_scores).to(self.args.DEVICE)
        return batch_answers, batch_scores, ent_distribution   
    
    
    def decode_reg_answer_id(self, text):
        '''
        Decode the answer id (str) from the text.
        '''
        pattern = re.compile(r'(\d+)\.(.+)\]')
        match = pattern.search(text)
        try:
            answer_text = match.group(2)
            answer_id = str(self.ent2id[answer_text.lower()])
        except:
            answer_id = "0"
        return answer_id
    
    
    def sort_answers(self, answers, scores):
        '''
        Sort answers by scores.
        '''
        k_dict = {}
        for k in range(len(answers)):
            if answers[k] not in k_dict:
                k_dict[answers[k]] = scores[k]
            else:
                k_dict[answers[k]] = max(k_dict[answers[k]], scores[k])
        #sort k_dict by score
        k_dict = dict(sorted(k_dict.items(), key=lambda item: item[1], reverse=True))
        result_answers = list(k_dict.keys())
        result_scores = list(k_dict.values())
        return result_answers, result_scores
    
    def turn_to_distribution(self, answers, scores):
        '''
        Generate the ent distribution [bsz, ent_num]
        answers: bsz lists of k answers
        scores: bsz lists of k scores
        '''
        ent_distribution = torch.zeros(len(answers), len(self.ent2id))
        for i in range(len(answers)):
            for j, ent in enumerate(answers[i]):
                try:
                    if ent.isdigit():
                        ent_id = int(ent)
                    elif "." in ent:
                        ent = ent.split(".")[-1]
                        ent_id = self.ent2id[ent.lower()]
                        answers[i][j] = str(ent_id)
                    else:
                        ent_id = self.ent2id[ent.lower()]
                    ent_distribution[i][ent_id] = max(scores[i][j], ent_distribution[i][ent_id])
                except:
                    continue
        return ent_distribution