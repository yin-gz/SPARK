import argparse
import wandb
import os
from datetime import datetime
from load_data import prepare_data, prepare_graph, load_vocab, prepare_graph_xERTE
from loops_train import Trainer
import torch.multiprocessing as mp


def generate_run_name(model_name, time=None):
    run_name = model_name + "_"
    if time is None:
        run_name = run_name + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else:
        run_name = run_name + time
    return run_name


def parse_args():
    parser = argparse.ArgumentParser(description="Configs")
    
    #! LLM parameters
    parser.add_argument('--TARGET_LEN', type=int, default=8, help='Token length of target sequence')
    parser.add_argument('--LABEL_TYPE', type=str, default="id", choices= ["id", "text", "both"], help='Label type for entities')
    parser.add_argument('--WITH_INSTRUCT', type=bool, default=False, help='Use instruct or not')
    parser.add_argument('--MODEL_NAME', type=str, default="../LLM/lmsys/llama-2-7b-hf", help='Model name') #gpt-neox-20b, llama-2-7b-hf, internlm2-7b
    parser.add_argument('--NUM_SEQS', type=int, default=10, help='Number of top-K generated seqs for llm')
    parser.add_argument('--SAVE_LLM', default=False, action="store_true", help='Only use LLM and pre-save its output')
    parser.add_argument('--LOAD_LLM', default=False, action="store_true", help='Load LLM results')
    parser.add_argument('--RAG', type=str, default="TLR", help='ICL or TLR to generate prompt')
    parser.add_argument('--GEN_MODE', type=str, default="beam", choices=["beam", "iterative"], help='Beam generate or iterative generate')
    
    #! Path and data parameters
    parser.add_argument('--OUTPUT_DIR', type=str, default="./model_checkpoints", help='Output dir')
    parser.add_argument('--DATA_PATH', type=str, default="./data/processed", help='Input dir of processed trainset')
    parser.add_argument('--ORI_DATA_PATH', type=str, default="./data/original", help='Input dir of original trainset')
    parser.add_argument('--RULE_DATA_PATH', type=str, default="./data/rule", help='Input dir to store rule information of each instance')
    parser.add_argument('--DATASET', type=str, default="GDELT", choices=["icews14", "icews18", "GDELT"], help='Name of dataset')
    
    #! Adapter parameters
    parser.add_argument('--ADAPTER_NAME', type=str, default=None, choices=["TLogic", "xERTE", None], help='Adapter model name')
    parser.add_argument('--LOAD_ADAPTER', type=bool, default=False, help='Load adapter checkpoint')
    parser.add_argument('--TEST_ADAPTER', type=bool, default=False, help='Only use and test adapter or not')
    parser.add_argument('--EMBEDDING_DIM', type=int, default=200, help='Embedding dimension')
    parser.add_argument('--LOAD_RULE', type=bool, default=False, help='Load rules learned by TLogic')
    
    #! Training parameters
    parser.add_argument("--TRAIN_BS", type=int, default=128, choices = [128, 256, 512], help="Batch size for train")
    parser.add_argument("--EVAL_BS", type=int, default=128, choices = [128, 256, 512], help="Batch size for evaluating")
    parser.add_argument('--EPOCHS', type=int, default=10, help='Training epochs')
    parser.add_argument('--DR', type=str, default="plateau", help="stepLR/cosine/plateau/onecycle")
    parser.add_argument('--WARMUP_STEPS', type=int, default=100, help='Warmup steps')
    parser.add_argument('--LEARNING_RATE', type=float, default= 1e-4, choices = [1e-5, 5e-5, 1e-4], help='Training learning rate')
    parser.add_argument('--LOGGING_STEPS', type=int, default=1, help='Logging steps in training')
    parser.add_argument('--EVAL_STEPS', type=int, default=10, help='Evaluate the model according to steps')
    parser.add_argument('--SAVE_STEPS', type=int, default=20, help='Save the model according to steps')
    parser.add_argument('--NUM_WORKERS', type=int, default=0, help='Number of workers in dataloader')
    parser.add_argument('--LOSS_TYPE', type=str, default = "target_loss", choices= ["target_loss", "all_loss"], help = 'Calculate loss only for candidate targets (predicting prob != 0) or for all entities')
    parser.add_argument('--ONLY_TEST', default = False, action="store_true", help = 'Only restore model and test')
    parser.add_argument('--TRAIN_PROP', type=float, default = 1, help = 'Train data proportion')
    parser.add_argument('--DEVICE', type=str, default="cuda:0", choices=["cuda:0", "cuda:1", "cpu"], help='Device to use')
    parser.add_argument('--RESTORE', type=str, default=None, help='Set None to no restore')
    
    #! Logging parameters
    parser.add_argument('--WANDB', type=bool, default=False, help='Logging to wandb')
    parser.add_argument('--RUN_NAME', type=str, default=None, help='Run name for wandb')

    #! xERTE
    parser.add_argument('--max_attended_edges', type=int, default=40, help='Max attended edges for xERTE') #icews14:40, icews18:60
    parser.add_argument('--ratio_update', type=float, default=0, help='Ratio update for xERTE') #icews14:0, icews18:0.75
    
    return parser.parse_args()

if __name__ == "__main__":
    mp.set_start_method('spawn')
    args = parse_args()
    if args.RUN_NAME is None:
            model_name = f"{args.DATASET}-{args.MODEL_NAME.split('/')[-1].split('-')[0]}-{args.ADAPTER_NAME}, bs:{args.TRAIN_BS},lr:{args.LEARNING_RATE}, loadLLM:{args.LOAD_LLM}, TEST_ADAPTER: {args.TEST_ADAPTER}, {args.TRAIN_PROP}"
            args.RUN_NAME = model_name
            print("gen_name", args.RUN_NAME)
    print(args)
   
    # wandb settings             
    if args.WANDB:
        os.environ["WANDB_MODE"] = "offline"
        wandb.init(project = "TKG-SPARK-" + args.DATASET, config=args)
        wandb.run.name = args.RUN_NAME
        
    # check parameters
    if args.LABEL_TYPE == "text":
        args.TARGET_LEN = 16
    if args.WITH_INSTRUCT:
        args.TARGET_LEN = 64
    if args.ADAPTER_NAME == "TLogic":
        args.LOAD_RULE = True
    if args.SAVE_LLM:
        args.EPOCHS = 1
            
    # load id2text mappings
    vocab_path = os.path.join(args.ORI_DATA_PATH, args.DATASET)
    vocab_dict = load_vocab(vocab_path)
    
    # load instruct data preprocessed by GENTKG, split the orginal train to 0.8 train and 0.2 valid
    tlr_data = prepare_data(args) 
    # load original facts and graphs (USED FOR xERTE AND EXTRACT RECENT FACTS)
    if args.ADAPTER_NAME == "xERTE":
        ori_facts, graphs = prepare_graph_xERTE(args, rel_num = len(vocab_dict["rel2id"])/2)
    else:
        ori_facts, graphs = prepare_graph(args, rel_num = len(vocab_dict["rel2id"])/2)
    

    trainer = Trainer(args, tlr_data, ori_facts, graphs, vocab_dict)
    trainer.train(resume_from_checkpoint=args.RESTORE)