# Ignite Forecasting with SPARK: An Efficient Generative Framework for Refining LLMs in Temporal Knowledge Graph Forecasting

### Installing Dependencies

- Dependencies can be installed using `requirements.txt`.

### Preparation

- Make directories: `./model_checkpoints` 、`./rule_output`
- Download datasets from [here](https://figshare.com/s/b327c9e306e28b710c9b), put them in  `./data` directory.
  - `original` denotes the original datasets, `processed` includes contextual information using TLR (in GenTKG).
- **Optional:** Download LLM model `gpt-neox-20b`、`llama-2-7b-hf`、`internlm2-7b` from Hugging face.

### Run SPARK (eg. dataset: icews14)

- Step1: Precompute the LLMs' output distributions and store the results.

```shell
# With TLR
python main.py --DATASET "icews14" --MODEL_NAME your_LLM_path --RAG "TLR" --GEN_MODE "beam" --SAVE_LLM

# With ICL
python main.py --DATASET "icews14" --MODEL_NAME your_LLM_path --RAG "ICL" --GEN_MODE "beam" --SAVE_LLM
```

- Step2: Train SPARK(G) or SPARK(R) as adapters, and then evaluate on test dataset.

```shell
# SPARK(G)
python main.py --DATASET "icews14" --MODEL_NAME your_LLM_path --RAG "TLR" --GEN_MODE "beam" --LOAD_LLM --ADAPTER_NAME "xERTE" 

# SPARK(R)
python main.py --DATASET "icews14" --MODEL_NAME your_LLM_path --RAG "TLR" --GEN_MODE "beam" --LOAD_LLM --ADAPTER_NAME "TLogic"

```

### Reproduce Other Analysis

- In-domain generalization

  ```
  # Modify Step2 to:
  python main.py --DATASET "icews14" --MODEL_NAME your_LLM_path --RAG "TLR" --GEN_MODE "beam" --LOAD_LLM --ADAPTER_NAME "xERTE" --TRAIN_PROP 0.2 
  ```
- Cross-domain generalization

  ```
  # Add Step3:
  python main.py --DATASET "icews18" --MODEL_NAME your_LLM_path --RAG "TLR" --GEN_MODE "beam" --LOAD_LLM --ADAPTER_NAME "xERTE" --RESTORE your_checkpoint --max_attended_edges 60 --ratio_update 0.75 --ONLY_TEST
  ```
