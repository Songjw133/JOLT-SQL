# JOLT-SQL
JOLT-SQL: Joint Loss Tuning of Text-to-SQL with Confusion-aware Noisy Schema Sampling (EMNLP 2025 Main)

📄 [arXiv:2301.12345](https://arxiv.org/abs/2505.14305)


We are in the process of organizing the full code for release.


# How to run
## Create a new environment

```
conda create -n jolt-sql python=3.11
conda activate jolt-sql
pip install -r requirements.txt
```

## Download model

(need git-lfs)
```
cd JOLT-SQL
git clone https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct
git clone https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct
```

## Download spider data

Download the Spider dataset from [here](https://yale-lily.github.io/spider) and place it in the spider_data folder. The path ./JOLT-SQL/spider_data/dev_gold.sql should be valid.

## Running
(Fine-tuning the 14B model requires a GPU with ≥48GB of VRAM.)
```
CUDA_VISIBLE_DEVICES=0 python JOLT_spider_v1.py
```

If you encounter an NLTK-related error, try the following:
```
import nltk  
nltk.download('punkt_tab') 
```

# Citation
```
@misc{song2025joltsqljointlosstuning,
      title={JOLT-SQL: Joint Loss Tuning of Text-to-SQL with Confusion-aware Noisy Schema Sampling}, 
      author={Jinwang Song and Hongying Zan and Kunli Zhang and Lingling Mu and Yingjie Han and Haobo Hua and Min Peng},
      year={2025},
      eprint={2505.14305},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.14305}, 
}
```