# JOLT-SQL
[EMNLP 2025 Main] JOLT-SQL: Joint Loss Tuning of Text-to-SQL with Confusion-aware Noisy Schema Sampling


[[arXiv]](https://arxiv.org/abs/2505.14305) [[ACL Anthology]](https://aclanthology.org/2025.emnlp-main.308/)


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
@inproceedings{song-etal-2025-jolt,
    title = "{JOLT}-{SQL}: Joint Loss Tuning of Text-to-{SQL} with Confusion-aware Noisy Schema Sampling",
    author = "Song, Jinwang  and
      Zan, Hongying  and
      Zhang, Kunli  and
      Mu, Lingling  and
      Han, Yingjie  and
      Hua, Haobo  and
      Peng, Min",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.308/",
    doi = "10.18653/v1/2025.emnlp-main.308",
    pages = "6051--6064",
    ISBN = "979-8-89176-332-6",
    abstract = "Text-to-SQL, which maps natural language to SQL queries, has benefited greatly from recent advances in Large Language Models (LLMs). While LLMs offer various paradigms for this task, including prompting and supervised fine-tuning (SFT), SFT approaches still face challenges such as complex multi-stage pipelines and poor robustness to noisy schema information. To address these limitations, we present JOLT-SQL, a streamlined single-stage SFT framework that jointly optimizes schema linking and SQL generation via a unified loss. JOLT-SQL employs discriminative schema linking, enhanced by local bidirectional attention, alongside a confusion-aware noisy schema sampling strategy with selective attention to improve robustness under noisy schema conditions. Experiments on the Spider and BIRD benchmarks demonstrate that JOLT-SQL achieves state-of-the-art execution accuracy among comparable-size open-source models, while significantly improving both training and inference efficiency."
}
```
