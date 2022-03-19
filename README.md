# HiStruct+ : Improving Extractive Text Summarization with Hierarchical Structure Information

## Abstract
Transformer-based language models usually treat texts as linear sequences. However, most texts also have an inherent hierarchical structure, i.\,e., parts of a text can be identified using their position in this hierarchy. In addition, section titles usually indicate the common topic of their respective sentences. We propose a novel approach to formulate, extract, encode and inject hierarchical structure information explicitly into an extractive summarization model based on a pre-trained, encoder-only Transformer language model (HiStruct+ model), which improves SOTA
ROUGEs for extractive summarization on PubMed and arXiv substantially. Using various experimental settings on three datasets (i.\,e., CNN/DailyMail, PubMed and arXiv), our HiStruct+ model outperforms a strong baseline collectively, which differs from our model only in that the hierarchical structure information is not injected.  It is also observed
that the more conspicuous hierarchical structure the dataset has, the larger improvements
our method gains. The ablation study demonstrates that the hierarchical position information is the main contributor to our model’s SOTA performance.

## Model architecture

![overview_CR_with_sidenotes](https://user-images.githubusercontent.com/28861305/158413092-657c34db-51c2-41d2-89de-7dcd2663d2ea.png)

Figure 1: Architecture of the HiStruct+ model. The model consists of a base TLM for sentence encoding and two stacked inter-sentence Transformer layers for hierarchical contextual learning with a sigmoid classifier for extractive summarization. The two blocks shaded in light-green are the HiStruct injection components

## ROUGE results on PubMed and arXiv



![image](https://user-images.githubusercontent.com/28861305/159101974-05dd09db-1672-4851-aa2c-aa0d782b0400.png)
![image](https://user-images.githubusercontent.com/28861305/159102002-1cc1bbcc-bef5-45f4-a520-d7c80e2051cd.png)





## Env. Setup

Requirements: Python 3.8 and Conda

```bash
# Create environment
conda create -n py38_pt18 python=3.8
conda activate py38_pt18

# Install dependencies
pip3 install -r requirements.txt

# Install pytorch
conda install pytorch==1.8.0 torchvision cudatoolkit=10.1 -c pytorch

# Setup pyrouge
pyrouge_set_rouge_path pyrouge/rouge/tools/ROUGE-1.5.5/
conda install -c bioconda perl-xml-parser 
conda install -c bioconda perl-lwp-protocol-https
conda install -c bioconda perl-db-file
```
## Preprocessing of data
- obtain HiStruct information 
- obatin gold labels for extractive summarization
- tokenize texts with the corresponding tokenizer

NOTE: Data preprocessing would take some time. It is recommended to use the preprocessed data if you experiment with CNN/DailyMail, PubMed or arXiv. (see links in Downloads).

```bash
#CNN/DailyMail####################################################################################################
#Make sure that you have the standford-corenlp toolkit downloaded
#export CLASSPATH=stanford-corenlp-4.2.0/stanford-corenlp-4.2.0.jar
#raw data saved in data_cnndm/data_cnndm_raw

# (1). tokenize the sentences and paragraphs respectively 
# output files: data_cnndm/data_cnndm_raw_tokenized_sent, data_cnndm/data_cnndm_raw_tokenized_para
python histruct/src/preprocess.py -mode tokenize -dataset cnndm  -raw_path data_cnndm/data_cnndm_raw -tok_sent_path data_cnndm/data_cnndm_raw_tokenized_sent -tok_para_path data_cnndm/data_cnndm_raw_tokenized_para -log_file data_cnndm/cnndm_prepro_tokenize.log

# (2). extract HiStruct info
# output path: data_cnndm/data_cnndm_raw_tokenized_histruct
python histruct/src/preprocess.py  -dataset cnndm -mode extract_histruct_items -histruct_path data_cnndm/data_cnndm_raw_tokenized_histruct  -tok_sent_path data_cnndm/data_cnndm_raw_tokenized_sent -tok_para_path data_cnndm/data_cnndm_raw_tokenized_para -lower true -log_file data_cnndm/cnndm_prepro_extract_histruct_items.log

# (3). merge data splits for training, validation and testing
#make sure that the mapping files are in the folder 'urls'
python histruct/src/preprocess.py -dataset cnndm -mode merge_data_splits -raw_path data_cnndm/data_cnndm_raw_tokenized_histruct -save_path data_cnndm/data_cnndm_splitted/cnndm -map_path urls -log_file data_cnndm/cnndm_prepro_merge_data_splits.log

# (4). convcert format for HiStruct+ training, perpare gold labels using ORACLE
#base_LM: the tokenizer used, should be consistent with the base TLM involved in the summarization model, choose from [roberta-base, bert-base]
#summ_size: how many sentences should be included in ORACLE summaries, default:0, no specific limitation
#obtain_tok_se: wehther to obatin token-level struture vectors (see Appendix A.5 in the paper), default: false 
python histruct/src/preprocess.py -mode format_to_histruct -dataset cnndm -base_LM roberta-base -raw_path data_cnndm/data_cnndm_splitted -save_path data_cnndm/data_cnndm_roberta  -log_file data_cnndm/cnndm_prepro_fth_roberta.log -summ_size 0 -n_cpus 1 -obtain_tok_se false


#PubMed####################################################################################################
#raw data saved in data_pubmed/data_pubmed_raw 
# (1). merge data splits for training, validation and testing
python histruct/src/preprocess.py -mode merge_data_splits -dataset pubmed -raw_path data_pubmed/data_pubmed_raw -save_path data_pubmed/data_pubmed_splitted/pubmed  -log_file data_pubmed/pubmed_prepro_merge_data_splits.log

# (2). convcert format for HiStruct+ training, perpare gold labels using ORACLE
#-base_LM: the tokenizer used, should be consistent with the base TLM involved in the summarization model, Longformer tokenizer is identical to roberta-base tokenizer
#-summ_size: how many sentences should be included in ORACLE summaries, default:0, no specific limitation
#-obtain_tok_se: wehther to obatin token-level struture vectors (see Appendix A.5 in the paper), default: false 
python histruct/src/preprocess.py -mode format_to_histruct -dataset pubmed -base_LM roberta-base -raw_path data_pubmed/data_pubmed_splitted -save_path data_pubmed/data_pubmed_roberta  -log_file data_pubmed/pubmed_prepro_fth_roberta.log -summ_size 0 -n_cpus 1 -obtain_tok_se false

# (3). obtain unique section titles from the raw data
python histruct/src/preprocess.py -mode obtain_section_names -dataset pubmed -raw_path data_pubmed/data_pubmed_raw  -save_path data_pubmed/data_pubmed_raw -log_file data_pubmed/pubmed_prepro_osn.log 

# (4). generate section title embeddings (STEs)
# -base_LM: the tokenizer used, should be consistent with the base TLM involved in the summarization model
# -sn_embed_comb_mode: how to convert the last hidden states at every token positions to a single vector, sum them up by default
python histruct/src/preprocess.py -mode encode_section_names -base_LM longformer-base-4096 -dataset pubmed -sn_embed_comb_mode sum -raw_path data_pubmed/data_pubmed_raw -save_path data_pubmed/data_pubmed_raw -log_file data_pubmed/pubmed_prepro_esn.log 

# (5). generate classified section title embeddings (classified STEs)
# base_LM: the tokenizer used, should be consistent with the base TLM involved in the summarization model
# -sn_embed_comb_mode: how to convert the last hidden states at every token positions to a single vector, sum them up by default
# -section_names_embed_path: the path to the original STE which is generated in the step (3)
# -section_names_cls_file: the predefined dictionary of typical section title classes and the in-class section titles
python histruct/src/preprocess.py -mode encode_section_names_cls -base_LM longformer-base-4096 -dataset pubmed -sn_embed_comb_mode sum -raw_path data_pubmed/data_pubmed_raw -save_path data_pubmed/data_pubmed_raw -log_file data_pubmed/pubmed_prepro_esnc.log  -section_names_embed_path data_pubmed/data_pubmed_raw/section_names_embed_longformerB_sum.pt -section_names_cls_file pubmed_SN_dic_8_Added.json


#arXiv####################################################################################################
#raw data saved in data_arxiv/data_arxiv_raw 
# (1). merge data splits for training, validation and testing
python histruct/src/preprocess.py -mode merge_data_splits -dataset arxiv -raw_path data_arxiv/data_arxiv_raw -save_path data_arxiv/data_arxiv_splitted/arxiv -log_file data_arxiv/arxiv_prepro_merge_data_splits.log

# (2). convcert format for HiStruct+ training, perpare gold labels using ORACLE
#-base_LM: the tokenizer used, should be consistent with the base TLM involved in the summarization model, Longformer tokenizer is identical to roberta-base tokenizer
#-summ_size: how many sentences should be included in ORACLE summaries, default:0, no specific limitation
#-obtain_tok_se: wehther to obatin token-level struture vectors (see Appendix A.5 in the paper), default: false 
python histruct/src/preprocess.py -mode format_to_histruct -dataset arxiv -base_LM roberta-base -raw_path data_arxiv/data_arxiv_splitted -save_path data_arxiv/data_arxiv_roberta  -log_file data_arxiv/arxiv_prepro_fth_roberta.log -summ_size 0 -n_cpus 1 -obtain_tok_se false

# (3). obtain unique section titles from the raw data
python histruct/src/preprocess.py -mode obtain_section_names -dataset arxiv -raw_path data_arxiv/data_arxiv_raw -save_path data_arxiv/data_arxiv_raw -log_file data_arxiv/arxiv_prepro_osn.log 

# (4). generate section title embeddings (STEs)
# -base_LM: the tokenizer used, should be consistent with the base TLM involved in the summarization model
# -sn_embed_comb_mode: how to convert the last hidden states at every token positions to a single vector, sum them up by default
python histruct/src/preprocess.py -mode encode_section_names -base_LM longformer-base-4096 -dataset arxiv -sn_embed_comb_mode sum -raw_path data_arxiv/data_arxiv_raw -save_path data_arxiv/data_arxiv_raw -log_file data_arxiv/arxiv_prepro_esn.log  

# (5). generate classified section title embeddings (classified STEs)
# -base_LM: the tokenizer used, should be consistent with the base TLM involved in the summarization model
# -sn_embed_comb_mode: how to convert the last hidden states at every token positions to a single vector, sum them up by default
# -section_names_embed_path: the path to the original STE which is generated in the step (3)
# -section_names_cls_file: the predefined dictionary of typical section title classes and the in-class section titles, saved in raw_path
python histruct/src/preprocess.py -mode encode_section_names_cls -base_LM longformer-base-4096 -dataset arxiv -sn_embed_comb_mode sum -raw_path data_arxiv/data_arxiv_raw -save_path data_arxiv/data_arxiv_raw -log_file data_arxiv/arxiv_prepro_esnc.log  -section_names_embed_path data_arxiv/data_arxiv_raw/section_names_embed_longformerB_sum.pt -section_names_cls_file arxiv_SN_dic_10_Added.json

```
## Root directory
- ./data_cnndm: the preprocessed cnndm data saved in this folder
- ./data_pubmed: the preprocessed pubmed data saved in this folder, the STE and classified STE in ./data_pubmed/data_pubmed_raw
- ./data_arxiv: the preprocessed arxiv data saved in this folder, the STE and classified STE in ./data_arxiv/data_arxiv_raw
- ./histruct: the github repository dowloaded in this folder
- ./models: the trained models saved in this folder


## Training and evaluation

See `run_exp_cnndm.py`, `run_exp_pubmed.py` and `run_exp_arxiv.py`. Arguments can be changed in the scripts.

```bash
#run experiments on CNN/DailyMail
python histruct/run_exp_cnndm.py

#run experiments on PubMed
python histruct/run_exp_pubmed.py

#run experiments on arXiv
python histruct/run_exp_arxiv.py
```


## Downloads
- the [raw CNN/DailyMail](https://cs.nyu.edu/~kcho/DMQA/) dataset
- the [raw PubMed & arXiv](https://github.com/armancohan/long-summarization) datasets
- the preprocessed CNN/DailyMail data containing hierarchical structure information (roberta tokenizer used)
- the preprocessed PubMed data containing hierarchical structure information (roberta tokenizer used)
- the preprocessed arXiv data containing hierarchical structure information (roberta tokenizer used)
- the [pre-defined dictionaries](https://drive.google.com/file/d/1fSHK6r9QIPXNG58p0kzdFIWeRpFcdlqn/view?usp=sharing) of the typical section classes and the corresponding alternative section titles 
- our best-performed HiStruct+RoBERTa model on CNN/DailyMail
- our best-performed HiStruct+Longformer model on PubMed
- our best-performed HiStruct+Longformer model on arXiv
