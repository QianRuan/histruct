# HiStruct+ : Improving Extractive Text Summarization with Hierarchical Structure Information

## Abstract
Transformer-based language models usually treat texts as linear sequences. However, most texts also have an inherent hierarchical structure, i.\,e., parts of a text can be identified using their position in this hierarchy. In addition, section titles usually indicate the common topic of their respective sentences. We propose a novel approach to formulate, extract, encode and inject hierarchical structure information explicitly into an extractive summarization model based on a pre-trained, encoder-only Transformer language model (HiStruct+ model), which improves SOTA
ROUGEs for extractive summarization on PubMed and arXiv substantially. Using various experimental settings on three datasets (i.\,e., CNN/DailyMail, PubMed and arXiv), our HiStruct+ model outperforms a strong baseline collectively, which differs from our model only in that the hierarchical structure information is not injected.  It is also observed
that the more conspicuous hierarchical structure the dataset has, the larger improvements
our method gains. The ablation study demonstrates that the hierarchical position information is the main contributor to our model’s SOTA performance.

## Model architecture

![overview_CR_with_sidenotes](https://user-images.githubusercontent.com/28861305/158413092-657c34db-51c2-41d2-89de-7dcd2663d2ea.png)

Figure 1: Architecture of the HiStruct+ model. The model consists of a base TLM for sentence encoding and two stacked inter-sentence Transformer layers for hierarchical contextual learning with a sigmoid classifier for extractive summarization. The two blocks shaded in light-green are the HiStruct injection components

## ROUGEs results on PubMed and arXiv
![image](https://user-images.githubusercontent.com/28861305/159074104-fdf15316-1c7d-4e7a-809b-935bd5d17965.png)
![image](https://user-images.githubusercontent.com/28861305/159074172-eb18dcc1-95e9-4b07-a3fa-43e9572bba46.png)

| Model $\downarrow$ / Metric $\rightarrow$                   | \multicolumn{1}{c}{R1}      | \multicolumn{1}{c}{R2}     | \multicolumn{1}{c}{RL}      |
|-------------------------------------------------------------|-----------------------------|----------------------------|-----------------------------|
| PEGASUS \citeyearpar{zhang2019pegasus}                      | 44.70                       | 17.27                      | 25.80                       |
| BigBird PEGASUS \citeyearpar{bigbird}                       | \underline{46.63}           | 19.02                      | \underline{41.77}           |
| DANCER  PEGASUS \citeyearpar{dancer2020}                    | 45.01                       | 17.60                      | 40.56                       |
| \citeyearpar{longformer}                                    | \underline{46.63}           | \underline{19.62}          | 41.48                       |
| Sent-CLF \citeyearpar{pilault-etal-2020-extractive}         | 34.01                       | 8.71                       | 30.41                       |
| Sent-PTR \citeyearpar{pilault-etal-2020-extractive}         | 42.32                       | 15.63                      | 38.06                       |
| ExtSum-LG + \citeyearpar{xiao-carenini-2020-systematically} |                             |                            |                             |
| ~ RLoss                                                     | \underline{44.01}           | \underline{17.79}          | \underline{39.09}           |
| ~ MMR-Select+                                               | 43.87                       | 17.50                      | 38.97                       |
| TLM-I+E(G,M) \citeyearpar{pilault-etal-2020-extractive}     | \underline{41.62}           | \underline{14.69}          | \underline{38.03}           |
| ORACLE (15k tok.)                                           | 53.58                       | 26.19                      | 47.76                       |
| ORACLE (28k tok.)                                           | 53.97                       | 26.42                      | 48.12                       |
| LEAD-10                                                     | 37.37                       | 10.85                      | 33.17                       |
| TransformerETS                                              |                             |                            |                             |
| ~~ \textit{Longformer-base (15k tok.)}                      | 38.49                       | 11.59                      | 33.85                       |
| ~~ \textit{Longformer-base (28k tok.)}                      | 38.47                       | 11.56                      | 33.82                       |
| HiStruct+                                                   |                             |                            |                             |
| ~~ \textit{Longformer-base (15k tok.)}                      |                             |                            |                             |
| ~~~~~~ sHE+STE(classified)                                  | \textbf{44.94*}             | \textbf{17.42}             | \textbf{39.90*}             |
| ~~~~~~ sHE+STE                                              | \textbf{45.02*}             | \textbf{17.48}             | \textbf{39.94*}             |
| ~~~~~~ sHE                                                  | \textbf{43.04}              | \textbf{15.87}             | \textbf{38.13}              |
| ~~ \textit{Longformer-base (28k tok.)}                      |                             |                            |                             |
| ~~~~~~ sHE+STE(classified)                                  | \textbf{45.17*}             | \textbf{17.61}             | \textbf{40.10*}             |
| ~~~~~~ sHE+STE                                              | \underline{\textbf{45.22*}} | \underline{\textbf{17.67}} | \underline{\textbf{40.16*}} |


```{=latex}

\begin{table}[ht]
\fontsize{9}{9}
\selectfont
\centering
\begin{tabular}[t]{@{}llll@{}}
\toprule
Model $\downarrow$ / Metric $\rightarrow$  & \multicolumn{1}{c}{R1} & \multicolumn{1}{c}{R2} & \multicolumn{1}{c}{RL} \\ \midrule
\multicolumn{4}{c}{Abstractive}               \\ \midrule
PEGASUS \citeyearpar{zhang2019pegasus}   & 45.49 & 19.90  & \underline{42.42} \\
BigBird PEGASUS \citeyearpar{bigbird}   & 46.32 & \underline{20.65} & 42.33 \\
DANCER PEGASUS \citeyearpar{dancer2020}   & \underline{46.34} & 19.97 & \underline{42.42} \\ \midrule
\multicolumn{4}{c}{Extractive}                \\ \midrule
Sent-CLF \citeyearpar{pilault-etal-2020-extractive}              & 45.01 & 19.91 & \underline{41.16} \\
Sent-PTR \citeyearpar{pilault-etal-2020-extractive}              & 43.30  & 17.92 & 39.47 \\
ExtSum-LG+ \citeyearpar{xiao-carenini-2020-systematically}   & & &\\
~ RLoss      & 45.30  & \underline{20.42} & 40.95 \\
~ MMR-Select+ & \underline{45.39} & 20.37 & 40.99 \\ \midrule
\multicolumn{4}{c}{Hybrid}                    \\ \midrule
TLM-I+E(G,M) \citeyearpar{pilault-etal-2020-extractive}       & \underline{42.13} & \underline{16.27} & \underline{39.21} \\ \midrule %
\multicolumn{4}{c}{Reproduced baselines}                           \\ \midrule
ORACLE (4,096 tok.)     & 49.73   & 27.29 & 45.26  \\
ORACLE (9,600 tok.)      & 52.80    & 28.95 & 48.08  \\
ORACLE (15k tok.)      & 53.04   & 29.08 & 48.31  \\
LEAD-7                       & 38.30    & 12.54 & 34.31  \\
LEAD-10                      & 38.59   & 13.05 & 34.81  \\
TransformerETS   & & & \\
~~ \textit{Longformer-base (15k tok.)}   & 41.69   & 15.76 & 37.48  \\
~~ \textit{Longformer-large (15k tok.)}  & 41.69   & 15.79 & 37.49  \\ \midrule
\multicolumn{4}{c}{Our models (Extractive)}   \\ \midrule
HiStruct+  &&&\\
~~ \textit{Longformer-base (15k tok.)}  &&&  \\ 
~~~~~~ sHE+STE(classified) & \underline{\textbf{46.59*’}} & \underline{\textbf{20.39}} & \underline{\textbf{42.11*}} \\
~~~~~~ sHE+STE                      & \textbf{46.49*’} & \textbf{20.29} & \textbf{42.02*} \\
~~~~~~ sHE                          & \textbf{45.76*}  & \textbf{19.64} & \textbf{41.34*} \\ %
~~ \textit{Longformer-large (15k tok.)} &&& \\ %
~~~~~~ sHE+STE(classified)          & \textbf{46.38*’} & \textbf{20.17} & \textbf{41.92*} \\
~~~~~~ sHE                          & \textbf{45.67*}  & \textbf{19.60}  & \textbf{41.26*} \\ 
\bottomrule
\end{tabular}
\caption[Results on PubMed]{F1 ROUGE results on PubMed. Bold are the scores of the HiStruct+ models that are better than the corresponding TransformerETS baseline. The symbol * indicates that the corresponding SOTA ROUGE for extractive summarization is improved by our model. The symbol ' indicates that the SOTA ROUGEs (incl. all types of summarization approaches) are outperformed.}
\label{tab:pubmed_result}
\end{table}

\begin{table}[ht]
\fontsize{9}{9}
\selectfont
\centering
\begin{tabular}[t]{@{}llll@{}}
\toprule
Model $\downarrow$ / Metric $\rightarrow$ & \multicolumn{1}{c}{R1} & \multicolumn{1}{c}{R2} & \multicolumn{1}{c}{RL} \\ \midrule
\multicolumn{4}{c}{Abstractive }               \\ \midrule
PEGASUS \citeyearpar{zhang2019pegasus}                & 44.70  & 17.27 & 25.80  \\
BigBird PEGASUS \citeyearpar{bigbird}        & \underline{46.63} & 19.02 & \underline{41.77} \\
DANCER  PEGASUS \citeyearpar{dancer2020}       & 45.01 & 17.60  & 40.56 \\
LED-large %
\citeyearpar{longformer}    & \underline{46.63} & \underline{19.62} & 41.48 \\ \midrule
\multicolumn{4}{c}{Extractive }                \\ \midrule
Sent-CLF \citeyearpar{pilault-etal-2020-extractive}              & 34.01 & 8.71  & 30.41 \\
Sent-PTR \citeyearpar{pilault-etal-2020-extractive}              & 42.32 & 15.63 & 38.06 \\
ExtSum-LG + \citeyearpar{xiao-carenini-2020-systematically}  &&&\\
~ RLoss & \underline{44.01} & \underline{17.79} & \underline{39.09} \\
~ MMR-Select+  & 43.87 & 17.50  & 38.97 \\ \midrule
\multicolumn{4}{c}{Hybrid }                    \\ \midrule
TLM-I+E(G,M) \citeyearpar{pilault-etal-2020-extractive}       & \underline{41.62} & \underline{14.69} & \underline{38.03} \\\midrule
\multicolumn{4}{c}{Reproduced baselines}                              \\ \midrule
ORACLE (15k tok.)   & 53.58    & 26.19   & 47.76   \\
ORACLE (28k tok.)    & 53.97    & 26.42   & 48.12   \\
LEAD-10                     & 37.37    & 10.85   & 33.17   \\
TransformerETS   & & & \\
~~ \textit{Longformer-base (15k tok.)}   & 38.49    & 11.59   & 33.85   \\
~~ \textit{Longformer-base (28k tok.)}   & 38.47    & 11.56   & 33.82   \\ \midrule
\multicolumn{4}{c}{Our models (Extractive)}      \\ \midrule
HiStruct+  &&&\\
~~ \textit{Longformer-base (15k tok.)}  &&&  \\ 
~~~~~~ sHE+STE(classified)         & \textbf{44.94*}   & \textbf{17.42}   & \textbf{39.90*}   \\
~~~~~~ sHE+STE                     & \textbf{45.02*}   & \textbf{17.48}   & \textbf{39.94*}  \\
~~~~~~ sHE                         & \textbf{43.04}    & \textbf{15.87}   & \textbf{38.13}   \\ 
~~ \textit{Longformer-base (28k tok.)}  &&&  \\ 
~~~~~~ sHE+STE(classified)         & \textbf{45.17*}   & \textbf{17.61}   & \textbf{40.10*}   \\
~~~~~~ sHE+STE                     & \underline{\textbf{45.22*}}   & \underline{\textbf{17.67}}   & \underline{\textbf{40.16*}}  \\ \bottomrule
\end{tabular}
\caption[Results on arXiv]{F1 ROUGE results on arXiv. Bold are the scores of the HiStruct+ models that are better than the corresponding TransformerETS baseline. The symbol * indicates that the corresponding SOTA ROUGE for extractive summarization is improved by our model. }
\label{tab:arxiv_result}
\end{table}


```



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
## Preprocessing of data, HiStruct information obtained

Data preprocessing would take some time. It is recommended to use the preprocessed data (see links in Downloads).



## Training and evaluation

See `src/train.py`.

## Downloads
- the [raw CNN/DailyMail](https://cs.nyu.edu/~kcho/DMQA/) dataset
- the [raw PubMed & arXiv](https://github.com/armancohan/long-summarization) datasets
- the preprocessed CNN/DailyMail data containing hierarchical structure information
- the preprocessed PubMed data containing hierarchical structure information
- the preprocessed arXiv data containing hierarchical structure information
- the [pre-defined dictionaries](https://drive.google.com/file/d/1fSHK6r9QIPXNG58p0kzdFIWeRpFcdlqn/view?usp=sharing) of the typical section classes and the corresponding alternative section titles 
- our best-performed HiStruct+RoBERTa model on CNN/DailyMail
- our best-performed HiStruct+Longformer model on PubMed
- our best-performed HiStruct+Longformer model on arXiv
