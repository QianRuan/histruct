# -*- coding: utf-8 -*-


import subprocess
import os


def name_hs_type(ADD_TOK_SE,TOK_POS_EMB_TYPE,TOK_SE_COMB_MODE,ADD_SENT_SE,SENT_POS_EMB_TYPE,SENT_SE_COMB_MODE):
    tok_hs_type=[]
    if ADD_TOK_SE=='true':
        tok_hs_type.append('t')
        if TOK_POS_EMB_TYPE=='sinusoidal':
            tok_hs_type.append('sin')
        if TOK_POS_EMB_TYPE=='learned_all':
            tok_hs_type.append('la')
        if TOK_SE_COMB_MODE=='sum':
            tok_hs_type.append('sum')
        if TOK_SE_COMB_MODE=='mean':
            tok_hs_type.append('mean')
        if TOK_SE_COMB_MODE=='concat':
            tok_hs_type.append('concat')
    
       
    sent_hs_type=[]
    if ADD_SENT_SE=='true':
        sent_hs_type.append('s')
        if SENT_POS_EMB_TYPE=='sinusoidal':
            sent_hs_type.append('sin')
        if SENT_POS_EMB_TYPE=='learned_all':
            sent_hs_type.append('la')
        if SENT_POS_EMB_TYPE=='learned_pos':
            sent_hs_type.append('lp')
        if SENT_SE_COMB_MODE=='sum':
            sent_hs_type.append('sum')
        if SENT_SE_COMB_MODE=='mean':
            sent_hs_type.append('mean')
        if SENT_SE_COMB_MODE=='concat':
            sent_hs_type.append('concat') 
    
    tok_hs_type ='_'.join(tok_hs_type)  
    sent_hs_type ='_'.join(sent_hs_type)   
    hs_type = ''      
    
    if tok_hs_type!='' and tok_hs_type!='' :
        if tok_hs_type[2:] == sent_hs_type[2:]:
            hs_type = 'ts_'+tok_hs_type[2:]
        else:
            hs_type = '_'.join([tok_hs_type,sent_hs_type])
    else:
        hs_type = ''.join([tok_hs_type,sent_hs_type])
        
    
    return hs_type

######Eval Arguments############################################################################################################## 
#how many sentences are etracted as the final summary
SELECT_TOP_N_SENT = 7
#BLOCK TRIGRAM is not applied by default on PubMed and arXiv
BLOCK_TRIGRAM = 'false'
TEST_BATCH_SIZE = 500
ALPHA=0.95
######Training Arguments##############################################################################################################
DATASET = 'pubmed'
PATH = '' #root dir
DATA_PATH = PATH+'data_pubmed/data_pubmed_roberta/pubmed'
TEMP_DIR = PATH+'temp'
#the script for training and evaluation
TRAIN_PY = PATH+'histruct/src/train.py'
#the sin/la PE method for hierarchical position encoding
POS_EMB_TYPE = ['learned_all', 'sinusoidal'] 
#the combination modes for hierarchical position encoding
COMB_MODE = ['sum', 'mean', 'concat']
#dropout
EXT_DROPOUT = 0.1
REPORT_EVERY = 50
#saving steps, save checkpoints every k steps, see the HiStruct+ paper, Appendix A.4
SAVE_CP_STEPS = 1000
#input length
MAX_POS = 15000
# #PE, the numbers of the learned position embeddings for each hierarchy-level of the hierarchical positions of sentences AND the linear sentence positions,when using the learnable position encoding method. 
# We set them to a same value during training by using the default value 0 for MAX_NPARA and MAX_NSENT_IN_PARA
MAX_NSENT = 450 #should be set to a number that is larger than the number of input sentences
MAX_NPARA = 0 #0 means same as Max_NSENT
MAX_NSENT_IN_PARA = 0 #0 means same as Max_NSENT
# the number of the Transformer layers stacked upon the base TLM for extractive summarization.
EXT_LAYERS = 2
#whether the linear sentence position information (sPE) is not injected, default: 'false'
WITHOUT_SENT_POS = 'false'
#whether only the first dimension of SSVs is encoded, default: 'false'
PARA_ONLY = 'false'
#whether the involved TLM is finetune, default:'false' when using longformer
FINETUNE = 'false'

#longformer arguments
#please refer to the longformer paper
#whether global attention is used at BOS tokens, default:'true'
USE_GLOBAL_ATTENTION = 'true'
#the local attention window size, default:1024
LOCAL_ATTENTION_WINDOW = 1024


#whether section title embeddings are injected (STE or classified STE)
#'': not injected
#SECTION_NAMES_EMBED_PATH = ''
#the original STE injected
#SECTION_NAMES_EMBED_PATH = 'data_pubmed/data_pubmed_raw/section_names_embed_longformerB_sum.pt'
#the classified STE injected
SECTION_NAMES_EMBED_PATH = 'data_pubmed/data_pubmed_raw/section_names_embed_longformerB_sumCLS8.pt'

SEED = 666  
#training batch size
BATCH_SIZE = 500
#accumulation count, gradient accumulation every k steps
ACCUM_COUNT = 2
#steps for warmup
WARMUP_STEPS = 10000
#the total training steps
TRAIN_STEPS = 70000
#learning rate, default 2e-3
LR = 2e-3
#GPUs used
VISIBLE_GPUS = 0,1,2
if VISIBLE_GPUS != -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(VISIBLE_GPUS)

#the base TLM used in the summarization model, choose from ['longformer-base-4096','longformer-large-4096']
BASE_LM = 'longformer-base-4096'#'longformer-large-4096'

#whether the token-level hierarchical position information is injected, default:'false'
ADD_TOK_SE = 'false'
TOK_POS_EMB_TYPE = 'learned_all'
TOK_SE_COMB_MODE = 'sum'

#whether the sentence hierarchical position information (sHE) is injected
ADD_SENT_SE ='true'
#the PE method used, choose from ['learned_all', 'sinusoidal'] 
SENT_POS_EMB_TYPE = 'learned_all'
#the combination mode used, choose from ['sum', 'mean', 'concat']
SENT_SE_COMB_MODE = 'sum'

#name the model with arguments
#the hierarchical position encoding setting
hs_type = name_hs_type(ADD_TOK_SE,TOK_POS_EMB_TYPE,TOK_SE_COMB_MODE,ADD_SENT_SE,SENT_POS_EMB_TYPE,SENT_SE_COMB_MODE)
#other arguments
paras = 'bs'+str(BATCH_SIZE)+'ac'+str(ACCUM_COUNT)+'ws'+str(WARMUP_STEPS)+'ts'+str(TRAIN_STEPS)
if LR!=2e-3:
    paras = paras+'_lr'+str(LR)

paras = paras+'_mp'+str(MAX_POS)
paras=paras+'mns'+str(MAX_NSENT)
if MAX_NPARA!=0:
    paras=paras+'mnp'+str(MAX_NPARA)
if MAX_NSENT_IN_PARA!=0:
    paras=paras+'mnsp'+str(MAX_NSENT_IN_PARA)
    
paras=paras+'_law'+str(LOCAL_ATTENTION_WINDOW)


if FINETUNE=='false':
    paras=paras+'_F-finetune'
elif FINETUNE=='true':
    paras=paras+'_T-finetune'
    
if USE_GLOBAL_ATTENTION=='false':
    paras=paras+'_F-globatt_'
elif USE_GLOBAL_ATTENTION=='true':
    paras=paras+'_T-globatt_'
    
    
if SECTION_NAMES_EMBED_PATH!='':
    sn=SECTION_NAMES_EMBED_PATH.split('_')[-2:]
    paras=paras+'sn-'+sn[0]+'-'+sn[1][:-3]

if EXT_LAYERS !=2:
    paras=paras+'_el'+str(EXT_LAYERS)
if WITHOUT_SENT_POS == 'true':
    paras=paras+'_woSentPos'
if PARA_ONLY == 'true':
    paras=paras+'_paraOnly'

if VISIBLE_GPUS==-1:
    n_gpus = 'cpu'
elif VISIBLE_GPUS==0:
    n_gpus = '1gpu'
else:
    n_gpus = str(len(VISIBLE_GPUS))+'gpu'


base_lm_name = BASE_LM.split('-')[0]+BASE_LM.split('-')[1][0].upper()


MODELS_PATH = PATH+'models/'
if ADD_TOK_SE == 'false' and ADD_SENT_SE =='false':
    if SECTION_NAMES_EMBED_PATH!='':
        MODEL_NAME = '_'.join([DATASET,'hs',base_lm_name, hs_type, paras, n_gpus])
    else:   
        MODEL_NAME = '_'.join([DATASET, base_lm_name, hs_type, paras, n_gpus])  
else:   
    MODEL_NAME = '_'.join([DATASET,'hs',base_lm_name, hs_type, paras, n_gpus])
MODEL_PATH = MODELS_PATH + MODEL_NAME




train_args = [TRAIN_PY, '-task', 'ext', '-mode', 'train', 
             '-base_LM', BASE_LM, 
             '-add_tok_struct_emb', ADD_TOK_SE, 
             '-tok_pos_emb_type', TOK_POS_EMB_TYPE, 
             '-tok_se_comb_mode', TOK_SE_COMB_MODE,
             '-add_sent_struct_emb', ADD_SENT_SE,
             '-sent_pos_emb_type', SENT_POS_EMB_TYPE, 
             '-sent_se_comb_mode', SENT_SE_COMB_MODE,
             '-ext_dropout', str(EXT_DROPOUT),
             '-model_path', MODEL_PATH,
             '-batch_size', str(BATCH_SIZE),
             '-accum_count', str(ACCUM_COUNT),
             '-warmup_steps', str(WARMUP_STEPS),
             '-train_steps', str(TRAIN_STEPS),
             '-lr', str(LR),
             '-data_path', DATA_PATH,
             '-temp_dir', TEMP_DIR,
             '-visible_gpus', str(VISIBLE_GPUS)[1:-1],
             '-report_every', str(REPORT_EVERY),
             '-save_checkpoint_steps', str(SAVE_CP_STEPS),
             '-max_pos', str(MAX_POS),
             '-ext_layers',str(EXT_LAYERS),
             '-without_sent_pos', WITHOUT_SENT_POS,
             '-max_nsent',str(MAX_NSENT),
             '-max_npara',str(MAX_NPARA),
             '-max_nsent_in_para',str(MAX_NSENT_IN_PARA),
             '-para_only', PARA_ONLY,
             '-finetune_bert', FINETUNE,
             '-use_global_attention', USE_GLOBAL_ATTENTION,
             '-local_attention_window', str(LOCAL_ATTENTION_WINDOW),
             '-section_names_embed_path', SECTION_NAMES_EMBED_PATH,
             '-seed',str(SEED)
             ]

#-test_all:true, validate all checkpoints, 3 best checkpoints then tested on test data, ROUGEs reported, avg. ROUGEs reported
eval_args = [TRAIN_PY,'-task', 'ext', '-mode', 'validate', '-test_all', 'true', 
             '-select_top_n_sent', str(SELECT_TOP_N_SENT),
             '-batch_size', str(BATCH_SIZE),
             '-test_batch_size',str(TEST_BATCH_SIZE),
             '-data_path', DATA_PATH,
             '-model_path', MODEL_PATH,
             '-temp_dir', TEMP_DIR,
             '-visible_gpus', str(VISIBLE_GPUS)[1:-1],
             '-base_LM', BASE_LM, 
             '-add_tok_struct_emb', ADD_TOK_SE, 
             '-tok_pos_emb_type', TOK_POS_EMB_TYPE, 
             '-tok_se_comb_mode', TOK_SE_COMB_MODE,
             '-add_sent_struct_emb', ADD_SENT_SE,
             '-sent_pos_emb_type', SENT_POS_EMB_TYPE, 
             '-sent_se_comb_mode', SENT_SE_COMB_MODE,
             '-ext_dropout', str(EXT_DROPOUT),
             '-max_pos', str(MAX_POS),
             '-alpha', str(ALPHA),
             '-ext_layers',str(EXT_LAYERS),
             '-without_sent_pos', WITHOUT_SENT_POS,
             '-max_nsent',str(MAX_NSENT),
             '-max_npara',str(MAX_NPARA),
             '-max_nsent_in_para',str(MAX_NSENT_IN_PARA),
             '-para_only', PARA_ONLY,
             '-finetune_bert', FINETUNE,
             '-use_global_attention', USE_GLOBAL_ATTENTION,
             '-local_attention_window', str(LOCAL_ATTENTION_WINDOW),
             '-section_names_embed_path', SECTION_NAMES_EMBED_PATH,
             '-block_trigram', BLOCK_TRIGRAM
             ]




if __name__ == '__main__':
    
    #TRAIN
    command_Train = ['python']+train_args
    c_Train = " ".join([c for c in command_Train])
    
    
    #EVAL
    command_EVAL = ['python']+eval_args
    c_Eval = " ".join([c for c in command_EVAL])
    
    print('##########################################################################')
    print('Starting experiment...', MODEL_NAME)
    print('##########################################################################')
    print("Training with command... {}".format(c_Train))
    print('##########################################################################')
    subprocess.run(command_Train, check=True)
    
    print("Evaluating with command... {}".format(c_Eval))
    subprocess.run(command_EVAL, check=True)
    
    #record commands
    with open(MODEL_PATH+'/run_exp.txt','w+') as f:
        print('Saving commands...')
        f.write('###################################TRAIN')
        f.write('\n')
        f.write(c_Train)
        f.write('\n')
        f.write('###################################TEST')
        f.write('\n')
        f.write(c_Eval)
        print('DONE')
    f.close()
    
        

    
    
    

    