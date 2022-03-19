#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import argparse
import os

from train_extractive import train_ext, validate_ext, test_ext
from train_extractive import test_steps, baseline_ext, get_cand_list_ext





def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-task", default='ext', type=str, choices=['ext', 'abs'])
    parser.add_argument("-encoder", default='bert', type=str, choices=['bert', 'baseline'])
    parser.add_argument("-mode", default='train', type=str, choices=['train', 'validate', 'test','test_steps','get_cand_list','lead','oracle'])
    parser.add_argument("-base_LM", default='bert-base', type=str, choices=['bert-base', 'bert-large', 'roberta-base','roberta-large','bart-base','bart-large','longformer-base-4096','longformer-large-4096','bigbird-pegasus-large-arxiv','bigbird-pegasus-large-pubmed'])
    parser.add_argument("-add_tok_struct_emb", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-add_sent_struct_emb", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-tok_se_comb_mode", default='concat', type=str, choices=['sum', 'mean', 'concat'])
    parser.add_argument("-sent_se_comb_mode", default='concat', type=str, choices=['sum', 'mean', 'concat'])
    parser.add_argument("-tok_pos_emb_type", default='sinusoidal', type=str, choices=['learned_pos', 'learned_all', 'sinusoidal'])
    parser.add_argument("-sent_pos_emb_type", default='sinusoidal', type=str, choices=['learned_pos', 'learned_all', 'sinusoidal'])
    parser.add_argument("-max_nsent", default=512, type=int)
    parser.add_argument("-max_npara", default=0, type=int)
    parser.add_argument("-max_nsent_in_para", default=0, type=int)
    parser.add_argument('-eval_folder', default='eval')
    parser.add_argument('-eval_path', default='')
    parser.add_argument('-log_file', default='')
    parser.add_argument("-result_path", default='')
    parser.add_argument("-select_top_n_sent", default=3, type=int)
    parser.add_argument("-without_sent_pos", type=str2bool, nargs='?',const=False,default=False)
    parser.add_argument("-para_only", type=str2bool, nargs='?',const=False,default=False)
    
    #longformer
    parser.add_argument("-use_global_attention", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-local_attention_window", default=1024, type=int)
    
    #bigbird_pegasus, bart to encoder-only model
    parser.add_argument("-is_encoder_decoder", type=str2bool, nargs='?',const=False,default=False)
    parser.add_argument("-pooled_encoder_output", type=str2bool, nargs='?',const=True,default=True)
    
    
    #section names embeddings
    parser.add_argument("-section_names_embed_path", default='')
    
    #get_cand_list(for matchsum)
    parser.add_argument("-corpus_type", default='')
    parser.add_argument("-save_path", default='')
    
    
    
    parser.add_argument("-data_path", default='')
    parser.add_argument("-model_path", default='')
    parser.add_argument("-temp_dir", default='temp')

    parser.add_argument("-batch_size", default=140, type=int)
    parser.add_argument("-test_batch_size", default=200, type=int)

    parser.add_argument("-max_pos", default=512, type=int)
    parser.add_argument("-use_interval", type=str2bool, nargs='?',const=True,default=True)
    
    
    parser.add_argument("-load_from_extractive", default='', type=str)

    parser.add_argument("-sep_optim", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-lr_bert", default=2e-3, type=float)
    parser.add_argument("-lr_dec", default=2e-3, type=float)
    parser.add_argument("-use_bert_emb", type=str2bool, nargs='?',const=True,default=False)

    parser.add_argument("-share_emb", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-finetune_bert", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-dec_dropout", default=0.2, type=float)
    parser.add_argument("-dec_layers", default=6, type=int)
    parser.add_argument("-dec_hidden_size", default=768, type=int)
    parser.add_argument("-dec_heads", default=8, type=int)
    parser.add_argument("-dec_ff_size", default=2048, type=int)
    parser.add_argument("-enc_hidden_size", default=512, type=int)
    parser.add_argument("-enc_ff_size", default=512, type=int)
    parser.add_argument("-enc_dropout", default=0.2, type=float)
    parser.add_argument("-enc_layers", default=6, type=int)

    # params for EXT
    
    parser.add_argument("-ext_dropout", default=0.2, type=float)
    parser.add_argument("-ext_layers", default=2, type=int)
    parser.add_argument("-ext_hidden_size", default=768, type=int)
    parser.add_argument("-ext_heads", default=8, type=int)
    parser.add_argument("-ext_ff_size", default=2048, type=int)

    parser.add_argument("-label_smoothing", default=0.1, type=float)
    parser.add_argument("-generator_shard_size", default=32, type=int)
    parser.add_argument("-alpha",  default=0.6, type=float)
    parser.add_argument("-beam_size", default=5, type=int)




    parser.add_argument("-param_init", default=0, type=float)
    parser.add_argument("-param_init_glorot", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-optim", default='adam', type=str)
    parser.add_argument("-lr", default=1, type=float)
    parser.add_argument("-beta1", default= 0.9, type=float)
    parser.add_argument("-beta2", default=0.999, type=float)
    parser.add_argument("-warmup_steps", default=8000, type=int)
    parser.add_argument("-warmup_steps_bert", default=8000, type=int)
    parser.add_argument("-warmup_steps_dec", default=8000, type=int)
    parser.add_argument("-max_grad_norm", default=0, type=float)

    parser.add_argument("-save_checkpoint_steps", default=5, type=int)
    parser.add_argument("-accum_count", default=1, type=int)
    parser.add_argument("-report_every", default=1, type=int)
    parser.add_argument("-train_steps", default=1000, type=int)
    parser.add_argument("-recall_eval", type=str2bool, nargs='?',const=True,default=False)#


    parser.add_argument('-visible_gpus', default='-1', type=str)
    parser.add_argument('-gpu_ranks', default='0', type=str)
    
    parser.add_argument('-seed', default=666, type=int)

    parser.add_argument("-test_all", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-test_from", default='')
    parser.add_argument("-test_steps", default='')
    parser.add_argument("-test_start_from", default=-1, type=int)

    parser.add_argument("-train_from", default='')
    parser.add_argument("-report_rouge", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)

    args = parser.parse_args()
    args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
    args.world_size = len(args.gpu_ranks)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    device_id = 0 if device == "cuda" else -1
    



    if (args.task == 'ext'):
        if (args.mode == 'train'):
            train_ext(args, device_id)
            os.mkdir(args.model_path+'/DONE')
        elif (args.mode == 'validate'):
            validate_ext(args, device_id)
            os.mkdir(args.model_path+'/eval/DONE')
        elif (args.mode == 'lead'):
            baseline_ext(args, cal_lead=True)
            os.mkdir(args.model_path+'/DONE')
            os.mkdir(args.model_path+'/eval/DONE')
        elif (args.mode == 'oracle'):
            baseline_ext(args, cal_oracle=True)
            os.mkdir(args.model_path+'/DONE')
            os.mkdir(args.model_path+'/eval/DONE')
        elif (args.mode == 'test_steps'):#test many steps
            test_steps(args, device_id)
            os.mkdir(args.eval_path+'/DONE') 
        elif (args.mode == 'test'):#test one checkpoint
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
            test_ext(args, device_id, cp, step)
            os.mkdir(args.model_path+'/eval/DONE')
        elif (args.mode == 'get_cand_list'):#generate a list of candidate sentence indices for each document (for MatchSum)
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
            get_cand_list_ext(args, device_id, cp, step)
            
        elif (args.mode == 'test_text'):
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
                
