#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import argparse
import glob
import os
import random
import signal
import time
import shutil
import json
import statistics
import torch
import numpy as np
import distributed
from models import data_loader, model_builder
from models.data_loader import load_dataset
from models.model_builder import ExtSummarizer
from models.trainer_ext import build_trainer
from others.logging import logger, init_logger
from others.utils import rouge_results_to_str

model_flags = ['-pooled_encoder_output','-is_encoder_decoder','-section_names_embed_path','local_attention_window','use_global_attention','max_npara','max_nsent_in_para','para_only','max_nsent','without_sent_pos','ext_layers','base_LM','add_tok_struct_emb', 'add_sent_struct_emb' ,'tok_pos_emb_type','sent_pos_emb_type', 'tok_se_comb_mode' ,'sent_se_comb_mode',
               'hidden_size', 'ff_size', 'heads', 'inter_layers', 'encoder', 'ff_actv', 'use_interval', 'rnn_size']#


def train_multi_ext(args):
    """ Spawns 1 process per GPU """

    nb_gpu = args.world_size
    mp = torch.multiprocessing.get_context('spawn')

    # Create a thread to listen for errors in the child processes.
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)

    # Train with multiprocessing.
    procs = []
    for i in range(nb_gpu):
        device_id = i
        procs.append(mp.Process(target=run, args=(args,
                                                  device_id, error_queue,), daemon=True))
        procs[i].start()
        logger.info(" Starting process pid: %d  " % procs[i].pid)
        error_handler.add_child(procs[i].pid)
    for p in procs:
        p.join()


def run(args, device_id, error_queue):
    """ run process """
    setattr(args, 'gpu_ranks', [int(i) for i in args.gpu_ranks])

    try:
        gpu_rank = distributed.multi_init(device_id, args.world_size, args.gpu_ranks)

        if gpu_rank != args.gpu_ranks[device_id]:
            print("An error occurred in \
                  Distributed initialization")
            raise AssertionError("An error occurred in \
                  Distributed initialization")
        
        train_single_ext(args, device_id)
        
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        pass  # killed by parent, do nothing
    except Exception:
        print("Traceback")
        # propagate exception to parent process, keeping original traceback
        import traceback
        print(traceback.format_exc())
        error_queue.put((args.gpu_ranks[device_id], traceback.format_exc()))


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        """ init error handler """
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(
            target=self.error_listener, daemon=True)
        self.error_thread.start()
        #signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """ error handler """
        self.children_pids.append(pid)

    def error_listener(self):
        """ error listener """
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
       # os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        """ signal handler """
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = """\n\n-- Tracebacks above this line can probably
                 be ignored --\n\n"""
        msg += original_trace
        raise Exception(msg)
        

def validate_ext(args, device_id):
    
    if args.eval_path=='':
        args.eval_path=args.model_path+'/'+args.eval_folder
        
    if args.log_file=='':
        args.log_file=args.eval_path+'/eval.log'
    
    if args.result_path=='':
        args.result_path=args.eval_path+'/eval.results'
        
    elif '/'.join(args.result_path.split('/')[:-1]) != args.eval_path:
        raise ValueError("Evaluation result path not in the eval folder")
        
    #create eval folder if not exists, delete if exists
    if os.path.exists(args.eval_path):
        logger.info('Eval folder already exists, remove it!')
        shutil.rmtree(args.eval_path)
        os.mkdir(args.eval_path)
    else:
        os.mkdir(args.eval_path)
    
    init_logger(args.log_file)
    logger.info(args)
    
    timestep = 0
    
    if (args.test_all):
        cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
        cp_files.sort(key=os.path.getmtime)
        logger.info('There are %i checkpoints'%(len(cp_files)))
        

        xent_lst = []
        for i, cp in enumerate(cp_files):
            step = int(cp.split('.')[-2].split('_')[-1])
            xent = validate(args, device_id, cp, step)
            xent_lst.append((xent, cp))

        
        #save info for post analysis
        with open(args.eval_path+'/validation_xent.json', 'w+') as f:
            xents = sorted(xent_lst, key=lambda x: x[0])
            json.dump(xents,f) 
        
        
        xent_lst = sorted(xent_lst, key=lambda x: x[0])[:3]
        logger.info('PPL %s' % str(xent_lst))
        
       
        test_xent_lst=[]
        test_rouge_lst=[]
        for xent, cp in xent_lst:
            step = int(cp.split('.')[-2].split('_')[-1])
            xent, rouges = test_ext(args, device_id, cp, step)
            test_xent_lst.append((xent,cp))
            test_rouge_lst.append((cp,rouges))
            
        
        #save info for post analysis
        with open(args.eval_path+'/test_xent.json', 'w+') as f:
            xents = sorted(test_xent_lst, key=lambda x: x[0])
            json.dump(xents,f)
        with open(args.eval_path+'/test_rouges.json', 'w+') as f:
            json.dump(test_rouge_lst,f)
        with open(args.eval_path+'/test_avg_rouges.json', 'w+') as f:
            metrics=['rouge_1_f_score','rouge_2_f_score','rouge_l_f_score',
                     'rouge_1_recall','rouge_2_recall','rouge_l_recall',
                     'rouge_1_precision','rouge_2_precision','rouge_l_precision']
            dic={}    
            for m in metrics:
                li=[]
                for item in test_rouge_lst:
                    li.append(item[1][m])
                avg=statistics.mean(li)
                dic.update({m:avg})
        
            json.dump(dic,f)
            logger.info('Avg. rouges of the model______%s \n%s' % (args.model_path.split('/')[1], rouge_results_to_str(dic)))
        
       
            
    else:
        while (True):

            cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
            cp_files.sort(key=os.path.getmtime)
            if (cp_files):
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)
                if (not os.path.getsize(cp) > 0):
                    time.sleep(60)
                    continue
                if (time_of_cp > timestep):
                    timestep = time_of_cp
                    step = int(cp.split('.')[-2].split('_')[-1])
                    validate(args, device_id, cp, step)
                    test_ext(args, device_id, cp, step)

            cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
            cp_files.sort(key=os.path.getmtime)
            if (cp_files):
                
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)
                
                if (time_of_cp > timestep):

                    continue
            else:

                time.sleep(300)


def validate(args, device_id, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
            
    
    model = ExtSummarizer(args, device, checkpoint)
    model.eval()   
    valid_iter = data_loader.Dataloader(args, load_dataset(args, 'valid', shuffle=False),
                                        args.batch_size, device,
                                        shuffle=False, is_test=False)
    trainer = build_trainer(args, device_id, model, None)
    stats = trainer.validate(valid_iter, step)
    return stats.xent()

def test_steps(args, device_id):
     if args.eval_folder=='':
        args.eval_path=args.model_path+'/test_steps'
        print('args.eval_path',args.eval_path)
     else:
        args.eval_path=args.model_path+'/'+args.eval_folder
        print('args.eval_path',args.eval_path)
         
        
     args.result_path=args.eval_path+'/eval.results'
     print('args.result_path',args.result_path)
        
     if os.path.exists(args.eval_path):
        logger.info('Eval folder already exists, remove it!')
        shutil.rmtree(args.eval_path)
        os.mkdir(args.eval_path)
     else:
        os.mkdir(args.eval_path)
        
     if args.log_file=='':
        args.log_file=args.eval_path+'/eval.log'
        
     init_logger(args.log_file)
    
     if args.test_steps=='all':
        cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
        cp_files.sort(key=os.path.getmtime)
        steps = [cp.split('.')[-2].split('_')[-1] for cp in cp_files]
     else:
        steps = args.test_steps.split(',')
     logger.info('Testing step models in the model folder %s, steps: %s '%(args.model_path, steps))
     test_rouge_lst=[]
     for step in steps:
         cp = args.model_path+'/model_step_'+step+'.pt'
         step=int(step)
         xent, rouges = test_ext(args, device_id, cp, step)
         test_rouge_lst.append((cp,rouges))
         
    
        
     with open(args.eval_path+'/test_rouges.json', 'w+') as f:
         json.dump(test_rouge_lst,f)
     with open(args.eval_path+'/test_avg_rouges.json', 'w+') as f:
         metrics=['rouge_1_f_score','rouge_2_f_score','rouge_l_f_score',
                     'rouge_1_recall','rouge_2_recall','rouge_l_recall',
                     'rouge_1_precision','rouge_2_precision','rouge_l_precision']
         dic={}    
         for m in metrics:
            li=[]
            for item in test_rouge_lst:
                li.append(item[1][m])
            avg=statistics.mean(li)
            dic.update({m:avg})
        
         json.dump(dic,f) 
         logger.info('Avg. rouges of the model______%s \n%s' % (args.model_path.split('/')[1], rouge_results_to_str(dic)))
         
    

def test_ext(args, device_id, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    
    model = ExtSummarizer(args, device, checkpoint)
   
    model.eval()
    
    test_iter = data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                       args.test_batch_size, device,
                                       shuffle=False, is_test=True)
   
    trainer = build_trainer(args, device_id, model, None)
    
    stats, rouges =trainer.test(test_iter, step)
    
    return stats.xent(), rouges

def get_cand_list_ext(args, device_id, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    
    test_from = pt
    if args.eval_path=='':
        args.eval_path=args.model_path+'/matchsum'
    if args.result_path=='':
        args.result_path=args.eval_path+'/eval.results'
        
    if not os.path.exists(args.eval_path):
        os.mkdir(args.eval_path)
   
        
    
    
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    
  
    model = ExtSummarizer(args, device, checkpoint)
   
    model.eval()
    
    test_iter = data_loader.Dataloader(args, load_dataset(args, args.corpus_type, shuffle=False),
                                       args.test_batch_size, device,
                                       shuffle=False, is_test=True)
   
    trainer = build_trainer(args, device_id, model, None)
    
    trainer.get_cand_list(test_iter, step)
    


def baseline_ext(args, cal_lead=False, cal_oracle=False):
    
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)
    
    if args.eval_path=='':
        args.eval_path=args.model_path+'/eval'
        
    if args.log_file=='':
        args.log_file=args.eval_path+'/eval.log'
    
    if args.result_path=='':
        args.result_path=args.eval_path+'/eval.results'
        
    elif '/'.join(args.result_path.split('/')[:-1]) != args.eval_path:
        raise ValueError("Evaluation result path not in the eval folder")
        
    #create eval folder if not exists, delete if exists
    if os.path.exists(args.eval_path):
        logger.info('Eval folder already exists, remove it!')
        shutil.rmtree(args.eval_path)
        os.mkdir(args.eval_path)
    else:
        os.mkdir(args.eval_path)
    
    init_logger(args.log_file)
    logger.info(args)
    
    test_iter = data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                       args.batch_size, 'cpu',
                                       shuffle=False, is_test=True)

    trainer = build_trainer(args, -1, None, None)
   
    if (cal_lead):
        stats, rouges = trainer.test(test_iter, 0, cal_lead=True)
    elif (cal_oracle):
        stats, rouges = trainer.test(test_iter, 0, cal_oracle=True)
        
        
     
    #save info for post analysis
    with open(args.model_path+'/eval/test_rouges.json', 'w+') as f:
        json.dump(rouges,f)
         
    with open(args.model_path+'/eval/test_avg_rouges.json', 'w+') as f:
        metrics=['rouge_1_f_score','rouge_2_f_score','rouge_l_f_score',
                 'rouge_1_recall','rouge_2_recall','rouge_l_recall',
                 'rouge_1_precision','rouge_2_precision','rouge_l_precision']
        dic={}    
        for m in metrics:
            dic.update({m:rouges[m]})
    
        json.dump(dic,f)
        logger.info('Avg. rouges of the model______%s \n%s' % (args.model_path.split('/')[1], rouge_results_to_str(dic)))
        
        

def train_ext(args, device_id):
    #check if the model already exists
    #create save folder if not exisits
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)
        logger.info('Model folder created.')
    else:
        if len(os.listdir(args.model_path))!= 0:
            text = input('Model folder already exisits and is not empty. Do you want to remove it and redo training (yes or no) ?')
            if text.lower()=='yes':
                shutil.rmtree(args.model_path)
                os.mkdir(args.model_path)
                logger.info('YES: Model folder removed and recreated.')
            else:
                logger.info('NO: Program stopped.')
                exit()
                
    
    if args.log_file=='':
        args.log_file=args.model_path+'/train.log'
        
    init_logger(args.log_file)
    logger.info(args)
    
    if (args.world_size > 1):
        logger.info("Training (train_multi_ext)...")#
        train_multi_ext(args)
    else:
        logger.info("Training (train_single_ext)...")#
        train_single_ext(args, device_id)


def train_single_ext(args, device_id):

    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    logger.info('Device ID %d' % device_id)
    logger.info('Device %s' % device)
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        torch.cuda.manual_seed(args.seed)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if args.train_from != '':
        logger.info('Loading checkpoint from %s' % args.train_from)
        checkpoint = torch.load(args.train_from,
                                map_location=lambda storage, loc: storage)
        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if (k in model_flags):
                setattr(args, k, opt[k])
    else:
        checkpoint = None

    def train_iter_fct():
        return data_loader.Dataloader(args, load_dataset(args, 'train', shuffle=True), args.batch_size, device,
                                      shuffle=True, is_test=False)

    model = ExtSummarizer(args, device, checkpoint)
    optim = model_builder.build_optim(args, model, checkpoint)

    logger.info(model)

    trainer = build_trainer(args, device_id, model, optim)
    trainer.train(train_iter_fct, args.train_steps)
