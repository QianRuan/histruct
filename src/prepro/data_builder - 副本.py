import gc
import glob
import hashlib
import itertools
import json
import os
import random
import re
import subprocess
from collections import Counter
from os.path import join as pjoin

import torch
from multiprocess import Pool

from others.logging import logger
#from others.tokenization import BertTokenizer
from pytorch_transformers import XLNetTokenizer
from pytorch_transformers import BertTokenizer

from others.utils import clean
from prepro.utils import _get_word_ngrams

import xml.etree.ElementTree as ET

nyt_remove_words = ["photo", "graph", "chart", "map", "table", "drawing"]


def recover_from_corenlp(s):
    s = re.sub(r' \'{\w}', '\'\g<1>', s)
    s = re.sub(r'\'\' {\w}', '\'\'\g<1>', s)



def load_json(p, lower):
    source = []
    tgt = []
    flag = False
    for sent in json.load(open(p))['sentences']:
        tokens = [t['word'] for t in sent['tokens']]
        if (lower):
            tokens = [t.lower() for t in tokens]
        if (tokens[0] == '@highlight'):
            flag = True
            tgt.append([])
            continue
        if (flag):
            tgt[-1].extend(tokens)
        else:
            source.append(tokens)

    source = [clean(' '.join(sent)).split() for sent in source]
    tgt = [clean(' '.join(sent)).split() for sent in tgt]
    return source, tgt

def obtain_histruct_info(args):
    tok_sent_dir = os.path.abspath(args.tok_sent_path)
    tok_para_dir = os.path.abspath(args.tok_para_path)
    tok_histruct_dir = os.path.abspath(args.save_path)
    
   
    
    tok_sent_stories = os.listdir(tok_sent_dir)
    #tok_para_stories = os.listdir(tok_para_dir)
    
    for sent_story in  tok_sent_stories:
        
        sent_story_path = tok_sent_dir+'\\'+sent_story
        para_story_path = tok_para_dir+'\\'+sent_story #same name
        print("###############sent_story_path",sent_story_path)
        print("###############para_story_path",para_story_path)
        
        #get list of tokens of sentences from source text
        source_sent_tokens=[]
        flag =False
        for sent in json.load(open(sent_story_path))['sentences']:
            tokens = [t['word'] for t in sent['tokens']]
            if(tokens[0] == '@highlight'):
                flag = True
                #print("tokens",tokens)
                #print("flag",flag)
                continue
            if (not flag):
                source_sent_tokens.append(tokens)
        #print (len(source_sent_tokens),source_sent_tokens)
        
        #get list of tokens of paragraphs from source text
        source_para_tokens=[]
        flag =False
        for sent in json.load(open(para_story_path))['sentences']:
            tokens = [t['word'] for t in sent['tokens']]
            if(tokens[0] == '@highlight'):
                flag = True
                #print("tokens",tokens)
                #print("flag",flag)
                continue
            if (not flag):
                source_para_tokens.append(tokens)
        #print (len(source_para_tokens),source_para_tokens)
        
        
        #j=0
        #token_struct_vec = []
        sent_struct_vec = []
        #overall_sent_pos = []
        for i in range(len(source_para_tokens)):
            #print("###1")
            sent_idx_in_para=0
            for j in range(len(source_sent_tokens)):
                #print("###2")
                #print(source_sent_tokens[j])
                #print(source_para_tokens[i])
                #print("###3")
                
                #print("###4")
                if (source_sent_tokens[j]!=[]) and (source_para_tokens[i][:len(source_sent_tokens[j])] == source_sent_tokens[j]):
                    #print("###5")
                    sent_struct_vec.append([i,sent_idx_in_para])
                    
                    source_para_tokens[i] = source_para_tokens[i][len(source_sent_tokens[j]):]
                    source_sent_tokens[j] = []
                    
                    #print(source_sent_tokens)
                    #print(source_para_tokens)
                    #print(sent_struct_vec)
                    if source_para_tokens[i] != []:
                        sent_idx_in_para+=1
                        #print("continue")
                        continue
                    if source_para_tokens[i] == []:
                        #print("break")
                        break
                    
        print(sent_struct_vec)
        
        
        #read original json data, inject histruct info
        flag =False
        doc = json.load(open(sent_story_path))
        #sents = doc['sentences']
        for i in range(len(doc['sentences'])):
            
            tokens = [t['word'] for t in  doc['sentences'][i]['tokens']]
            if(tokens[0] == '@highlight'):
                flag = True
                #print("tokens",tokens)
                #print("flag",flag)
                continue
            if (not flag):
                doc['sentences'][i]['overall_sent_pos'] = i
                doc['sentences'][i]['sent_struct_vec'] = sent_struct_vec[i]
                a = sent_struct_vec[i][0]
                b = sent_struct_vec[i][1]
                
                for t in doc['sentences'][i]['tokens']:
                    t['token_struct_vec'] = [a,b,int(t['index'])]
                    
                #doc['sentences'][i]['tokens']['token_struct_vec'] = [a,b,int(doc['sentences'][i]['tokens']['index'])]
                
                
                
                
        #save modified json data
        if not os.path.exists(tok_histruct_dir):
           os.makedirs(tok_histruct_dir)
           
        save_file = tok_histruct_dir+'\\'+sent_story
        
        with open(save_file, 'w+') as outfile:
            json.dump(doc, outfile)

    



def load_xml(p):
    tree = ET.parse(p)
    root = tree.getroot()
    title, byline, abs, paras = [], [], [], []
    title_node = list(root.iter('hedline'))
    if (len(title_node) > 0):
        try:
            title = [p.text.lower().split() for p in list(title_node[0].iter('hl1'))][0]
        except:
            print(p)

    else:
        return None, None
    byline_node = list(root.iter('byline'))
    byline_node = [n for n in byline_node if n.attrib['class'] == 'normalized_byline']
    if (len(byline_node) > 0):
        byline = byline_node[0].text.lower().split()
    abs_node = list(root.iter('abstract'))
    if (len(abs_node) > 0):
        try:
            abs = [p.text.lower().split() for p in list(abs_node[0].iter('p'))][0]
        except:
            print(p)

    else:
        return None, None
    abs = ' '.join(abs).split(';')
    abs[-1] = abs[-1].replace('(m)', '')
    abs[-1] = abs[-1].replace('(s)', '')

    for ww in nyt_remove_words:
        abs[-1] = abs[-1].replace('(' + ww + ')', '')
    abs = [p.split() for p in abs]
    abs = [p for p in abs if len(p) > 2]

    for doc_node in root.iter('block'):
        att = doc_node.get('class')
        # if(att == 'abstract'):
        #     abs = [p.text for p in list(f.iter('p'))]
        if (att == 'full_text'):
            paras = [p.text.lower().split() for p in list(doc_node.iter('p'))]
            break
    if (len(paras) > 0):
        if (len(byline) > 0):
            paras = [title + ['[unused3]'] + byline + ['[unused4]']] + paras
        else:
            paras = [title + ['[unused3]']] + paras

        return paras, abs
    else:
        return None, None


def tokenize(args):
    stories_dir = os.path.abspath(args.raw_path)
    tok_sent_dir = os.path.abspath(args.tok_sent_path)
    tok_para_dir = os.path.abspath(args.tok_para_path)
    corenlp_path = args.corenlp_path

    print("Preparing to tokenize %s" % (stories_dir))
    stories = os.listdir(stories_dir)
    # make IO list file
    print("Making list of files to tokenize...")
    with open("mapping_for_corenlp.txt", "w") as f:
        for s in stories:
            if (not s.endswith('story')):
                continue
            f.write("%s\n" % (os.path.join(stories_dir, s)))
#    command = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit',
#               '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat',
#               'json', '-outputDirectory', tokenized_stories_dir]
    command_sent = [corenlp_path, 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit',
               '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat',
               'json', '-outputDirectory', tok_sent_dir]#
    command_para = [corenlp_path, 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit',
               '-ssplit.eolonly', 'true', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat',
               'json', '-outputDirectory', tok_para_dir]#
    
    print("Tokenizing %i files in %s and saving in %s..." % (len(stories), stories_dir, tok_sent_dir))
    subprocess.call(command_sent,shell=True)
    print("Stanford CoreNLP Tokenizer has finished.")
    #os.remove("mapping_for_corenlp.txt")
    
    print("Tokenizing %i files in %s and saving in %s..." % (len(stories), stories_dir, tok_para_dir))
    subprocess.call(command_para,shell=True)#
    print("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping_for_corenlp.txt")

    # Check that the tokenized stories directory contains the same number of files as the original directory
    num_orig = len(os.listdir(stories_dir))
    num_tokenized_para = len(os.listdir(tok_para_dir))
    num_tokenized_sent = len(os.listdir(tok_sent_dir))
    if num_orig != num_tokenized_sent:
        raise Exception(
            "The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
                tok_sent_dir, num_tokenized_sent, stories_dir, num_orig))
    if num_orig != num_tokenized_para:
        raise Exception(
            "The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
                tok_para_dir, num_tokenized_para, stories_dir, num_orig))
    print("Successfully finished tokenizing %s to %s.\n" % (stories_dir, tok_para_dir))
    print("Successfully finished tokenizing %s to %s.\n" % (stories_dir, tok_sent_dir))
    
def tokenize_bert(args):
    stories_dir = os.path.abspath(args.raw_path)
    tok_sent_dir = os.path.abspath(args.tok_sent_path)
    tok_para_dir = os.path.abspath(args.tok_para_path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    print("Preparing to tokenize %s" % (stories_dir))
    stories = os.listdir(stories_dir)
    
    tok_para_list=[]
    for story in stories:
        #doc['docId']=story
        with open(stories_dir+"\\"+story) as s:
            with open(tok_para_dir+"\\"+story,"w+") as out:
                for line in s.readlines(s):
                    tok_para = tokenizer.tokenize(s)
                    print(tok_para)
                    tok_para_list.append(tok_para)
                out.save(tok_para_list)
#    # make IO list file
#    print("Making list of files to tokenize...")
#    with open("mapping_for_corenlp.txt", "w") as f:
#        for s in stories:
#            if (not s.endswith('story')):
#                continue
#            f.write("%s\n" % (os.path.join(stories_dir, s)))
#    
#    
#    command_sent = [corenlp_path, 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit',
#               '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat',
#               'json', '-outputDirectory', tok_sent_dir]#
#    command_para = [corenlp_path, 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit',
#               '-ssplit.eolonly', 'true', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat',
#               'json', '-outputDirectory', tok_para_dir]#
#    
#    print("Tokenizing %i files in %s and saving in %s..." % (len(stories), stories_dir, tok_sent_dir))
#    subprocess.call(command_sent,shell=True)
#    print("Stanford CoreNLP Tokenizer has finished.")
#    #os.remove("mapping_for_corenlp.txt")
#    
#    print("Tokenizing %i files in %s and saving in %s..." % (len(stories), stories_dir, tok_para_dir))
#    subprocess.call(command_para,shell=True)#
#    print("Stanford CoreNLP Tokenizer has finished.")
#    os.remove("mapping_for_corenlp.txt")
#
#    # Check that the tokenized stories directory contains the same number of files as the original directory
#    num_orig = len(os.listdir(stories_dir))
#    num_tokenized_para = len(os.listdir(tok_para_dir))
#    num_tokenized_sent = len(os.listdir(tok_sent_dir))
#    if num_orig != num_tokenized_sent:
#        raise Exception(
#            "The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
#                tok_sent_dir, num_tokenized_sent, stories_dir, num_orig))
#    if num_orig != num_tokenized_para:
#        raise Exception(
#            "The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
#                tok_para_dir, num_tokenized_para, stories_dir, num_orig))
#    print("Successfully finished tokenizing %s to %s.\n" % (stories_dir, tok_para_dir))
#    print("Successfully finished tokenizing %s to %s.\n" % (stories_dir, tok_sent_dir))

def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()


class BertData():
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.tgt_bos = '[unused0]'
        self.tgt_eos = '[unused1]'
        self.tgt_sent_split = '[unused2]'
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]

    def preprocess(self, doc, src, tgt, sent_labels, use_bert_basic_tokenizer=False, is_test=False):
        #src:list of sentences within a document
        #tgt:list of summaries (sentences) within a document
        #token_struct_vec: List (sentences) of list of token_struct_vec (3-dim)
        #sentence_struct_vec: List of sentence_struct_vec (2-dim)
        #sentence_overall_pos: List of overall positions of the sentences
        

        if ((not is_test) and len(src) == 0):
            return None
        
        print("#################src",len(src),src)
        original_src_txt = [' '.join(s) for s in src]
        print("#################original_src_txt",len(original_src_txt),original_src_txt)

        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens_per_sent)]
        print("#################self.args.min_src_ntokens_per_sent",self.args.min_src_ntokens_per_sent)
        print("#################idxs",len(idxs),idxs)

        _sent_labels = [0] * len(src)
        for l in sent_labels:
            _sent_labels[l] = 1

        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]
        print("#################self.args.max_src_ntokens_per_sent",self.args.max_src_ntokens_per_sent)
        print("#################idxs",len(src),src)
        sent_labels = [_sent_labels[i] for i in idxs]
        print("#################sent_labels",len(sent_labels),sent_labels)
        src = src[:self.args.max_src_nsents]
        print("#################idxs",len(src),src)
        sent_labels = sent_labels[:self.args.max_src_nsents]
        print("#################self.args.max_src_nsents",self.args.max_src_nsents)
        print("#################sent_labels",len(sent_labels),sent_labels)
        

        if ((not is_test) and len(src) < self.args.min_src_nsents):
            print("#################self.args.min_src_nsents",self.args.min_src_nsents)
            return None

#        src_txt = [' '.join(sent) for sent in src]#
#        print("#################src_txt",len(src_txt),src_txt)
#        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)#
#        print("#################text",len(text), text)
#
#        src_subtokens = self.tokenizer.tokenize(text)#
        src_subtokens=[]
        for sent in src:
            src_subtokens.append(self.cls_token)
            for t in sent:
                src_subtokens.append(t)
            src_subtokens.append(self.sep_token)
                
        #print("#################self.tokenizer",self.tokenizer)
        print("#################src_subtokens",len(src_subtokens), src_subtokens)

        #src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
        #print("#################src_subtokens",len(src_subtokens), src_subtokens)
        
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        print("#################src_subtokens",len( src_subtoken_idxs),  src_subtoken_idxs)
        
        
        
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
                
                
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid] #index of [CLS]
        print("#################cls_ids",len(cls_ids), cls_ids)
        sent_labels = sent_labels[:len(cls_ids)]

        tgt_subtokens_str = '[unused0] ' + ' [unused2] '.join(
            [' '.join(self.tokenizer.tokenize(' '.join(tt), use_bert_basic_tokenizer=use_bert_basic_tokenizer)) for tt in tgt]) + ' [unused1]'
        tgt_subtoken = tgt_subtokens_str.split()[:self.args.max_tgt_ntokens]
        if ((not is_test) and len(tgt_subtoken) < self.args.min_tgt_ntokens):
            return None

        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)

        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]

        return src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt


def format_to_bert(args):
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['train', 'valid', 'test']
    for corpus_type in datasets:
        a_lst = []
        for json_f in glob.glob(pjoin(args.raw_path, '*' + corpus_type + '.*.json')):       
            #real_name = json_f.split('/')[-1]
            real_name = json_f.split('/')[-1].split('\\')[-1]
            print("##########################real_name",real_name)
            a_lst.append((corpus_type, json_f, args, pjoin(args.save_path, real_name.replace('json', 'bert.pt'))))
        print(a_lst)
        pool = Pool(args.n_cpus)
        for d in pool.imap(_format_to_bert, a_lst):
            pass

        pool.close()
        pool.join()


def _format_to_bert(params):
    corpus_type, json_file, args, save_file = params
    print("##########################save_file",save_file)
    is_test = corpus_type == 'test'
    if (os.path.exists(save_file)):
#        logger.info('Ignore %s' % save_file)
#        return
        print("save_file already exisits, remove it")
        os.remove(save_file)

    bert = BertData(args)

    logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file)) #nr. of documents
    print("####################jobs",len(jobs))
    datasets = []
    for d in jobs:
        source, tgt = d['src'], d['tgt']
        #docId = d['docId']
        sents = d['sentences']
        overall_sent_pos=[]
        sent_struct_vec=[]
        token_struct_vec1=[]
        token_struct_vec_by_sent_list = []
        for sent in sents:
            token_struct_vec_by_sent=[]
            if 'overall_sent_pos' in sent.keys() and 'sent_struct_vec' in sent.keys():
                overall_sent_pos.append(sent['overall_sent_pos'])
                sent_struct_vec.append(sent['sent_struct_vec'])
                for t in sent['tokens']:
                    token_struct_vec_by_sent.append(t['token_struct_vec'])
                    token_struct_vec1.append(t['token_struct_vec'])
                token_struct_vec_by_sent_list.append(token_struct_vec_by_sent)
        #print(overall_sent_pos)    
        #print(sent_struct_vec) 
        print("########",len(token_struct_vec1)) 
        print("########",len(token_struct_vec_by_sent_list))
            
        
        #print("####################d,",d)
        #print("####################d['docID'],",d['docId'])
        #print("############print(d['sentences'])",d['sentences'])
        #print("####################source",len(source),source)
        #print("####################tgt",len(tgt),tgt)

        sent_labels = greedy_selection(source[:args.max_src_nsents], tgt, 3)
        tgt_sent_idx = sent_labels
        #print("####################sent_labels",len(sent_labels),sent_labels)#?????????2
        if (args.lower):
            source = [' '.join(s).lower().split() for s in source]
            tgt = [' '.join(s).lower().split() for s in tgt]
            #print("####################sourcelower",len(source),source)
            #print("####################tgtlower",len(tgt),tgt)
                  
        b_data = bert.preprocess(d, source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
                                 is_test=is_test)
        # b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer)

        if (b_data is None):
            continue
        src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data
        
        #src:list of sentences within a document
        #tgt:list of summaries (sentences) within a document
        #token_struct_vec: List (sentences) of list of token_struct_vec (3-dim)
        #sentence_struct_vec: List of sentence_struct_vec (2-dim)
        #sentence_overall_pos: List of overall positions of the sentences
        #token_overall_pos:
#        b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
#                       "src_sent_labels": sent_labels, "segs": segments_ids, 'clss': cls_ids,
#                       'src_txt': src_txt, "tgt_txt": tgt_txt}
        
        #add token_struct_vec for [CLS] and [SEP]
        token_struct_vec=[]
        for sent in token_struct_vec_by_sent_list:
            
            #sent = token_struct_vec_by_sent_list[i]
            #print("sent",sent)
            cls = [sent[0][0],sent[0][1],sent[0][2]-1]
            #print("cls",cls)
            sep = [sent[-1][0],sent[-1][1],sent[-1][2]+1]
            #print("sep",sep)
            new_sent = [cls] + sent + [sep]
            #print("new_sent",new_sent)
            token_struct_vec+=new_sent
        
        
        b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                       "src_sent_labels": sent_labels, "segs": segments_ids, 'clss': cls_ids,
                       'src_txt': src_txt, "tgt_txt": tgt_txt, 
                       "tgt_sent_idx":tgt_sent_idx,"overall_sent_pos":overall_sent_pos,
                       "sent_struct_vec":sent_struct_vec, "token_struct_vec":token_struct_vec}
        
        
        print("#################src_sent_labels",len(b_data_dict['src_sent_labels']),b_data_dict['src_sent_labels'])
        print("#################segs",len(b_data_dict['segs']),b_data_dict['segs'])
        print("#################clss",len(b_data_dict['clss']),b_data_dict['clss'])
        print("#################src_txt",len(b_data_dict['src_txt']),b_data_dict['src_txt'])
        print("#################tgt_txt",len(b_data_dict['tgt_txt']),b_data_dict['tgt_txt'])
        print("#################tgt_sent_idx",len(b_data_dict['tgt_sent_idx']),b_data_dict['tgt_sent_idx'])
        print("#################src",len(b_data_dict['src']),b_data_dict['src'])
        print("#################tgt",len(b_data_dict['tgt']),b_data_dict['tgt'])
        print("#################overall_sent_pos",len(b_data_dict['overall_sent_pos']),b_data_dict['overall_sent_pos'])
        print("#################sent_struct_vec",len(b_data_dict['sent_struct_vec']),b_data_dict['sent_struct_vec'])
        print("#################token_struct_vec",len(b_data_dict['token_struct_vec']),b_data_dict['token_struct_vec'])
    
             
        datasets.append(b_data_dict)
        
        #print("######################b_data_dict",b_data_dict)
        #break
    logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
    
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()


def format_to_lines(args):
    corpus_mapping = {}
    for corpus_type in ['valid', 'test', 'train']:
        temp = []
        for line in open(pjoin(args.map_path, 'mapping_' + corpus_type + '.txt')):
            temp.append(hashhex(line.strip()))
        corpus_mapping[corpus_type] = {key.strip(): 1 for key in temp}
    train_files, valid_files, test_files = [], [], []
    for f in glob.glob(pjoin(args.raw_path, '*.json')):
        
        #real_name = f.split('/')[-1].split('.')[0]
        real_name = f.split('/')[-1].split('\\')[1].split('.')[0]#
        
        print('##########################')
        print(real_name)
        #print(corpus_mapping['valid'])
        if (real_name in corpus_mapping['valid']):
            print(real_name,'in valid')
            valid_files.append(f)
        elif (real_name in corpus_mapping['test']):
            print(real_name,'in test')
            test_files.append(f)
        elif (real_name in corpus_mapping['train']):
            print(real_name,'in train')
            train_files.append(f)
        # else:
        #     train_files.append(f)

    corpora = {'train': train_files, 'valid': valid_files, 'test': test_files}
    for corpus_type in ['train', 'valid', 'test']:
        a_lst = [(f, args) for f in corpora[corpus_type]]
        pool = Pool(args.n_cpus)
        dataset = []
        p_ct = 0
        for d in pool.imap_unordered(_format_to_lines, a_lst):
            dataset.append(d)
            if (len(dataset) > args.shard_size):
                pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
                with open(pt_file, 'w') as save:
                    # save.write('\n'.join(dataset))
                    save.write(json.dumps(dataset))
                    p_ct += 1
                    dataset = []

        pool.close()
        pool.join()
        if (len(dataset) > 0):
            pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
            with open(pt_file, 'w') as save:
                # save.write('\n'.join(dataset))
                save.write(json.dumps(dataset))
                p_ct += 1
                dataset = []


def _format_to_lines(params):
    f, args = params
    print("#####f",f)
    source, tgt = load_json(f, args.lower)
    doc = json.load(open(f))
    doc['src'] = source
    doc['tgt'] = tgt
    return doc #{'src': source, 'tgt': tgt}




