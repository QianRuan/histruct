import math

import torch
import torch.nn as nn
from others.logging import logger,init_logger
from models.neural import MultiHeadedAttention, PositionwiseFeedForward
from models.histruct.sentStructEmb import LASentAddEmb, SINSentAddEmb, LPSentAddEmb, SinPositionalEncoding


class Classifier(nn.Module):
    def __init__(self, hidden_size):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask_cls):
        h = self.linear1(x).squeeze(-1)
        sent_scores = self.sigmoid(h) * mask_cls.float()
        return sent_scores


class PositionalEncoding(nn.Module):

    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if (step):
            emb = emb + self.pe[:, step][:, None, :]

        else:
            emb = emb + self.pe[:, :emb.size(1)]
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pe[:, :emb.size(1)]



class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, query, inputs, mask):
        if (iter != 0):
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs

        mask = mask.unsqueeze(1)
        context = self.self_attn(input_norm, input_norm, input_norm,
                                 mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)




class ExtTransformerEncoder(nn.Module):
    def __init__(self, model, args, d_model, d_ff, heads, dropout, num_inter_layers=0):
        super(ExtTransformerEncoder, self).__init__()
        self.args = args
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers
        
        init_logger(self.args.log_file)
        
        if (args.add_sent_struct_emb):
            
            logger.info("#####Sentence embeddings_add sentence hierarchical position embeddings (sHE): TRUE") 
            if (args.sent_pos_emb_type == 'learned_all'):
                logger.info("-----Type of positional embeddings...learnable")
                logger.info("-----Sequential position and hiarchical positions...different PosEmbs ")
                logger.info("-----Sentence Structure Embeddings_combination mode ... "+args.sent_se_comb_mode)
                
                self.add_emb = LASentAddEmb(args,model.config)
               
            elif (args.sent_pos_emb_type == 'learned_pos'):
                logger.info("-----Type of positional embeddings...learnable")
                logger.info("-----Sequential position and hiarchical positions...one same PosEmb")
                logger.info("-----Sentence Structure Embeddings_combination mode ... "+args.sent_se_comb_mode)
                
                self.add_emb = LPSentAddEmb(args,model.config)
               
                    
            elif (args.sent_pos_emb_type == 'sinusoidal'):
                logger.info("-----Type of positional embeddings...sinusoidal")
                logger.info("-----Sentence Structure Embeddings_combination mode ... "+args.sent_se_comb_mode)
                
                self.add_emb = SINSentAddEmb(args,model.config)
                
            else:
                raise ValueError("args.sent_pos_emb_type must be one of ['learned_pos', 'learned_all', 'sinusoidal'] ")
        else:
            self.add_emb = SinPositionalEncoding(d_model, max_len = args.max_nsent)#
            logger.info("#####Sentence embeddings_add sentence hierarchical structure embeddings: FALSE") 
            logger.info("-----only add sentence sinusoidal positional embeddings") 
            
        #section_names    
        if self.args.section_names_embed_path!='':
                self.sn_emb_dict = torch.cuda.FloatTensor(list(torch.load(self.args.section_names_embed_path).values())) 
                     
        else:
                self.sn_emb_dict=None
 
        
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_inter_layers)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.wo = nn.Linear(d_model, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        
        

    def forward(self, top_vecs, mask, sent_struct_vec, section_names):#!#
        """ See :obj:`EncoderBase.forward()`"""
        
        sent_pos_emb = self.add_emb(top_vecs, sent_struct_vec)

        
        #if not using hierarchical structure embeddings
        if type(sent_pos_emb) == tuple:
            sent_pos_emb = sent_pos_emb[1]
        #add sentence hierarchical structure embeddings
        x = top_vecs * mask[:, :, None].float()
        x = x + sent_pos_emb
        
        
        #add section names embeddings
        #section_names: index of unique section name according to the file at section_names_embed_path
        sn_emb=None
        if section_names is not None and self.args.section_names_embed_path!='':
            
            #section_pos: section index of each sentences in the document
            section_pos = sent_struct_vec[:,:,0]

            
            
            #sn_index: convert section_pos into section names index 
            indices = torch.tensor(list(range(section_pos.shape[0])))
            sn_index=section_names[:,section_pos][indices,indices,:]

            
            #sn_emb: get section name embedding from the file at section_names_embed_path
            sn_emb=self.sn_emb_dict[sn_index,:]

        if sn_emb is not None:
            x = x + sn_emb

        for i in range(self.num_inter_layers):
            x = self.transformer_inter[i](i, x, x, ~mask)
        x = self.layer_norm(x)
        sent_scores = self.sigmoid(self.wo(x))
        sent_scores = sent_scores.squeeze(-1) * mask.float()

        return sent_scores
    

