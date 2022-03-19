import math
import torch
from torch import nn
from pytorch_transformers.modeling_bert import BertLayerNorm


def compute_se(pos, pe, position_embeddings): 

    x_position_embeddings = torch.zeros_like(position_embeddings)
 
    for i in range(pos.size(0)):
  
        for j in range (pos.size(1)):
        
            idx = int(pos[i][j].item())
            x_position_embeddings[i][j] = pe[0][idx]
            

    return x_position_embeddings



class SinPositionalEncoding(nn.Module):

    def __init__(self, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(SinPositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dim = dim

    def forward(
        self,
        inputs,
        position_ids=None,
    ):

        batch_size = inputs.size(0)
        n = inputs.size(1)

        pe = self.pe[:, :n]

        
        pos_embs = pe.expand(batch_size,-1,-1)

               
        return pe, pos_embs

class LASentAddEmb(nn.Module):
    
    def __init__(self, args, config):
       
        super(LASentAddEmb, self).__init__()
        
        self.args = args
        self.position_embeddings = nn.Embedding(args.max_nsent, config.hidden_size)
        
        
        
        if(self.args.sent_se_comb_mode == 'concat'):
            if args.max_npara==0:
                self.a_position_embeddings = nn.Embedding(args.max_nsent, int(config.hidden_size/2))
            else:
                self.a_position_embeddings = nn.Embedding(args.max_npara, int(config.hidden_size/2))
            if args.max_nsent_in_para==0:
                self.b_position_embeddings = nn.Embedding(args.max_nsent, int(config.hidden_size/2))
            else:
                self.b_position_embeddings = nn.Embedding(args.max_nsent_in_para, int(config.hidden_size/2))
                
        else:
            if args.max_npara==0:
                self.a_position_embeddings = nn.Embedding(args.max_nsent, config.hidden_size)
            else:
                self.a_position_embeddings = nn.Embedding(args.max_npara, config.hidden_size)
            if args.max_nsent_in_para==0:
                self.b_position_embeddings = nn.Embedding(args.max_nsent, config.hidden_size)
            else:
                self.b_position_embeddings = nn.Embedding(args.max_nsent_in_para, config.hidden_size)
                
        
        if args.base_LM.startswith('bigbird-pegasus') or args.base_LM.startswith('bart'):
            self.LayerNorm = nn.LayerNorm(config.d_model)
            self.dropout = nn.Dropout(config.dropout)
        else:
            self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        top_vecs,
        sent_struct_vec,
        position_ids=None,
    ):
        
        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
       
        if position_ids is None:
            position_ids = torch.arange(n_sents, dtype=torch.long, device=top_vecs.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size,-1)

        
        position_embeddings = self.position_embeddings(position_ids)
        
              
        para_pos = sent_struct_vec[:,:,0]
        sent_pos = sent_struct_vec[:,:,1]
        
        
        para_position_embeddings = self.a_position_embeddings(para_pos)
        sent_position_embeddings = self.b_position_embeddings(sent_pos)
        
        
        if(self.args.sent_se_comb_mode == 'sum'):
            sent_struct_embeddings = para_position_embeddings+sent_position_embeddings
            
        elif(self.args.sent_se_comb_mode == 'mean'):
            sent_struct_embeddings = (para_position_embeddings+sent_position_embeddings)/2
            
        elif(self.args.sent_se_comb_mode == 'concat'):
            sent_struct_embeddings = torch.cat((para_position_embeddings,sent_position_embeddings,),2)
            
        else:
            raise ValueError ("args.sent_se_comb_mode must be one of ['sum', 'mean', 'concat']")
            
       
        
        if self.args.without_sent_pos and self.args.para_only:
            
            embeddings = para_position_embeddings
            
        elif self.args.without_sent_pos:
            
            embeddings = sent_struct_embeddings
            
        elif self.args.para_only:
            
            embeddings = (           
                 position_embeddings          
                + para_position_embeddings
            )     
        else:
            
            embeddings = (           
                 position_embeddings          
                + sent_struct_embeddings
            )
            
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings
    
class SINSentAddEmb(nn.Module):
    def __init__(self, args, config):
       
        super(SINSentAddEmb, self).__init__()
        
        self.args = args
       
        self.position_embeddings = SinPositionalEncoding(config.hidden_size, max_len=args.max_nsent)
        
        if(self.args.sent_se_comb_mode == 'concat'):
            self.histruct_position_embeddings = SinPositionalEncoding(int(config.hidden_size/2),max_len=args.max_nsent)
        else:
            self.histruct_position_embeddings = None
        
        if args.base_LM.startswith('bigbird-pegasus') or args.base_LM.startswith('bart'):
            self.LayerNorm = nn.LayerNorm(config.d_model)
            self.dropout = nn.Dropout(config.dropout)
        else:
            self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            
            


    def forward(
        self,
        top_vecs,
        sent_struct_vec,
        position_ids=None,
    ):
        

        batch_size = top_vecs.size(0)
        n = top_vecs.size(1)
        pe = self.position_embeddings.pe[:, :n]      
        position_embeddings = pe.expand(batch_size,-1,-1)
  
        if (self.histruct_position_embeddings != None):

            hs_pe = self.histruct_position_embeddings.pe[:, :n]   
            hs_position_embeddings = hs_pe.expand(batch_size,-1,-1)
            
        else:
            hs_pe,hs_position_embeddings = None, None
        
    
        para_pos = sent_struct_vec[:,:,0]
        sent_pos = sent_struct_vec[:,:,1]

                    
        if(self.args.sent_se_comb_mode == 'concat'):
            para_position_embeddings = compute_se(para_pos, self.histruct_position_embeddings.pe, hs_position_embeddings)
            sent_position_embeddings = compute_se(sent_pos, self.histruct_position_embeddings.pe, hs_position_embeddings)
        else:
            para_position_embeddings = compute_se(para_pos, self.position_embeddings.pe, position_embeddings)
            sent_position_embeddings = compute_se(sent_pos, self.position_embeddings.pe, position_embeddings)
            

        
        if(self.args.sent_se_comb_mode == 'sum'):
            sent_struct_embeddings = para_position_embeddings+sent_position_embeddings
            
        elif(self.args.sent_se_comb_mode == 'mean'):
            sent_struct_embeddings = (para_position_embeddings+sent_position_embeddings)/2
            
        elif(self.args.sent_se_comb_mode == 'concat'):
            sent_struct_embeddings = torch.cat((para_position_embeddings,sent_position_embeddings),2)
            
        else:
            raise ValueError("args.sent_se_comb_mode must be one of ['sum','mean','concat']")
            
        
        if self.args.without_sent_pos and self.args.para_only:
            
            embeddings = para_position_embeddings
            
        elif self.args.without_sent_pos:
            
            embeddings = sent_struct_embeddings
            
        elif self.args.para_only:
           
            embeddings = (           
                 position_embeddings          
                + para_position_embeddings
            )     
        else:
            
            embeddings = (           
                 position_embeddings          
                + sent_struct_embeddings
            )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings




    
class LPSentAddEmb(nn.Module):
    def __init__(self, args, config):
       
        super(LPSentAddEmb, self).__init__()
        
        self.args =args
        self.position_embeddings = nn.Embedding(args.max_nsent, config.hidden_size)
        
        if(self.args.sent_se_comb_mode == 'concat'):

            raise ValueError ("Concat mode can not be used when we only learn one PosEmb for all positions")
            
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        top_vecs,

        sent_struct_vec,
        position_ids=None,
    ):
        
        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        
        if position_ids is None:
            position_ids = torch.arange(n_sents, dtype=torch.long, device=top_vecs.device)

            position_ids = position_ids.unsqueeze(0).expand(batch_size,-1)

        
        position_embeddings = self.position_embeddings(position_ids)
        
              
        para_pos = sent_struct_vec[:,:,0]
        sent_pos = sent_struct_vec[:,:,1]
        
       
        para_position_embeddings = self.position_embeddings(para_pos)
        sent_position_embeddings = self.position_embeddings(sent_pos)
        

        
        if(self.args.sent_se_comb_mode == 'sum'):
            sent_struct_embeddings = para_position_embeddings+sent_position_embeddings
            
        elif(self.args.sent_se_comb_mode == 'mean'):
            sent_struct_embeddings = (para_position_embeddings+sent_position_embeddings)/2
            
        else:
            raise ValueError ("args.sent_se_comb_mode must be one of ['sum', 'mean']")
            
        
        if self.args.without_sent_pos and self.args.para_only:
            
            embeddings = para_position_embeddings
            
        elif self.args.without_sent_pos:
            
            embeddings = sent_struct_embeddings
            
        elif self.args.para_only:
            
            embeddings = (           
                 position_embeddings          
                + para_position_embeddings
            )     
        else:
            
            embeddings = (           
                 position_embeddings          
                + sent_struct_embeddings
            )
        

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings
    
    
class LPSentAddEmbPOS(nn.Module):
     def __init__(self, args, config):
       
        super(LPSentAddEmbPOS, self).__init__()
    
        self.position_embeddings = nn.Embedding(
            args.max_nsent, config.hidden_size
        )
              
        
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

     def forward(
        self,
        top_vecs,
        position_ids=None,
    ):
        
        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        
        if position_ids is None:
            position_ids = torch.arange(
                n_sents, dtype=torch.long, device=top_vecs.device
            )

            position_ids = position_ids.unsqueeze(0).expand(batch_size,-1)

        
        position_embeddings = self.position_embeddings(position_ids)
        
        return position_embeddings
    


        



    

