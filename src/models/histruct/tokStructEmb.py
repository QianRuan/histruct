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
        
        return pe,pos_embs

class LATokInputEmb(nn.Module):
    def __init__(self, config, args):
        super(LATokInputEmb, self).__init__()
        
        self.args=args
        
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        
        if(self.args.tok_se_comb_mode == 'concat'):
            self.a_position_embeddings = nn.Embedding(args.max_nsent, int(config.hidden_size/3))
            self.b_position_embeddings = nn.Embedding(args.max_nsent, int(config.hidden_size/3))
            self.c_position_embeddings = nn.Embedding(args.max_pos, int(config.hidden_size/3))
        else:
            print('args.max_pos',args.max_pos)
            self.a_position_embeddings = nn.Embedding(args.max_nsent, config.hidden_size)
            self.b_position_embeddings = nn.Embedding(args.max_nsent, config.hidden_size)
            self.c_position_embeddings = nn.Embedding(args.max_pos, config.hidden_size)
        
       
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids,
        tok_struct_vec,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
    ):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
              
        para_pos = tok_struct_vec[:,:,0]
        sent_pos = tok_struct_vec[:,:,1]
        tok_pos = tok_struct_vec[:,:,2]
        
        para_position_embeddings = self.a_position_embeddings(para_pos)
        sent_position_embeddings = self.b_position_embeddings(sent_pos)
        tok_position_embeddings = self.c_position_embeddings(tok_pos)

        
        if(self.args.tok_se_comb_mode == 'sum'):
            tok_struct_embeddings = para_position_embeddings+sent_position_embeddings+tok_position_embeddings
            
        elif(self.args.tok_se_comb_mode == 'mean'):
            tok_struct_embeddings = (para_position_embeddings+sent_position_embeddings+tok_position_embeddings)/3
            
        elif(self.args.tok_se_comb_mode == 'concat'):
            tok_struct_embeddings = torch.cat((para_position_embeddings,sent_position_embeddings,tok_position_embeddings),2)
        else:
            raise ValueError ("args.tok_se_comb_mode must be one of ['sum', 'mean', 'concat']")
            

        embeddings = (
            words_embeddings
            + position_embeddings
            + token_type_embeddings
            + tok_struct_embeddings
        )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    

class SINTokInputEmb(nn.Module):
    def __init__(self, config,args):
        super(SINTokInputEmb, self).__init__()
        
        self.args=args
        
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.position_embeddings = SinPositionalEncoding(config.hidden_size, max_len=args.max_pos)
        
        if(self.args.tok_se_comb_mode == 'concat'):
            self.histruct_position_embeddings = SinPositionalEncoding(int(config.hidden_size/3),max_len=args.max_pos)
        else:
            self.histruct_position_embeddings = None
        
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids,
        tok_struct_vec,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
    ):
        

            
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
 
        batch_size = input_ids.size(0)
        n = input_ids.size(1)
        pe = self.position_embeddings.pe[:, :n]      
        position_embeddings = pe.expand(batch_size,-1,-1)
        
        
       
        if (self.histruct_position_embeddings != None):
           
            hs_pe = self.histruct_position_embeddings.pe[:, :n]   
            hs_position_embeddings = hs_pe.expand(batch_size,-1,-1)
        else:
            hs_pe,hs_position_embeddings = None, None
              
        para_pos = tok_struct_vec[:,:,0]
        sent_pos = tok_struct_vec[:,:,1]
        tok_pos = tok_struct_vec[:,:,2]
        
        
        
                    
        if(self.args.tok_se_comb_mode == 'concat'):
            para_position_embeddings = compute_se(para_pos, hs_pe, hs_position_embeddings)
            sent_position_embeddings = compute_se(sent_pos, hs_pe, hs_position_embeddings)
            tok_position_embeddings = compute_se(tok_pos, hs_pe, hs_position_embeddings)
        else:
            para_position_embeddings = compute_se(para_pos, pe, position_embeddings)
            sent_position_embeddings = compute_se(sent_pos, pe, position_embeddings)
            tok_position_embeddings = compute_se(tok_pos, pe, position_embeddings)

        
        if(self.args.tok_se_comb_mode == 'sum'):
            tok_struct_embeddings = para_position_embeddings+sent_position_embeddings+tok_position_embeddings
            
        elif(self.args.tok_se_comb_mode == 'mean'):
            tok_struct_embeddings = (para_position_embeddings+sent_position_embeddings+tok_position_embeddings)/3
            
        elif(self.args.tok_se_comb_mode == 'concat'):
            tok_struct_embeddings = torch.cat((para_position_embeddings,sent_position_embeddings,tok_position_embeddings),2)
            
        else:
            raise ValueError ("args.tok_se_comb_mode must be one of ['sum', 'mean', 'concat']")

        embeddings = (
            words_embeddings
            + position_embeddings
            + token_type_embeddings
            + tok_struct_embeddings
        )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings   

    



class LPTokInputEmb(nn.Module):
    def __init__(self, config,args):
        super(LPTokInputEmb, self).__init__()
        
        self.args=args
        
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        
        
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids,
        tok_struct_vec,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
    ):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        

              
        para_pos = tok_struct_vec[:,:,0]
        sent_pos = tok_struct_vec[:,:,1]
        tok_pos = tok_struct_vec[:,:,2]
        
        para_position_embeddings = self.position_embeddings(para_pos)
        sent_position_embeddings = self.position_embeddings(sent_pos)
        tok_position_embeddings = self.position_embeddings(tok_pos)

        if(self.args.tok_se_comb_mode == 'sum'):
            tok_struct_embeddings = para_position_embeddings+sent_position_embeddings+tok_position_embeddings
            
        elif(self.args.tok_se_comb_mode == 'mean'):
            tok_struct_embeddings = (para_position_embeddings+sent_position_embeddings+tok_position_embeddings)/3
            
        elif(self.args.tok_se_comb_mode == 'concat'):
            raise ValueError("Concat mode is impossible when we only learn one positional embedding, since the dimension is fixed")
        else:
            raise ValueError ("args.tok_se_comb_mode must be one of ['sum', 'mean']")

        embeddings = (
            words_embeddings
            + position_embeddings
            + token_type_embeddings
            + tok_struct_embeddings
        )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    

    

