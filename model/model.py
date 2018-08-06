import sys
sys.path.insert(0, '../')

from base import BaseModel


import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch
import numpy as np
from torch.autograd import Variable

if torch.cuda.is_available():
    USE_CUDA=True
else : 
    USE_CUDA = False


class BLMPM_model(BaseModel):
    def __init__(self, config):
        super(BLMPM_model, self).__init__(config)
        # configuration parameters
        self.config = config
        self.embeddings_dim = self.config['embeddings_dim']
        self.hidden_size = self.config['hidden_size']
        self.num_layers= self.config['num_layers']
        self.num_perspectives = self.config['num_perspectives']

        # weight matrices
        self.W_perspectives_fwd = nn.Parameter(torch.zeros([self.num_perspectives, self.hidden_size]).normal_(0,1))
        self.W_perspectives_bwd = nn.Parameter(torch.zeros([self.num_perspectives, self.hidden_size]).normal_(0,1))

        self.context_gru = nn.GRU(self.embeddings_dim, self.hidden_size, num_layers=self.num_layers, 
        batch_first=True, bidirectional=True)
        self.aggregate_gru = nn.GRU(self.num_perspectives*2, self.hidden_size, num_layers=self.num_layers,
        bidirectional=True, batch_first=True)
        self.fc = nn.Linear(4* self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        # initializing weights
        
        for param in self.context_gru.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

   

    def forward(self, p_sents, q_sents, word_embeddings, lengths=None):
        """
        x : (BxPxd, BxQ,d) 
        lengths : (B, B)
        """

        #print (self.W_perspectives_fwd)
        p_seqs, q_seqs = self.get_pretrained_embeddings(p_sents, q_sents, word_embeddings)
        
        batch_size = p_seqs.size(0)
        p_seq_len = p_seqs.size(1)
        q_seq_len = q_seqs.size(1)

        p_h_outputs, _ = self.context_gru(p_seqs)
        q_h_outputs, _ = self.context_gru(q_seqs)
        # extracting forward and backward states of P 

        p_h_outputs = p_h_outputs.view(-1, p_seq_len, 2, self.hidden_size)
        p_h_forward_outputs = p_h_outputs[:,:,0,:]
        p_h_backward_outputs = p_h_outputs[:,:,1,:]
        
        # extracting forward and backward states of Q 
        q_h_outputs = q_h_outputs.view(batch_size, q_seq_len, 2, self.hidden_size)
        q_h_forward_outputs = q_h_outputs[:,:,0,:]
        q_h_backward_outputs = q_h_outputs[:,:,1,:]

        # computing full matching vectors
        p_q_fullm, q_p_fullm = self.full_matching(p_h_forward_outputs,p_h_backward_outputs, q_h_forward_outputs, 
        q_h_backward_outputs)

        # aggregating matching sequences into 4 vectors
        p_q_aggregate_outputs ,_ = self.aggregate_gru(p_q_fullm)
        q_p_aggregate_outputs, _ = self.aggregate_gru(q_p_fullm)

        p_q_aggregate_outputs = p_q_aggregate_outputs.view(-1,p_seq_len,2,self.hidden_size)
        q_p_aggregate_outputs = q_p_aggregate_outputs.view(-1,q_seq_len,2,self.hidden_size)
        
        v1 = p_q_aggregate_outputs[:,-1,0,:]
        v2 = p_q_aggregate_outputs[:,-1,1,:]
        v3 = q_p_aggregate_outputs[:,-1,0,:]
        v4 = q_p_aggregate_outputs[:,-1,1,:]

        fixed_length_aggregation = torch.cat([v1,v2,v3,v4], -1) # B x 4H
        output = self.sigmoid(self.fc(fixed_length_aggregation))
        
        return output


    def get_pretrained_embeddings(self, p_sents, q_sents, word_embeddings):
        # get pre-trained word embeddings
        p_embeddings= np.zeros([len(p_sents), len(p_sents[0]), word_embeddings.shape[-1]])
        q_embeddings= np.zeros([len(q_sents), len(q_sents[0]), word_embeddings.shape[-1]])

        for i, seq in enumerate(p_sents):
            for j, idd in enumerate(seq):
                p_embeddings[i,j,:]= word_embeddings[idd]
        
        for i, seq in enumerate(q_sents):
            for j, idd in enumerate(seq):
                q_embeddings[i,j,:]= word_embeddings[idd]
        
        p_seqs, q_seqs = torch.Tensor(p_embeddings), torch.Tensor(q_embeddings)
       
        
        return p_seqs.cuda(), q_seqs.cuda()

    def multiperspective_matching (self, V1, V2, W):
        """
        takes V1: (B, d)
        V2 : (B, d)
        W : (l, d)
        returns mutliperspetive matching tensor of size B x l
        """
        V1.cuda()
        V2.cuda()
        W.cuda()

        d = V1.size(1)
        batch_size = V1.size(0)
        l = W.size(0)

        W = W.repeat(batch_size, 1) # repeat W batch size times -> (B*l, d)
        V1 = V1.repeat(1, l).view(-1, d) # repeat V1 -> (B*l, d)
        V2 = V2.repeat(1, l).view(-1, d) # repeat V2 -> (B*l, d)

        W_V1 = W * V1
        W_V2 = W * V2
        cos = nn.CosineSimilarity(dim=1)
        sim = cos(W_V1, W_V2).view(-1, l)
        
        return sim

    def full_matching_(self, a_h, b_h, fwd=True):
        """
        computes full matching sequence of vectors by matching each step of a against the last step of b
        
        returns a_b_fwd (B, A_MAX_LEN, L)
        """
        a_max_len = a_h.size(1)
        h = a_h.size(2)

        b_last = b_h[:,-1,:] # B x h

        matching_vectors = []
        for step in range(a_max_len):
            a_step = a_h[:,step,:] # B x h
            if fwd:
                m_i = self.multiperspective_matching(a_step, b_last, self.W_perspectives_fwd)
            else : 
                m_i = self.multiperspective_matching(a_step, b_last, self.W_perspectives_bwd) # BxL
            matching_vectors.append(m_i)
        
        matching_vectors = torch.cat(matching_vectors, 1).view(-1, a_max_len, self.num_perspectives)
        return matching_vectors

    def full_matching(self, p_h_fwd, p_h_bwd, q_h_fwd, q_h_bwd):
        
        p_match_q_fwd = self.full_matching_(p_h_fwd, q_h_fwd)
        p_match_q_bwd = self.full_matching_(p_h_bwd, q_h_bwd, fwd=False)
        q_match_p_fwd = self.full_matching_(q_h_fwd, p_h_fwd)
        q_match_p_bwd = self.full_matching_(q_h_bwd, p_h_bwd, fwd=False)

        p_match_q = torch.cat([p_match_q_fwd, p_match_q_bwd], 2) 
        q_match_p = torch.cat([q_match_p_fwd, q_match_p_bwd], 2) # B x MAX_Q x 2L

        return p_match_q, q_match_p


if __name__=='__main__':
    m =BLMPM_model(config={'embedding_dim':100, 'hidden_size':50, 'num_layers':1})

    