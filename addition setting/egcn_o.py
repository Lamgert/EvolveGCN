from numpy.lib.function_base import append
from torch.autograd import grad
import utils as u
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import math
from torch.nn import functional as F

class EGCN(torch.nn.Module):
    def __init__(self, args, activation, device='cpu', skipfeats=False ):
        super().__init__()
        GRCU_args = u.Namespace({})
        # GRCU_args = u.Namespace({'head' : 4,
        #                          'windows' : 3})
       
        feats = [args.feats_per_node,
                 args.layer_1_feats,
                 args.layer_2_feats]

        
        print(args)
        # windowsize = args.windowsize

        self.device = device
        self.skipfeats = skipfeats
        self.GRCU_layers = []
        self._parameters = nn.ParameterList()
        for i in range(1,len(feats)):
            GRCU_args = u.Namespace({'in_feats' : feats[i-1],
                                     'out_feats': feats[i],
                                     'activation': activation,
                                     'head': 4,
                                     'device': self.device})

            grcu_i = GRCU(GRCU_args)
            #print (i,'grcu_i', grcu_i)
            self.GRCU_layers.append(grcu_i.to(self.device))
            self._parameters.extend(list(self.GRCU_layers[-1].parameters()))

    def parameters(self):
        return self._parameters

    def forward(self,A_list, Nodes_list,nodes_mask_list):
        node_feats= Nodes_list[-1]

        for unit in self.GRCU_layers:
            Nodes_list = unit(A_list,Nodes_list)#,nodes_mask_list)

        out = Nodes_list[-1]
        if self.skipfeats:
            out = torch.cat((out,node_feats), dim=1)   # use node_feats.to_dense() if 2hot encoded input 
        return out


class GRCU(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        cell_args = u.Namespace({})
        cell_args.rows = args.in_feats
        cell_args.cols = args.out_feats
        # self.dropout = nn.Dropout(dropout)
        self.device = args.device
        self.aggr_weight = self_Atten(args)
        
        self.evolve_weights = mat_GRU_cell(cell_args)

        # self.V = Parameter(torch.randn(self.args.in_feats, self.args.in_feats, requires_grad=True))

        self.activation = self.args.activation
        self.GCN_init_weights = Parameter(torch.randn(self.args.in_feats,self.args.out_feats))
        # self.GCN_init_weights = torch.randn(self.args.in_feats,self.args.out_feats)
        nn.init.orthogonal_(self.GCN_init_weights)
        # self.reset_param(self.GCN_init_weights)

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)

    def forward(self,A_list,node_embs_list):#,mask_list):
        GCN_weights = self.GCN_init_weights
        out_seq = []
        GCN_weights_list = []
        time_step = len(A_list)
        GCN_weights_list.append(GCN_weights)

        for t,Ahat in enumerate(A_list):
            node_embs = node_embs_list[t]
            #first evolve the weights from the initial and use the new weights with the node_embs
            GCN_weights = self.aggr_weight(GCN_weights_list, time_step)

            GCN_weights = self.evolve_weights(GCN_weights, GCN_weights_list[-1] )#,node_embs,mask_list[t])
            # GCN_weights = self.V.matmul(GCN_weights)
            GCN_weights_list.append(GCN_weights)

            node_embs = self.activation(Ahat.matmul(node_embs.matmul(GCN_weights)))

            # GCN_weights = self.aggr_weight(GCN_weights_list)
            # print(GCN_weights)
            # print(len(GCN_weights_list))
            print('time_step:', time_step)
            out_seq.append(node_embs)

        return out_seq

class mat_GRU_cell(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.update = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Sigmoid())

        self.reset = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Sigmoid())

        self.htilda = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Tanh())
        
        # self.choose_topk = TopK(feats = args.rows,
        #                         k = args.cols)

    def forward(self,atten, prev_Q):#,prev_Z,mask):
        # z_topk = self.choose_topk(prev_Z,mask)
        # z_topk = prev_Q

        update = self.update(atten,prev_Q)
        reset = self.reset(atten,prev_Q)

        h_cap = reset * prev_Q
        h_cap = self.htilda(atten, h_cap)

        new_Q = (1 - update) * prev_Q + update * h_cap

        return new_Q

        
# class mat_LSTM_cell(torch.nn.Module):
#     def __init__(self,args):
#         super().__init__()
#         self.args = args
#         self.forget = mat_GRU_gate(args.rows,
#                                    args.cols,
#                                    torch.nn.Sigmoid())

#         self.input = mat_GRU_gate(args.rows,
#                                    args.cols,
#                                    torch.nn.Sigmoid())
        
#         self.output = mat_GRU_gate(args.rows,
#                                    args.cols,
#                                    torch.nn.Sigmoid())

#         self.c_cap = mat_GRU_gate(args.rows,
#                                    args.cols,
#                                    torch.nn.Tanh())

#     def forward(self,atten, prev_Q, c_hist):
        
#         forget = self.forget(atten,prev_Q)
#         input = self.input(atten,prev_Q)
#         output = self.output(atten,prev_Q)
#         c_cap = self.c_cap

#         c = (forget * c_hist) + (input * c_cap)

#         new_Q = output * torch.nn.Tanh(c)

#         return new_Q, c


class mat_GRU_gate(torch.nn.Module):
    def __init__(self,rows,cols,activation):
        super().__init__()
        self.activation = activation
        #the k here should be in_feats which is actually the rows
        self.W = Parameter(torch.Tensor(rows,rows))
        nn.init.orthogonal_(self.W)
        # self.reset_param(self.W)

        self.U = Parameter(torch.Tensor(rows,rows))
        nn.init.orthogonal_(self.U)
        # self.reset_param(self.U)

        self.bias = Parameter(torch.zeros(rows,cols))

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)

    def forward(self,x,hidden):
        out = self.activation(self.W.matmul(x) + \
                              self.U.matmul(hidden) + \
                              self.bias)

        return out

# class TopK(torch.nn.Module):
#     def __init__(self,feats,k):
#         super().__init__()
#         self.scorer = Parameter(torch.randn(feats,1))
#         # self.reset_param(self.scorer)
        
#         self.k = k

#     def reset_param(self,t):
#         #Initialize based on the number of rows
#         stdv = 1. / math.sqrt(t.size(0))
#         t.data.uniform_(-stdv,stdv)

#     def forward(self,node_embs,mask):
#         scores = node_embs.matmul(self.scorer) / self.scorer.norm()
#         scores = scores + mask

#         vals, topk_indices = scores.view(-1).topk(self.k)
#         topk_indices = topk_indices[vals > -float("Inf")]

#         if topk_indices.size(0) < self.k:
#             topk_indices = u.pad_with_last_val(topk_indices,self.k)
            
#         tanh = torch.nn.Tanh()

#         if isinstance(node_embs, torch.sparse.FloatTensor) or \
#            isinstance(node_embs, torch.cuda.sparse.FloatTensor):
#             node_embs = node_embs.to_dense()

#         out = node_embs[topk_indices] * tanh(scores[topk_indices].view(-1,1))

#         #we need to transpose the output
#         return out.t()

class self_Atten(torch.nn.Module):
    def __init__(self, args, eps=1e-6, dropout=0):
        super().__init__()
        self.args = args
        self.head = args.head
        # self.windows = args.windowsize
        self.key_list = []
        self.query_list = []
        self.value_list = []
        self.device = args.device
        self.eps = eps
        
        print('head=', self.head)
        print('device=', self.device)

        self.w_1 = nn.Linear(args.out_feats, int(args.out_feats * 2))
        self.w_2 = nn.Linear(int(args.out_feats * 2), args.out_feats)

        self.dropout = nn.Dropout(dropout)

        self.a1 = nn.Parameter(torch.ones(args.in_feats, args.out_feats, requires_grad=True)).to(self.device)
        self.b1 = nn.Parameter(torch.zeros(args.in_feats, args.out_feats, requires_grad=True)).to(self.device)

        # a1 = nn.Parameter(torch.ones(args.in_feats, args.out_feats, requires_grad=True)).to('cuda:3')
        # b1 = nn.Parameter(torch.zeros(args.in_feats, args.out_feats, requires_grad=True)).to('cuda:3')

        for i in range(self.head):
            key = Parameter(torch.randn(self.args.in_feats,self.args.in_feats,requires_grad=True)).to(self.device)
            # self.reset_param(key)
            nn.init.orthogonal_(key)

            query = Parameter(torch.randn(self.args.in_feats,self.args.out_feats,requires_grad=True)).to(self.device)
            # self.reset_param(query)
            nn.init.orthogonal_(query)

            value = Parameter(torch.randn(self.args.in_feats,self.args.in_feats,requires_grad=True)).to(self.device)
            # self.reset_param(value)
            nn.init.orthogonal_(value)

            self.key_list.append(key)
            self.query_list.append(query)
            self.value_list.append(value)

        if self.head != 1:
            self.W_O = Parameter(torch.randn(self.args.out_feats * self.head, self.args.out_feats,requires_grad=True)).to(self.device)
            # self.reset_param(self.W_O)
            nn.init.orthogonal_(self.W_O)

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)  

    def PositionalEncoding(self, GCN_weights_list):

        GCN_weights_shape = list(GCN_weights_list[0].size())
        GCN_weighes_dim = GCN_weights_shape[0] * GCN_weights_shape[1]

        # div_term = torch.exp(torch.arange(0, GCN_weighes_dim, 2) * (-(math.log(10000.0) / GCN_weighes_dim)))
        for pos in range(len(GCN_weights_list)):
            pos_encode = torch.zeros(1, GCN_weighes_dim)
            # pos_encode[:, 0::2] = torch.sin(i * div_term)
            # pos_encode[:, 1::2] = torch.cos(i * div_term)
            for i in range(0, GCN_weighes_dim):
                 if i % 2 == 0:
                     p = torch.sin(pos * math.exe(-(i / GCN_weighes_dim) * math.log(10000.0)))
                 else:
                     p = torch.cos(pos * math.exe(-((i-1) / GCN_weighes_dim) * math.log(10000.0)))
                 
            pos_encode.view(GCN_weights_shape[0], GCN_weights_shape[1])
            
            print(GCN_weights_shape[0] )
            print(GCN_weights_shape[1])

            print(pos_encode.shape)
            print(GCN_weights_list[i].shape)
            GCN_weights_list[i] = GCN_weights_list[i] + pos_encode

        return GCN_weights_list

    def cal_attention(self, query, key, value, GCN_weights_list):
        
        query_mat_key_list = []
        value_list = []
        
        for i in range(len(GCN_weights_list)):
            # query_mat_key_list.append((query.matmul(GCN_weights_list[-1])).mul((key.matmul(GCN_weights_list[i]))))
            # target_query = query.matmul(GCN_weights_list[-1])
            target_query = query
            target_key = key.matmul(GCN_weights_list[i])

            # print(target_query.shape)
            # print(target_key.shape)

            query_mat_key_list.append(torch.mul(target_query, target_key))
            value_list.append((value.matmul(GCN_weights_list[i])))

            query_mat_key_list[i] = torch.sum(query_mat_key_list[i]) / (self.args.in_feats * self.args.out_feats)
            # print(query_mat_key_list[i])
        
        query_mat_key_list = torch.tensor(query_mat_key_list)

        p = F.softmax(query_mat_key_list, dim=0)
        aggr_value = p[0] * value_list[0]

        for j in range(1, len(GCN_weights_list)):
            aggr_value = aggr_value + (p[j] * value_list[j])

        return aggr_value

    def mult_head(self, key_list, query_list, value_list, GCN_weights_list):
        heads = self.head
        cat_value = self.cal_attention(query_list[0], key_list[0], value_list[0], GCN_weights_list)

        for i in range(1, heads):
            
            cat_value = torch.cat((cat_value, self.cal_attention(query_list[i], key_list[i], value_list[i], GCN_weights_list)), 1)
            
        if heads == 1:
            return cat_value + self.dropout(self.LayerNorm(cat_value, self.a1, self.b1))
        else:
            return cat_value.matmul(self.W_O) + self.dropout(self.LayerNorm(cat_value.matmul(self.W_O), self.a1, self.b1))


    def LayerNorm(self, final_weight, a, b):
        row = list(final_weight.size())[0]
        col = list(final_weight.size())[1]
        mean = torch.sum(final_weight) / (row * col)
        mean_matrix = torch.ones(row, col).to(self.device) * mean
        
        std = math.sqrt(torch.sum((final_weight - mean_matrix) * (final_weight - mean_matrix))) / (row * col)

        # a_2 = nn.Parameter(torch.ones(row, col, torch.zeros(row, col, requires_grad=True)).to('cuda:3')
        # b_2 = nn.Parameter(torch.zeros(row, col,requires_grad=True)).to('cuda:3')

        return a * (final_weight - mean_matrix) / (std + self.eps) + b
        # mean = final_weight.mean(-1, keepdim=True)
        # std = final_weight.std(-1, keepdim=True)
        # return a * (final_weight - mean) / (std + self.eps) + b


    def forward(self, GCN_weights_list, time_step):
        
        # math.ceil(self.windows * time_step)
        if len(GCN_weights_list) <= 10:
            new_GCN_weights_list = GCN_weights_list
        else:
            new_GCN_weights_list = GCN_weights_list[(len(GCN_weights_list) - 10) : len(GCN_weights_list)]
        
        # new_GCN_weights_list = GCN_weights_list
        
        # new_GCN_weights_list = self.PositionalEncoding(new_GCN_weights_list)
        after_mult_head = self.mult_head(self.key_list, self.query_list, self.value_list, new_GCN_weights_list)
        final_weight = after_mult_head 
        
        return final_weight

        # return self.w_2(self.dropout(F.relu(self.w_1(final_weight))))
        # return self.w_2(self.dropout(F.relu(self.w_1(final_weight))))
