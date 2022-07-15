import torch.nn as nn
import torch
import math

class LSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz, num_layers = 1, bidirectional = False, dropout = 0.0):

        super().__init__()

        ##unused:
        self.dropout = dropout
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        ##used:
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.weight_ih_l0 = nn.Parameter(torch.Tensor(hidden_sz * 4, input_sz))
        self.weight_hh_l0 = nn.Parameter(torch.Tensor(hidden_sz * 4, hidden_sz))
        self.bias_ih_l0 = nn.Parameter(torch.Tensor(hidden_sz * 4))
        self.bias_hh_l0 = nn.Parameter(torch.Tensor(hidden_sz * 4))


        self.init_weights()
                
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
         
    def forward(self, x, 
                init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        
        x = x.permute(1,0,2)
        ##seq_len = x[1]
        ##x = x[0]

        self.guess_i = []
        self.guess_h = []

        bs, seq_sz, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device), 
                        torch.zeros(bs, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states
         
        HS = self.hidden_size
        for t in range(seq_sz):
            x_t = x[:, t, :]
            # batch the computations into a single matrix multiplication

            input_mul = x_t @ self.weight_ih_l0.permute(1,0) +  self.bias_ih_l0
            hidden_mul = h_t @ self.weight_hh_l0.permute(1,0) + self.bias_hh_l0
            
            def hook_fn_i(grad):
                self.guess_i.append(grad.clone())
                return None

            def hook_fn_h(grad):
                self.guess_h.append(grad.clone())
                return None
            ##hook_fn = lambda grad: self.guess.append(grad)
            ##hook_fn = lambda grad: breakpoint()

            input_mul = torch.tensor(input_mul, requires_grad = True)         
            ##hidden_mul = torch.tensor(input_mul, requires_grad = True)           
  
            input_mul.register_hook(hook_fn_i)
            ##hidden_mul.register_hook(hook_fn_h)

            gates = input_mul + hidden_mul
            
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]), # input
                torch.sigmoid(gates[:, HS:HS*2]), # forget
                torch.tanh(gates[:, HS*2:HS*3]),
                torch.sigmoid(gates[:, HS*3:]), # output
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)

    def pop_guess(self):
        tmp = self.guess_i
        self.guess_i = []
        return tmp

    def permute_hidden(self, hx, permutation):
        assert(permutation is None)
        pass