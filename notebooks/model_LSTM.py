from socket import gaierror
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence

from torch.distributions.multivariate_normal import MultivariateNormal

from torch.nn import RNN
from rnn import LSTM

def normalize(x):
    eps = 1E-8
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def split_by_4(x):
    return torch.split(x, x.shape[0] // 4)


def apply_fwd_grad_no_batch(dFg, vw):
    return dFg * vw

def apply_fwd_grad_batch(dFg, vw):
    if len(vw.shape) == 3:
        return torch.matmul(dFg, vw.view(vw.shape[0], -1)).view(vw.shape[1], vw.shape[2])
    elif len(vw.shape) == 2:
        return torch.matmul(dFg, vw.view(vw.shape[0], -1)).squeeze()
    else:
        return torch.matmul(dFg, vw)


def apply_fwd_grad_reduce_batch(dFg, vw):
    return dFg.sum() * (vw.mean(dim=0) if vw.ndim == 3 else vw)


def apply_fwd_grad_reduce_batch_w_biases(dFg, vw):
    return dFg.sum() * vw.mean(dim=0)


def random(weight, binary, batch_size):
    if batch_size == None:
        get_shape = lambda w: (w.shape[0], 1)
    else:
        get_shape = lambda w: (batch_size, w.shape[0], 1)
    if binary:
        return torch.randint(0, 2, get_shape(weight), dtype=torch.float32, device=weight.device) * 2 - 1
    else: # gaussian
        return torch.randn(get_shape(weight), device=weight.device)


def create_new_Vs(rnn, j, device, epsilon):
    W_ii, W_if, W_ig, W_io = split_by_4(rnn.__getattr__(f"weight_ih_l{j}"))
    W_hi, W_hf, W_hg, W_ho = split_by_4(rnn.__getattr__(f"weight_hh_l{j}"))
    b_ii, b_if, b_ig, b_io = split_by_4(rnn.__getattr__(f"bias_ih_l{j}"))
    b_hi, b_hf, b_hg, b_ho = split_by_4(rnn.__getattr__(f"bias_hh_l{j}"))

    _vw_ii = torch.randn(W_ii.shape, device=device) * epsilon
    _vw_if = torch.randn(W_if.shape, device=device) * epsilon
    _vw_ig = torch.randn(W_ig.shape, device=device) * epsilon
    _vw_io = torch.randn(W_io.shape, device=device) * epsilon
    _vw_hi = torch.randn(W_hi.shape, device=device) * epsilon
    _vw_hf = torch.randn(W_hf.shape, device=device) * epsilon
    _vw_hg = torch.randn(W_hg.shape, device=device) * epsilon
    _vw_ho = torch.randn(W_ho.shape, device=device) * epsilon
    _vb_ii = torch.randn(b_ii.shape, device=device) * epsilon
    _vb_if = torch.randn(b_if.shape, device=device) * epsilon
    _vb_ig = torch.randn(b_ig.shape, device=device) * epsilon
    _vb_io = torch.randn(b_io.shape, device=device) * epsilon
    _vb_hi = torch.randn(b_hi.shape, device=device) * epsilon
    _vb_hf = torch.randn(b_hf.shape, device=device) * epsilon
    _vb_hg = torch.randn(b_hg.shape, device=device) * epsilon
    _vb_ho = torch.randn(b_ho.shape, device=device) * epsilon

    _vw_i = (_vw_ii, _vw_if, _vw_ig, _vw_io)
    _vw_h = (_vw_hi, _vw_hf, _vw_hg, _vw_ho)
    _vb_i = (_vb_ii, _vb_if, _vb_ig, _vb_io)
    _vb_h = (_vb_hi, _vb_hf, _vb_hg, _vb_ho)

    return _vw_i, _vw_h, _vb_i, _vb_h

def create_new_Vs_binary(rnn, j, device, epsilon):
    W_ii, W_if, W_ig, W_io = split_by_4(rnn.__getattr__(f"weight_ih_l{j}"))
    W_hi, W_hf, W_hg, W_ho = split_by_4(rnn.__getattr__(f"weight_hh_l{j}"))
    b_ii, b_if, b_ig, b_io = split_by_4(rnn.__getattr__(f"bias_ih_l{j}"))
    b_hi, b_hf, b_hg, b_ho = split_by_4(rnn.__getattr__(f"bias_hh_l{j}"))

    _vw_ii = torch.randint(0, 2, W_ii.shape, dtype=torch.float32, device=device) * 2 - 1
    _vw_if = torch.randint(0, 2, W_if.shape, dtype=torch.float32, device=device) * 2 - 1
    _vw_ig = torch.randint(0, 2, W_ig.shape, dtype=torch.float32, device=device) * 2 - 1
    _vw_io = torch.randint(0, 2, W_io.shape, dtype=torch.float32, device=device) * 2 - 1
    _vw_hi = torch.randint(0, 2, W_hi.shape, dtype=torch.float32, device=device) * 2 - 1
    _vw_hf = torch.randint(0, 2, W_hf.shape, dtype=torch.float32, device=device) * 2 - 1
    _vw_hg = torch.randint(0, 2, W_hg.shape, dtype=torch.float32, device=device) * 2 - 1
    _vw_ho = torch.randint(0, 2, W_ho.shape, dtype=torch.float32, device=device) * 2 - 1
    _vb_ii = torch.randint(0, 2, b_ii.shape, dtype=torch.float32, device=device) * 2 - 1
    _vb_if = torch.randint(0, 2, b_if.shape, dtype=torch.float32, device=device) * 2 - 1
    _vb_ig = torch.randint(0, 2, b_ig.shape, dtype=torch.float32, device=device) * 2 - 1
    _vb_io = torch.randint(0, 2, b_io.shape, dtype=torch.float32, device=device) * 2 - 1
    _vb_hi = torch.randint(0, 2, b_hi.shape, dtype=torch.float32, device=device) * 2 - 1
    _vb_hf = torch.randint(0, 2, b_hf.shape, dtype=torch.float32, device=device) * 2 - 1
    _vb_hg = torch.randint(0, 2, b_hg.shape, dtype=torch.float32, device=device) * 2 - 1
    _vb_ho = torch.randint(0, 2, b_ho.shape, dtype=torch.float32, device=device) * 2 - 1

    _vw_i = (_vw_ii, _vw_if, _vw_ig, _vw_io)
    _vw_h = (_vw_hi, _vw_hf, _vw_hg, _vw_ho)
    _vb_i = (_vb_ii, _vb_if, _vb_ig, _vb_io)
    _vb_h = (_vb_hi, _vb_hf, _vb_hg, _vb_ho)

    return _vw_i, _vw_h, _vb_i, _vb_h

def create_new_Vs_mage_guess(x_t, h_part, rnn, guess, alpha, t, j, device, with_batch=False):
    W_ii, W_if, W_ig, W_io = split_by_4(rnn.__getattr__(f"weight_ih_l{j}"))
    W_hi, W_hf, W_hg, W_ho = split_by_4(rnn.__getattr__(f"weight_hh_l{j}"))
    b_ii, b_if, b_ig, b_io = split_by_4(rnn.__getattr__(f"bias_ih_l{j}"))
    b_hi, b_hf, b_hg, b_ho = split_by_4(rnn.__getattr__(f"bias_hh_l{j}"))

    get_shape = lambda w: (w.shape[0], 1) if not with_batch else (x_t.shape[0], w.shape[0], 1)
    get_shape_bias = lambda b: (x_t.shape[0], b.shape[0]) if with_batch else b.shape



    z = guess[t]
    I = torch.eye(z.shape[1], device = device)
    Mu = torch.zeros([z.shape[1]],device = device)

    ##breakpoint()
    g = torch.cat([MultivariateNormal(Mu, I + alpha* torch.matmul(z[i:i+1,:].t(),z[i:i+1,:])).sample().unsqueeze(0) for i in range(z.shape[0]) ],dim = 0).unsqueeze(2)



    H = g.shape[1]//4
    g0 = g[:,:H,:]
    g1 = g[:,H:2*H,:]
    g2 = g[:,2*H:3*H,:]
    g3 = g[:,3*H:,:]

    norm = torch.sqrt((x_t**2).sum(dim = 1, keepdim = True) + (h_part**2).sum(dim = 1, keepdim = True) + 1 + 1)

    pw_ii = g0
    pw_if = g1
    pw_ig = g2
    pw_io = g3
    pw_hi = g0
    pw_hf = g1
    pw_hg = g2
    pw_ho = g3

    _vw_ii = torch.matmul(pw_ii, (x_t/norm).unsqueeze(1))
    _vw_hi = torch.matmul(pw_hi, (h_part/norm).unsqueeze(1))
    _vw_if = torch.matmul(pw_if, (x_t/norm).unsqueeze(1))
    _vw_hf = torch.matmul(pw_hf, (h_part/norm).unsqueeze(1))
    _vw_ig = torch.matmul(pw_ig, (x_t/norm).unsqueeze(1))
    _vw_hg = torch.matmul(pw_hg, (h_part/norm).unsqueeze(1))
    _vw_io = torch.matmul(pw_io, (x_t/norm).unsqueeze(1))
    _vw_ho = torch.matmul(pw_ho, (h_part/norm).unsqueeze(1))

    _vb_ii = torch.matmul(g0, (1.0/norm).unsqueeze(1)).squeeze(2)
    _vb_hi = torch.matmul(g1, (1.0/norm).unsqueeze(1)).squeeze(2)
    _vb_if = torch.matmul(g2, (1.0/norm).unsqueeze(1)).squeeze(2)
    _vb_hf = torch.matmul(g3, (1.0/norm).unsqueeze(1)).squeeze(2)
    _vb_ig = _vb_ii
    _vb_hg = _vb_hi
    _vb_io = _vb_if
    _vb_ho = _vb_hf


    _vw_i = (_vw_ii, _vw_if, _vw_ig, _vw_io)
    _vw_h = (_vw_hi, _vw_hf, _vw_hg, _vw_ho)
    _vb_i = (_vb_ii, _vb_if, _vb_ig, _vb_io)
    _vb_h = (_vb_hi, _vb_hf, _vb_hg, _vb_ho)

    return _vw_i, _vw_h, _vb_i, _vb_h

def create_new_Vs_mage_random_t_separately(x_t, h_part, gs, device, binary=False):
    def rand():
        # keep the overall distribution gaussian
        return torch.randint(0, 2, (1,), dtype=torch.float32, device=device) * 2 - 1 if not binary \
            else torch.randn((1,), device=device)

    g0, g1, g2, g3 = [g * rand() for g in gs]

    norm = torch.sqrt((x_t**2).sum(dim = 1, keepdim = True) + (h_part**2).sum(dim = 1, keepdim = True) + 1 + 1)

    pw_ii = g0
    pw_if = g1
    pw_ig = g2
    pw_io = g3
    pw_hi = g0
    pw_hf = g1
    pw_hg = g2
    pw_ho = g3

    _vw_ii = torch.matmul(pw_ii, (x_t/norm).unsqueeze(1))
    _vw_hi = torch.matmul(pw_hi, (h_part/norm).unsqueeze(1))
    _vw_if = torch.matmul(pw_if, (x_t/norm).unsqueeze(1))
    _vw_hf = torch.matmul(pw_hf, (h_part/norm).unsqueeze(1))
    _vw_ig = torch.matmul(pw_ig, (x_t/norm).unsqueeze(1))
    _vw_hg = torch.matmul(pw_hg, (h_part/norm).unsqueeze(1))
    _vw_io = torch.matmul(pw_io, (x_t/norm).unsqueeze(1))
    _vw_ho = torch.matmul(pw_ho, (h_part/norm).unsqueeze(1))

    _vb_ii = torch.matmul(g0, (1.0/norm).unsqueeze(1)).squeeze(2)
    _vb_hi = torch.matmul(g1, (1.0/norm).unsqueeze(1)).squeeze(2)
    _vb_if = torch.matmul(g2, (1.0/norm).unsqueeze(1)).squeeze(2)
    _vb_hf = torch.matmul(g3, (1.0/norm).unsqueeze(1)).squeeze(2)
    _vb_ig = _vb_ii
    _vb_hg = _vb_hi
    _vb_io = _vb_if
    _vb_ho = _vb_hf

    _vw_i = (_vw_ii, _vw_if, _vw_ig, _vw_io)
    _vw_h = (_vw_hi, _vw_hf, _vw_hg, _vw_ho)
    _vb_i = (_vb_ii, _vb_if, _vb_ig, _vb_io)
    _vb_h = (_vb_hi, _vb_hf, _vb_hg, _vb_ho)

    return _vw_i, _vw_h, _vb_i, _vb_h


def create_new_Vs_mage(x_t, h_part, rnn, j, with_batch=False, binary=False):
    W_ii, W_if, W_ig, W_io = split_by_4(rnn.__getattr__(f"weight_ih_l{j}"))

    batch = None if not with_batch else x_t.shape[0]

    g0 = random(W_ii, binary, batch)
    g1 = random(W_if, binary, batch)
    g2 = random(W_ig, binary, batch)
    g3 = random(W_io, binary, batch)

    norm = torch.sqrt((x_t**2).sum(dim = 1, keepdim = True) + (h_part**2).sum(dim = 1, keepdim = True) + 1 + 1)

    pw_ii = g0
    pw_if = g1
    pw_ig = g2
    pw_io = g3
    pw_hi = g0
    pw_hf = g1
    pw_hg = g2
    pw_ho = g3

    _vw_ii = torch.matmul(pw_ii, (x_t/norm).unsqueeze(1))
    _vw_hi = torch.matmul(pw_hi, (h_part/norm).unsqueeze(1))
    _vw_if = torch.matmul(pw_if, (x_t/norm).unsqueeze(1))
    _vw_hf = torch.matmul(pw_hf, (h_part/norm).unsqueeze(1))
    _vw_ig = torch.matmul(pw_ig, (x_t/norm).unsqueeze(1))
    _vw_hg = torch.matmul(pw_hg, (h_part/norm).unsqueeze(1))
    _vw_io = torch.matmul(pw_io, (x_t/norm).unsqueeze(1))
    _vw_ho = torch.matmul(pw_ho, (h_part/norm).unsqueeze(1))

    _vb_ii = torch.matmul(g0, (1.0/norm).unsqueeze(1)).squeeze(2)
    _vb_hi = torch.matmul(g1, (1.0/norm).unsqueeze(1)).squeeze(2)
    _vb_if = torch.matmul(g2, (1.0/norm).unsqueeze(1)).squeeze(2)
    _vb_hf = torch.matmul(g3, (1.0/norm).unsqueeze(1)).squeeze(2)
    _vb_ig = _vb_ii
    _vb_hg = _vb_hi
    _vb_io = _vb_if
    _vb_ho = _vb_hf

    _vw_i = (_vw_ii, _vw_if, _vw_ig, _vw_io)
    _vw_h = (_vw_hi, _vw_hf, _vw_hg, _vw_ho)
    _vb_i = (_vb_ii, _vb_if, _vb_ig, _vb_io)
    _vb_h = (_vb_hi, _vb_hf, _vb_hg, _vb_ho)

    return _vw_i, _vw_h, _vb_i, _vb_h

def create_new_Vs_mage_old(x_t, h_part, rnn, j, device, epsilon, with_batch=False, with_batch_for_biases=False):
    W_ii, W_if, W_ig, W_io = split_by_4(rnn.__getattr__(f"weight_ih_l{j}"))
    W_hi, W_hf, W_hg, W_ho = split_by_4(rnn.__getattr__(f"weight_hh_l{j}"))
    b_ii, b_if, b_ig, b_io = split_by_4(rnn.__getattr__(f"bias_ih_l{j}"))
    b_hi, b_hf, b_hg, b_ho = split_by_4(rnn.__getattr__(f"bias_hh_l{j}"))

    get_shape = lambda w: (w.shape[0], 1) if not with_batch else (x_t.shape[0], w.shape[0], 1)
    get_shape_bias = lambda b: (x_t.shape[0], b.shape[0]) if with_batch and with_batch_for_biases else b.shape

    pw_ii = torch.randn(get_shape(W_ii), device=device) * epsilon
    pw_if = torch.randn(get_shape(W_if), device=device) * epsilon
    pw_ig = torch.randn(get_shape(W_ig), device=device) * epsilon
    pw_io = torch.randn(get_shape(W_io), device=device) * epsilon
    pw_hi = torch.randn(get_shape(W_hi), device=device) * epsilon
    pw_hf = torch.randn(get_shape(W_hf), device=device) * epsilon
    pw_hg = torch.randn(get_shape(W_hg), device=device) * epsilon
    pw_ho = torch.randn(get_shape(W_ho), device=device) * epsilon
    _vb_ii = torch.randn(get_shape_bias(b_ii), device=device) * epsilon
    _vb_if = torch.randn(get_shape_bias(b_if), device=device) * epsilon
    _vb_ig = torch.randn(get_shape_bias(b_ig), device=device) * epsilon
    _vb_io = torch.randn(get_shape_bias(b_io), device=device) * epsilon
    _vb_hi = torch.randn(get_shape_bias(b_hi), device=device) * epsilon
    _vb_hf = torch.randn(get_shape_bias(b_hf), device=device) * epsilon
    _vb_hg = torch.randn(get_shape_bias(b_hg), device=device) * epsilon
    _vb_ho = torch.randn(get_shape_bias(b_ho), device=device) * epsilon


    _vw_ii = torch.matmul(pw_ii, normalize(x_t).unsqueeze(1))
    _vw_hi = torch.matmul(pw_hi, normalize(h_part).unsqueeze(1))
    _vw_if = torch.matmul(pw_if, normalize(x_t).unsqueeze(1))
    _vw_hf = torch.matmul(pw_hf, normalize(h_part).unsqueeze(1))
    _vw_ig = torch.matmul(pw_ig, normalize(x_t).unsqueeze(1))
    _vw_hg = torch.matmul(pw_hg, normalize(h_part).unsqueeze(1))
    _vw_io = torch.matmul(pw_io, normalize(x_t).unsqueeze(1))
    _vw_ho = torch.matmul(pw_ho, normalize(h_part).unsqueeze(1))


    _vw_i = (_vw_ii, _vw_if, _vw_ig, _vw_io)
    _vw_h = (_vw_hi, _vw_hf, _vw_hg, _vw_ho)
    _vb_i = (_vb_ii, _vb_if, _vb_ig, _vb_io)
    _vb_h = (_vb_hi, _vb_hf, _vb_hg, _vb_ho)

    return _vw_i, _vw_h, _vb_i, _vb_h



class RNN(nn.Module):
    def __init__(self, rnn_type, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx, train_embedding=True, save_correlations=False, trunc = -1):
        super().__init__()
        self.rnn_type = rnn_type
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.save_correlations = save_correlations
        self.input_correlation_matrix = None
        self.correlation_matrixes = [[] for _ in range(n_layers + 1)]
        self.trunc = trunc
        self.output_correlation_matrix = None


        # Cell states
        if rnn_type == 'RNN':
            self.keys_rnn = ["c"]
        elif rnn_type == 'GRU':
            self.keys_rnn = ["r", "z", "c"]  # c = n, but we need this label.
        elif rnn_type == 'LSTM':
            self.keys_rnn = ["i", "f", "c", "o"]  # c = g.
        else:
            raise NotImplementedError()
        # Number of states
        self.n_states = len(self.keys_rnn)

        # Number of directions
        n_directions = 1 + int(bidirectional)
        self.n_directions = n_directions
        # Total number of distinguishable weights
        self.n_rnn = self.n_states * n_directions

        # Define layers
        # Embedding. Set padding_idx so that the <pad> is not updated. 
        self.encoder = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        # RNN
        if rnn_type == 'RNN':
            self.rnn = RNN(embedding_dim, hidden_dim, num_layers=n_layers,
                              bidirectional=bidirectional, dropout=dropout)
        elif rnn_type == 'LSTM':
            self.rnn = LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                               bidirectional=bidirectional, dropout=dropout)





        # Decoder: fully-connected
        self.decoder = nn.Linear(hidden_dim * n_directions, output_dim)

        # Train the embedding?
        if not train_embedding:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Setup dropout
        self.drop = nn.Dropout(dropout)

    def forward(self, batch_text):

        text, text_lengths = batch_text
        # text = [sentence len, batch size]

        embedded = self.encoder(text)
        # embedded = [sent len, batch size, emb dim]
        T = self.trunc

        if (T <= 0) or (T >= min(text_lengths)):

            # Pack sequence
            ##packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
            packed_embedded = embedded
            if self.rnn_type == 'LSTM':
                _, (hidden, _) = self.rnn(packed_embedded)
            else:
                _, hidden = self.rnn(packed_embedded)
        else:
            ##print(min(text_lengths))
            # Pack sequences
            ##import pdb; pdb.set_trace();
            packed_embedded0 = nn.utils.rnn.pack_padded_sequence(embedded[:-self.trunc], text_lengths - T)
            packed_embedded1 = nn.utils.rnn.pack_padded_sequence(embedded[-self.trunc:], torch.zeros_like(text_lengths) + T)

            if self.rnn_type == 'LSTM':
                with torch.no_grad():
                    _, state0 = self.rnn(packed_embedded0)
                _, (hidden, _) = self.rnn(packed_embedded1, state0)
            else:
                with torch.no_grad():
                    _, hidden = self.rnn(packed_embedded0)
                _, hidden = self.rnn(packed_embedded1, hidden)

        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]
        # Unpack sequence
        # output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        # output = [sentence len, batch size, hid dim * num directions]

        # Concatenate the final hidden layers (for bidirectional)
        # hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        # breakpoint()
        if hidden.ndim == 2:
            hidden = hidden.unsqueeze(0)
        hidden = (torch.transpose(hidden[-self.n_directions:], 0, 1)).reshape(
            (-1, self.hidden_dim * self.n_directions))

        # Dropout
        # hidden = self.drop(hidden)

        # Decode
        decoded = self.decoder(hidden).squeeze(1)

        return decoded

    def batch_text_to_input(self, batch_text):
        text, text_lengths = batch_text
        # text = [sentence len, batch size]
        with torch.no_grad():
            embedded = self.encoder(text)
        # embedded = [sent len, batch size, emb dim]

        # Pack sequence
        # packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)

        # if isinstance(packed_embedded, PackedSequence):
        #     input, batch_sizes, sorted_indices, unsorted_indices = packed_embedded
        #     max_batch_size = batch_sizes[0]
        #     max_batch_size = int(max_batch_size)

        num_directions = 2 if self.rnn.bidirectional else 1
        zeros = torch.zeros(self.rnn.num_layers * num_directions,
                            embedded.shape[1], self.rnn.hidden_size,
                            dtype=embedded.dtype, device=embedded.device)
        hx = (zeros, zeros)


        return hx, embedded

    def fwd_mode(self, batch_text, y, loss, mage=False, grad_div=1, g_with_batch=False, reduce_batch=False,
                 random_binary=False, vanilla_V_per_timestep=False,
                 random_t_separately=False, guess=None, ig=-1):
        hx, embedding = self.batch_text_to_input(batch_text)
        x = embedding
        device = x.device
        epsilon = 1
        V = {}
        grad = 0
        h_stack = []
        c_stack = []
        relevant_Vs = [(j, seq) for j in range(self.rnn.num_layers) for seq in range(len(x))]
        # relevant_Vs = [(j, seq) for j in range(self.rnn.num_layers) for seq in range(len(x))[-3:]]
        with torch.no_grad():
            for j in range(self.rnn.num_layers):
                h, c_t_1 = hx[j] if hx[0].shape[0] > 1 else (hx[0][0], hx[1][0])
                h_list = [h]
                h_grad_list = []
                h_full_batch = torch.zeros_like(h)
                dh_t_dW_full_batch = torch.zeros_like(h)
                accumulated_grad = None

                W_ii, W_if, W_ig, W_io = split_by_4(self.rnn.__getattr__(f"weight_ih_l{j}"))
                W_hi, W_hf, W_hg, W_ho = split_by_4(self.rnn.__getattr__(f"weight_hh_l{j}"))
                b_ii, b_if, b_ig, b_io = split_by_4(self.rnn.__getattr__(f"bias_ih_l{j}"))
                b_hi, b_hf, b_hg, b_ho = split_by_4(self.rnn.__getattr__(f"bias_hh_l{j}"))

                get_shape = lambda weight: weight.shape if not mage else [x[0].shape[0]] + list(weight.shape)
                vw_ii = torch.zeros(get_shape(W_ii)).to(device)
                vw_hi = torch.zeros(get_shape(W_hi)).to(device)
                vw_if = torch.zeros(get_shape(W_if)).to(device)
                vw_hf = torch.zeros(get_shape(W_hf)).to(device)
                vw_ig = torch.zeros(get_shape(W_ig)).to(device)
                vw_hg = torch.zeros(get_shape(W_hg)).to(device)
                vw_io = torch.zeros(get_shape(W_io)).to(device)
                vw_ho = torch.zeros(get_shape(W_ho)).to(device)
                vb_ii = torch.zeros(get_shape(b_ii)).to(device)
                vb_hi = torch.zeros(get_shape(b_hi)).to(device)
                vb_if = torch.zeros(get_shape(b_if)).to(device)
                vb_hf = torch.zeros(get_shape(b_hf)).to(device)
                vb_ig = torch.zeros(get_shape(b_ig)).to(device)
                vb_hg = torch.zeros(get_shape(b_hg)).to(device)
                vb_io = torch.zeros(get_shape(b_io)).to(device)
                vb_ho = torch.zeros(get_shape(b_ho)).to(device)
                dh_t_dW = None
                dc_dW = None
                dc_dh_t_1 = None

                # todo: remove V creation from time step
                if mage:
                    if random_t_separately:
                        batch = None
                        g0 = random(W_ii, random_binary, batch)
                        g1 = random(W_if, random_binary, batch)
                        g2 = random(W_ig, random_binary, batch)
                        g3 = random(W_io, random_binary, batch)
                elif not vanilla_V_per_timestep:
                    if not random_binary:
                        _vw_i, _vw_h, _vb_i, _vb_h = create_new_Vs(self.rnn, j, device, epsilon)
                    else:
                        _vw_i, _vw_h, _vb_i, _vb_h = create_new_Vs_binary(self.rnn, j, device, epsilon)
                    vw_ii, vw_if, vw_ig, vw_io = _vw_i
                    vw_hi, vw_hf, vw_hg, vw_ho = _vw_h
                    vb_ii, vb_if, vb_ig, vb_io = _vb_i
                    vb_hi, vb_hf, vb_hg, vb_ho = _vb_h
                    _vw_ii, _vw_if, _vw_ig, _vw_io = _vw_i
                    _vw_hi, _vw_hf, _vw_hg, _vw_ho = _vw_h
                    _vb_ii, _vb_if, _vb_ig, _vb_io = _vb_i
                    _vb_hi, _vb_hf, _vb_hg, _vb_ho = _vb_h

                for seq in range(len(x)):
                    x_t = x[seq]
                    h_part = h
                    dz_dW = z_grad_list[seq] if j > 0 else None
                    if vanilla_V_per_timestep:
                        _vw_i, _vw_h, _vb_i, _vb_h = create_new_Vs(self.rnn, j, device, epsilon)
                        _vw_ii, _vw_if, _vw_ig, _vw_io = _vw_i
                        _vw_hi, _vw_hf, _vw_hg, _vw_ho = _vw_h
                        _vb_ii, _vb_if, _vb_ig, _vb_io = _vb_i
                        _vb_hi, _vb_hf, _vb_hg, _vb_ho = _vb_h
                    if mage:
                        if guess is not None:
                            _vw_i, _vw_h, _vb_i, _vb_h = create_new_Vs_mage_guess(x_t, h_part, self.rnn, guess, ig, seq , j, device,
                                                                            epsilon, reduce_batch)
                        elif random_t_separately:
                            _vw_i, _vw_h, _vb_i, _vb_h = create_new_Vs_mage_random_t_separately(x_t, h_part,
                                                                                            (g0, g1, g2, g3),
                                                                                            device, binary=random_binary)
                        else:
                            _vw_i, _vw_h, _vb_i, _vb_h = create_new_Vs_mage(x_t, h_part, self.rnn, j,
                                                                            with_batch=g_with_batch, binary=random_binary)
                        _vw_ii, _vw_if, _vw_ig, _vw_io = _vw_i
                        _vw_hi, _vw_hf, _vw_hg, _vw_ho = _vw_h
                        _vb_ii, _vb_if, _vb_ig, _vb_io = _vb_i
                        _vb_hi, _vb_hf, _vb_hg, _vb_ho = _vb_h

                    i_p = x_t @ W_ii.T + b_ii + h_part @ W_hi.T + b_hi

                    di_p_dW = (x_t.unsqueeze(1) @ torch.transpose(_vw_ii, -1, -2)).squeeze(1) + \
                              (dz_dW @ W_ii.T if dz_dW is not None else 0) + \
                              _vb_ii + \
                              (h_part.unsqueeze(1) @ torch.transpose(_vw_hi, -1, -2)).squeeze(1) + \
                              (dh_t_dW @ W_hi.T if dh_t_dW is not None else 0) + \
                              _vb_hi
                    i = torch.sigmoid(i_p)
                    di_dW = (i * (1 - i)) * di_p_dW

                    f_p = x_t @ W_if.T + b_if + h_part @ W_hf.T + b_hf
                    df_p_dW = (x_t.unsqueeze(1) @ torch.transpose(_vw_if, -1, -2)).squeeze(1) + \
                              (dz_dW @ W_if.T if dz_dW is not None else 0) + \
                              _vb_if + \
                              (h_part.unsqueeze(1) @ torch.transpose(_vw_hf, -1, -2)).squeeze(1) + \
                              (dh_t_dW @ W_hf.T if dh_t_dW is not None else 0) + \
                              _vb_hf
                    f = torch.sigmoid(f_p)
                    df_dW = (f * (1 - f)) * df_p_dW


                    g_p = x_t @ W_ig.T + b_ig + h_part @ W_hg.T + b_hg
                    dg_p_dW = (x_t.unsqueeze(1) @ torch.transpose(_vw_ig, -1, -2)).squeeze(1) + \
                              (dz_dW @ W_ig.T if dz_dW is not None else 0) + \
                              _vb_ig + \
                              (h_part.unsqueeze(1) @ torch.transpose(_vw_hg, -1, -2)).squeeze(1) + \
                              (dh_t_dW @ W_hg.T if dh_t_dW is not None else 0) + \
                              _vb_hg
                    g = torch.tanh(g_p)
                    dg_dW = (1 - g ** 2) * dg_p_dW

                    o_p = x_t @ W_io.T + b_io + h_part @ W_ho.T + b_ho
                    do_p_dW = (x_t.unsqueeze(1) @ torch.transpose(_vw_io, -1, -2)).squeeze(1) + \
                              (dz_dW @ W_io.T if dz_dW is not None else 0) + \
                              _vb_io + \
                              (h_part.unsqueeze(1) @ torch.transpose(_vw_ho, -1, -2)).squeeze(1) + \
                              (dh_t_dW @ W_ho.T if dh_t_dW is not None else 0) + \
                              _vb_ho
                    o = torch.sigmoid(o_p)
                    do_dW = (o * (1 - o)) * do_p_dW

                    c_t = f * c_t_1 + i * g
                    dc_dW = df_dW * c_t_1 + (dc_dW * f if dc_dW is not None else 0) + di_dW * g + dg_dW * i

                    tanh_c_t = torch.tanh(c_t)
                    d_tanh_c_t_dW = dc_dW * (1 - tanh_c_t ** 2)

                    h = o * tanh_c_t
                    dh_t_dW = do_dW * tanh_c_t + d_tanh_c_t_dW * o

                    h_list.append(h)
                    # h_full_batch = combine_batch(h, h_full_batch)
                    h_grad_list.append(dh_t_dW)
                    c_t_1 = c_t

                    if (mage or vanilla_V_per_timestep) and (j, seq) in relevant_Vs:
                        vw_ii += _vw_ii
                        vw_hi += _vw_hi
                        vw_if += _vw_if
                        vw_hf += _vw_hf
                        vw_ig += _vw_ig
                        vw_hg += _vw_hg
                        vw_io += _vw_io
                        vw_ho += _vw_ho
                        vb_ii += _vb_ii
                        vb_hi += _vb_hi
                        vb_if += _vb_if
                        vb_hf += _vb_hf
                        vb_ig += _vb_ig
                        vb_hg += _vb_hg
                        vb_io += _vb_io
                        vb_ho += _vb_ho
                        # todo: add dropout as in nn.LSTM

                x = torch.stack(h_list[1:], dim=0)
                z_grad_list = torch.stack(h_grad_list, dim=0)
                vw_ih = torch.cat([vw_ii, vw_if, vw_ig, vw_io], dim=-2)
                vw_hh = torch.cat([vw_hi, vw_hf, vw_hg, vw_ho], dim=-2)
                vb_ih = torch.cat([vb_ii, vb_if, vb_ig, vb_io], dim=-1)
                vb_hh = torch.cat([vb_hi, vb_hf, vb_hg, vb_ho], dim=-1)
                V[j] = (vw_ih, vw_hh, vb_ih, vb_hh)
                h_stack.append(h)
                c_stack.append(c_t)

            grad = dh_t_dW

        with torch.no_grad():
            output, (hidden, _) = x, (torch.stack(h_stack, dim=0), torch.stack(c_stack, dim=0))

            self.rnn.permute_hidden(hidden, None)

            hidden = (torch.transpose(hidden[-self.n_directions:], 0, 1)).reshape(
                (-1, self.hidden_dim * self.n_directions))

            # # Dropout
            # hidden = self.drop(hidden)

            if mage:
                # get_shape = lambda w: (w.shape[0], 1) if not reduce_batch else (hidden.shape[0], w.shape[0], 1)
                norm = torch.sqrt((hidden ** 2).sum(dim=1, keepdim=True) + 1)

                pw = random(self.decoder.weight, random_binary, hidden.shape[0] if g_with_batch else None)
                vw = torch.matmul(pw, (hidden/norm).unsqueeze(1))
                vb = torch.matmul(pw, (1.0 / norm).unsqueeze(1)).squeeze(-1).squeeze(-1)
            else:
                if random_binary:
                    vw = torch.randint(0, 2, self.decoder.weight.shape, device=device, dtype=torch.float32) * 2 - 1
                else:
                    vw = torch.randn(self.decoder.weight.shape, device=device, dtype=torch.float32) * epsilon
                vb = torch.randn(self.decoder.bias.shape, device=device, dtype=torch.float32) * epsilon
            new_grad = (hidden.unsqueeze(1) @ torch.transpose(vw, -1, -2)).squeeze() + vb
            grad = torch.matmul(grad, self.decoder.weight.permute(1, 0)).squeeze() + new_grad

            # Decode
            decoded = self.decoder(hidden).squeeze(1)

        dLdout = torch.zeros_like(decoded)

        out = torch.autograd.Variable(decoded, requires_grad=True)
        out.grad = torch.zeros_like(out)
        L = loss(out, y)

        L.backward()
        ##import pdb; pdb.set_trace()
        dLdout = out.grad

        ##grad_transfer = dLdout.permute(1, 0) ## Batch x n_classes
        ##tot_norm = torch.sqrt(tot_norm)
        with torch.no_grad():
            dFg = (dLdout * grad) if mage else (dLdout * grad).sum()
            apply_fwd_grad = apply_fwd_grad_batch if mage else apply_fwd_grad_no_batch
            if mage and reduce_batch:
                apply_fwd_grad = apply_fwd_grad_reduce_batch

            for i in range(self.rnn.num_layers):
                for w in [self.rnn.__getattr__(f"weight_ih_l{i}"),
                          self.rnn.__getattr__(f"weight_hh_l{i}"),
                          self.rnn.__getattr__(f"bias_ih_l{i}"),
                          self.rnn.__getattr__(f"bias_hh_l{i}")]:
                    if w.grad is None:
                        w.grad = torch.zeros_like(w)

                vw_ih, vw_hh, vb_ih, vb_hh = V[i]
                ##breakpoint()
                self.rnn.__getattr__(f"weight_ih_l{i}").grad += apply_fwd_grad(dFg, vw_ih) / grad_div
                self.rnn.__getattr__(f"weight_hh_l{i}").grad += apply_fwd_grad(dFg, vw_hh) / grad_div
                self.rnn.__getattr__(f"bias_ih_l{i}").grad += apply_fwd_grad(dFg, vb_ih) / grad_div
                self.rnn.__getattr__(f"bias_hh_l{i}").grad += apply_fwd_grad(dFg, vb_hh) / grad_div
            if self.decoder.weight.grad is None:
                self.decoder.weight.grad = torch.zeros_like(self.decoder.weight)
            if self.decoder.bias.grad is None:
                self.decoder.bias.grad = torch.zeros_like(self.decoder.bias)
            self.decoder.weight.grad += apply_fwd_grad(dFg, vw) / grad_div
            self.decoder.bias.grad += apply_fwd_grad(dFg, vb) / grad_div

        return decoded

