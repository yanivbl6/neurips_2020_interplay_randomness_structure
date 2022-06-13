import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence


def normalize(x):
    eps = 1E-6
    return x / (x.norm(dim=1, keepdim=True) + eps)


def split_by_4(x):
    return torch.split(x, x.shape[0] // 4)


def combine_batch(new, old):
    if new.shape == old.shape:
        return new
    b = new.shape[0]
    return torch.cat([new, torch.zeros_like(old[b:])], dim=0) + torch.cat([torch.zeros_like(old[:b]), old[b:]], dim=0)

def apply_fwd_grad_no_batch(dFg, vw):
    return dFg * vw

def apply_fwd_grad_batch(dFg, vw):
    if len(vw.shape) == 3:
        return torch.matmul(dFg, vw.view(vw.shape[0], -1)).view(vw.shape[1], vw.shape[2])
    else:
        return dFg.sum() * vw

def compute_corr_matrix(padded_activations, batch_sizes):
    norms = torch.norm(padded_activations[0], dim=-1).unsqueeze(-1)
    corr_matrices = padded_activations[0] @ padded_activations[0].transpose(-2, -1) / (1e-8 + norms @ norms.transpose(-2, -1))
    seqs_in_batch = (batch_sizes.unsqueeze(1) @ torch.ones_like(batch_sizes).unsqueeze(0)).to(corr_matrices.device)
    seqs_in_batch = torch.min(seqs_in_batch, seqs_in_batch.T)
    return torch.sum(corr_matrices / seqs_in_batch.unsqueeze(0), dim=0)

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


def create_new_Vs_mage(x_t, h_part, rnn, j, device, epsilon):
    W_ii, W_if, W_ig, W_io = split_by_4(rnn.__getattr__(f"weight_ih_l{j}"))
    W_hi, W_hf, W_hg, W_ho = split_by_4(rnn.__getattr__(f"weight_hh_l{j}"))
    b_ii, b_if, b_ig, b_io = split_by_4(rnn.__getattr__(f"bias_ih_l{j}"))
    b_hi, b_hf, b_hg, b_ho = split_by_4(rnn.__getattr__(f"bias_hh_l{j}"))

    pw_ii = torch.randn((W_ii.shape[0], 1), device=device) * epsilon
    pw_if = torch.randn((W_if.shape[0], 1), device=device) * epsilon
    pw_ig = torch.randn((W_ig.shape[0], 1), device=device) * epsilon
    pw_io = torch.randn((W_io.shape[0], 1), device=device) * epsilon
    pw_hi = torch.randn((W_hi.shape[0], 1), device=device) * epsilon
    pw_hf = torch.randn((W_hf.shape[0], 1), device=device) * epsilon
    pw_hg = torch.randn((W_hg.shape[0], 1), device=device) * epsilon
    pw_ho = torch.randn((W_ho.shape[0], 1), device=device) * epsilon
    _vb_ii = torch.randn(b_ii.shape, device=device) * epsilon
    _vb_if = torch.randn(b_if.shape, device=device) * epsilon
    _vb_ig = torch.randn(b_ig.shape, device=device) * epsilon
    _vb_io = torch.randn(b_io.shape, device=device) * epsilon
    _vb_hi = torch.randn(b_hi.shape, device=device) * epsilon
    _vb_hf = torch.randn(b_hf.shape, device=device) * epsilon
    _vb_hg = torch.randn(b_hg.shape, device=device) * epsilon
    _vb_ho = torch.randn(b_ho.shape, device=device) * epsilon

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

def create_new_Vs_mage_all_times(x, hx, rnn, j, device, epsilon):
    W_ii, W_if, W_ig, W_io = split_by_4(rnn.__getattr__(f"weight_ih_l{j}"))
    W_hi, W_hf, W_hg, W_ho = split_by_4(rnn.__getattr__(f"weight_hh_l{j}"))
    b_ii, b_if, b_ig, b_io = split_by_4(rnn.__getattr__(f"bias_ih_l{j}"))
    b_hi, b_hf, b_hg, b_ho = split_by_4(rnn.__getattr__(f"bias_hh_l{j}"))

    relevant_Vs = [(j, seq) for j in range(rnn.num_layers) for seq in range(len(x))]
    with torch.no_grad():
        h, c_t_1 = hx[j] if hx[0].shape[0] > 1 else (hx[0][0], hx[1][0])
        h_list = [h]
        h_full_batch = torch.zeros_like(h)

        vw_ii = torch.zeros([x[0].shape[0]] + list(W_ii.shape)).to(device)
        vw_hi = torch.zeros([x[0].shape[0]] + list(W_hi.shape)).to(device)
        vw_if = torch.zeros([x[0].shape[0]] + list(W_if.shape)).to(device)
        vw_hf = torch.zeros([x[0].shape[0]] + list(W_hf.shape)).to(device)
        vw_ig = torch.zeros([x[0].shape[0]] + list(W_ig.shape)).to(device)
        vw_hg = torch.zeros([x[0].shape[0]] + list(W_hg.shape)).to(device)
        vw_io = torch.zeros([x[0].shape[0]] + list(W_io.shape)).to(device)
        vw_ho = torch.zeros([x[0].shape[0]] + list(W_ho.shape)).to(device)
        vb_ii = torch.zeros_like(b_ii).to(device)
        vb_hi = torch.zeros_like(b_hi).to(device)
        vb_if = torch.zeros_like(b_if).to(device)
        vb_hf = torch.zeros_like(b_hf).to(device)
        vb_ig = torch.zeros_like(b_ig).to(device)
        vb_hg = torch.zeros_like(b_hg).to(device)
        vb_io = torch.zeros_like(b_io).to(device)
        vb_ho = torch.zeros_like(b_ho).to(device)

        for seq in range(len(x)):
            x_t = x[seq]
            h_part = h[:x_t.shape[0]]
            c_t_1 = c_t_1[:x_t.shape[0]]

            pw_ii = torch.randn((W_ii.shape[0], 1), device=device) * epsilon
            pw_if = torch.randn((W_if.shape[0], 1), device=device) * epsilon
            pw_ig = torch.randn((W_ig.shape[0], 1), device=device) * epsilon
            pw_io = torch.randn((W_io.shape[0], 1), device=device) * epsilon
            pw_hi = torch.randn((W_hi.shape[0], 1), device=device) * epsilon
            pw_hf = torch.randn((W_hf.shape[0], 1), device=device) * epsilon
            pw_hg = torch.randn((W_hg.shape[0], 1), device=device) * epsilon
            pw_ho = torch.randn((W_ho.shape[0], 1), device=device) * epsilon
            _vb_ii = torch.randn(b_ii.shape, device=device) * epsilon
            _vb_if = torch.randn(b_if.shape, device=device) * epsilon
            _vb_ig = torch.randn(b_ig.shape, device=device) * epsilon
            _vb_io = torch.randn(b_io.shape, device=device) * epsilon
            _vb_hi = torch.randn(b_hi.shape, device=device) * epsilon
            _vb_hf = torch.randn(b_hf.shape, device=device) * epsilon
            _vb_hg = torch.randn(b_hg.shape, device=device) * epsilon
            _vb_ho = torch.randn(b_ho.shape, device=device) * epsilon


            i_p = x_t @ W_ii.T + b_ii + h_part @ W_hi.T + b_hi
            i = torch.sigmoid(i_p)
            f_p = x_t @ W_if.T + b_if + h_part @ W_hf.T + b_hf
            f = torch.sigmoid(f_p)
            g_p = x_t @ W_ig.T + b_ig + h_part @ W_hg.T + b_hg
            g = torch.tanh(g_p)
            o_p = x_t @ W_io.T + b_io + h_part @ W_ho.T + b_ho
            o = torch.sigmoid(o_p)
            c_t = f * c_t_1 + i * g
            tanh_c_t = torch.tanh(c_t)
            h = o * tanh_c_t

            _vw_ii = torch.matmul(pw_ii, normalize(x_t).unsqueeze(1))
            _vw_hi = torch.matmul(pw_hi, normalize(h_part).unsqueeze(1))
            _vw_if = torch.matmul(pw_if, normalize(x_t).unsqueeze(1))
            _vw_hf = torch.matmul(pw_hf, normalize(h_part).unsqueeze(1))
            _vw_ig = torch.matmul(pw_ig, normalize(x_t).unsqueeze(1))
            _vw_hg = torch.matmul(pw_hg, normalize(h_part).unsqueeze(1))
            _vw_io = torch.matmul(pw_io, normalize(x_t).unsqueeze(1))
            _vw_ho = torch.matmul(pw_ho, normalize(h_part).unsqueeze(1))

            h_list.append(h)
            h_full_batch = combine_batch(h, h_full_batch)
            c_t_1 = c_t

            if (j, seq) in relevant_Vs:
                vw_ii += combine_batch(_vw_ii, torch.zeros_like(vw_ii))
                vw_hi += combine_batch(_vw_hi, torch.zeros_like(vw_hi))
                vw_if += combine_batch(_vw_if, torch.zeros_like(vw_if))
                vw_hf += combine_batch(_vw_hf, torch.zeros_like(vw_hf))
                vw_ig += combine_batch(_vw_ig, torch.zeros_like(vw_ig))
                vw_hg += combine_batch(_vw_hg, torch.zeros_like(vw_hg))
                vw_io += combine_batch(_vw_io, torch.zeros_like(vw_io))
                vw_ho += combine_batch(_vw_ho, torch.zeros_like(vw_ho))
                vb_ii += _vb_ii
                vb_hi += _vb_hi
                vb_if += _vb_if
                vb_hf += _vb_hf
                vb_ig += _vb_ig
                vb_hg += _vb_hg
                vb_io += _vb_io
                vb_ho += _vb_ho

    vw_i = (vw_ii, vw_if, vw_ig, vw_io)
    vw_h = (vw_hi, vw_hf, vw_hg, vw_ho)
    vb_i = (vb_ii, vb_if, vb_ig, vb_io)
    vb_h = (vb_hi, vb_hf, vb_hg, vb_ho)

    return vw_i, vw_h, vb_i, vb_h


class RNN(nn.Module):
    def __init__(self, rnn_type, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx, train_embedding=True, save_correlations=False):
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
            self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=n_layers,
                              bidirectional=bidirectional, dropout=dropout)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
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

        # Pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        if self.rnn_type == 'LSTM':
            packed_output, (hidden, cell) = self.rnn(packed_embedded)
        else:
            packed_output, hidden = self.rnn(packed_embedded)
        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]
        # Unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        # output = [sentence len, batch size, hid dim * num directions]

        # Concatenate the final hidden layers (for bidirectional)
        # hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        hidden = (torch.transpose(hidden[-self.n_directions:], 0, 1)).reshape(
            (-1, self.hidden_dim * self.n_directions))

        # Dropout
        hidden = self.drop(hidden)

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
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)

        if isinstance(packed_embedded, PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = packed_embedded
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
        num_directions = 2 if self.rnn.bidirectional else 1
        zeros = torch.zeros(self.rnn.num_layers * num_directions,
                            max_batch_size, self.rnn.hidden_size,
                            dtype=input.dtype, device=input.device)
        hx = (zeros, zeros)
        self.rnn.check_forward_args(input, hx, batch_sizes)

        return hx, input, batch_sizes, sorted_indices, unsorted_indices, packed_embedded

    def fwd_mode(self, batch_text, y, loss, mage=False, grad_div=1):
        hx, input, batch_sizes, sorted_indices, unsorted_indices, packed_embedded = self.batch_text_to_input(batch_text)
        x = torch.split(input, tuple(batch_sizes))
        device = x[0].device
        if self.save_correlations:
            self.input_correlation_matrix = compute_corr_matrix(torch.nn.utils.rnn.pad_packed_sequence(packed_embedded, batch_first=True), batch_sizes)
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

                vw_ii = torch.zeros(W_ii.shape if not mage else [x[0].shape[0]] + list(W_ii.shape)).to(device)
                vw_hi = torch.zeros(W_hi.shape if not mage else [x[0].shape[0]] + list(W_hi.shape)).to(device)
                vw_if = torch.zeros(W_if.shape if not mage else [x[0].shape[0]] + list(W_if.shape)).to(device)
                vw_hf = torch.zeros(W_hf.shape if not mage else [x[0].shape[0]] + list(W_hf.shape)).to(device)
                vw_ig = torch.zeros(W_ig.shape if not mage else [x[0].shape[0]] + list(W_ig.shape)).to(device)
                vw_hg = torch.zeros(W_hg.shape if not mage else [x[0].shape[0]] + list(W_hg.shape)).to(device)
                vw_io = torch.zeros(W_io.shape if not mage else [x[0].shape[0]] + list(W_io.shape)).to(device)
                vw_ho = torch.zeros(W_ho.shape if not mage else [x[0].shape[0]] + list(W_ho.shape)).to(device)
                vb_ii = torch.zeros_like(b_ii).to(device)
                vb_hi = torch.zeros_like(b_hi).to(device)
                vb_if = torch.zeros_like(b_if).to(device)
                vb_hf = torch.zeros_like(b_hf).to(device)
                vb_ig = torch.zeros_like(b_ig).to(device)
                vb_hg = torch.zeros_like(b_hg).to(device)
                vb_io = torch.zeros_like(b_io).to(device)
                vb_ho = torch.zeros_like(b_ho).to(device)
                dh_t_dW = None
                dc_dW = None
                dc_dh_t_1 = None

                # todo: remove V creation from time step
                if mage:
                    pass
                    # _vw_i, _vw_h, _vb_i, _vb_h = create_new_Vs_mage_all_times(x, hx, self.rnn, j, device, epsilon)
                else:
                    _vw_i, _vw_h, _vb_i, _vb_h = create_new_Vs(self.rnn, j, device, epsilon)
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
                    # if x_t.shape[0] != 64:
                    #     continue
                    h_part = h[:x_t.shape[0]]
                    c_t_1 = c_t_1[:x_t.shape[0]]
                    dz_dW = z_grad_list[seq] if j > 0 else None
                    # if mage:
                    #     _vw_ii, _vw_if, _vw_ig, _vw_io = [v[:x_t.shape[0]] for v in _vw_i]
                    #     _vw_hi, _vw_hf, _vw_hg, _vw_ho = [v[:x_t.shape[0]] for v in _vw_h]
                    if mage:
                        _vw_i, _vw_h, _vb_i, _vb_h = create_new_Vs_mage(x_t, h_part, self.rnn, j, device, epsilon)
                        _vw_ii, _vw_if, _vw_ig, _vw_io = _vw_i
                        _vw_hi, _vw_hf, _vw_hg, _vw_ho = _vw_h
                        _vb_ii, _vb_if, _vb_ig, _vb_io = _vb_i
                        _vb_hi, _vb_hf, _vb_hg, _vb_ho = _vb_h

                    i_p = x_t @ W_ii.T + b_ii + h_part @ W_hi.T + b_hi
                    di_p_dW = (x_t.unsqueeze(1) @ torch.transpose(_vw_ii, -1, -2)).squeeze(1) + \
                              (dz_dW @ W_ii.T if dz_dW is not None else 0) + \
                              _vb_ii + \
                              (h_part.unsqueeze(1) @ torch.transpose(_vw_hi, -1, -2)).squeeze(1) + \
                              (dh_t_dW[:x_t.shape[0]] @ W_hi.T if dh_t_dW is not None else 0) + \
                              _vb_hi
                    # di_p_dh_t_1 = W_hi
                    i = torch.sigmoid(i_p)
                    di_dW = (i * (1 - i)) * di_p_dW
                    # di_dh_t_1 = (i * (1 - i)).unsqueeze(-1) * di_p_dh_t_1

                    f_p = x_t @ W_if.T + b_if + h_part @ W_hf.T + b_hf
                    df_p_dW = (x_t.unsqueeze(1) @ torch.transpose(_vw_if, -1, -2)).squeeze(1) + \
                              (dz_dW @ W_if.T if dz_dW is not None else 0) + \
                              _vb_if + \
                              (h_part.unsqueeze(1) @ torch.transpose(_vw_hf, -1, -2)).squeeze(1) + \
                              (dh_t_dW[:x_t.shape[0]] @ W_hf.T if dh_t_dW is not None else 0) + \
                              _vb_hf
                    # df_p_dh_t_1 = W_hf
                    f = torch.sigmoid(f_p)
                    df_dW = (f * (1 - f)) * df_p_dW
                    # df_dh_t_1 = (f * (1 - f)).unsqueeze(-1) * df_p_dh_t_1


                    g_p = x_t @ W_ig.T + b_ig + h_part @ W_hg.T + b_hg
                    dg_p_dW = (x_t.unsqueeze(1) @ torch.transpose(_vw_ig, -1, -2)).squeeze(1) + \
                              (dz_dW @ W_ig.T if dz_dW is not None else 0) + \
                              _vb_ig + \
                              (h_part.unsqueeze(1) @ torch.transpose(_vw_hg, -1, -2)).squeeze(1) + \
                              (dh_t_dW[:x_t.shape[0]] @ W_hg.T if dh_t_dW is not None else 0) + \
                              _vb_hg
                    # dg_p_dh_t_1 = W_hg
                    g = torch.tanh(g_p)
                    dg_dW = (1 - g ** 2) * dg_p_dW
                    # dg_dh_t_1 = (1 - g ** 2).unsqueeze(-1) * dg_p_dh_t_1

                    o_p = x_t @ W_io.T + b_io + h_part @ W_ho.T + b_ho
                    do_p_dW = (x_t.unsqueeze(1) @ torch.transpose(_vw_io, -1, -2)).squeeze(1) + \
                              (dz_dW @ W_io.T if dz_dW is not None else 0) + \
                              _vb_io + \
                              (h_part.unsqueeze(1) @ torch.transpose(_vw_ho, -1, -2)).squeeze(1) + \
                              (dh_t_dW[:x_t.shape[0]] @ W_ho.T if dh_t_dW is not None else 0) + \
                              _vb_ho
                    # do_p_dh_t_1 = W_ho
                    o = torch.sigmoid(o_p)
                    do_dW = (o * (1 - o)) * do_p_dW
                    # do_dh_t_1 = (o * (1 - o)).unsqueeze(-1) * do_p_dh_t_1

                    c_t = f * c_t_1 + i * g
                    dc_dW = df_dW * c_t_1 + (dc_dW[:x_t.shape[0]] * f if dc_dW is not None else 0) + di_dW * g + dg_dW * i
                    # dc_dh_t_1 = df_dh_t_1 * c_t_1.unsqueeze(-1) + (dc_dh_t_1[:x_t.shape[0]] * f.unsqueeze(-1) if dc_dh_t_1 is not None else 0) + di_dh_t_1 * g.unsqueeze(-1) + dg_dh_t_1 * i.unsqueeze(-1)

                    tanh_c_t = torch.tanh(c_t)
                    d_tanh_c_t_dW = dc_dW * (1 - tanh_c_t ** 2)
                    # d_tanh_c_t_dh_t_1 = dc_dh_t_1 * (1 - tanh_c_t ** 2).unsqueeze(-1)

                    h = o * tanh_c_t
                    dh_t_dW = do_dW * tanh_c_t + d_tanh_c_t_dW * o
                    # dh_t_dh_t_1 = do_dh_t_1 * tanh_c_t.unsqueeze(-1) + d_tanh_c_t_dh_t_1 * o.unsqueeze(-1)

                    # if accumulated_grad is None:
                    #     accumulated_grad = dh_t_dW
                    # else:
                    #     # todo: remove dh_t_dh_t_1
                    #     accumulated_grad = combine_batch((dh_t_dh_t_1 @ accumulated_grad[:x_t.shape[0]].unsqueeze(
                    #             -1)).squeeze() + dh_t_dW, accumulated_grad)

                    h_list.append(h)
                    h_full_batch = combine_batch(h, h_full_batch)
                    h_grad_list.append(dh_t_dW)
                    dh_t_dW_full_batch = combine_batch(dh_t_dW, dh_t_dW_full_batch)
                    c_t_1 = c_t

                    if (j, seq) in relevant_Vs:
                        vw_ii += combine_batch(_vw_ii, torch.zeros_like(vw_ii))
                        vw_hi += combine_batch(_vw_hi, torch.zeros_like(vw_hi))
                        vw_if += combine_batch(_vw_if, torch.zeros_like(vw_if))
                        vw_hf += combine_batch(_vw_hf, torch.zeros_like(vw_hf))
                        vw_ig += combine_batch(_vw_ig, torch.zeros_like(vw_ig))
                        vw_hg += combine_batch(_vw_hg, torch.zeros_like(vw_hg))
                        vw_io += combine_batch(_vw_io, torch.zeros_like(vw_io))
                        vw_ho += combine_batch(_vw_ho, torch.zeros_like(vw_ho))
                        vb_ii += _vb_ii
                        vb_hi += _vb_hi
                        vb_if += _vb_if
                        vb_hf += _vb_hf
                        vb_ig += _vb_ig
                        vb_hg += _vb_hg
                        vb_io += _vb_io
                        vb_ho += _vb_ho


                # todo: add dropout as in nn.LSTM
                x = tuple(h_list[1:])
                z_grad_list = h_grad_list
                vw_ih = torch.cat([vw_ii, vw_if, vw_ig, vw_io], dim=-2)
                vw_hh = torch.cat([vw_hi, vw_hf, vw_hg, vw_ho], dim=-2)
                vb_ih = torch.cat([vb_ii, vb_if, vb_ig, vb_io], dim=-1)
                vb_hh = torch.cat([vb_hi, vb_hf, vb_hg, vb_ho], dim=-1)
                V[j] = (vw_ih, vw_hh, vb_ih, vb_hh)
                h_stack.append(h_full_batch)
                c_stack.append(c_t)

            grad = dh_t_dW_full_batch

        with torch.no_grad():
            packed_output, (hidden, _) = x, (torch.stack(h_stack, dim=0), torch.zeros_like(c_stack[0]))

            self.rnn.permute_hidden(hidden, unsorted_indices)

            if self.save_correlations:
                padded = torch.nn.utils.rnn.pad_packed_sequence(torch.nn.utils.rnn.PackedSequence(torch.cat(h_list, dim=0),
                                                                                                  batch_sizes=batch_sizes,
                                                                                                  sorted_indices=sorted_indices,
                                                                                                  unsorted_indices=unsorted_indices),
                                                                batch_first=True)
                self.output_correlation_matrix = compute_corr_matrix(padded, batch_sizes)

            hidden = (torch.transpose(hidden[-self.n_directions:], 0, 1)).reshape(
                (-1, self.hidden_dim * self.n_directions))

            # # Dropout
            # hidden = self.drop(hidden)

            if mage:
                pw = torch.randn((self.decoder.weight.shape[0], 1), device=device, dtype=torch.float32) * epsilon
                vw = torch.matmul(pw, normalize(hidden).unsqueeze(1))
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

            for i in range(self.rnn.num_layers):
                for w in [self.rnn.__getattr__(f"weight_ih_l{i}"),
                          self.rnn.__getattr__(f"weight_hh_l{i}"),
                          self.rnn.__getattr__(f"bias_ih_l{i}"),
                          self.rnn.__getattr__(f"bias_hh_l{i}")]:
                    if w.grad is None:
                        w.grad = torch.zeros_like(w)

                vw_ih, vw_hh, vb_ih, vb_hh = V[i]
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

