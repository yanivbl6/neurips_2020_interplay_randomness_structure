from socket import gaierror
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence

from torch.distributions.multivariate_normal import MultivariateNormal 

from torch.nn import RNN
from rnn import LSTM

import math
import wandb

def angle(x,y):
    ##breakpoint()
    return (x.squeeze()*y.squeeze()).sum(dim=1)/(torch.norm(x.squeeze(),dim=1)*torch.norm(y.squeeze(),dim=1))

def decompose(z, alpha):
    device = z.device

    # I = torch.eye(z.shape[2], device = device).unsqueeze(0)
    # Sigma = I + alpha* torch.matmul(z,z.permute(0,2,1))
    # L2 = torch.cholesky(Sigma)


    z = z / z.norm(dim=2, keepdim = True)

    znorm2 = (z.norm(dim=2, keepdim = True))**2
    
    factor = torch.sqrt(alpha**2+ (1-alpha**2)* (znorm2))  - alpha
    factor = factor/ znorm2
    
    L =  alpha*torch.eye(z.size(2), device = device).unsqueeze(0) + (1-alpha**2) * (z.permute(0,2,1) @ z)
    ##L =  alpha*torch.eye(z.size(2), device = device).unsqueeze(0) + factor * z



    I = torch.eye(z.shape[2], device = device).unsqueeze(0)
    # Sigma = (alpha**2)*I + (1-alpha**2)* torch.matmul(z,z.permute(0,2,1))
    # Sigma2 = (L @ L.permute(0,2,1))
    # diff = ((Sigma - Sigma2 )**2).sum()
    # if (diff > 1E-5):
    #     breakpoint()

    return L


def batched_sherman_morrison(guess,t, alpha):
    z = guess[t].unsqueeze(1)
    device = z.device
    to_inv = torch.matmul(z.permute(0,2,1), z) + 1e-6 * torch.eye(z.shape[2], device = device).unsqueeze(0)
    return torch.inverse(to_inv)

    z = z / z.norm(dim=2, keepdim = True)



    I = torch.eye(z.shape[2], device = device).unsqueeze(0)
    if alpha > 0.0:

        if alpha == 0.0:
            return I

        factor = (1-alpha**2)/(alpha**2)
        den = 1 + factor* torch.matmul(z,z.permute(0,2,1))
        sol = I - factor* torch.matmul(z.permute(0,2,1),z)/den
        return sol/alpha**2
    else:
        return I    



def normalize(x):
    eps = 1E-8
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def invert_sigma(A):
    eps = 1E-8
    return torch.linalg.inv(A.T @ A + eps * torch.eye(A.shape[0]))


def split_by_4(x):
    return torch.split(x, x.shape[0] // 4)

def split_by_4(x):
    return torch.split(x, x.shape[0] // 4)



def combine_batch(new, old):
    if new.shape == old.shape:
        return new
    try:
        b = new.shape[0]
        return torch.cat([new, torch.zeros_like(old[b:])], dim=0) + torch.cat([torch.zeros_like(old[:b]), old[b:]], dim=0)
    except:
        breakpoint()
        pass
def apply_fwd_grad_no_batch(dFg, vw):
    return dFg * vw

def apply_fwd_grad_batch(dFg, vw):
    if len(vw.shape) == 3:
        return torch.matmul(dFg, vw.view(vw.shape[0], -1)).view(vw.shape[1], vw.shape[2])
    elif len(vw.shape) == 2:
        return torch.matmul(dFg, vw.view(vw.shape[0], -1)).squeeze()
    else:
        return (dFg * vw).sum()

def apply_fwd_grad_reduce_batch(dFg, vw):
    return dFg.sum() * (vw.mean(dim=0) if vw.ndim == 3 else vw)

def apply_fwd_grad_reduce_batch_w_biases(dFg, vw):
    return dFg.sum() * vw.mean(dim=0)

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

def create_new_Vs_mage_guess(x_t, h_part, rnn, guess, alpha, t, j, device, epsilon, with_batch=False, with_batch_for_biases=False):
    # W_ii, W_if, W_ig, W_io = split_by_4(rnn.__getattr__(f"weight_ih_l{j}"))
    # W_hi, W_hf, W_hg, W_ho = split_by_4(rnn.__getattr__(f"weight_hh_l{j}"))
    # b_ii, b_if, b_ig, b_io = split_by_4(rnn.__getattr__(f"bias_ih_l{j}"))
    # b_hi, b_hf, b_hg, b_ho = split_by_4(rnn.__getattr__(f"bias_hh_l{j}"))

    # get_shape = lambda w: (w.shape[0], 1) if not with_batch else (x_t.shape[0], w.shape[0], 1)
    # get_shape_bias = lambda b: (x_t.shape[0], b.shape[0]) if with_batch and with_batch_for_biases else b.shape

    

    if False:
        z = guess[t]
        I = torch.eye(z.shape[1], device = device)
        Mu = torch.zeros([z.shape[1]],device = device)

        ##breakpoint()
        g = torch.cat([MultivariateNormal(Mu, I + alpha* torch.matmul(z[i:i+1,:].t(),z[i:i+1,:])).sample().unsqueeze(0) for i in range(z.shape[0]) ],dim = 0).unsqueeze(2)
    elif False:
        z = guess[t]
        I = torch.eye(z.shape[1], device = device)
        Mu = torch.zeros([z.shape[1]],device = device)
        
        g = MultivariateNormal(Mu, I + alpha* torch.matmul(z.t(),z)).sample((x_t.shape[0],)).unsqueeze(2)
    elif True:
        z = guess[t]

        g = rnn.rand_rand * z.unsqueeze(2)

        # g = guess[t].unsqueeze(2)
    else:
        z = guess[t].unsqueeze(1)

        if alpha > 0.0:
            ##breakpoint()

            L = decompose(z,alpha )



            v = torch.randn([L.shape[0],L.shape[1],1], device=device)

            g= L @ v
            


        else:
            g = guess[t].unsqueeze(2)

        ang = angle(z,g).abs().mean()
        ##wandb.log({f"Angle_{t}": ang })

        ##breakpoint()




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

def create_new_Vs_mage(x_t, h_part, rnn, j, device, epsilon, with_batch=False, with_batch_for_biases=False):
    W_ii, W_if, W_ig, W_io = split_by_4(rnn.__getattr__(f"weight_ih_l{j}"))
    # W_hi, W_hf, W_hg, W_ho = split_by_4(rnn.__getattr__(f"weight_hh_l{j}"))
    # b_ii, b_if, b_ig, b_io = split_by_4(rnn.__getattr__(f"bias_ih_l{j}"))
    # b_hi, b_hf, b_hg, b_ho = split_by_4(rnn.__getattr__(f"bias_hh_l{j}"))

    get_shape = lambda w: (x_t.shape[0], w.shape[0], 1)
    get_shape_bias = lambda b: (x_t.shape[0], b.shape[0]) if with_batch and with_batch_for_biases else b.shape

    # pw_ii = torch.randn(get_shape(W_ii), device=device) * epsilon
    # pw_if = torch.randn(get_shape(W_if), device=device) * epsilon
    # pw_ig = torch.randn(get_shape(W_ig), device=device) * epsilon
    # pw_io = torch.randn(get_shape(W_io), device=device) * epsilon
    # pw_hi = torch.randn(get_shape(W_hi), device=device) * epsilon
    # pw_hf = torch.randn(get_shape(W_hf), device=device) * epsilon
    # pw_hg = torch.randn(get_shape(W_hg), device=device) * epsilon
    # pw_ho = torch.randn(get_shape(W_ho), device=device) * epsilon
    # _vb_ii = torch.randn(get_shape_bias(b_ii), device=device) * epsilon
    # _vb_if = torch.randn(get_shape_bias(b_if), device=device) * epsilon
    # _vb_ig = torch.randn(get_shape_bias(b_ig), device=device) * epsilon
    # _vb_io = torch.randn(get_shape_bias(b_io), device=device) * epsilon
    # _vb_hi = torch.randn(get_shape_bias(b_hi), device=device) * epsilon
    # _vb_hf = torch.randn(get_shape_bias(b_hf), device=device) * epsilon
    # _vb_hg = torch.randn(get_shape_bias(b_hg), device=device) * epsilon
    # _vb_ho = torch.randn(get_shape_bias(b_ho), device=device) * epsilon

    g0 = torch.randn(get_shape(W_ii), device=device) * epsilon
    g1 = torch.randn(get_shape(W_if), device=device) * epsilon
    g2 = torch.randn(get_shape(W_ig), device=device) * epsilon
    g3 = torch.randn(get_shape(W_io), device=device) * epsilon

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

def create_new_Vs_mage2(x_t, h_part, rnn, j, device, epsilon, with_batch=False, with_batch_for_biases=False):
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

# def create_new_Vs_mage_learn_sigmas(x_t, h_part, rnn, j, device, epsilon, A, with_batch=False):
#     W_ii, W_if, W_ig, W_io = split_by_4(rnn.__getattr__(f"weight_ih_l{j}"))
#     W_hi, W_hf, W_hg, W_ho = split_by_4(rnn.__getattr__(f"weight_hh_l{j}"))
#     b_ii, b_if, b_ig, b_io = split_by_4(rnn.__getattr__(f"bias_ih_l{j}"))
#     b_hi, b_hf, b_hg, b_ho = split_by_4(rnn.__getattr__(f"bias_hh_l{j}"))
#     A_ii, A_if, A_ig, A_io = split_by_4(A[f"weight_ih_l{j}"])
#     A_hi, A_hf, A_hg, A_ho = split_by_4(A[f"weight_hh_l{j}"])
#
#     get_shape = lambda w : (w.shape[0], 1) if not with_batch else (x_t.shape[0], w.shape[0], 1)
#
#     pw_ii = A_ii.unsqueeze(0) @ torch.randn(get_shape(W_ii), device=device) * epsilon
#     pw_if = A_if.unsqueeze(0) @ torch.randn(get_shape(W_if), device=device) * epsilon
#     pw_ig = A_ig.unsqueeze(0) @ torch.randn(get_shape(W_ig), device=device) * epsilon
#     pw_io = A_io.unsqueeze(0) @ torch.randn(get_shape(W_io), device=device) * epsilon
#     pw_hi = A_hi.unsqueeze(0) @ torch.randn(get_shape(W_hi), device=device) * epsilon
#     pw_hf = A_hf.unsqueeze(0) @ torch.randn(get_shape(W_hf), device=device) * epsilon
#     pw_hg = A_hg.unsqueeze(0) @ torch.randn(get_shape(W_hg), device=device) * epsilon
#     pw_ho = A_ho.unsqueeze(0) @ torch.randn(get_shape(W_ho), device=device) * epsilon
#     _vb_ii = torch.randn(b_ii.shape, device=device) * epsilon
#     _vb_if = torch.randn(b_if.shape, device=device) * epsilon
#     _vb_ig = torch.randn(b_ig.shape, device=device) * epsilon
#     _vb_io = torch.randn(b_io.shape, device=device) * epsilon
#     _vb_hi = torch.randn(b_hi.shape, device=device) * epsilon
#     _vb_hf = torch.randn(b_hf.shape, device=device) * epsilon
#     _vb_hg = torch.randn(b_hg.shape, device=device) * epsilon
#     _vb_ho = torch.randn(b_ho.shape, device=device) * epsilon
#
#     _vw_ii = torch.matmul(pw_ii, normalize(x_t).unsqueeze(1))
#     _vw_hi = torch.matmul(pw_hi, normalize(h_part).unsqueeze(1))
#     _vw_if = torch.matmul(pw_if, normalize(x_t).unsqueeze(1))
#     _vw_hf = torch.matmul(pw_hf, normalize(h_part).unsqueeze(1))
#     _vw_ig = torch.matmul(pw_ig, normalize(x_t).unsqueeze(1))
#     _vw_hg = torch.matmul(pw_hg, normalize(h_part).unsqueeze(1))
#     _vw_io = torch.matmul(pw_io, normalize(x_t).unsqueeze(1))
#     _vw_ho = torch.matmul(pw_ho, normalize(h_part).unsqueeze(1))
#
#     _vw_i = (_vw_ii, _vw_if, _vw_ig, _vw_io)
#     _vw_h = (_vw_hi, _vw_hf, _vw_hg, _vw_ho)
#     _vb_i = (_vb_ii, _vb_if, _vb_ig, _vb_io)
#     _vb_h = (_vb_hi, _vb_hf, _vb_hg, _vb_ho)
#
#     return _vw_i, _vw_h, _vb_i, _vb_h


def create_new_Vs_mage_binary(x_t, h_part, rnn, j, device, epsilon, with_batch=False):
    W_ii, W_if, W_ig, W_io = split_by_4(rnn.__getattr__(f"weight_ih_l{j}"))
    W_hi, W_hf, W_hg, W_ho = split_by_4(rnn.__getattr__(f"weight_hh_l{j}"))
    b_ii, b_if, b_ig, b_io = split_by_4(rnn.__getattr__(f"bias_ih_l{j}"))
    b_hi, b_hf, b_hg, b_ho = split_by_4(rnn.__getattr__(f"bias_hh_l{j}"))

    get_shape = lambda w : (w.shape[0], 1) if not with_batch else (x_t.shape[0], w.shape[0], 1)

    pw_ii = torch.randint(0, 2, get_shape(W_ii), dtype=torch.float32, device=device) * 2 - 1
    pw_if = torch.randint(0, 2, get_shape(W_if), dtype=torch.float32, device=device) * 2 - 1
    pw_ig = torch.randint(0, 2, get_shape(W_ig), dtype=torch.float32, device=device) * 2 - 1
    pw_io = torch.randint(0, 2, get_shape(W_io), dtype=torch.float32, device=device) * 2 - 1
    pw_hi = torch.randint(0, 2, get_shape(W_hi), dtype=torch.float32, device=device) * 2 - 1
    pw_hf = torch.randint(0, 2, get_shape(W_hf), dtype=torch.float32, device=device) * 2 - 1
    pw_hg = torch.randint(0, 2, get_shape(W_hg), dtype=torch.float32, device=device) * 2 - 1
    pw_ho = torch.randint(0, 2, get_shape(W_ho), dtype=torch.float32, device=device) * 2 - 1
    _vb_ii = torch.randint(0, 2, b_ii.shape, dtype=torch.float32, device=device) * 2 - 1
    _vb_if = torch.randint(0, 2, b_if.shape, dtype=torch.float32, device=device) * 2 - 1
    _vb_ig = torch.randint(0, 2, b_ig.shape, dtype=torch.float32, device=device) * 2 - 1
    _vb_io = torch.randint(0, 2, b_io.shape, dtype=torch.float32, device=device) * 2 - 1
    _vb_hi = torch.randint(0, 2, b_hi.shape, dtype=torch.float32, device=device) * 2 - 1
    _vb_hf = torch.randint(0, 2, b_hf.shape, dtype=torch.float32, device=device) * 2 - 1
    _vb_hg = torch.randint(0, 2, b_hg.shape, dtype=torch.float32, device=device) * 2 - 1
    _vb_ho = torch.randint(0, 2, b_ho.shape, dtype=torch.float32, device=device) * 2 - 1

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

def create_new_Vs_mage_no_batch(x_t, h_part, rnn, j, device, epsilon):
    W_ii, W_if, W_ig, W_io = split_by_4(rnn.__getattr__(f"weight_ih_l{j}"))
    W_hi, W_hf, W_hg, W_ho = split_by_4(rnn.__getattr__(f"weight_hh_l{j}"))
    b_ii, b_if, b_ig, b_io = split_by_4(rnn.__getattr__(f"bias_ih_l{j}"))
    b_hi, b_hf, b_hg, b_ho = split_by_4(rnn.__getattr__(f"bias_hh_l{j}"))

    get_shape = lambda w: (x_t.shape[0], w.shape[0], 1)

    pw_ii = torch.randn(get_shape(W_ii), device=device) * epsilon
    pw_if = torch.randn(get_shape(W_if), device=device) * epsilon
    pw_ig = torch.randn(get_shape(W_ig), device=device) * epsilon
    pw_io = torch.randn(get_shape(W_io), device=device) * epsilon
    pw_hi = torch.randn(get_shape(W_hi), device=device) * epsilon
    pw_hf = torch.randn(get_shape(W_hf), device=device) * epsilon
    pw_hg = torch.randn(get_shape(W_hg), device=device) * epsilon
    pw_ho = torch.randn(get_shape(W_ho), device=device) * epsilon
    _vb_ii = torch.randn(b_ii.shape, device=device) * epsilon
    _vb_if = torch.randn(b_if.shape, device=device) * epsilon
    _vb_ig = torch.randn(b_ig.shape, device=device) * epsilon
    _vb_io = torch.randn(b_io.shape, device=device) * epsilon
    _vb_hi = torch.randn(b_hi.shape, device=device) * epsilon
    _vb_hf = torch.randn(b_hf.shape, device=device) * epsilon
    _vb_hg = torch.randn(b_hg.shape, device=device) * epsilon
    _vb_ho = torch.randn(b_ho.shape, device=device) * epsilon

    _vw_ii = torch.matmul(pw_ii, normalize(x_t).unsqueeze(1)).mean(dim=0)
    _vw_hi = torch.matmul(pw_hi, normalize(h_part).unsqueeze(1)).mean(dim=0)
    _vw_if = torch.matmul(pw_if, normalize(x_t).unsqueeze(1)).mean(dim=0)
    _vw_hf = torch.matmul(pw_hf, normalize(h_part).unsqueeze(1)).mean(dim=0)
    _vw_ig = torch.matmul(pw_ig, normalize(x_t).unsqueeze(1)).mean(dim=0)
    _vw_hg = torch.matmul(pw_hg, normalize(h_part).unsqueeze(1)).mean(dim=0)
    _vw_io = torch.matmul(pw_io, normalize(x_t).unsqueeze(1)).mean(dim=0)
    _vw_ho = torch.matmul(pw_ho, normalize(h_part).unsqueeze(1)).mean(dim=0)

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

        self.guess_decoder = []

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
        
        ##output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        
        # output = [sentence len, batch size, hid dim * num directions]

        # Concatenate the final hidden layers (for bidirectional)
        # hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        # breakpoint()
        # hidden = (torch.transpose(hidden[-self.n_directions:], 0, 1)).reshape(
        #     (-1, self.hidden_dim * self.n_directions))

        # Dropout
        hidden = self.drop(hidden)

        def hook_fn_d(grad):
            self.guess_decoder.append(grad.clone())
            return grad

        self.decoder.bias.register_hook(hook_fn_d)

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
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu())

        if isinstance(packed_embedded, PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = packed_embedded
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
        num_directions = 2 if self.rnn.bidirectional else 1
        zeros = torch.zeros(self.rnn.num_layers * num_directions,
                            max_batch_size, self.rnn.hidden_size,
                            dtype=input.dtype, device=input.device)
        hx = (zeros, zeros)
        ##self.rnn.check_forward_args(input, hx, batch_sizes)

        return hx, input, batch_sizes, sorted_indices, unsorted_indices, packed_embedded

    def fwd_mode(self, batch_text, y, loss, mage=False, guess = None, ig = -1, grad_div=1, reduce_batch=False, mage_no_batch=False,
                 random_binary=False, reduce_batch_biases=False, vanilla_biases=False):
        hx, input, batch_sizes, sorted_indices, unsorted_indices, packed_embedded = self.batch_text_to_input(batch_text)
        x = torch.split(input, tuple(batch_sizes))
        device = x[0].device
        self.rnn.rand_rand = torch.randn((1,), device=device)
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

                get_shape = lambda weight: weight.shape if not mage or mage_no_batch else [x[0].shape[0]] + list(weight.shape)
                get_shape_biases = lambda bias: [x[0].shape[0]] + list(bias.shape)
                vw_ii = torch.zeros(get_shape(W_ii)).to(device)
                vw_hi = torch.zeros(get_shape(W_hi)).to(device)
                vw_if = torch.zeros(get_shape(W_if)).to(device)
                vw_hf = torch.zeros(get_shape(W_hf)).to(device)
                vw_ig = torch.zeros(get_shape(W_ig)).to(device)
                vw_hg = torch.zeros(get_shape(W_hg)).to(device)
                vw_io = torch.zeros(get_shape(W_io)).to(device)
                vw_ho = torch.zeros(get_shape(W_ho)).to(device)
                vb_ii = torch.zeros(get_shape_biases(b_ii)).to(device)
                vb_hi = torch.zeros(get_shape_biases(b_hi)).to(device)
                vb_if = torch.zeros(get_shape_biases(b_if)).to(device)
                vb_hf = torch.zeros(get_shape_biases(b_hf)).to(device)
                vb_ig = torch.zeros(get_shape_biases(b_ig)).to(device)
                vb_hg = torch.zeros(get_shape_biases(b_hg)).to(device)
                vb_io = torch.zeros(get_shape_biases(b_io)).to(device)
                vb_ho = torch.zeros(get_shape_biases(b_ho)).to(device)
                dh_t_dW = None
                dc_dW = None
                dc_dh_t_1 = None

                # todo: remove V creation from time step
                if mage:
                    if vanilla_biases:
                        _vw_i, _vw_h, _vb_i, _vb_h = create_new_Vs(self.rnn, j, device, epsilon)
                        vb_ii, vb_if, vb_ig, vb_io = _vb_i
                        vb_hi, vb_hf, vb_hg, vb_ho = _vb_h
                        _vb_ii, _vb_if, _vb_ig, _vb_io = _vb_i
                        _vb_hi, _vb_hf, _vb_hg, _vb_ho = _vb_h
                else:
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
                    h_part = h[:x_t.shape[0]]
                    c_t_1 = c_t_1[:x_t.shape[0]]
                    dz_dW = z_grad_list[seq] if j > 0 else None
                    if mage:
                        if mage_no_batch:
                            _vw_i, _vw_h, _vb_i, _vb_h = create_new_Vs_mage_no_batch(x_t, h_part, self.rnn, j, device, epsilon)
                        else:
                            if guess is not None:
                                _vw_i, _vw_h, _vb_i, _vb_h = create_new_Vs_mage_guess(x_t, h_part, self.rnn, guess, ig, seq , j, device,
                                                                                epsilon, reduce_batch,
                                                                                reduce_batch_biases)
                            elif not random_binary:
                                _vw_i, _vw_h, _vb_i, _vb_h = create_new_Vs_mage(x_t, h_part, self.rnn, j, device,
                                                                                epsilon, reduce_batch,
                                                                                reduce_batch_biases)
                            else:
                                _vw_i, _vw_h, _vb_i, _vb_h = create_new_Vs_mage_binary(x_t, h_part, self.rnn, j, device,
                                                                                epsilon, reduce_batch)
                        _vw_ii, _vw_if, _vw_ig, _vw_io = _vw_i
                        _vw_hi, _vw_hf, _vw_hg, _vw_ho = _vw_h
                        if not vanilla_biases:
                            _vb_ii, _vb_if, _vb_ig, _vb_io = _vb_i
                            _vb_hi, _vb_hf, _vb_hg, _vb_ho = _vb_h

                    i_p = x_t @ W_ii.T + b_ii + h_part @ W_hi.T + b_hi

                    di_p_dW = (x_t.unsqueeze(1) @ torch.transpose(_vw_ii, -1, -2)).squeeze(1) + \
                              (dz_dW @ W_ii.T if dz_dW is not None else 0) + \
                              _vb_ii + \
                              (h_part.unsqueeze(1) @ torch.transpose(_vw_hi, -1, -2)).squeeze(1) + \
                              (dh_t_dW[:x_t.shape[0]] @ W_hi.T if dh_t_dW is not None else 0) + \
                              _vb_hi
                    i = torch.sigmoid(i_p)
                    di_dW = (i * (1 - i)) * di_p_dW

                    f_p = x_t @ W_if.T + b_if + h_part @ W_hf.T + b_hf
                    df_p_dW = (x_t.unsqueeze(1) @ torch.transpose(_vw_if, -1, -2)).squeeze(1) + \
                              (dz_dW @ W_if.T if dz_dW is not None else 0) + \
                              _vb_if + \
                              (h_part.unsqueeze(1) @ torch.transpose(_vw_hf, -1, -2)).squeeze(1) + \
                              (dh_t_dW[:x_t.shape[0]] @ W_hf.T if dh_t_dW is not None else 0) + \
                              _vb_hf
                    f = torch.sigmoid(f_p)
                    df_dW = (f * (1 - f)) * df_p_dW


                    g_p = x_t @ W_ig.T + b_ig + h_part @ W_hg.T + b_hg
                    dg_p_dW = (x_t.unsqueeze(1) @ torch.transpose(_vw_ig, -1, -2)).squeeze(1) + \
                              (dz_dW @ W_ig.T if dz_dW is not None else 0) + \
                              _vb_ig + \
                              (h_part.unsqueeze(1) @ torch.transpose(_vw_hg, -1, -2)).squeeze(1) + \
                              (dh_t_dW[:x_t.shape[0]] @ W_hg.T if dh_t_dW is not None else 0) + \
                              _vb_hg
                    g = torch.tanh(g_p)
                    dg_dW = (1 - g ** 2) * dg_p_dW

                    o_p = x_t @ W_io.T + b_io + h_part @ W_ho.T + b_ho
                    do_p_dW = (x_t.unsqueeze(1) @ torch.transpose(_vw_io, -1, -2)).squeeze(1) + \
                              (dz_dW @ W_io.T if dz_dW is not None else 0) + \
                              _vb_io + \
                              (h_part.unsqueeze(1) @ torch.transpose(_vw_ho, -1, -2)).squeeze(1) + \
                              (dh_t_dW[:x_t.shape[0]] @ W_ho.T if dh_t_dW is not None else 0) + \
                              _vb_ho
                    o = torch.sigmoid(o_p)
                    do_dW = (o * (1 - o)) * do_p_dW

                    c_t = f * c_t_1 + i * g
                    dc_dW = df_dW * c_t_1 + (dc_dW[:x_t.shape[0]] * f if dc_dW is not None else 0) + di_dW * g + dg_dW * i

                    tanh_c_t = torch.tanh(c_t)
                    d_tanh_c_t_dW = dc_dW * (1 - tanh_c_t ** 2)

                    h = o * tanh_c_t
                    dh_t_dW = do_dW * tanh_c_t + d_tanh_c_t_dW * o

                    h_list.append(h)
                    h_full_batch = combine_batch(h, h_full_batch)
                    h_grad_list.append(dh_t_dW)
                    dh_t_dW_full_batch = combine_batch(dh_t_dW, dh_t_dW_full_batch)
                    c_t_1 = c_t

                    # if not guess is None and ig > 0.0:
                    #     transform = batched_sherman_morrison(guess, seq, ig)
                    #     vw_ih2 = torch.cat([_vw_ii, _vw_if, _vw_ig, _vw_io], dim=-2)
                    #     vw_hh2 = torch.cat([_vw_hi, _vw_hf, _vw_hg, _vw_ho], dim=-2)
                    #     vb_ih2 = torch.cat([_vb_ii, _vb_if, _vb_ig, _vb_io], dim=-1).unsqueeze(2)
                    #     vb_hh2 = torch.cat([_vb_hi, _vb_hf, _vb_hg, _vb_ho], dim=-1).unsqueeze(2)
                    #     vw_ih2 = transform @ vw_ih2
                    #     vw_hh2 = transform @ vw_hh2
                    #     vb_ih2 = transform @ vb_ih2
                    #     vb_hh2 = transform @ vb_hh2
                    #     # breakpoint()
                    #
                    #
                    #     _vw_ii, _vw_if, _vw_ig, _vw_io = torch.split(vw_ih2, vw_ih2.shape[1] // 4, dim = 1)
                    #     _vw_hi, _vw_hf, _vw_hg, _vw_ho = torch.split(vw_hh2, vw_ih2.shape[1] // 4, dim = 1)
                    #     _vb_ii, _vb_if, _vb_ig, _vb_io = torch.split(vb_ih2.squeeze(2), vw_ih2.shape[1] // 4, dim = 1)
                    #     _vb_hi, _vb_hf, _vb_hg, _vb_ho = torch.split(vb_hh2.squeeze(2), vw_ih2.shape[1] // 4, dim = 1)
                    if mage and (j, seq) in relevant_Vs:
                        vw_ii += combine_batch(_vw_ii, torch.zeros_like(vw_ii))
                        vw_hi += combine_batch(_vw_hi, torch.zeros_like(vw_hi))
                        vw_if += combine_batch(_vw_if, torch.zeros_like(vw_if))
                        vw_hf += combine_batch(_vw_hf, torch.zeros_like(vw_hf))
                        vw_ig += combine_batch(_vw_ig, torch.zeros_like(vw_ig))
                        vw_hg += combine_batch(_vw_hg, torch.zeros_like(vw_hg))
                        vw_io += combine_batch(_vw_io, torch.zeros_like(vw_io))
                        vw_ho += combine_batch(_vw_ho, torch.zeros_like(vw_ho))

                        vb_ii += combine_batch(_vb_ii, torch.zeros_like(vb_ii))
                        vb_hi += combine_batch(_vb_hi, torch.zeros_like(vb_hi))
                        vb_if += combine_batch(_vb_if, torch.zeros_like(vb_if))
                        vb_hf += combine_batch(_vb_hf, torch.zeros_like(vb_hf))
                        vb_ig += combine_batch(_vb_ig, torch.zeros_like(vb_ig))
                        vb_hg += combine_batch(_vb_hg, torch.zeros_like(vb_hg))
                        vb_io += combine_batch(_vb_io, torch.zeros_like(vb_io))
                        vb_ho += combine_batch(_vb_ho, torch.zeros_like(vb_ho))





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
                norm = torch.sqrt((x_t ** 2).sum(dim=1, keepdim=True) + 1)
                if guess is not None:
                    pw = self.guess_decoder[-1]

                else:
                    get_shape = lambda w: (w.shape[0], 1) if not reduce_batch else (hidden.shape[0], w.shape[0], 1)

                    pw = torch.randn(get_shape(self.decoder.weight), device=device, dtype=torch.float32)
                vw = torch.matmul(pw, (hidden / norm).unsqueeze(1)).unsqueeze(1)
                vb = torch.matmul(pw, (1.0 / norm).unsqueeze(1)).squeeze(-1).squeeze(-1)
            else:
                if random_binary:
                    vw = torch.randint(0, 2, self.decoder.weight.shape, device=device, dtype=torch.float32) * 2 - 1
                else:
                    vw = torch.randn(self.decoder.weight.shape, device=device, dtype=torch.float32) * epsilon
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
            dFg = (dLdout * grad) if mage and not mage_no_batch else (dLdout * grad).sum()
            apply_fwd_grad = apply_fwd_grad_batch if mage else apply_fwd_grad_no_batch
            # if mage and reduce_batch:
            #     apply_fwd_grad = apply_fwd_grad_reduce_batch if not reduce_batch_biases else \
            #         apply_fwd_grad_reduce_batch_w_biases

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

