import torch
import torch.nn as nn
import numpy as np
from warnings import warn
import time
from tqdm import tqdm
class RNN(nn.Module):
    def __init__(self, dims, noise_std, dt=0.5, 
                 nonlinearity='tanh', readout_nonlinearity=None,
                 g=None, wi_init=None, wrec_init=None, wo_init=None, brec_init=None, h0_init=None,
                 train_wi=False, train_wrec=True, train_wo=False, train_brec=False, train_h0=False, 
                 ML_RNN=False,
                ):
        """
        :param dims: list = [input_size, hidden_size, output_size]
        :param noise_std: float
        :param dt: float, integration time step
        :param nonlinearity: str, nonlinearity. Choose 'tanh' or 'id'
        :param readout_nonlinearity: str, nonlinearity. Choose 'tanh' or 'id'
        :param g: float, std of gaussian distribution for initialization
        :param wi_init: torch tensor of shape (input_dim, hidden_size)
        :param wo_init: torch tensor of shape (hidden_size, output_dim)
        :param wrec_init: torch tensor of shape (hidden_size, hidden_size)
        :param brec_init: torch tensor of shape (hidden_size)
        :param h0_init: torch tensor of shape (hidden_size)
        :param train_wi: bool
        :param train_wo: bool
        :param train_wrec: bool
        :param train_brec: bool
        :param train_h0: bool
        :param ML_RNN: bool; whether forward pass is ML convention f(Wr)
        """
        super(RNN, self).__init__()
        self.dims = dims
        input_size, hidden_size, output_size = dims
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.noise_std = noise_std
        self.dt = dt
        self.train_wi = train_wi
        self.train_wo = train_wo
        self.train_wrec = train_wrec
        self.train_brec = train_brec
        self.train_h0 = train_h0
        self.ML_RNN = ML_RNN
        assert(not ML_RNN)
        # Either set g or choose initial parameters. Otherwise, there's a conflict!
        assert (g is not None) or (wrec_init is not None), "Choose g or initial wrec!"
        if (g is not None) and (wrec_init is not None):
            g_wrec = wrec_init.std() * np.sqrt(hidden_size)
            tol_g = 0.01
            if np.abs(g_wrec - g) > tol_g:
                warn("Nominal g and wrec_init disagree: g = %.2f, g_wrec = %.2f" % (g, g_wrec))
        self.g = g
        
        # Nonlinearity
        if nonlinearity == 'tanh':
            self.nonlinearity = torch.tanh
        elif nonlinearity == 'id':
            self.nonlinearity = lambda x: x
            if g is not None:
                if g > 1:
                    warn("g > 1. For a linear network, we need stable dynamics!")
        elif nonlinearity.lower() == 'relu':
            self.nonlinearity = nn.ReLU()
        elif nonlinearity == 'softplus':
            softplus_scale = 1 # Note that scale 1 is quite far from relu
            self.nonlinearity = lambda x: torch.log(1. + torch.exp(softplus_scale * x)) / softplus_scale
        elif type(nonlinearity) == str:
            raise NotImplementedError("Nonlinearity not yet implemented.")
        else:
            self.nonlinearity = nonlinearity
            
        # Readout nonlinearity
        if readout_nonlinearity is None:
            # Same as recurrent nonlinearity
            self.readout_nonlinearity = self.nonlinearity
        elif readout_nonlinearity == 'tanh':
            self.readout_nonlinearity = torch.tanh
        elif readout_nonlinearity == 'logistic':
            # Note that the range is [0, 1]. otherwise, 'logistic' is a scaled and shifted tanh
            self.readout_nonlinearity = lambda x: 1. / (1. + torch.exp(-x))
        elif readout_nonlinearity == 'id':
            self.readout_nonlinearity = lambda x: x
        elif type(readout_nonlinearity) == str:
            raise NotImplementedError("readout_nonlinearity not yet implemented.")
        else:
            self.readout_nonlinearity = readout_nonlinearity

        # Define parameters
        self.wi = nn.Parameter(torch.Tensor(input_size, hidden_size))
        if not train_wi:
            self.wi.requires_grad = False
        self.wrec = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        if not train_wrec:
            self.wrec.requires_grad = False
        self.wo = nn.Parameter(torch.Tensor(hidden_size, output_size))
        if not train_wo:
            self.wo.requires_grad = False
        self.brec = nn.Parameter(torch.Tensor(hidden_size))
        if not train_brec:
            self.brec.requires_grad = False
        self.h0 = nn.Parameter(torch.Tensor(hidden_size))
        if not train_h0:
            self.h0.requires_grad = False

        # Initialize parameters
        with torch.no_grad():
            if wi_init is None:
                self.wi.normal_()
            else:
                if type(wi_init) == np.ndarray:
                    wi_init = torch.from_numpy(wi_init)
                self.wi.copy_(wi_init)
            if wrec_init is None:
                self.wrec.normal_(std=g / np.sqrt(hidden_size))
            else:
                if type(wrec_init) == np.ndarray:
                    wrec_init = torch.from_numpy(wrec_init)
                self.wrec.copy_(wrec_init)
            if wo_init is None:
                self.wo.normal_(std=1 / hidden_size)
            else:
                if type(wo_init) == np.ndarray:
                    wo_init = torch.from_numpy(wo_init)
                self.wo.copy_(wo_init)
            if brec_init is None:
                self.brec.zero_()
            else:
                if type(brec_init) == np.ndarray:
                    brec_init = torch.from_numpy(brec_init)
                self.brec.copy_(brec_init)
            if h0_init is None:
                self.h0.zero_()
            else:
                if type(h0_init) == np.ndarray:
                    h0_init = torch.from_numpy(h0_init)
                self.h0.copy_(h0_init)
            
            
    def forward(self, input, return_dynamics=False, h_init=None):
        """
        :param input: tensor of shape (batch_size, #timesteps, input_dimension)
        Important: the 3 dimensions need to be present, even if they are of size 1.
        :param return_dynamics: bool
        :return: if return_dynamics=False, output tensor of shape (batch_size, #timesteps, output_dimension)
                 if return_dynamics=True, (output tensor, trajectories tensor of shape (batch_size, #timesteps, #hidden_units))
        """
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        if h_init is None:
            h = self.h0
        else:
            h_init_torch = nn.Parameter(torch.Tensor(batch_size, self.hidden_size))
            h_init_torch.requires_grad = False
            # Initialize parameters
            with torch.no_grad():
                h = h_init_torch.copy_(torch.from_numpy(h_init))
        noise = torch.randn(batch_size, seq_len, self.hidden_size, device=self.wrec.device)
        output = torch.zeros(batch_size, seq_len, self.output_size, device=self.wrec.device)
        if return_dynamics:
            trajectories = torch.zeros(batch_size, seq_len, self.hidden_size, device=self.wrec.device)
        assert(not self.ML_RNN)

        # simulation loop
        for i in range(seq_len):
            if self.ML_RNN:
                rec_input = self.nonlinearity(
                    h.matmul(self.wrec.t()) 
                    + input[:, i, :].matmul(self.wi)
                    + self.brec)
                     # Note that if noise is added inside the nonlinearity, the amplitude should be adapted to the slope...
                     # + np.sqrt(2. / self.dt) * self.noise_std * noise[:, i, :])
                h = ((1 - self.dt) * h 
                     + self.dt * rec_input
                     + np.sqrt(self.dt) * self.noise_std * noise[:, i, :])
                out_i = self.readout_nonlinearity(h.matmul(self.wo))
                
            else:
                rec_input = (
                    self.nonlinearity(h).matmul(self.wrec.t()) 
                    + input[:, i, :].matmul(self.wi) 
                    + self.brec)
                h = ((1 - self.dt) * h 
                     + self.dt * rec_input
                     + np.sqrt(self.dt) * self.noise_std * noise[:, i, :])
                out_i = self.readout_nonlinearity(h).matmul(self.wo)

            output[:, i, :] = out_i

            if return_dynamics:
                trajectories[:, i, :] = h

        if not return_dynamics:
            return output
        else:
            return output, trajectories

    
    def fwd_mode(self,x, loss_fn , mage = False, h_init=None):

        batch_size = x.shape[0]
        seq_len = x.shape[1]
        eps = 1E-6

        device =x.device
        dtype = x.dtype
        V = {}

        if h_init is None:
            h = self.h0
        else:
            h_init_torch = nn.Parameter(torch.Tensor(batch_size, self.hidden_size))
            h_init_torch.requires_grad = False
            # Initialize parameters
            with torch.no_grad():
                h = h_init_torch.copy_(torch.from_numpy(h_init))

        noise = torch.randn(batch_size, seq_len, self.hidden_size, device=self.wrec.device)
        output = torch.zeros(batch_size, seq_len, self.output_size, device=self.wrec.device)
        output_grad = torch.zeros(batch_size, seq_len, self.output_size, device=self.wrec.device)


        assert(not self.ML_RNN)

        h_grad = None

        Wrec = self.wrec
        Wi = self.wi
        Brec = self.brec
        Wo = self.wo

        if mage:
            Vrec = None
            Vi = None
            Vbrec = None
            Vo = None

            V= {}
        else:
            Vrec = torch.randn_like(Wrec)
            Vi = torch.randn_like(Wi)
            Vbrec = torch.randn_like(Brec)
            Vo = torch.randn_like(Wo)

    
        for i in range(seq_len):
        
            if mage:
                with torch.no_grad():
                    a_i = x[:, i, :].matmul(Wi)+ Brec    ##   [B,1] x [1,M]  -> B X M

                    if Vi is None:
                        vn = torch.randn([Wi.shape[1],1],device =device, dtype = dtype)   ## [M,1]   
                        Vbrec = vn.clone().squeeze().expand(Brec.shape)   ## [M]
                        z = x[:, i, :]/(x[:, i, :].norm(dim=1,keepdim = True)+ eps)  ## B X 1
                        Vi = torch.matmul(vn, z.unsqueeze(1)).permute(0,2,1)##   M X 1    mm    B X 1 X 1  -> B X 1 X M
                    
                    ai_grad = torch.matmul(x[:, i, :].unsqueeze(1) , Vi ).squeeze(1) + Vbrec   ## B x 1 x 1   mm B x 1 X M -> B x M

                if not h_grad is None:
                    h_var =torch.autograd.Variable(h.detach(), requires_grad = True)
                    act_h = self.nonlinearity(h_var)
                    rec_grad = torch.autograd.grad(act_h, h_var, h_grad)[0]                    
                    with torch.no_grad():
                        rec_grad = torch.matmul(rec_grad, Wrec.t())

                else:
                    with torch.no_grad():
                        act_h = self.nonlinearity(h)
                        rec_grad = 0

                with torch.no_grad():
                    rec_input = (act_h.matmul(Wrec.t()))  +  a_i
                    act_h = act_h.reshape(batch_size,-1)

                    if Vrec is None:
                        ##import pdb; pdb.set_trace()##broken, need fix  

                        vn = torch.randn([Wrec.shape[0],1],device =device, dtype = dtype) 
                        z = act_h/(act_h.norm(dim=1,keepdim = True)+ eps)  ## B X N 
                        Vrec = torch.matmul(vn,z.unsqueeze(1))

                    rec_grad = rec_grad + torch.matmul(Vrec, act_h.unsqueeze(2)).squeeze(2) + ai_grad 



                    h = ((1 - self.dt) * h + self.dt * rec_input + np.sqrt(self.dt) * self.noise_std * noise[:, i, :])


                    if h_grad is None:
                        h_grad = self.dt * rec_grad
                    else:
                        h_grad = ((1 - self.dt)* h_grad + self.dt * rec_grad)

                h_var =torch.autograd.Variable(h.detach(), requires_grad = True)
                act_h2 = self.readout_nonlinearity(h_var)
                act_h_grad = torch.autograd.grad(act_h2, h_var, h_grad)[0]
                
                with torch.no_grad():
                    out_i = act_h2.matmul(Wo)   ## [1,M] X [M,1] - > [1,1]

                    if Vo is None:
                        vn = torch.randn([Wo.shape[1],1],device =device, dtype = dtype) ## [N,1] 
                        z = act_h2/(act_h2.norm(dim=1,keepdim = True)+ eps)  ## B X N
                        Vo = torch.matmul(vn, z.unsqueeze(1)).permute(0,2,1)
                        
                    
                    ##out_i_grad =  act_h_grad.matmul(Wo) +  torch.matmul(Vo, act_h2.unsqueeze(2)).squeeze(2)
                    out_i_grad = act_h_grad.matmul(Wo) + torch.matmul(act_h2.unsqueeze(1) , Vo ).squeeze(1)

                ##V[i] = (Vrec, Vi, Vbrec, Vo)
                output[:, i, :] = out_i
                output_grad[:, i, :] = out_i_grad



            else:

                with torch.no_grad():
                    a_i = x[:, i, :].matmul(Wi)+ Brec

                    ai_grad = x[:, i, :].matmul(Vi) + Vbrec

                if not h_grad is None:
                    h_var =torch.autograd.Variable(h.detach(), requires_grad = True)
                    act_h = self.nonlinearity(h_var)
                    rec_grad = torch.autograd.grad(act_h, h_var, h_grad)[0]
                    
                    with torch.no_grad():
                        rec_grad = torch.matmul(rec_grad, Wrec.t())

                else:
                    with torch.no_grad():
                        act_h = self.nonlinearity(h)
                        rec_grad = 0

                with torch.no_grad():
                    rec_input = (act_h.matmul(Wrec.t()))  +  a_i

                with torch.no_grad():
                    rec_grad = rec_grad + act_h.matmul(Vrec.t()) + ai_grad
                    h = ((1 - self.dt) * h + self.dt * rec_input + np.sqrt(self.dt) * self.noise_std * noise[:, i, :])


                    if h_grad is None:
                        h_grad = self.dt * rec_grad
                    else:
                        h_grad = ((1 - self.dt)* h_grad + self.dt * rec_grad)

                h_var =torch.autograd.Variable(h.detach(), requires_grad = True)
                act_h = self.readout_nonlinearity(h_var)
                act_h_grad = torch.autograd.grad(act_h, h_var, h_grad)[0]
                
                with torch.no_grad():
                    out_i = act_h.matmul(Wo)
                    out_i_grad = act_h_grad.matmul(Wo) +  act_h.matmul(Vo)

                ##V[i] = (Vrec, Vi, Vbrec, Vo)
                output[:, i, :] = out_i
                output_grad[:, i, :] = out_i_grad

        ##import pdb; pdb.set_trace()

        out =torch.autograd.Variable(output, requires_grad = True)
        out.grad = torch.zeros_like(out)
        L = loss_fn(out)
        
        if torch.isnan(L):
            raise ValueError("Nan")
        
        L.backward()
        dLdout = out.grad
        



        with torch.no_grad():

            if mage:
                dFg = (dLdout*output_grad).sum([1,2]).unsqueeze(1)
            else:
                dFg = (dLdout*output_grad).sum()

            if Wrec.grad is None:
                Wrec.grad = torch.zeros_like(Wrec)
                Wi.grad = torch.zeros_like(Wi)
                Brec.grad = torch.zeros_like(Brec)
                Wo.grad = torch.zeros_like(Wo)

            if mage:
                Wrec.grad +=  torch.matmul(dFg.permute(1,0), Vrec.view(Vrec.shape[0],-1)).view(Vrec.shape[1],Vrec.shape[2])
                Wi.grad +=    torch.matmul(dFg.permute(1,0), Vi.view(Vi.shape[0],-1)).view(Vi.shape[1],Vi.shape[2])
                Brec.grad +=   dFg.sum() * Vbrec
                Wo.grad +=   torch.matmul(dFg.permute(1,0), Vo.view(Vo.shape[0],-1)).view(Vo.shape[1],Vo.shape[2])
            else:
                Wrec.grad +=   dFg* Vrec
                Wi.grad +=   dFg* Vi
                Brec.grad +=   dFg* Vbrec
                Wo.grad +=   dFg* Vo

        return output




    # def fwd_mode2(self,x, y, loss , mage = False, epsilon=1e-5, per_batch = False, normalize_v = False, replicates = 0, resample = True, sparsity = 0.0):

    #     tot_norm = 0
    #     epsilon = 1
    #     eps = 1E-6
    #     batch_size = x.shape[0]
    #     old_grad = None
    #     delta = 1

    #     device =x.device
    #     dtype = x.dtype



    #     with torch.no_grad():
    #         assert(self.actstr == "relu")
    #         src_shape = x.shape
    #         x = x.view(src_shape[0],-1) 
    #         V = {

    #         for i,linop in enumerate(self.linops):
    #             W = linop.weight
    #             B = linop.bias

    #             ##vb = torch.randn(B.shape,device =device, dtype = dtype) *epsilon

    #             if mage:
    #                 ##import pdb; pdb.set_trace()

    #                 if resample:

    #                     vb = torch.randn([batch_size,W.shape[0]],device =device, dtype = dtype) *epsilon 
    #                     if False and sparsity > 0.0:
    #                         vnmask = (torch.rand(vb.shape,device =device, dtype = dtype) >= sparsity).to(torch.float)
    #                         vb = vb*vnmask
    #                     else:
    #                         ##import pdb; pdb.set_trace()
    #                         w_end_idx =  w_start_idx + W.shape[0]
    #                         vnmask = mmapping[:,w_start_idx:w_end_idx]
    #                         w_start_idx = w_end_idx
    #                         vb = vb*vnmask

    #                     if normalize_v:
    #                         z = x/(x.norm(dim=1,keepdim = True)+ eps)  ## B X N
    #                         vw = torch.matmul(vb.unsqueeze(2),z.unsqueeze(1))   ##   B X M X 1    mm    B X 1 X N
    #                     else:
    #                         vw = torch.matmul(vb.unsqueeze(2),x.unsqueeze(1))


    #                 else:
    #                     vn = torch.randn([W.shape[0],1],device =device, dtype = dtype) *epsilon 
    #                     if sparsity > 0.0:
    #                         vnmask = (torch.rand(vn.shape,device =device, dtype = dtype) >= sparsity).to(torch.float)
    #                         vn = vn*vnmask

    #                     vb = vn.clone().squeeze().expand(B.shape)

    #                     if normalize_v:
    #                         z = x/(x.norm(dim=1,keepdim = True)+ eps)
    #                         vw = torch.matmul(vn,z.unsqueeze(1))
    #                     else:
    #                         vw = torch.matmul(vn,x.unsqueeze(1))

    #                 new_grad = torch.matmul(vw, x.unsqueeze(2)).squeeze() + vb   ## B x N1

    #                 if per_batch:
    #                     vw = vw.mean(0)

    #             else:
    #                 vb = torch.randn(B.shape,device =device, dtype = dtype) *epsilon
    #                 vw = torch.randn(W.shape,device =device) *epsilon
    #                 new_grad = F.linear(x, vw) + vb  ## B x 1 X N1

    #             if not old_grad is None:
    #                 old_grad = torch.matmul(old_grad, linop.weight.permute(1,0)) ## B X N0  mm  N0 X N1 -> B X N1
    #             else:
    #                 old_grad = 0
    #             old_grad = old_grad * delta + new_grad


    #             # if not per_batch and mage:
    #             #     tot_norm = tot_norm + vw.norm(1, keepdim = True)**2 + vb.norm()**2
    #             # else:
    #             #     tot_norm = tot_norm + vw.norm()**2 + vb.norm()**2

    #             x = linop(x)

    #             if i < len(self.linops) - 1:
    #                 mask  =  ((x >= 0).to(torch.float))

    #                 old_grad = old_grad * mask ## B X N1
    #                 if not mage:
    #                     maskd3 = mask.unsqueeze(2).expand([batch_size, vw.size(0), vw.size(1) ])
    #                     maskd3 = (mask.sum(0) > 0).to(torch.float).unsqueeze(1)
    #                 else:
    #                     maskd3 = mask.unsqueeze(2).expand( vw.shape)

    #                 vw = vw * maskd3   ##  B X N2 X N1  mm  

    #                 x = self.act(x)
    #             V[i] = (vw, vb)

                


    #     ##import pdb; pdb.set_trace()

    #     if mage:
    #         dLdout = torch.zeros_like(x)
        


    #     out =torch.autograd.Variable(x, requires_grad = True)
    #     out.grad = torch.zeros_like(out)
    #     L = loss(out, y)


    #     L.backward()
    #     ##import pdb; pdb.set_trace()
    #     dLdout = out.grad

    #     ##grad_transfer = dLdout.permute(1, 0) ## Batch x n_classes
    #     ##tot_norm = torch.sqrt(tot_norm)

    #     if not per_batch and mage:
    #         dFg = (dLdout*old_grad).sum(1, keepdim = True)
    #     else:
    #         dFg = (dLdout*old_grad).sum()

    #     ##dFg = dFg * ((dFg >= 0).to(torch.float)) ## DELETE ME, I AM ERROR

    #     ##import pdb; pdb.set_trace()
    #     ##import pdb; pdb.set_trace();

    #     for i in range(len(self.linops)):

    #         linop = self.linops[i]
    #         if linop.weight.grad is None:
    #             linop.weight.grad = torch.zeros_like(linop.weight)
    #             if not linop.bias is None:
    #                 linop.bias.grad = torch.zeros_like(linop.bias)



    #         vw, vb = V[i]
    #         K = len(self.linops) -1 - i
    #         if not per_batch and mage:
    #             linop.weight.grad +=  torch.matmul(dFg.permute(1,0), vw.view(vw.shape[0],-1)).view(vw.shape[1],vw.shape[2]) * (delta**K)   ## 1 x B   mm   Bx(N1xN2)  >   N1 x N2
    #             if not linop.bias is None:
    #                 if resample:
    #                     linop.bias.grad +=  (dFg* vb).sum(0) * (delta**K)
    #                 else:
    #                     linop.bias.grad +=dFg.sum() * vb * (delta**K) 

    #         else:
    #             linop.weight.grad +=   dFg* vw * (delta**K) 
    #             if not linop.bias is None:
    #                 linop.bias.grad +=  dFg * vb  * (delta**K) 



    #     return x


def loss_mse(output, target, mask):
    """
    Mean squared error loss
    :param output: torch tensor of shape (num_trials, num_timesteps, output_dim)
    :param target: idem
    :param mask: idem -- or torch tensor of shape (num_trials, num_timesteps, 1)
    :return: float
    """
    # If mask has the same shape as output:
    if output.shape == mask.shape:
        loss = (mask * (target - output).pow(2)).sum() / mask.sum()
    else:
        raise Exception("This is problematic...")
        output_dim = output.shape[-1]
        loss = (mask * (target - output).pow(2)).sum() / (mask.sum() * output_dim)
    # Take half:
    loss = 0.5 * loss
    return loss

def train(net, task, n_epochs, batch_size=32, learning_rate=1e-2, clip_gradient=None, cuda=False, rec_step=1, 
          optimizer='sgd', h_init=None, verbose=True, fwd_mode = False, mage = False, n_directions = 1):
    """
    Train a network
    :param net: nn.Module
    :param task: function; generates input, target, mask for a single batch
    :param n_epochs: int
    :param batch_size: int
    :param learning_rate: float
    :param clip_gradient: None or float, if not None the value at which gradient norm is clipped
    :param cuda: bool
    :param rec_step: int; record weights after these steps
    :return: res
    """
    # CUDA management
    if cuda:
        if not torch.cuda.is_available():
            print("Warning: CUDA not available on this machine, switching to CPU")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    net.to(device=device)
    
    # Optimizer
    if optimizer == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    elif optimizer == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    else:
        raise Exception("Optimizer not known.")
    
    # Save initial weights
    wi_init = net.wi.detach().cpu().numpy().copy()
    wrec_init = net.wrec.detach().cpu().numpy().copy()
    wo_init = net.wo.detach().cpu().numpy().copy()
    brec_init = net.brec.detach().cpu().numpy().copy()
    weights_init = [wi_init, wrec_init, wo_init, brec_init]
    
    # Record
    dim_rec = net.hidden_size
    dim_in = net.input_size
    dim_out = net.output_size
    n_rec_epochs = n_epochs // rec_step
    
    losses = np.zeros((n_epochs), dtype=np.float32)
    gradient_norm_sqs = np.zeros((n_epochs), dtype=np.float32)
    epochs = np.zeros((n_epochs))
    rec_epochs = np.zeros((n_rec_epochs))
    if net.train_wi:
        wis = np.zeros((n_rec_epochs, dim_in, dim_rec), dtype=np.float32)
    if net.train_wrec:
        wrecs = np.zeros((n_rec_epochs, dim_rec, dim_rec), dtype=np.float32)
    if net.train_wo:
        wos = np.zeros((n_rec_epochs, dim_rec, dim_out), dtype=np.float32)
    if net.train_brec:
        brecs = np.zeros((n_rec_epochs, dim_rec), dtype=np.float32)

    time0 = time.time()
    if verbose:
        print("Training...")
        
    pbar = tqdm(range(n_epochs))
        
    for i in pbar :
        # Save weights (before update)
        if i % rec_step == 0:
            k = i // rec_step
            rec_epochs[k] = i
            if net.train_wi:
                wis[k] = net.wi.detach().cpu().numpy()
            if net.train_wrec:
                wrecs[k] = net.wrec.detach().cpu().numpy()
            if net.train_wo:
                wos[k] = net.wo.detach().cpu().numpy()
            if net.train_brec:
                brecs[k] = net.brec.detach().cpu().numpy()
        
        # Generate batch
        _input, _target, _mask = task(batch_size)
        # Convert training data to pytorch tensors
        _input = torch.from_numpy(_input)
        _target = torch.from_numpy(_target)
        _mask = torch.from_numpy(_mask)
        # Allocate
        input = _input.to(device=device)
        target = _target.to(device=device)
        mask = _mask.to(device=device)
        
        optimizer.zero_grad()

        output = net(input, h_init=h_init)
        loss = loss_mse(output, target, mask)
        if fwd_mode: 
            for _ in range(n_directions):
                output = net.fwd_mode(x = input, loss_fn= lambda x: loss_mse(x, target, mask)/n_directions, mage = mage, h_init=h_init)
        else:
            loss.backward()

        pbar.set_description("loss: %.2e" % loss)


        if clip_gradient is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip_gradient)
        gradient_norm_sq = sum([(p.grad ** 2).sum() for p in net.parameters() if p.requires_grad])
        
        # Update weights
        optimizer.step()
        
        # These 2 lines important to prevent memory leaks
        loss.detach_()
        output.detach_()
        
        # Save
        epochs[i] = i
        losses[i] = loss.item()
        gradient_norm_sqs[i] = gradient_norm_sq
        
        if verbose:
            print("epoch %d / %d:  loss=%.6f, time=%.1f sec." % (i+1, n_epochs, np.mean(losses), time.time() - time0), end="\r")
    if verbose:
        print("\nDone. Training took %.1f sec." % (time.time() - time0))
    
    # Obtain gradient norm
    gradient_norms = np.sqrt(gradient_norm_sqs)
    
    # Final weights
    wi_last = net.wi.detach().cpu().numpy().copy()
    wrec_last = net.wrec.detach().cpu().numpy().copy()
    wo_last = net.wo.detach().cpu().numpy().copy()
    brec_last = net.brec.detach().cpu().numpy().copy()
    weights_last = [wi_last, wrec_last, wo_last, brec_last]
    
    # Weights throughout training: 
    weights_train = {}
    if net.train_wi:
        weights_train["wi"] = wis
    if net.train_wrec:
        weights_train["wrec"] = wrecs
    if net.train_wo:
        weights_train["wo"] = wos
    if net.train_brec:
        weights_train["brec"] = brecs
    
    res = [losses, gradient_norms, weights_init, weights_last, weights_train, epochs, rec_epochs]
    return res

        
def run_net(net, task, batch_size=32, return_dynamics=False, h_init=None):
    # Generate batch
    input, target, mask = task(batch_size)
    # Convert training data to pytorch tensors
    input = torch.from_numpy(input)
    target = torch.from_numpy(target)
    mask = torch.from_numpy(mask)
    with torch.no_grad():
        # Run dynamics
        if return_dynamics:
            output, trajectories = net(input, return_dynamics, h_init=h_init)
        else:
            output = net(input, h_init=h_init)
        loss = loss_mse(output, target, mask)
    res = [input, target, mask, output, loss]
    if return_dynamics:
        res.append(trajectories)
    res = [r.numpy() for r in res]
    return res