#!/usr/bin/python3


# %%

import random
import os
import time
import warnings
import copy
import pickle
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchtext import data
from torchtext import datasets
from tqdm import tqdm

import wandb

import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import os

from torch import nn
from torch.nn import LSTM
from copytask import dataloader
from model_LSTM_for_copy import RNN




parser = argparse.ArgumentParser(description='PyTorch Mage Sentiment Analysis')
parser.add_argument('--epochs', default=500, type=int,
					help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64, type=int,
					help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
					help='initial learning rate')

parser.add_argument('--use-mage', action='store_true', default=False,
					help='Use mage')
parser.add_argument('--use-fwd', action='store_true', default=False,
					help='Use Forward mode')

parser.add_argument('-d', '--num-directions', default=10, type=int,
					help='number of directions to average on in forward mode (default: 10)')

parser.add_argument('--rnn-type', type=str, default="LSTM", help='Type of RNN (default: LSTM)')

parser.add_argument('--name', type=str, default="", help='network name')

parser.add_argument('--scheduler', type=str, default="", help='scheduler')

parser.add_argument('--vec-dim', default=8, type=int,
					help='bit vector size')

parser.add_argument('--hidden-dim', default=64, type=int,
					help='hidden dimension')

parser.add_argument('-L', '--num-layers', default=1, type=int,
					help='number of layers')

parser.add_argument('--bidirectional', action='store_true', default=False,
					help='bidirectional RNN')

parser.add_argument('--weight-decay', '--wd', default=0, type=float,
					help='weight decay (default: 0)')

parser.add_argument('--droprate', default=0.0, type=float,
					help='rate for dropout')


parser.add_argument('--no-wandb', action='store_true', default=False,
					help='disables W&B logging')

parser.add_argument('--save-corr', action='store_true', default=False,
					help='save correlation between output of the LSTM for different time steps')

parser.add_argument('--reduce-batch', action='store_true', default=False,
					help='average on the batch dimension on Mage')

parser.add_argument('--reduce-batch-biases', action='store_true', default=False,
					help='average on the batch dimension on Mage for biases as well')

parser.add_argument('--batched-mage', action='store_true', default=False,
					help='average on the batch dimension on activations for Mage')

parser.add_argument('--binary', action='store_true', default=False,
					help='use binary random instead of gaussian')

parser.add_argument('--vanilla-biases', action='store_true', default=False,
					help='use the same random for all time steps for the biases')

parser.add_argument('--random-t-separately', action='store_true', default=False,
					help='use different random to decorrelate t')

parser.add_argument('--fwd-V-per-timestep', action='store_true', default=False,
					help='use a different V for each timestep')

parser.add_argument('--g-with-batch', action='store_true', default=False,
					help='add batch dimension to the gs')

parser.add_argument('--gpu', default=0, type=int,
					help='which GPU to use')

parser.add_argument('-t', '--trunc', default=-1, type=int,
					help='which GPU to use')

parser.add_argument('--pretrained', default="", type=str,
					help='pretrained model')

parser.add_argument('--ig', default=-1.0, type=float,
					help='weight decay (default: 0)')

args = parser.parse_args()

args.use_ig = args.ig > 0

if args.use_ig:
	args.use_mage = True

if args.use_mage:
	args.use_fwd = True

RNN_TYPE = args.rnn_type
HIDDEN_DIM = args.hidden_dim
N_LAYERS = args.num_layers
BIDIRECTIONAL = args.bidirectional
WEIGHT_DECAY = args.weight_decay
DROPOUT = args.droprate
SAVE_CORR = args.save_corr
BATCH_SIZE = args.batch_size
G_REC = None
N_EPOCHS = args.epochs
VEC_DIM = args.vec_dim

stop_at = 0.0080  # End training at required loss

seq_len = 20 # Change this to change the sequence length (Kept at 20 for initial training)
bits=VEC_DIM # The actual vector size for copying task
in_bits = bits+2 # The extra side track
out_bits = bits

grad_clip=10
act_seq_len = (seq_len*2)+2 # Actual sequence lenght which includes the delimiters (Start and Stop bits on the side tracks)





RNN_TYPE = args.rnn_type
HIDDEN_DIM = args.hidden_dim
N_LAYERS = args.num_layers
BIDIRECTIONAL = args.bidirectional
WEIGHT_DECAY = args.weight_decay
DROPOUT = args.droprate
SAVE_CORR = args.save_corr

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data directory for saving
from data_dir import data_dir as model_dir

# Directory for datasets and pretrained word vectors
datasets_dir = os.path.join(model_dir, 'sentiment_analysis/data')


def set_deterministic(seed=42):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.enabled = False


set_deterministic()


# Random seed
SEED = random.randint(0, 9999)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
###########################################################################
# File name for saved model (model.pt)
if G_REC is not None:
    g_str = ''.join(('_g_%.1f' % G_REC).split('.'))
# Embedding
emb_str = '_emb'

# Regularization
dropout_str = ''.join(('_dropout_%.1f' % DROPOUT).split('.'))
weight_decay_str = ''.join(('_weight_decay_%.3f' % WEIGHT_DECAY).split('.'))
# Join
model_name = ("" + '_' + RNN_TYPE.lower()
              + '_nlayers_%d' % N_LAYERS
              + '_nhid_%d' % HIDDEN_DIM
              + (g_str if G_REC is not None else '')
              + dropout_str
              + weight_decay_str
              + '_seed_%d' % SEED)

if not args.name == "":
    model_name = args.name

model_name += '.pt'
SAVE = os.path.join(model_dir, model_name)

print("Will save model to \n  '%s'" % SAVE)

# A 3-layer LSTM
model = RNN(hidden_dim=args.hidden_dim, output_dim=VEC_DIM, n_layers=1,
                 bidirectional=False, dropout=0)

# Save the initial model connectivity
state_dict_init = copy.deepcopy(model.state_dict())

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'\nThe model has {count_parameters(model):,} trainable parameters')

train_length, val_length, test_length = 1000, 100, 100

train_iterator = list(dataloader(num_batches=train_length,
							batch_size=args.batch_size,
							seq_width=VEC_DIM,
							min_len=1,
							max_len=10,
							device=device))

valid_iterator = list(dataloader(num_batches=val_length,
							batch_size=args.batch_size,
							seq_width=VEC_DIM,
							min_len=1,
							max_len=10,
							device=device))

test_iterator = list(dataloader(num_batches=test_length,
						   batch_size=args.batch_size,
						   seq_width=VEC_DIM,
						   min_len=1,
						   max_len=10,
						   device=device))


# Move model to GPU before choosing the optimizer
model = model.to(device)

# %%

# Training
import torch.optim as optim
import torch.nn as nn

# Choose optimizer
# optimizer = optim.SGD(model.parameters(), lr=1e-3)
# optimizer = optim.Adagrad(model.parameters(), lr=0.05)
# optimizer = optim.Adam(model.parameters())

# Learning rate for adam should be scaled by network size!
lr0 = args.lr
lr = lr0 / HIDDEN_DIM
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

if args.scheduler.lower() == "steplr":
	n_phases = 5
	gamma = math.pow(0.1, 1.0 / (n_phases - 1))
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs / n_phases, gamma=gamma)
else:
	scheduler = None

criterion = nn.BCEWithLogitsLoss().to(device)

# Categorical accuracy
def accuracy(preds, y):
	""" Categorical accuracy for multiple classes."""
	correct = (preds >= 0).float().eq(y).reshape(-1)
	return correct.sum() / torch.FloatTensor([y.reshape(-1).shape[0]]).to(device)



def train(model, iterator, optimizer, criterion, length):
	epoch_loss = 0
	epoch_acc = 0
	model.train()
	corr_mats = ([], [])
	input_corr_matrices = []
	output_corr_matrices = []
	for batch in tqdm(iterator, total=length):
		_, x, y = batch
		optimizer.zero_grad()

		if args.use_ig:
			predictions = model(batch.text)
			loss = criterion(predictions, batch.label)
			acc = accuracy(predictions, batch.label)
			loss.backward()

			guess = model.rnn.pop_guess()
			optimizer.zero_grad()

			for _ in range(args.num_directions):
				predictions = model.fwd_mode(batch.text, batch.label, criterion, True, args.num_directions,
											 g_with_batch=args.g_with_batch,
											 reduce_batch=args.reduce_batch,
											 random_binary=args.binary,
											 vanilla_V_per_timestep=args.fwd_V_per_timestep,
											 random_t_separately=args.random_t_separately, guess=guess, ig=args.ig)
			if model.save_correlations:
				input_corr_matrices.append(model.input_correlation_matrix)
				output_corr_matrices.append(model.output_correlation_matrix)

			loss = criterion(predictions, batch.label)
			acc = accuracy(predictions, batch.label)


		elif args.use_fwd:
			for _ in range(args.num_directions):
				predictions = model.fwd_mode(x, y, criterion, args.use_mage, args.num_directions,
											 g_with_batch=args.g_with_batch,
											 reduce_batch=args.reduce_batch,
											 random_binary=args.binary,
											 vanilla_V_per_timestep=args.fwd_V_per_timestep,
											 random_t_separately=args.random_t_separately,
											 guess=None, ig=-1.0)
			predictions, y = predictions.reshape(-1, VEC_DIM), y.reshape(-1, VEC_DIM)
			loss = criterion(predictions, y)
			acc = accuracy(predictions, y)

		else:
			predictions = model(x)
			predictions, y = predictions.reshape(-1, VEC_DIM), y.reshape(-1, VEC_DIM)
			loss = criterion(predictions, y)
			acc = accuracy(predictions, y)

			loss.backward()

		optimizer.step()
		epoch_loss += loss.item()
		epoch_acc += acc.item()

	return epoch_loss / length, epoch_acc / length, corr_mats


def evaluate(model, iterator, criterion, length):
	epoch_loss = 0
	epoch_acc = 0
	model.eval()
	with torch.no_grad():
		for batch in iterator:
			_, x, y = batch
			predictions = model(x)
			predictions, y = predictions.reshape(-1, VEC_DIM), y.reshape(-1, VEC_DIM)
			loss = criterion(predictions, y)
			acc = accuracy(predictions, y)
			epoch_loss += loss.item()
			epoch_acc += acc.item()
	return epoch_loss / length, epoch_acc / length


def epoch_time(start_time, end_time):
	elapsed_time = end_time - start_time
	elapsed_mins = int(elapsed_time / 60)
	elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
	return elapsed_mins, elapsed_secs


# %%

# Run training

if not args.no_wandb:
	wandb.init(project="mage-copy", entity="dl-projects", config={"INPUT_DIM": VEC_DIM})

	wandb.config.update(args)

best_valid_loss = float('inf')
train_losses, valid_losses = np.zeros((2, N_EPOCHS))
loss_acc = np.zeros((N_EPOCHS, 4))

valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, val_length)

print(f'Epoch: 0 | Epoch Time: N/A')
print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

for epoch in range(N_EPOCHS):
	start_time = time.time()
	train_loss, train_acc, corr_mats = train(model, train_iterator, optimizer, criterion, train_length)
	valid_loss, valid_acc = evaluate(model, valid_iterator, criterion,val_length)
	end_time = time.time()
	epoch_mins, epoch_secs = epoch_time(start_time, end_time)
	# Save losses and accuracy
	loss_acc[epoch] = train_loss, valid_loss, train_acc, valid_acc

	if not scheduler is None:
		scheduler.step()

	lr_sample = optimizer.param_groups[0]['lr']
	# # Save the best model so far
	# if valid_loss < best_valid_loss:
	#     best_valid_loss = valid_loss
	#     state_dict_best = copy.deepcopy(model.state_dict())

	show_step = N_EPOCHS // N_EPOCHS

	if not args.no_wandb:
		if SAVE_CORR:
			figs = {}
			for name, corr_mat in zip(["inputs", "outputs"], corr_mats):
				fig, ax = plt.subplots()
				corr_mat = corr_mat.cpu().numpy()
				im = ax.matshow(corr_mat)
				fig.colorbar(im)
				figs[name] = fig

			wandb.log({"Epoch": epoch + 1,
					   "Train Loss": train_loss,
					   "Validation Loss": valid_loss,
					   "Train Acc": train_acc * 100,
					   "Validation Acc": valid_acc * 100,
					   "Learning Rate": lr_sample,
					   f"inputs correlation matrix": figs["inputs"],
					   f"outputs correlation matrix": figs["outputs"]})
		else:
			wandb.log({"Epoch": epoch + 1,
					   "Train Loss": train_loss,
					   "Validation Loss": valid_loss,
					   "Train Acc": train_acc * 100,
					   "Validation Acc": valid_acc * 100,
					   "Learning Rate": lr_sample})

	if (epoch + 1) % show_step == 0 or epoch == 0:
		print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
		print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
		print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
		# Save
		state_dict_final = model.state_dict()
		with open(SAVE, 'wb') as f:
			torch.save({'state_dict_init': state_dict_init,
						'state_dict_final': state_dict_final,
						'loss_acc': loss_acc,
						}, f)

#######################################################################
# Save the initial and last model
state_dict_final = model.state_dict()
with open(SAVE, 'wb') as f:
	torch.save({'state_dict_init': state_dict_init,
				'state_dict_final': state_dict_final,
				'loss_acc': loss_acc,
				}, f)
# print("Saved last model to '%s'" % SAVE)
print("Saved initial and final model to '%s'" % SAVE)
# print("Saved best model to '%s'" % SAVE)

# Save the initial and last model
state_dict_final = model.state_dict()
with open(SAVE, 'wb') as f:
	torch.save({'state_dict_init': state_dict_init,
				'state_dict_final': state_dict_final,
				'loss_acc': loss_acc,
				}, f)
# print("Saved last model to '%s'" % SAVE)
print("Saved initial and final model to '%s'" % SAVE)
