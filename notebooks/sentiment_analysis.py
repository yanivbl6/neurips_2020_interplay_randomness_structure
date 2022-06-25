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

parser.add_argument('--emb-dim', default=50, type=int,
                    help='embedding dimension [50, 100, 200, 300]')

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

parser.add_argument('--train-emb', action='store_true', default=False,
                    help='traing the embedding layer')

parser.add_argument('--no-wandb', action='store_true', default=False,
                    help='disables W&B logging')

parser.add_argument('--save-corr', action='store_true', default=False,
                    help='save correlation between output of the LSTM for different time steps')

parser.add_argument('--reduce-batch', action='store_true', default=False,
                    help='average on the batch dimension on Mage')

parser.add_argument('--batched-mage', action='store_true', default=False,
                    help='average on the batch dimension on activations for Mage')

parser.add_argument('--gpu', default=0, type=int,
                    help='which GPU to use')


args = parser.parse_args()
RNN_TYPE = args.rnn_type
EMB_DIM = args.emb_dim
HIDDEN_DIM = args.hidden_dim
N_LAYERS = args.num_layers
BIDIRECTIONAL = args.bidirectional
WEIGHT_DECAY = args.weight_decay
DROPOUT = args.droprate
TRAIN_EMB = args.train_emb
SAVE_CORR = args.save_corr



os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from model_LSTM import RNN

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

# %%

# Parameters
# TSM: Tai, Socher, Manning, 2015 for SST-2
# Choose dataset
DATASET = ['SST', 'IMDB'][0]
# Choose pretrained embedding? (GloVe)
PRETRAINED_EMB = True  # TSM: T
# Scale pretrained embedding by 1/sqrt(N)
SCALE_EMB = False
# Train the embedding?


# Data set and training parameters
if DATASET == 'SST':
    MAX_VOCAB_SIZE = None  # max: 15431; TSM: None


    BATCH_SIZE = args.batch_size  # TSM: 25 (paper) or 5 (github repo)
    G_REC = None
    N_EPOCHS = args.epochs

    # RNN_TYPE = 'RNN'
    # BATCH_SIZE = 64
    # EMB_DIM = [50, 100, 200, 300][1]
    # HIDDEN_DIM = 1024
    # G_REC = None
    # N_EPOCHS = 200
    # DROPOUT = 0.0

elif DATASET == 'IMDB':
    MAX_VOCAB_SIZE = 25_000  # for IMDB
    BATCH_SIZE = args.batch_size
    N_EPOCHS = args.epochs
    G_REC = None
    assert(False)


# Network parameters

# Initial weight statistics from previous training?
LOAD_WB_STAT = False

# Use binary or multiple labels?
# Use multiple for SST with neutral labels or SST-5: fine-grained sentiments).
BINARY_LABELS = True

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
if PRETRAINED_EMB:
    emb_str += "_pretrained"
else:
    emb_str += "_random"
if SCALE_EMB:
    emb_str += "_scaled"
if TRAIN_EMB:
    emb_str += "_train"
else:
    emb_str += "_fix"
emb_str += '_dim_%d' % EMB_DIM
# Regularization
dropout_str = ''.join(('_dropout_%.1f' % DROPOUT).split('.'))
weight_decay_str = ''.join(('_weight_decay_%.3f' % WEIGHT_DECAY).split('.'))
# Join
model_name = (DATASET.lower()
              + ('_nvocab_' + str(MAX_VOCAB_SIZE)
                 if MAX_VOCAB_SIZE is not None else '')
              + '_' + RNN_TYPE.lower()
              + '_nlayers_%d' % N_LAYERS
              + '_nhid_%d' % HIDDEN_DIM
              + (g_str if G_REC is not None else '')
              + emb_str
              + dropout_str
              + weight_decay_str
              + '_seed_%d' % SEED)
model_name += '.pt'
SAVE = os.path.join(model_dir, model_name)
print("Will save model to \n  '%s'" % SAVE)

# %%

# Define how the dataset is split.
# Tokenize = act of splitting the string into discrete 'tokens'.
# Include length for packed padded sequences
TEXT = data.Field(tokenize='spacy', include_lengths=True)
if BINARY_LABELS:
    LABEL = data.LabelField(dtype=torch.float)
else:
    LABEL = data.LabelField()

# Load dataset (downloads first if not provided).
if DATASET == 'SST':
    train_data, valid_data, test_data = datasets.SST.splits(TEXT, LABEL,
                                                            root=datasets_dir, fine_grained=False,
                                                            filter_pred=lambda ex: ex.label != 'neutral')
elif DATASET == 'IMDB':
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL, root=datasets_dir)
    print('   Loading dataset complete.')
    # Split train into train and validation set
    train_data, valid_data = train_data.split(random_state=random.seed(SEED))
else:
    raise NotImplementedError()

# Understand the dataset
print(f'\nNumber of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')

# Total number of words in the entire dataset (counting doubles)
n_total_words = 0
for ex in train_data.examples:
    n_total_words += len(ex.text)
print(f'\nTotal number of words in training dataset: {n_total_words:,}')
print("Average sentence length: %.1f words." % (n_total_words / len(train_data)))

# One example sentence
ex = train_data.examples[0].text
print("\nExample sentence (len = %d):" % len(ex))
print(' '.join(ex))

# %%


# %%

# Build a vocabulary from training data.
# Limit the max size of vocabulary (otherwise > 100,000).

# # Randomly initialized:
if PRETRAINED_EMB:
    # Pretrained word embeddings (~1 GB to download!)
    TEXT.build_vocab(train_data,
                     max_size=MAX_VOCAB_SIZE,
                     vectors="glove.6B.%dd" % EMB_DIM,
                     vectors_cache=datasets_dir,
                     unk_init=torch.Tensor.normal_)
else:
    TEXT.build_vocab(train_data, max_size=MAX_VOCAB_SIZE)

LABEL.build_vocab(train_data)

# Create iterators over datasets
# The BucketIterator will return batches with examples of similar lengths to minimize padding.
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    sort_within_batch=True,
    device=device)

# For the vocabulary, note that there are 2 extra tokens, for <unk> and <pad>.
print(f"Full length of vocab without capping at "
      + f"MAX_VOCAB_SIZE: {len(TEXT.vocab.freqs.most_common()):,}")
print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab):,}")
print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab):,}")
print("LABELS:", LABEL.vocab.stoi)

assert BINARY_LABELS == (len(LABEL.vocab.stoi) == 2), '\n\nLabels are not binary!\n'

# Print most and least common words
n_show = 10
print("\nFirst %d words of entire vocab:" % n_show)
print(np.array(TEXT.vocab.freqs.most_common(n_show)))
n_show = 5
print("\nLast %d words included in the vocab:" % n_show)
print(np.array(TEXT.vocab.freqs.most_common(MAX_VOCAB_SIZE)[-n_show:]))
print("\nLast %d words of entire vocab:" % n_show)
print(np.array(TEXT.vocab.freqs.most_common()[-n_show:]))

# %%


# %%

# Create RNN instance
INPUT_DIM = len(TEXT.vocab)
if BINARY_LABELS:
    OUTPUT_DIM = 1
else:
    OUTPUT_DIM = len(LABEL.vocab)
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

# Instantiate
model = RNN(RNN_TYPE, INPUT_DIM, EMB_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS,
            BIDIRECTIONAL, DROPOUT, PAD_IDX, TRAIN_EMB, SAVE_CORR)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'\nThe model has {count_parameters(model):,} trainable parameters')

# Embedding
if PRETRAINED_EMB:
    # Apply pretrained embeddings
    pretrained_embeddings = TEXT.vocab.vectors
    model.encoder.weight.data.copy_(pretrained_embeddings)
if SCALE_EMB:
    model.encoder.weight.data *= 1 / np.sqrt(HIDDEN_DIM)
# Initialize <unk> and <pad> to zero:
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
model.encoder.weight.data[UNK_IDX] = torch.zeros(EMB_DIM)
model.encoder.weight.data[PAD_IDX] = torch.zeros(EMB_DIM)
# print(model.embedding.weight.data)

# RNN and decoder
if LOAD_WB_STAT:
    # Choose file
    wb_stat_name = 'sst_emb_pretrained_scaled_fix_dim_300_lstm_nlayers_1_nhid_150_dropout_05_seed_1234'
    # Load and set weights
    wb_stat_file = os.path.join(model_dir, wb_stat_name)
    with open(args.wb_stat_file, 'rb') as f:
        wb_stat = pickle.load(f)
    for key, param in model.named_parameters():
        if 'encoder' in key:
            # Leave as is!
            pass
        elif 'rnn' in key:
            # RNN weights and biases come in blocks of n_rnn * N x N. Separate these blocks.
            for k, key_rnn in enumerate(model.keys_rnn):
                wb_mean, wb_std = wb_stat[key][key_rnn]
                mul = param.shape[0] // model.n_rnn
                sub_w = param[k * mul: (k + 1) * mul]
                sub_w.data.normal_(wb_mean, wb_std)
                print(key, key_rnn, ' ', wb_mean, wb_std)
        elif 'decoder' in key:
            wb_mean, wb_std = wb_stat[key]
            param.data.normal_(wb_mean, wb_std)
            print(key, ' ', wb_mean, wb_std)
elif G_REC is not None:
    # Recurrent weights
    for key, param in model.rnn.named_parameters():
        if 'weight' in key:
            # Input vs. recurrent
            hh_or_ih, layer_str = key.split('_')[1:]
            layer = int(layer_str[1:])
            # Input layer: scale by 1/sqrt(embedding_dim)
            print(hh_or_ih, layer)
            if hh_or_ih == 'ih' and layer == 0:
                param.data.normal_(0, 1. / np.sqrt(model.embedding_dim))
            else:
                # LSTM and GRU cells have combined weights for the effect of the
                # last hidden state on the new state and the gates.
                # We scale only the hidden-to-hidden weights ('hh') by
                # a factor g controlling the radius of the spectrum.
                for k, key_rnn in enumerate(model.keys_rnn):
                    mul = param.shape[0] // model.n_states
                    sub_w = param[k * mul: (k + 1) * mul]
                    if hh_or_ih == 'hh' and key_rnn == 'c':
                        # Scale only the recurrent state weights by g
                        sub_w.data.normal_(0, G_REC / np.sqrt(model.hidden_dim))
                    else:
                        sub_w.data.normal_(0, 1. / np.sqrt(model.hidden_dim))
        else:
            # Biases are set to zero by default.
            pass

        # # Decoder: leave as is...
        # decoder_max = math.sqrt(3 / model.hidden_dim)
        # model.decoder.weight.data.uniform_(-decoder_max, decoder_max)
        # model.decoder.bias.data.zero_()
else:
    # Use standard initialization, but rescale the input weights of layer 0.
    # model.rnn.weight_ih_l0.data.uniform_(0, 1. / np.sqrt(model.embedding_dim))
    a = 1. / np.sqrt(model.embedding_dim)
    model.rnn.weight_ih_l0.data.uniform_(-a, a)

# Save the initial model connectivity
state_dict_init = copy.deepcopy(model.state_dict())

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

if BINARY_LABELS:
    # Binary cross-entropy loss with logits
    criterion = nn.BCEWithLogitsLoss().to(device)


    # Binary accuracy
    def accuracy(preds, y):
        # round predictions to the closest integer
        rounded_preds = torch.round(torch.sigmoid(preds))
        correct = (rounded_preds == y).float()  # convert into float for division
        acc = correct.sum() / len(correct)
        return acc
else:
    # Cross-entropy loss
    criterion = nn.CrossEntropyLoss().to(device)


    # Categorical accuracy
    def accuracy(preds, y):
        """ Categorical accuracy for multiple classes."""
        max_preds = preds.argmax(dim=1, keepdim=True)  # get the index of the max probability
        correct = max_preds.squeeze(1).eq(y)
        return correct.sum() / torch.FloatTensor([y.shape[0]]).to(device)

def corr_mat_w_mismatching_dims(matrices):
    with torch.no_grad():
        ones_like = [torch.ones_like(m) for m in matrices]
        max_shape = max([m.shape[0] for m in matrices])
        matrices = [torch.nn.functional.pad(m, (0, max_shape-m.shape[0], 0, max_shape-m.shape[1])) for m in matrices]
        ones_like = [torch.nn.functional.pad(m, (0, max_shape-m.shape[0], 0, max_shape-m.shape[1])) for m in ones_like]
        stack = torch.stack(matrices, dim=0)
        num_in_each_time_step = torch.sum(torch.stack(ones_like, dim=0), dim=0, keepdim=True)
        corr_mat = torch.sum(stack / num_in_each_time_step, dim=0)
    return corr_mat



def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    corr_mats = ([], [])
    input_corr_matrices = []
    output_corr_matrices = []
    for batch in tqdm(iterator):
        optimizer.zero_grad()
        if args.use_fwd:
            for _ in range(args.num_directions):
                predictions = model.fwd_mode(batch.text, batch.label, criterion, args.use_mage, args.num_directions, args.reduce_batch, args.batched_mage)
            if model.save_correlations:
                input_corr_matrices.append(model.input_correlation_matrix)
                output_corr_matrices.append(model.output_correlation_matrix)
            loss = criterion(predictions, batch.label)
            acc = accuracy(predictions, batch.label)

        else:
            predictions = model(batch.text)
            loss = criterion(predictions, batch.label)
            acc = accuracy(predictions, batch.label)
            loss.backward()

        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    if model.save_correlations:
        corr_mats = (corr_mat_w_mismatching_dims(input_corr_matrices),
                     corr_mat_w_mismatching_dims(output_corr_matrices))
    return epoch_loss / len(iterator), epoch_acc / len(iterator), corr_mats


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text)
            loss = criterion(predictions, batch.label)
            acc = accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# %%

# Run training

if not args.no_wandb:
    wandb.init(project="mage-text", entity="dl-projects", config = {"PAD_IDX": PAD_IDX, "INPUT_DIM": INPUT_DIM} )

    wandb.config.update(args)


best_valid_loss = float('inf')
train_losses, valid_losses = np.zeros((2, N_EPOCHS))
loss_acc = np.zeros((N_EPOCHS, 4))
for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss, train_acc, corr_mats = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    # Save losses and accuracy
    loss_acc[epoch] = train_loss, valid_loss, train_acc, valid_acc

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
                   f"inputs correlation matrix": figs["inputs"],
                   f"outputs correlation matrix": figs["outputs"]})
        else:
            wandb.log({"Epoch": epoch + 1,
                       "Train Loss": train_loss,
                       "Validation Loss": valid_loss,
                       "Train Acc": train_acc * 100,
                       "Validation Acc": valid_acc * 100,})

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



