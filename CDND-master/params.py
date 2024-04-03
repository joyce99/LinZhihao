import json

dataset = 'data/assist09/'
# dataset = 'data/mooper/'
# dataset = 'data/CSEDM-F/'

batch_size = 128
lr = 0.002
epoch = 50
src = dataset + 'train.json'
tgt = dataset + 'val.json'
test = tgt

with open(dataset + 'config.txt') as f:
    f.readline()
    un, en, kn = f.readline().split(',')
    un, en, kn = int(un), int(en), int(kn)
latent_dim = kn

pass
