import json


# dataset = 'data/mooper/'
dataset = 'data/junyi/'

with open(dataset + 'config.txt') as f:
    f.readline()
    un, en, kn = f.readline().split(',')
    un, en, kn = int(un), int(en), int(kn)

# type = "none"
type = "equal"

if type != "none":
    if dataset=='data/mooper/':
        with open(dataset + 'grain.txt') as f:
            f.readline()
            exn, cn, sn, dn = f.readline().split(',')
            exn, cn, sn, dn = int(exn), int(cn), int(sn), int(dn)
        if type == "equal":
            kn = kn + exn + cn + sn + dn
    elif dataset=='data/junyi/':
        with open(dataset + 'grain.txt') as f:
            f.readline()
            tn, an = f.readline().split(',')
            tn, an = int(tn), int(an)
        if type == "equal":
            kn = kn + tn + an


src = dataset + '0.8/train_d.json'
tgt = dataset + '0.8/val_d.json'
test = tgt

batch_size = 128
lr = 0.002
epoch = 200

# latent_dim = kn
latent_dim = 100
# latent_dim = 384

