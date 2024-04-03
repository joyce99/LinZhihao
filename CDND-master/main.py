import torch
from model import cdnd
import params
import dataloader


src, tgt = dataloader.CD_DL()
device = 'cuda:0'


def CDND_main():
    cdm = cdnd.ACD(params.un, params.en, params.kn, params.latent_dim)
    e, auc, acc, rmse = cdm.train(train_data=src, test_data=tgt, epoch=200, device=device, lr=params.lr)
    with open('result/CDND.txt', 'a', encoding='utf8') as f:
        f.write('epoch= %d, accuracy= %f, auc= %f\n' % (e, acc, auc))


if __name__ == '__main__':
    CDND_main()

