import torch
import DL_junyi, DL_mooper
from model import MGCD
from model import MGCD_NCD

import params

if params.dataset == 'data/junyi/':
    src, tgt = DL_junyi.CD_DL()
else:
    src, tgt = DL_mooper.CD_DL()
device = 'cuda:0'


def MGCD_main():
    cdm = MGCD.MGCD(params.un, params.en, params.kn, params.latent_dim)
    e, auc, acc, rmse = cdm.train(train_data=src, test_data=tgt, epoch=20, device=device, lr=0.001)
    with open('result/MGCD.txt', 'a', encoding='utf8') as f:
        f.write('epoch= %d, accuracy= %f, auc= %f\n' % (e, acc, auc))

def MGCD_eval():
    cdm = MGCD.MGCD(params.un, params.en, params.kn, params.latent_dim)
    cdm.load("params/MGCD.params")
    e, auc, acc, rmse = cdm.eval(test_data=tgt, device=device)


def MGCD_NCD_main():
    cdm = MGCD_NCD.MGCD(params.un, params.en, params.kn, params.latent_dim)
    e, auc, acc, rmse = cdm.train(train_data=src, test_data=tgt, epoch=200, device=device, lr=0.002)
    with open('result/MGCD.txt', 'a', encoding='utf8') as f:
        f.write('epoch= %d, accuracy= %f, auc= %f\n' % (e, acc, auc))

if __name__ == '__main__':
    MGCD_main()
    # MGCD_eval()
    # MGCD_NCD_main()

