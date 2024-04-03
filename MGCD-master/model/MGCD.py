import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import sys
import logging
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from MultiGrain.Grain_junyi import adjNT,adjTA
from MultiGrain.Grain_junyi import BERT_emb
from pygcn.utils import load_data, accuracy
from pygcn.layers import GraphConvolution


class PCA:
    def __init__(self, Data):
        self.Data = Data

    def __repr__(self):
        return f'PCA({self.Data})'

    @staticmethod
    def Center(Data):
        # Convert to torch Tensor and keep the number of rows and columns
        # t = torch.from_numpy(Data)
        t = Data
        no_rows, no_columns = t.size()
        row_means = torch.mean(t, 1).unsqueeze(1)
        # Expand the matrix in order to have the same shape as X and substract, to center
        for_subtraction = row_means.expand(no_rows, no_columns)
        X = t - for_subtraction  # centered
        return (X)

    @classmethod
    def Decomposition(cls, Data, k):
        # Center the Data using the static method within the class
        X = cls.Center(Data)
        U, S, V = torch.svd(X)
        eigvecs = U.t()[:, :k]  # the first k vectors will be kept
        y = torch.mm(U, eigvecs)

        # Save variables to the class object, the eigenpair and the centered data
        cls.eigenpair = (eigvecs, S)
        cls.data = X
        return (y)


class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)


class Net(nn.Module):
    def __init__(self, stu_n, exer_n, k_n, emb_dim):
        super(Net, self).__init__()
        self.stu_n = stu_n
        self.exer_n = exer_n
        self.k_n = k_n
        self.emb_dim = emb_dim

        self.student_v = nn.Embedding(self.stu_n, self.emb_dim)
        self.exercise_v = nn.Embedding(self.exer_n, self.emb_dim)
        self.knowledge_v = nn.Embedding(self.k_n, self.emb_dim)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.k_index = torch.LongTensor(list(range(self.k_n))).to(self.device)

        self.guess = nn.Linear(self.emb_dim, self.emb_dim)
        self.slip = nn.Linear(384, self.emb_dim)
        self.pass_func = nn.Linear(self.emb_dim, self.emb_dim)
        self.nopass_func = nn.Linear(self.emb_dim, self.emb_dim)

        self.wz = nn.Linear(self.emb_dim+384, self.emb_dim)
        self.wr = nn.Linear(2*self.emb_dim, self.emb_dim)
        self.ww = nn.Linear(2*self.emb_dim, self.emb_dim)

        self.DR1 = nn.Linear(self.emb_dim + 384, self.emb_dim)
        self.DR2 = nn.Linear(2*self.emb_dim + 384, self.emb_dim)
        self.DR_BERT = nn.Linear(384, self.emb_dim)

        self.prednet_full1 = PosLinear(2 * self.emb_dim, self.emb_dim)
        self.prednet_full2 = PosLinear(2 * self.emb_dim, self.emb_dim)
        self.prednet_full3 = PosLinear(self.emb_dim, 1)
        self.prednet_full4 = nn.Linear(2*self.emb_dim, self.emb_dim)
        self.prednet_full5 = nn.Linear(2*self.emb_dim, self.emb_dim)

        self.gc1 = GraphConvolution(emb_dim, emb_dim)
        self.gc2 = GraphConvolution(emb_dim, emb_dim)
        # self.dropout = 0.5

        self.adj1 = adjNT.to(self.device)
        self.adj2 = adjTA.to(self.device)
        self.Bert_emb = torch.Tensor(BERT_emb).to(self.device)
        # self.Bert_emb = torch.Tensor(BERT_emb[:self.k_n]).to(self.device)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, exer_id, kq):
        stu_v = self.student_v(stu_id)
        exer_v = self.exercise_v(exer_id)
        self.k_v = self.knowledge_v(self.k_index)

        x0 = self.k_v
        # x2 = x0
        # x2 = self.DR_BERT(self.Bert_emb)
        x0 = self.aggregation(x0)
        #
        x1 = F.relu(self.gc1(x0, self.adj1))
        x1 = self.aggregation(x1)
        #
        #
        x2 = F.relu(self.gc2(x1, self.adj2))
        x2 = self.aggregation(x2)


        batch_stu_vector = stu_v.repeat(1, x0.shape[0]).reshape(stu_v.shape[0], x0.shape[0], stu_v.shape[1])
        batch_exer_vector = exer_v.repeat(1, x0.shape[0]).reshape(exer_v.shape[0], x0.shape[0], exer_v.shape[1])
        kn_vector = x2.repeat(stu_v.shape[0], 1).reshape(stu_v.shape[0], x0.shape[0], x0.shape[1])

        preference = torch.sigmoid(self.prednet_full4(torch.cat((batch_stu_vector, kn_vector), dim=2)))
        diff = torch.sigmoid(self.prednet_full5(torch.cat((batch_exer_vector, kn_vector), dim=2)))

        input_x = preference - diff
        # input_x = preference

        # ka = self.advantage(input_x[:,:835,:],kq[:,:835])
        # ta = self.advantage(input_x[:,835:835+40,:],kq[:,835:835+40])
        # aa = self.advantage(input_x[:,-8:,:],kq[:,-8:])

        # profvalue = torch.sigmoid(self.prednet_full3(preference)) * kq.unsqueeze(2)  # get scalar pro
        o = self.prednet_full3(input_x)

        ko = self.diag(o[:,:835,:],kq[:,:835])
        to = self.diag(o[:,835:835+40,:],kq[:,835:835+40])
        ao = self.diag(o[:,-8:,:],kq[:,-8:])




        sum_out = torch.sum(o * kq.unsqueeze(2), dim=1)
        count_of_concept = torch.sum(kq, dim=1).unsqueeze(1)
        output = torch.sigmoid(sum_out / count_of_concept)

        # output = (ko+to+ao)/3

        # return torch.squeeze(output), ka, ta, aa
        return torch.squeeze(output), ko, to, ao
        # return output, ko, to, ao
        # return torch.squeeze(output)

    def aggregation(self, x):
        # guess_prob = torch.sigmoid(self.guess(x))  # distribution of guess
        # slip_prob = torch.sigmoid(self.slip(self.Bert_emb))  # distribution of slip
        # gate = guess_prob + slip_prob
        # pass_results = torch.tanh(self.pass_func(self.k_v))
        # no_pass_results = self.nopass_func(x)
        # output = pass_results + gate * no_pass_results

        # zt = torch.sigmoid(self.wz(torch.cat((self.Bert_emb, x), dim=1)))  # add weight
        # rt = torch.sigmoid(self.wr(torch.cat((self.k_v, x), dim=1)))  # forget weight
        # h = torch.tanh(self.ww(torch.cat((rt * self.k_v, x), dim=1)))  # forget information
        # output = (1-zt) * h + zt * h  # add information

        x = torch.cat((x, self.Bert_emb), dim=1)  # BERT加在这里
        output = torch.tanh(self.DR2(torch.cat((x, self.k_v), dim=1)))

        # Bert_emb = torch.sigmoid(self.DR_BERT(self.Bert_emb))
        # output = x*self.Bert_emb*self.k_v

        # # x1 = PCA.Decomposition(x1, self.emb_dim)
        # # x1 = F.dropout(x1, self.dropout, training=self.training)  # 没有self.training吧

        return output

    def diag(self,o,kq):
        sum_out = torch.sum(o * kq.unsqueeze(2), dim=1)
        count_of_concept = torch.sum(kq, dim=1)+1e-6
        output = torch.sigmoid(sum_out / count_of_concept.unsqueeze(1))
        return torch.squeeze(output)

    def advantage(self,input_x,kq):
        sum_out = torch.sum(input_x * kq.unsqueeze(2), dim=1)
        count_of_concept = torch.sum(kq, dim=1)+1e-6
        output = sum_out / count_of_concept.unsqueeze(1)
        return torch.squeeze(output)



class MGCD:
    def __init__(self, student_n, exer_n, k_n, emb_dim):
        super(MGCD, self).__init__()
        self.model = Net(student_n, exer_n, k_n, emb_dim)

    def train(self, train_data, test_data=None, epoch=10, device="cpu", lr=0.002):
        self.model = self.model.to(device)
        self.model.train()
        loss_function = nn.BCELoss()
        MSE = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        best_epoch = 0
        best_auc = 0.
        acc1 = 0.
        best_f1 = 0.
        rmse1 = 1.
        for epoch_i in range(epoch):
            epoch_losses = []
            batch_count = 0
            for batch_data in tqdm(train_data, "Epoch %s" % epoch_i, file=sys.stdout):
                batch_count += 1
                user_id, item_id, kq, y = batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                kq = kq.to(device)
                y: torch.Tensor = y.to(device)

                # pred = self.model(user_id, item_id, kq)
                pred, ko, to, ao = self.model(user_id, item_id, kq)

                loss = 0
                loss = loss + loss_function(pred, y)
                loss1 = loss_function(ko, y)
                loss += 5*loss1
                loss2 = loss_function(to, y)
                loss += 5*loss2
                loss3 = loss_function(ao, y)
                loss += 5*loss3

                # lossKT = MSE(ko,to)
                # loss = loss + lossKT
                # lossTA = MSE(to,ao)
                # loss = loss + lossTA

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.mean().item())

            print("[Epoch %d] average loss: %.6f" % (epoch_i, float(np.mean(epoch_losses))))

            if test_data is not None:
                auc, accuracy, rmse, f1 = self.eval(test_data, device=device)
                # print("[Epoch %d] auc: %.6f, accuracy: %.6f" % (epoch_i, auc, accuracy))
                print("[Epoch %d] auc: %.6f, accuracy: %.6f, rmse: %.6f, f1: %.6f" % (epoch_i, auc, accuracy, rmse, f1))

                # if rmse < rmse1:
                if best_auc < auc:
                    best_epoch = epoch_i
                    best_auc = auc
                    acc1 = accuracy
                    rmse1 = rmse
                    best_f1 = f1
                    # self.save("params/MGCD.params") #这里保存**********
            # print('BEST epoch<%d>, auc: %s, acc: %s' % (best_epoch, best_auc, acc1))
            print('BEST epoch<%d>, auc: %s, acc: %s, rmse: %.6f, f1: %.6f' % (best_epoch, best_auc, acc1, rmse1, best_f1))

        return best_epoch, best_auc, acc1, rmse1

    def eval(self, test_data, device="cpu"):
        self.model = self.model.to(device)
        self.model.eval()
        y_true, y_pred = [], []
        rmse = 0.
        for batch_data in tqdm(test_data, "Evaluating", file=sys.stdout):
            user_id, item_id, kq, y = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            kq = kq.to(device)
            # pred = self.model(user_id, item_id, kq)
            pred, ko, to, ao = self.model(user_id, item_id, kq)
            # pred = ko
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())
        rmse = np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))
        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5), rmse, f1_score(y_true,
                                                                                                              np.array(
                                                                                                                  y_pred) >= 0.5)

    def save(self, filepath):
        torch.save(self.model.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.model.load_state_dict(torch.load(filepath))  # , map_location=lambda s, loc: s
        logging.info("load parameters from %s" % filepath)

    def get_params(self, device="cpu"):
        u, i, disc = self.model.params(device)
        return u, i, disc

    def advantage(self, test_data, device="cpu"):
        self.model = self.model.to(device)
        self.model.eval()
        label, feature = [], []
        for batch_data in tqdm(test_data, "Get advantage", file=sys.stdout):
            user_id, item_id, kq, y = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            kq = kq.to(device)
            advantage = self.model.advantage(user_id, item_id, kq)

            feature.extend(advantage.detach().cpu().tolist())
            label.extend(y.tolist())
        return feature, label

    def pro_case(self, test_data, device="cpu"):
        self.model = self.model.to(device)
        self.model.eval()
        feature, feature_s = [], []
        for batch_data in tqdm(test_data, "Get some pro", file=sys.stdout):
            user_id, item_id, kq, _ = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            kq = kq.to(device)
            pro, pro_s = self.model.pro_case(user_id, item_id, kq)

            feature.extend(pro.detach().cpu().tolist())
            feature_s.extend(pro_s.detach().cpu().tolist())

        return feature, feature_s

    def diff_case(self, test_data, device="cpu"):
        self.model = self.model.to(device)
        self.model.eval()
        feature_s = []
        for batch_data in tqdm(test_data, "Get some diff", file=sys.stdout):
            user_id, item_id, kq, _ = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            kq = kq.to(device)
            diff_s = self.model.diff_case(user_id, item_id, kq)

            feature_s.extend(diff_s.detach().cpu().tolist())

        return feature_s
