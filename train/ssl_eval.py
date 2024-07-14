# In this file, I implement binary classification with binary cross entropy
# It is different from Evaluation and Testing  in ssl_train.py
# which  deal whth binary classification problem by typicall cross entropy 

import time
import os
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import sklearn.metrics as metrics

import torch

class Evaluation():
    def __init__(self, device, model, dataset, dataloader, criterion, optimizer, scheduler):
        self.device = device
        self.model = model
        self.dataset = dataset
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.len = self.dataset.__len__() # length of  evaluation dataset
        self.loss = []                    # evaluation loss
        self.lr = []                      # evaluation learning rate(備用)
        self.acc = []
        self.fscore = []

    def eval_fn(self, epoch):
        since = time.time()
        total_loss = 0
        total_len = 0
        correct = 0

        y_pred = np.array([], dtype=int)
        y_true = np.array([], dtype=int)

        self.model.train()
        for img, lbl in self.dataloader:
            img = img.to(self.device)
            lbl = lbl.to(self.device)         # size = [n, ]
            lbl = torch.unsqueeze(lbl, dim=1) # size = [batch, 1]

            # forward-propagation
            output = self.model(img)          # output size = [batch, 1]
            output = torch.sigmoid(output)

            # compute loss
            loss = self.criterion(output.squeeze(dim=1), lbl.squeeze(dim=1))
            # loss = self.criterion(output, lbl)

            # backward-propagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # compute prediction and correct
            pred = torch.where(output > 0.5, 1, 0) # size = [batch, 1]
            correct += torch.sum(pred == lbl.data)

            # compute y_pred and y_true
            y_pred = np.concatenate([y_pred, pred.squeeze(dim=1).cpu().numpy()])
            y_true = np.concatenate([y_true, lbl.squeeze(dim=1).cpu().numpy()])

            # compute avg loss
            total_loss += loss.item() * img.size(0)
            total_len += img.size(0)
            cur_loss = total_loss / total_len

            # print imformation in batch
            print(f"Epoch: [{epoch}] [{total_len:4d}/{self.len}] | Loss: {cur_loss:.6f}")
        
        # learning rate decay
        self.lr.append(self.scheduler.get_last_lr())
        self.scheduler.step()

        # compute loss, acc, f-score, and time
        epoch_loss = total_loss / total_len
        epoch_acc = correct.item() / self.len
        epoch_fscore = metrics.f1_score(y_true, y_pred)
        epoch_time = int(time.time() - since)

        self.loss.append(epoch_loss)
        self.acc.append(epoch_acc)
        self.fscore.append(epoch_fscore)

        return epoch_loss, epoch_acc, epoch_fscore, epoch_time
     
    def save_checkpoint(self, path: str, phase: str, epoch: int, best_param=None):
        """ save checkpoint or best parameter """
        if best_param:
            dst = os.path.join(path, f"{phase}-Epoch[{(epoch+1):02d}]-Loss[{self.loss[epoch]:.6f}]-Fscore[{self.fscore[epoch]:.3f}](Best).pt")
            torch.save(best_param, dst)
        else:
            dst = os.path.join(path, f"{phase}-Epoch[{(epoch+1):02d}]-Loss[{self.loss[epoch]:.6f}]-Fscore[{self.fscore[epoch]:.3f}].pt")
            param = self.model.state_dict()
            torch.save(param, dst)
    
    def save_log(self, file: str, phase: str, epoch: int, Epoch: int):
        """ save experiment log in epoch
        Parameter
        ------------
        epoch : int, current epoch
        Epoch : int, entire training epoch
        """
        with open(file, mode="a") as f:
            f.write(f"Phase: {phase} | Epoch: {epoch+1}/{Epoch} | Loss: {self.loss[epoch]:.6f} | F-1 score: {self.fscore[epoch]:.3f} | Acc: {self.acc[epoch]:.2f}\n")
    
    def save_fscore(self, file: str):
        """ save entire F-1 socre """
        with open(file, mode="a") as f:
            for fscore in self.fscore:
                f.write(f"{fscore}, ")
    
    def save_loss(self, file: str):
        """ save entire loss """
        with open(file, mode="a") as f:
            for loss in self.loss:
                f.write(f"{loss}, ")
    
    def save_acc(self, file: str):
        """ save entire loss """
        with open(file, mode="a") as f:
            for acc in self.acc:
                f.write(f"{acc}, ")

class Testing(Evaluation):
    def __init__(self, device, model, dataset, dataloader, criterion, optimizer=None, scheduler=None):
        super().__init__(device, model, dataset, dataloader, criterion, optimizer, scheduler)
        #---------- inherit these attributes ----------#
        # self.device = device
        # self.model = model
        # self.dataset = dataset
        # self.dataloader = dataloader
        # self.criterion = criterion
        # self.optimizer = optimizer
        # self.scheduler = scheduler

        # self.len = self.dataset.__len__() # length of  evaluation dataset
        # self.loss = []                    # evaluation loss
        # self.lr = []                      # evaluation learning rate(備用)
        # self.acc = []
        # self.fscore = []
        #----------------------------------------------#
    
    def test_fn(self):
        since = time.time()
        total_loss = 0.0
        total_len = 0
        correct = 0

        self.y_prob = np.array([], dtype=np.float32) # predicted probability
        self.y_pred = np.array([], dtype=int)        # predicted label
        self.y_true = np.array([], dtype=int)        # true label

        self.model.eval()
        with torch.no_grad():
            for img, lbl in self.dataloader:
                img = img.to(self.device)
                lbl = lbl.to(self.device)
                lbl = torch.unsqueeze(lbl, dim=1)

                # forward-propagation
                output = self.model(img) # output size = [batch, 1]
                output = torch.sigmoid(output)

                # compute loss
                loss = self.criterion(output.squeeze(dim=1), lbl.squeeze(dim=1))
                # loss = self.criterion(output, lbl)
                
                # compute prediction and correct
                pred = torch.where(output > 0.5, 1, 0) # size = [batch, 1]
                correct += torch.sum(pred == lbl.data)

                # compute y_pred, y_true and y_prob
                self.y_pred = np.concatenate([self.y_pred, pred.squeeze(dim=1).cpu().numpy()])
                self.y_true = np.concatenate([self.y_true, lbl.squeeze(dim=1).cpu().numpy()])
                self.y_prob = np.concatenate([self.y_prob, output.squeeze(dim=1).cpu().numpy()])

                # print testing information in batch
                total_loss += loss.item() * img.size(0)
                total_len += img.size(0)
                cur_loss = total_loss / total_len
                print(f"Batch: [{total_len:4d}/{self.len}] | Loss: {cur_loss:.6f}")
            
        self.epoch_loss = total_loss / self.len
        self.epoch_acc = correct.item() / self.len
        self.epoch_fscore = metrics.f1_score(self.y_true, self.y_pred)
        epoch_time = int(time.time() - since)

        return self.epoch_loss, self.epoch_acc, self.epoch_fscore, epoch_time
    
    def save_log(self, file: str, phase: str, epoch: int, Epoch: int):
        """ save experiment log in epoch
        Parameter
        ------------
        epoch : int, current epoch
        Epoch : int, entire training epoch
        """
        with open(file, mode="a") as f:
            f.write(f"Phase: {phase} | Epoch: {epoch+1}/{Epoch} | Loss: {self.epoch_loss:.6f} | F-1 score: {self.epoch_fscore:.3f} |  Acc: {self.epoch_acc:.2f}\n")
    
    def save_fscore(self, file: str):
        with open(file, mode="a") as f:
            f.write(f"{self.epoch_fscore}, ")
    
    def save_loss(self, file: str):
        with open(file, mode="a") as f:
            f.write(f"{self.epoch_loss}, ")

    def save_acc(self, file: str):
        with open(file, mode="a") as f:
            f.write(f"{self.epoch_acc}, ")
    
    def confusion_matrix(self, pth):
        self.matrix = metrics.confusion_matrix(self.y_true, self.y_pred)
        ax = sns.heatmap(self.matrix, annot=True, xticklabels=['0', '1'], yticklabels=['0', '1'], cmap="OrRd")
        ax.set_title("Confusion matrix")
        ax.set_xlabel("Predict")
        ax.set_ylabel("True")
        plt.tight_layout()
        plt.savefig(os.path.join(pth, "confusion-matrix.png"))
        plt.clf()
    
    def compute_precision(self):
        return metrics.precision_score(self.y_true, self.y_pred)
    
    def compute_recall(self):
        return metrics.recall_score(self.y_true, self.y_pred)
    
    def compute_pr_curve(self, pth):
        self.precision, self.recall, _ = metrics.precision_recall_curve(self.y_true, self.y_prob)
        self.auprc = metrics.auc(self.recall, self.precision)

        # plot P-R curve
        plt.plot(self.recall, self.precision, label=f"AUC = {self.auprc:.2f}")
        plt.plot([0, 1], [0, 1], color="red", linestyle="--", label="Random classifier")
        plt.title("P-R curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.savefig(os.path.join(pth, "pr-curve.png"))
        plt.clf()
    
    def compute_roc(self, pth):
        self.fpr, self.tpr, _ = metrics.roc_curve(self.y_true, self.y_prob)
        self.auroc = metrics.roc_auc_score(self.y_true, self.y_prob)

        # plot ROC curve
        plt.plot(self.fpr, self.tpr, label=f"AUC = {self.auroc:.2f}")
        plt.plot([0, 1], [0, 1], color="red", linestyle="--", label="Random classifier")
        plt.title("ROC curve")
        plt.legend()
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.savefig(os.path.join(pth, "roc.png"))
        # plt.show()
        plt.clf()
    
    def save_indicator(self, file: str):
        precision = self.compute_precision()
        recall = self.compute_recall()
        
        with open(file, mode="a") as f:
            f.write(f"Loss: {self.epoch_loss:.6f}\n")
            f.write(f"Acc: {self.epoch_acc*100:.2f}%\n")
            f.write(f"Precision: {precision:.2f}\n")
            f.write(f"Recall: {recall:.2f}\n")
            f.write(f"F-1 score: {self.epoch_fscore:.2f}\n")
    





