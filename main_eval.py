import torch
import models_vit
from util.pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_

import os
import math
import time
import matplotlib.pyplot as plt

import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset.eval_dataset import EvaluationDataset

from train.ssl_eval import Evaluation, Testing
from focal_loss.focal_loss import FocalLoss
from train.freeze import freeze
from torchlars import LARS

# setup
record_path = "/home/chenze/graduated/thesis/record/MAE"
batch_size = 32
Epochs = 30
################################
base = 0.001                 # default = 0.001 for AdamW
lr = base * batch_size / 256 # default = 0.000125
# lr = 0.001
weight_decay = 0.05
################################
alpha = 5.0
gamma_search = [1.2, 1.4, 1.6, 1.8, 2.0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# augmentation
evaluation = [
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]
testing = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]

# dataset
eval_pth = "/home/chenze/graduated/thesis/dataset/evaluation-Large/denoisy"
test_pth = "/home/chenze/graduated/thesis/dataset/testing-Large/balance-denoisy"
transform = {
    "eval": transforms.Compose(evaluation),
    "test": transforms.Compose(testing)
}
dataset = {
    "eval":EvaluationDataset(eval_pth, transform["eval"], mode="Both"),
    "test":EvaluationDataset(test_pth, transform["test"], mode="Both")
}
dataloader = {
        "eval":DataLoader(dataset["eval"], batch_size=batch_size, shuffle=True),
        "test":DataLoader(dataset["test"], batch_size=batch_size, shuffle=False),
}

since = time.time()
for gamma in gamma_search:
    # create folder
    dst = os.path.join(record_path, f"alpha({alpha})-gamma({gamma})")
    os.makedirs(dst, exist_ok=True)

    #==================== call the model ====================#
    model = models_vit.__dict__['vit_large_patch16'](
        num_classes = 1,
        drop_path_rate = 0.2, 
        global_pool = True,
    )
    checkpoint = torch.load('/home/chenze/graduated/thesis/RETFound_MAE/RETFound_cfp_weights.pth', map_location='cpu')
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    for k in ["head.weight", "head.bias"]:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
    interpolate_pos_embed(model, checkpoint_model)
    msg = model.load_state_dict(checkpoint_model, strict=False)
    assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
    trunc_normal_(model.head.weight, std=2e-5)
    model = model.to(device)
    #=========================================================#

    # criterion
    criterion = FocalLoss(gamma=gamma, weights=torch.tensor([1.0, alpha], device=device))

    # optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_lambda = lambda epoch: 0.5 * (1 + math.cos(epoch * math.pi / Epochs))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # evaluation
    fine_tune = Evaluation(device, model, dataset["eval"], dataloader["eval"], criterion, optimizer, scheduler)
    testing = Testing(device, model, dataset["test"], dataloader["test"], criterion)

    eval_loss_list = []
    test_loss_list = []
    eval_acc_list = []
    test_acc_list = []
    eval_fscore_list = []
    test_fscore_list = []
    best_loss = float("inf")
    best_fscore = 0
    best_acc = 0
    best_param = {}
    best_epoch = {}

    for epoch in range(Epochs):
        if epoch == 3:
            freeze(model)
        eval_loss, eval_acc, eval_fscore, eval_time = fine_tune.eval_fn(epoch)
        test_loss, test_acc, test_fscore, test_time = testing.test_fn()


        eval_loss_list.append(eval_loss)
        test_loss_list.append(test_loss)
        eval_acc_list.append(eval_acc)
        test_acc_list.append(test_acc)
        eval_fscore_list.append(eval_fscore)
        test_fscore_list.append(test_fscore)

        # find best parameter
        if test_acc > best_acc:
            best_epoch["acc"] = epoch
            best_acc = test_acc
            best_param["acc"] = fine_tune.model.state_dict()
        if test_loss < best_loss:
            best_epoch["loss"] = epoch
            best_loss = test_loss
            best_param["loss"] = fine_tune.model.state_dict()

        # save checkpoint
        # if (epoch+1) % 15 == 0:
        #     fine_tune.save_checkpoint(dst, "Eval", epoch)

        # save record
        fine_tune.save_log(os.path.join(dst, "eval-record.log"), "Eval", epoch, Epochs)
        testing.save_log(os.path.join(dst, "test-record.log"), "Test", epoch, Epochs)

        # print evaluation information for each epoch
        print("=" * 20)
        print(f"Epoch: {epoch+1}/{Epochs} for Gamma = {gamma}, Alpha = {alpha}")
        print(f"Phase: Eval | Loss: {eval_loss:.6f} | F-1 score: {eval_fscore:.4f} | Acc: {eval_acc*100:.3f}% | Times: {eval_time} sec")
        print(f"Phase: Test | Loss: {test_loss:.6f} | F-1 score: {test_fscore:.4f} | Acc: {test_acc*100:.3f}% | Times: {test_time} sec")
        print("=" * 20)

        # save testing loss and fscore
        testing.save_loss(os.path.join(dst, "test-loss.log"))
        testing.save_fscore(os.path.join(dst, "test-fscore.log"))
        testing.save_acc(os.path.join(dst, "test-acc.log"))

    # save best parameter and entire training loss, F-1 score
    fine_tune.save_checkpoint(dst, "eval(acc)", best_epoch["acc"], best_param["acc"]) # for acc
    fine_tune.save_checkpoint(dst, "eval(loss)", best_epoch["loss"], best_param["loss"]) # for loss
    fine_tune.save_loss(os.path.join(dst, "eval-loss.log"))
    fine_tune.save_fscore(os.path.join(dst, "eval-fscore.log"))
    fine_tune.save_acc(os.path.join(dst, "eval-acc.log"))

    # plot loss, learning rate, and f-score
    plt.plot(range(1, Epochs+1), eval_loss_list, label="eval")
    plt.plot(range(1, Epochs+1), test_loss_list, label="test")
    plt.legend()
    plt.title(f"Loss for MAE")
    plt.savefig(os.path.join(dst, "loss.png"))
    plt.clf()

    plt.plot(range(1, Epochs+1), fine_tune.lr, label="Learning rate")
    plt.legend()
    plt.title(f"Learning Rate with Gamma = {gamma}, Alpha = {alpha}")
    plt.savefig(os.path.join(dst, "learning-rate.png"))
    plt.clf()

    plt.plot(range(1, Epochs+1), eval_acc_list, label="eval")
    plt.plot(range(1, Epochs+1), test_acc_list, label="test")
    plt.legend()
    plt.title(f"Acc with Gamma = {gamma}, Alpha = {alpha}")
    plt.savefig(os.path.join(dst, "acc.png"))
    plt.clf()

    plt.plot(range(1, Epochs+1), eval_fscore_list, label="eval")
    plt.plot(range(1, Epochs+1), test_fscore_list, label="test")
    plt.legend()
    plt.title(f"F-1 score with Gamma = {gamma}, Alpha = {alpha}")
    plt.savefig(os.path.join(dst, "fscore.png"))
    plt.clf()

times = int(time.time() - since)
print(f"Times: {times//3600} hr {times//60%60} min {times%60} sec")




