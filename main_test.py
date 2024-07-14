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

# setup
record_path = "/home/chenze/graduated/thesis/record/MAE"
batch_size = 32
alpha = 5.0
gamma = 2.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eval = "eval-7.6"
parameters = "eval(acc)-Epoch[07]-Loss[0.133272]-Fscore[0.590](Best).pt"


# augmentation
testing = [
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

# dataset
test_pth = "/home/chenze/graduated/thesis/dataset/testing-Large/balance-denoisy"
transform = transforms.Compose(testing)
dataset = EvaluationDataset(test_pth, transform, mode="Both")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# create folder
dst = os.path.join(record_path, eval, f"alpha({alpha})-gamma({gamma})", "test")
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
#=========================================================#

# load weigth
weigth_pth = os.path.join(record_path, eval, f"alpha({alpha})-gamma({gamma})", parameters)
model.load_state_dict(torch.load(weigth_pth))
model = model.to(device)

# criterion
criterion = FocalLoss(gamma=gamma, weights=torch.tensor([1.0, alpha], device=device))

# evaluation
testing = Testing(device, model, dataset, dataloader, criterion)


test_loss, test_acc, test_fscore, _ = testing.test_fn()

# plot confusion matrix
testing.confusion_matrix(pth=dst)

# plot P-R curve
testing.compute_pr_curve(pth=dst)

# plot ROC curve
testing.compute_roc(pth=dst)

# save indicator
testing.save_indicator(os.path.join(dst, "indicator.log"))

print(f"Loss: {test_loss:.6f}")
print(f"Acc: {test_acc*100:.2f}%")
print(f"Precision: {testing.compute_precision():.2f}")
print(f"Recall: {testing.compute_recall():.2f}")
print(f"F-1 socre: {test_fscore:.2f}")





