# Source: https://blog.csdn.net/YI_SHU_JIA/article/details/127223374
import os
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn.functional as F
from torchvision import transforms


from dataset.eval_dataset import EvaluationDataset

import torch
import models_vit
from util.pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_
from vit_explain.vit_rollout import VITAttentionRollout

# basic configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 1
eval = "eval-7.1"

# record setting (設定實驗紀錄的儲存路徑與 log 檔)
path = f"/home/chenze/graduated/thesis/record/MAE/{eval}/alpha(5.0)-gamma(2.0)"
record_path = f"/home/chenze/graduated/thesis/record/MAE/{eval}/alpha(5.0)-gamma(2.0)/test"
os.makedirs(record_path, exist_ok=True)

# augmentation
augmentation = [
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

def show_mask_on_image(img, mask, alpha=0.8):
            img = np.float32(img) / 255
            heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            # cam = heatmap + np.float32(img)
            cam = (1 - alpha) * heatmap + alpha * np.float32(img)
            cam = cam / np.max(cam)
            return np.uint8(255 * cam)

# dataset
root = "/home/chenze/graduated/thesis/dataset/testing-Large/balance-denoisy/0_normal"
for side in ["Left", "Right"]:
    pth = os.path.join(root, side)

    for filename in os.listdir(pth):
        file_pth = os.path.join(pth, filename)

        transform = transforms.Compose(augmentation)
        # dataset = EvaluationDataset(root, transform, mode="Both")
        # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        img = Image.open(file_pth).convert("RGB")
        img = img.resize((224, 224))
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0)
        img_array = np.array(img)
        img_array = np.float32(img_array)/255

        ####################### load pre-trained model and revised it #####################
        model = models_vit.__dict__['vit_large_patch16'](
            num_classes=1,
            drop_path_rate=0.2,
            global_pool=True,
        )
        checkpoint = torch.load('RETFound_cfp_weights.pth', map_location='cpu')
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        interpolate_pos_embed(model, checkpoint_model)
        msg = model.load_state_dict(checkpoint_model, strict=False)
        assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        trunc_normal_(model.head.weight, std=2e-5)
        ####################################################################################


        # load weight
        weight_pth = os.path.join(path, "eval(acc)-Epoch[19]-Loss[0.105938]-Fscore[0.613](Best).pt")
        param = torch.load(weight_pth)
        model.load_state_dict(param)

        # compute output
        output = model(img_tensor)
        pred = torch.sigmoid(output)

        grad_rollout = VITAttentionRollout(model, discard_ratio=0.9)
        mask = grad_rollout(img_tensor)
        np_img = np.array(img)[:, :, ::-1]
        mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
        heatmap = show_mask_on_image(np_img, mask)

        if pred.item() > 0.5:
            os.makedirs(os.path.join(record_path, "1"), exist_ok=True)
            cv2.imwrite(os.path.join(record_path, "1", filename), heatmap)
        else:
            os.makedirs(os.path.join(record_path, "0"), exist_ok=True)
            cv2.imwrite(os.path.join(record_path, "0", filename), heatmap)

    

