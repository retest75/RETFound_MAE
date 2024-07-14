def freeze(model):
    for name, param in model.named_parameters():
        if (name != "head.weight") and (name != "head.bias"):
            param.requires_grad = False