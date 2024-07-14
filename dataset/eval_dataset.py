import os
import PIL.Image as Image
from torch.utils.data import Dataset

class EvaluationDataset(Dataset):
    def __init__(self, root, transform=None, mode=None):
        """ Create a evaluation dataset  to fine-tune or linear probe
        
        Parameter
        ------------
        mode : str, only for "Left", "Right" or "Both"

        Attribute
        ------------
        self.sides     : list, save selection mode for eyes direction
        self.classes   : dict, save folder name and its label
        """
        self.root = root
        self.transform = transform

        self.img = [] # image path in sequence
        self.lbl = [] # label in sequence

        self.classes = {"0_normal":0, "1_disease":1}
        self.sides = self.__sides(mode)

        for classes in self.classes.keys():
            for side in self.sides:
                path = os.path.join(self.root, classes, side) # Ex: /dataset/fine_tune/0_normal/Left
                
                for filename in os.listdir(path):
                    self.img.append(f"{path}/{filename}")
                    self.lbl.append(self.classes[classes])
    
    def __getitem__(self, index):
        img_pth = self.img[index]
        lbl = self.lbl[index]
        
        img = Image.open(img_pth)
        img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)
        
        return img, lbl
    
    def __len__(self):
        return len(self.lbl)
    
    def __sides(self, mode):
        """ decide both or single eyes """
        if (mode == "Left") or (mode == "Right"):
            return [mode]
        else:
            return ["Left", "Right"]

if __name__ == "__main__":
    from torchvision import transforms
    from torch.utils.data import DataLoader

    root = "/home/chenze/graduated/thesis/dataset/testing-Large/balance-denoisy"
    augmentation = [
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    transform = transforms.Compose(augmentation)
    
    dataset = EvaluationDataset(root=root, transform=transform, mode="Both")
    
    for (img, lbl) in DataLoader(dataset=dataset, batch_size=128):
        print(type(img))