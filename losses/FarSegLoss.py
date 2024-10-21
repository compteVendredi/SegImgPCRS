import torch
import numpy as np

class FarSegLoss(torch.nn.Module):
    def __init__(self, SOM=False, top_k=0, constant_z=1, gamma=1):
        super(FarSegLoss, self).__init__()
        self.SOM = SOM
        self.top_k = top_k
        self.bce = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.constant_z = constant_z
        self.gamma = gamma

    def forward(self, predictions, targets, batch_idx):
        mask = targets[0,0]==0

        y = targets[:, 1:].float()
        y_predit = predictions
        zeta = (1+torch.cos(batch_idx/91*torch.tensor(np.pi)))

        loss = ((1/self.constant_z)*(1-y)**self.gamma + zeta*(1-1/self.constant_z*(1-y)**self.gamma)) * self.bce(y_predit, y)

        loss = torch.masked_select(loss, mask)
        if self.SOM:
            loss = torch.sort(loss, descending=True)[0][:self.top_k]
        return torch.mean(loss)

    def __str__(self):
        if self.SOM:
            return "FarSegLoss_SOM_top_k="+str(self.top_k)+"_Z="+str(self.constant_z)+"_gamma="+str(self.gamma)
        else:
            return "FarSegLoss"+"_Z="+str(self.constant_z)+"_gamma="+str(self.gamma)



