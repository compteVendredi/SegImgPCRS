import torch
import torchvision


class CustomFocalLoss(torch.nn.Module):
    def __init__(self, pos_weight=None, SOM=False, top_k=0, alpha=0.25):
        super(CustomFocalLoss, self).__init__()
        self.SOM = SOM
        self.top_k = top_k
        self.alpha = alpha

    def forward(self, predictions, targets, batch_idx):
        mask = targets[0,0]==0
        loss = torchvision.ops.sigmoid_focal_loss(predictions, targets[:, 1:].float(), reduction="none", alpha=self.alpha)
        loss = torch.masked_select(loss, mask)
        if self.SOM:
            loss = torch.sort(loss, descending=True)[0][:self.top_k]

        return torch.mean(loss)

    def __str__(self):
        if self.SOM:
            return "FocalLoss_alpha="+str(self.alpha)+"_SOM_top_k="+str(self.top_k)
        else:
            return "FocalLoss_alpha="+str(self.alpha)

