#https://github.com/JunMa11/SegLossOdyssey/blob/master/losses_pytorch/dice_loss.py

import torch
import torchvision

class CustomDiceLoss(torch.nn.Module):
    def __init__(self, pos_weight=None, SOM=False, top_k=0, epsilon=1e-6):
        super(CustomDiceLoss, self).__init__()
        self.SOM = SOM
        self.top_k = top_k
        self.epsilon = epsilon

    def forward(self, predictions, targets, batch_idx):
        mask = targets[0,0]==0
        predictions = torch.nn.Sigmoid()(predictions)
        targets = targets[:, 1:]
        loss = torch.zeros((1,)).to('cuda')

        for c in range(predictions.shape[1]):
            input_flat = torch.masked_select(predictions[:, c, :, :], mask)
            target_flat = torch.masked_select(targets[:, c, :, :], mask)

            intersection = (input_flat * target_flat).sum()
            union = input_flat.sum() + target_flat.sum()
            dice_score = (2. * intersection + self.epsilon) / (union + self.epsilon)

            loss += dice_score

        loss /= predictions.shape[1]
        #if self.SOM:
            #loss = torch.sort(loss, descending=True)[0][:self.top_k]

        return loss

    def __str__(self):
        if self.SOM:
            return "DiceLoss_seuil_SOM_top_k="+str(self.top_k)
        else:
            return "DiceLoss_seuil"
