import torch


class CustomBCE(torch.nn.Module):
    def __init__(self, pos_weight=None, SOM=False, top_k=0):
        super(CustomBCE, self).__init__()
        if pos_weight != None:
            self.pos_weight=torch.unsqueeze(torch.unsqueeze(pos_weight,1),1)
        else:
            self.pos_weight=pos_weight
        self.SOM = SOM
        self.top_k = top_k
        self.bce = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=self.pos_weight)
        #self.bce = torch.nn.BCELoss(reduction="none")

    def forward(self, predictions, targets, batch_idx):
        mask = targets[0,0]==0
        loss = self.bce(predictions, targets[:, 1:].float())
        #loss[:, 1, :, :] *= self.pos_weight[1]
        loss = torch.masked_select(loss, mask)
        if self.SOM:
            loss = torch.sort(loss, descending=True)[0][:self.top_k]

        return torch.mean(loss)

    def __str__(self):
        if self.SOM:
            return "BCE_SOM_top_k="+str(self.top_k)+("_pos_weight="+str(torch.squeeze(torch.squeeze(self.pos_weight,1),1)) if self.pos_weight != None else "")
        else:
            return "BCE"+("_pos_weight="+str(torch.squeeze(torch.squeeze(self.pos_weight,1),1)) if self.pos_weight != None else "")


