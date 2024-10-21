import torch
import numpy as np

class CustomMetrics():
    def __init__(self, threshold=[0.5], num_classes=1):

        #self.tp = torch.zeros(num_classes,len(threshold), dtype=torch.int64)
        #self.fp = torch.zeros(num_classes,len(threshold), dtype=torch.int64)
        #self.fn = torch.zeros(num_classes,len(threshold), dtype=torch.int64)
        #self.tn = torch.zeros(num_classes,len(threshold), dtype=torch.int64)
        self.tp, self.fp, self.fn, self.tn = 0, 0, 0, 0

        self.threshold = threshold
        self.num_classes = num_classes

        self.labels = [torch.tensor([], dtype=torch.uint8).to("cuda") for _ in range(num_classes)]
        self.predictions = [torch.tensor([], dtype=torch.float).to("cuda")  for _ in range(num_classes)]

    def update(self, preds: torch.Tensor, target: torch.Tensor):

        mask = target[0,0]==0

        for c in range(self.num_classes):
            self.labels[c] = torch.concat((self.labels[c], torch.masked_select(target[:, c+1], mask)))
            self.predictions[c] = torch.concat((self.predictions[c], torch.masked_select(preds[:, c], mask)))

            #for i in range(len(self.threshold)):
                #i_preds = torch.masked_select(preds[:,c], mask)>=self.threshold[i]
                #i_target = torch.masked_select(target[:, c+1], mask)>=self.threshold[i]

                #self.tp[c][i] += torch.sum((torch.eq(i_preds,True)*torch.eq(i_target,True)))
                #self.fp[c][i] += torch.sum((torch.eq(i_preds,True)*torch.eq(i_target,False)))
                #self.fn[c][i] += torch.sum((torch.eq(i_preds,False)*torch.eq(i_target,True)))
                #self.tn[c][i] += torch.sum((torch.eq(i_preds,False)*torch.eq(i_target,False)))

        i_preds = torch.masked_select(torch.argmax(preds, dim=1), mask)
        i_targets = torch.masked_select(torch.argmax(target[:, 1:], dim=1), mask)

        self.tp += torch.sum((torch.eq(i_preds,1)*torch.eq(i_targets,1))).item()
        self.fp += torch.sum((torch.eq(i_preds,1)*torch.eq(i_targets,0))).item()
        self.fn += torch.sum((torch.eq(i_preds,0)*torch.eq(i_targets,1))).item()
        self.tn += torch.sum((torch.eq(i_preds,0)*torch.eq(i_targets,0))).item()




    def compute(self):
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp)!=0 else 0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn)!=0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall)!=0 else 0
        iou = self.tp / (self.fp + self.tp + self.fn) if (self.fp + self.tp + self.fn)!=0 else 0
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            #"mean_precision" : torch.mean(precision).item(),
            #"mean_recall" : torch.mean(recall).item(),
            #"mean_f1" : torch.mean(f1).item(),
            #"mean_iou" : torch.mean(iou).item()
        }


    def compute_and_log(self, logger, text=''):
        metrics = self.compute()

        logger(text+'_precision', metrics['precision'])
        logger(text+'_recall', metrics['recall'])
        logger(text+'_f1', metrics['f1'])
        logger(text+'_iou', metrics['iou'])
        #for i in range(len(self.threshold)):
            #for c in range(self.num_classes):
                #logger(text+'_precision'+'_seuil_'+str(self.threshold[i])+'_classes_'+str(c), metrics['precision'][c][i])
                #logger(text+'_recall'+'_seuil_'+str(self.threshold[i])+'_classes_'+str(c), metrics['recall'][c][i])
                #logger(text+'_f1'+'_seuil_'+str(self.threshold[i])+'_classes_'+str(c), metrics['f1'][c][i])
                #logger(text+'_iou'+'_seuil_'+str(self.threshold[i])+'_classes_'+str(c), metrics['iou'][c][i])

            #logger(text+'_mPrecision'+'_seuil_'+str(self.threshold[i]), metrics['mean_precision'])
            #logger(text+'_mRecall'+'_seuil_'+str(self.threshold[i]), metrics['mean_recall'])
            #logger(text+'_mF1'+'_seuil_'+str(self.threshold[i]), metrics['mean_f1'])
            #logger(text+'_mIoU'+'_seuil_'+str(self.threshold[i]), metrics['mean_iou'])



    def compute_and_log_pr_curve(self, tensorboard):
        for c in range(self.num_classes):
            tensorboard.add_pr_curve('pr_curve_'+str(c), self.labels[c], self.predictions[c], 0)


    def reset(self):
        #self.tp = torch.zeros(self.num_classes,len(self.threshold), dtype=torch.int64)
        #self.fp = torch.zeros(self.num_classes,len(self.threshold), dtype=torch.int64)
        #self.fn = torch.zeros(self.num_classes,len(self.threshold), dtype=torch.int64)
        #self.tn = torch.zeros(self.num_classes,len(self.threshold), dtype=torch.int64)
        self.tp, self.fp, self.fn, self.tn = 0, 0, 0, 0
        self.labels = [torch.tensor([], dtype=torch.uint8).to("cuda") for _ in range(self.num_classes)]
        self.predictions = [torch.tensor([], dtype=torch.float).to("cuda")  for _ in range(self.num_classes)]
