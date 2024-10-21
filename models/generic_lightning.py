import sys, os
sys.path.append(os.path.pardir)

import lightning as L
from metrics.CustomMetrics import CustomMetrics
import torch

class generic_lightning(L.LightningModule):
    def __init__(self, model, optimizer, loss_fn, vizualisation):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.val_metrics = CustomMetrics(threshold=[0.2], num_classes=2)
        self.test_metrics = CustomMetrics(threshold=[0.2], num_classes=2)

        self.example_input_array = torch.rand(4, 3, 512, 512)
        self.vizualisation = vizualisation

    def forward(self, inputs):
        return self.model(inputs)


    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        output = self(inputs)
        loss = self.loss_fn(output, targets, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        output = self(inputs)
        loss = self.loss_fn(output, targets, batch_idx)
        output = self.model.for_inference(output)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.val_metrics.update(output, targets)
        return loss

    def on_validation_epoch_end(self):
        #self.log("val_iou", self.val_metrics.compute()["iou"], on_step=False, on_epoch=True, prog_bar=False)
        self.val_metrics.compute_and_log(self.log, text='val')
        self.val_metrics.reset()
        pass


    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        output = self(inputs)
        loss = self.loss_fn(output, targets, batch_idx)
        output = self.model.for_inference(output)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.test_metrics.update(output, targets)

        tensorboard = self.logger.experiment
        tensorboard.add_image(str(batch_idx), self.vizualisation(inputs, targets, output), 0, dataformats="CHW")

        return loss

    def on_test_epoch_end(self):
        self.test_metrics.compute_and_log(self.log, text='test')
        self.test_metrics.compute_and_log_pr_curve(self.logger.experiment)
        self.test_metrics.reset()


    def configure_optimizers(self):
        return self.optimizer
