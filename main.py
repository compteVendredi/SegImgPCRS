import lightning as L
import torch
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from dataset.BMLQ_Datamodule import BMLQ_Datamodule
from dataset.vizualize_dataset import vizualize, vizualize_preprocess

from models.generic_lightning import generic_lightning
from models.unet_model import UNet
from models.farseg import CustomFarSeg
from models.netc import NetC
from models.deeplabv3 import CustomDeepLabV3
from models.FCN import CustomFCN
from models.pspnet_model import PSPNet
from models.aerialformer_model import AerialFormer

from losses.CustomBCE import CustomBCE
from losses.CustomFocalLoss import CustomFocalLoss
from losses.CustomDiceLoss import CustomDiceLoss
from losses.FarSegLoss import FarSegLoss


max_epochs=200
accelerator="gpu"
save_dir="logs/"
batch_size=4


def create_dataloader(data_train_dir, data_val_test_dir, preprocess_for_resnet, gaussian_ann):
    BMLQ_datamodule_train = BMLQ_Datamodule(data_train_dir, gaussian_ann=gaussian_ann, preprocess_for_resnet=preprocess_for_resnet)
    BMLQ_datamodule_train.prepare_data()

    BMLQ_datamodule_val_test = BMLQ_Datamodule(data_val_test_dir, gaussian_ann=gaussian_ann, preprocess_for_resnet=preprocess_for_resnet)
    BMLQ_datamodule_val_test.prepare_data()

    return BMLQ_datamodule_train.train_dataloader(batch_size=batch_size, num_workers=19),BMLQ_datamodule_val_test.val_dataloader(batch_size=1, num_workers=19),BMLQ_datamodule_val_test.test_dataloader(batch_size=1, num_workers=19)


def train_and_evaluate_one(type_model, optimizer, loss_fn, max_epochs, train_dataloader, val_dataloader, test_dataloader, name, version, vizualisator):
    model = generic_lightning(model=type_model, optimizer=optimizer, loss_fn=loss_fn, vizualisation=vizualisator)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=save_dir, name=name, version=version, log_graph=False)

    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_iou", mode="max")
    trainer = L.Trainer(max_epochs=max_epochs, accelerator=accelerator, logger=tb_logger, log_every_n_steps=1,callbacks=[checkpoint_callback])
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    best_model_path = checkpoint_callback.best_model_path
    #best_model_path = os.path.join(save_dir, name, version)
    model = generic_lightning.load_from_checkpoint(best_model_path, model=type_model, optimizer=optimizer, loss_fn=loss_fn, vizualisation=vizualisator)
    trainer.test(model, dataloaders=test_dataloader)


def train_and_evaluate_all(train_dataloader, val_dataloader, test_dataloader, type_models, loss_fns, max_epochs, vizualisator, folder):
    for i in range(len(type_models)):
        for j in range(len(loss_fns)):
            model = type_models[i][0](*type_models[i][1])
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            train_and_evaluate_one(model, optimizer, loss_fns[j], max_epochs, train_dataloader, val_dataloader, test_dataloader, str(model)+folder, str(loss_fns[j]), vizualisator)



def train_borneIncendie():

    ####################################################Preprocess resnet / Pretrained models

    #train_and_evaluate_all(*create_dataloader("data/dilate_reduitTOTAL_512px_0px_recouvrement", "data/dilate_reduitMOYEN_512px_0px_recouvrement", True, False),[
                            #(AerialFormer, (3, 96, 2, True)),
                            #(CustomFCN, (2, True)),
                        #],[CustomBCE(pos_weight=torch.tensor([1., (1/18968)*69860282]).to('cuda'))                         ], max_epochs, vizualize_preprocess, "/recouvrement0/Adam")

    #################################################### No pretrained models

    #train_and_evaluate_all(*create_dataloader("data/dilate_reduitTOTAL_512px_0px_recouvrement", "data/dilate_reduitMOYEN_512px_0px_recouvrement", False, False),[
                            #(UNet, (3, 2)), #31.0M
                            #(NetC, (3, 2)), #451K
                            #(CustomFCN, (2, False)), #35.3M
                            #(PSPNet, (2, False)), #49.1M
                            #(AerialFormer, (3, 96, 2, False)), #48.8M
                            #(CustomDeepLabV3, (2, False)), #39.6M
                            #(CustomFarSeg, (2, False)), #33.4M
                        #],[CustomBCE(pos_weight=torch.tensor([1.,(1/18968)*69860282]).to('cuda'))                         ],max_epochs, vizualize, "/recouvrement0/Adam")

    #################################################### Best model per loss

    train_and_evaluate_all(*create_dataloader("data/dilate_reduitTOTAL_512px_0px_recouvrement", "data/dilate_reduitMOYEN_512px_0px_recouvrement", True, False),[(CustomFCN, (2, True))],
                           [
                            FarSegLoss(gamma=0.25),
                            FarSegLoss(gamma=1/((1/18968)*69860282)),
                            CustomDiceLoss(),
                            CustomBCE(pos_weight=torch.tensor([1., (1/18968)*69860282]).to('cuda'), SOM=True, top_k=400),
                            CustomBCE(pos_weight=torch.tensor([1., (1/18968)*69860282]).to('cuda'), SOM=True, top_k=600),
                            CustomBCE(pos_weight=torch.tensor([1., (1/18968)*69860282]).to('cuda'), SOM=True, top_k=200),
                            CustomFocalLoss(alpha=0.25),
                            CustomFocalLoss(alpha=1/((1/18968)*69860282),),
                            ],
                        max_epochs, vizualize_preprocess, "/recouvrement0/Adam")

    #################################################### gaussian loss

    train_and_evaluate_all(*create_dataloader("data/dilate_reduitTOTAL_512px_0px_recouvrement", "data/dilate_reduitMOYEN_512px_0px_recouvrement", True, True), [(CustomFCN, (2, True))],[                  CustomBCE(pos_weight=torch.tensor([1., (1/18908)*69860107]).to('cuda'))],                   max_epochs, vizualize_preprocess, "/recouvrement0_gaussien/Adam")

    #################################################### recouvrement

    train_and_evaluate_all(*create_dataloader("data/dilate_reduitTOTAL_512px_300px_recouvrement", "data/dilate_reduitMOYEN_512px_0px_recouvrement", True, False), [(CustomFCN, (2, True))],[                  CustomBCE(pos_weight=torch.tensor([1., (1/17743)*70411731]).to('cuda'))],                   max_epochs, vizualize_preprocess, "/recouvrement300/Adam")

    #################################################### reduitMoyen

    train_and_evaluate_all(*create_dataloader("data/dilate_reduitMOYEN_512px_0px_recouvrement", "data/dilate_reduitMOYEN_512px_0px_recouvrement", True, False), [(CustomFCN, (2, True))],[                  CustomBCE(pos_weight=torch.tensor([1., (1/18801)*115157908]).to('cuda'))],                   max_epochs, vizualize_preprocess, "/recouvrement0_moyen/Adam")


def train_mur6():
    #####################################################Preprocess resnet / Pretrained models

    #train_and_evaluate_all(*create_dataloader("data/mur6_dilate_reduitTOTAL_512px_0px_recouvrement", "data/mur6_dilate_reduitMOYEN_512px_0px_recouvrement", True, False),[
                            #(AerialFormer, (3, 96, 2, True)),
                            #(CustomFCN, (2, True)),
                        #],[CustomBCE(pos_weight=torch.tensor([1., (1/103018)*28596674]).to('cuda'))                          ], max_epochs, vizualize_preprocess, "/mur6_recouvrement0/Adam")

    ##################################################### No pretrained models

    #train_and_evaluate_all(*create_dataloader("data/mur6_dilate_reduitTOTAL_512px_0px_recouvrement", "data/mur6_dilate_reduitMOYEN_512px_0px_recouvrement", False, False),[
                            #(UNet, (3, 2)), #31.0M
                            #(NetC, (3, 2)), #451K
                            #(CustomFCN, (2, False)), #35.3M
                            #(PSPNet, (2, False)), #49.1M
                            #(AerialFormer, (3, 96, 2, False)), #48.8M
                            #(CustomDeepLabV3, (2, False)), #39.6M
                            #(CustomFarSeg, (2, False)), #33.4M
                        #],[CustomBCE(pos_weight=torch.tensor([1.,(1/103018)*28596674]).to('cuda'))                         ],max_epochs, vizualize, "/mur6_recouvrement0/Adam")


    #################################################### Best model per loss

    train_and_evaluate_all(*create_dataloader("data/mur6_dilate_reduitTOTAL_512px_0px_recouvrement", "data/mur6_dilate_reduitMOYEN_512px_0px_recouvrement", True, False),[(CustomFCN, (2, True))],
                           [
                            FarSegLoss(gamma=1/((1/103018)*28596674)),
                            #CustomDiceLoss(),
                            CustomBCE(pos_weight=torch.tensor([1., (1/103018)*28596674]).to('cuda'), SOM=True, top_k=400),
                            #CustomFocalLoss(alpha=0.25),
                            ],
                        max_epochs, vizualize_preprocess, "/mur6_recouvrement0/Adam")


    ##################################################### gaussian loss

    #train_and_evaluate_all(*create_dataloader("data/mur6_dilate_reduitTOTAL_512px_0px_recouvrement", "data/mur6_dilate_reduitMOYEN_512px_0px_recouvrement", True, True), [(CustomFCN, (2, True))],[                  CustomBCE(pos_weight=torch.tensor([1., (1/102870)*28596696]).to('cuda'))],                   max_epochs, vizualize_preprocess, "/mur6_recouvrement0_gaussien/Adam")

    ##################################################### reduitMoyen

    #train_and_evaluate_all(*create_dataloader("data/mur6_dilate_reduitMOYEN_512px_0px_recouvrement", "data/mur6_dilate_reduitMOYEN_512px_0px_recouvrement", True, False), [(CustomFCN, (2, True))],[                  CustomBCE(pos_weight=torch.tensor([1., (1/109926)*51551613]).to('cuda'))],                   max_epochs, vizualize_preprocess, "/mur6_recouvrement0_moyen/Adam")



def train_haie():
    #####################################################Preprocess resnet / Pretrained models

    #train_and_evaluate_all(*create_dataloader("data/haie_dilate_reduitTOTAL_512px_0px_recouvrement", "data/haie_dilate_reduitTOTAL_512px_0px_recouvrement", True, False),[
                            #(AerialFormer, (3, 96, 2, True)),
                            #(CustomFCN, (2, True)),
                        #],[CustomBCE(pos_weight=torch.tensor([1., (1/709676)*146003833]).to('cuda'))                          ], max_epochs, vizualize_preprocess, "/haie_recouvrement0/Adam")

    ##################################################### No pretrained models

    train_and_evaluate_all(*create_dataloader("data/haie_dilate_reduitTOTAL_512px_0px_recouvrement", "data/haie_dilate_reduitTOTAL_512px_0px_recouvrement", False, False),[
                            #(UNet, (3, 2)), #31.0M
                            #(NetC, (3, 2)), #451K
                            #(CustomFCN, (2, False)), #35.3M
                            (PSPNet, (2, False)), #49.1M
                            (AerialFormer, (3, 96, 2, False)), #48.8M
                            (CustomDeepLabV3, (2, False)), #39.6M
                            (CustomFarSeg, (2, False)), #33.4M
                        ],[CustomBCE(pos_weight=torch.tensor([1.,(1/709676)*146003833]).to('cuda'))                         ],max_epochs, vizualize, "/haie_recouvrement0/Adam")


    #################################################### gaussian loss

    train_and_evaluate_all(*create_dataloader("data/haie_dilate_reduitTOTAL_512px_0px_recouvrement", "data/haie_dilate_reduitTOTAL_512px_0px_recouvrement", True, True), [(CustomFCN, (2, True))],[                  CustomBCE(pos_weight=torch.tensor([1., (1/708091)*146002650]).to('cuda'))],                   max_epochs, vizualize_preprocess, "/haie_recouvrement0_gaussien/Adam")

    #################################################### Best model per loss

    train_and_evaluate_all(*create_dataloader("data/haie_dilate_reduitTOTAL_512px_0px_recouvrement", "data/haie_dilate_reduitTOTAL_512px_0px_recouvrement", True, False),[(CustomFCN, (2, True))],
                           [
                            FarSegLoss(gamma=1/((1/709676)*146003833)),
                            CustomDiceLoss(),
                            CustomBCE(pos_weight=torch.tensor([1., (1/709676)*146003833]).to('cuda'), SOM=True, top_k=400),
                            CustomFocalLoss(alpha=0.25),
                            ],
                        max_epochs, vizualize_preprocess, "/haie_recouvrement0/Adam")


#train_borneIncendie()
#train_mur6()
train_haie()
