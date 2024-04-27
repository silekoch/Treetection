from ForestCoverDataset import ForestCoverDataset
import torch
import pytorch_lightning as pl
from DeepLab import TreepLab

if __name__ == "__main__":
    # preprocessing_fn = get_preprocessing_fn('resnet18', pretrained='imagenet')

    train_dataset = ForestCoverDataset(mode='train', overfitting_mode='2sample')
    val_dataset = ForestCoverDataset(mode='val', overfitting_mode='2sample')

    # train_dataset = ForestCoverDataset(mode='train', overfitting_mode='batch')
    # val_dataset = ForestCoverDataset(mode='val', overfitting_mode='batch')

    # train_dataset = ForestCoverDataset(mode='train')
    # val_dataset = ForestCoverDataset(mode='val')

    BATCH_SIZE = 2

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # model = smp.DeepLabV3(
    #     encoder_name="resnet18",
    #     encoder_weights="imagenet",
    #     in_channels=3,
    #     classes=1,
    # )

    # model = TreepLab("Unet", "resnet18", in_channels=3, out_classes=1)
    model = TreepLab("DeepLabV3", "resnet18", in_channels=3, out_classes=1)

    trainer = pl.Trainer(
        # gpus=1, 
        max_epochs=5,
    )

    trainer.fit(
        model, 
        train_dataloaders=train_loader, 
        val_dataloaders=val_loader,
    )

    # # Copied training code from kaggle notebook:

    # # Set flag to train the model or not. If set to 'False', only prediction is performed (using an older model checkpoint)
    # TRAINING = True

    # # Set num of epochs
    # EPOCHS = 5

    # # Set device: `cuda` or `cpu`
    # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # define loss function
    # loss = smp.losses.DiceLoss(mode='binary')

    # # define metrics
    # metrics = [
    #     smp.metrics.IoU(threshold=0.5),
    # ]

    # # define optimizer
    # optimizer = torch.optim.Adam([ 
    #     dict(params=model.parameters(), lr=0.00008),
    # ])

    # train_epoch = smp.utils.train.TrainEpoch(
    # model, 
    # loss=loss, 
    # metrics=metrics, 
    # optimizer=optimizer,
    # device=DEVICE,
    # verbose=True,
    # )

    # valid_epoch = smp.utils.train.ValidEpoch(
    #     model, 
    #     loss=loss, 
    #     metrics=metrics, 
    #     device=DEVICE,
    #     verbose=True,
    # )

    # if TRAINING:
    #     best_iou_score = 0.0
    #     train_logs_list, valid_logs_list = [], []

    #     for i in range(0, EPOCHS):

    #         # Perform training & validation
    #         print('\nEpoch: {}'.format(i))
    #         train_logs = train_epoch.run(train_loader)
    #         valid_logs = valid_epoch.run(val_loader)
    #         train_logs_list.append(train_logs)
    #         valid_logs_list.append(valid_logs)

    #         # Save model if a better val IoU score is obtained
    #         if best_iou_score < valid_logs['iou_score']:
    #             best_iou_score = valid_logs['iou_score']
    #             torch.save(model, './best_model.pth')
    #             print('Model saved!')