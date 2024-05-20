import os

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from dadaptation import DAdaptAdam

class CiliaModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, lr = 1.0, pooling = "max", dropout = None):
        super().__init__()
        aux_params = {"pooling": pooling, "dropout": dropout, "classes": 1}
        self.model = smp.create_model(
            arch = arch, # UnetPlusPlus | DeepLabV3Plus
            encoder_name = encoder_name, # resnet50 | densenet161
            in_channels = 1,
            classes = 1,
            aux_params = aux_params
        )
        self.lr = lr
        #self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits = True)
        self.train_step_outputs = []
        self.test_step_outputs = []
        self.val_step_outputs = []

    def forward(self, image):
        """
        Run a forward pass on an image.
        """
        return self.model.forward(image)
    
    def shared_step(self, batch, stage):
        """
        Since train, test, and validate are all basically the same.
        """
        image = batch["image"]

        # Shape of the image should be (batch_size, 1, height, width)
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32.
        # This should be the case--most images are 480x640, and the others are 
        # multiples of that--but still a good sanity check.
        #
        # The encoder and decoder stages are connected through skip connections
        # that downsample in five stages, each by a factor of two, which gives 2^5=32.
        # Trying to concatenate feature sets at the end, when the starting dimension
        # is NOT divisible by 32, will have some off-by-weird-amounts errors.
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch["mask"]

        # Shape of the mask should be identical to that of the image.
        assert mask.ndim == 4

        # Sanity check that the mask values' range is still correct.
        assert mask.max() <= 1.0 and mask.min() >= 0

        # Get the prediction!
        logits_mask = self.forward(image)
        
        # Compute the loss from the prediction.
        #loss = self.loss_fn(logits_mask[0], mask)
        loss = F.mse_loss(logits_mask[0], mask)

        # Compute a probability, then threshold to a predicted value.
        # Sigmoid is convenient for this.
        prob_mask = logits_mask[0].sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # Individual stats values will be aggregated for the whole epoch.
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), mask.long(), mode = "binary"
        )

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        #avgloss = torch.mean(torch.cat([x["loss"] for x in outputs]).mean())

        # This is an IOU average over all the images in the dataset.
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction = "micro-imagewise")
        
        # This is an IOU average over all the pixels in the dataset.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction = "micro")

        # This is an IOU weighted (to account for class imbalance) average over 
        # all the images per class in the dataset.
        ### For nvidia:
        # cilia: 0.101      -- weight: 9.86
        # non-cilia: 0.899  -- weight: 1.113
        ### For opencv:
        # cilia: 0.108      -- weight: 9.202
        # non-cilia: 0.892  -- weight: 1.122
        weighted_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction = "weighted-imagewise", 
            class_weights = [1.113, 9.86]
        )

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
            #f"{stage}_avg_loss": avgloss,
            f"{stage}_weighted_iou" : weighted_iou,
        }
        
        self.log_dict(metrics, prog_bar = True, sync_dist = True)

    def training_step(self, batch, batch_idx):
        outputs = self.shared_step(batch, "train")
        self.train_step_outputs.append(outputs)
        return outputs

    def on_training_epoch_end(self):
        return self.shared_epoch_end(self.train_step_outputs, "train")

    def validation_step(self, batch, batch_idx):
        outputs = self.shared_step(batch, "valid")
        self.val_step_outputs.append(outputs)
        return outputs

    def on_validation_epoch_end(self):
        return self.shared_epoch_end(self.val_step_outputs, "valid")

    def test_step(self, batch, batch_idx):
        outputs = self.shared_step(batch, "test")  
        self.test_step_outputs.append(outputs)
        return outputs

    def on_test_epoch_end(self):
        return self.shared_epoch_end(self.test_step_outputs, "test")

    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.parameters(), lr = self.lr, weight_decay = 1e-4)
        optimizer = DAdaptAdam(params = self.parameters(), lr = self.lr, weight_decay = 1e-4, log_every = 10)
        scheduler = {
            "scheduler": CosineAnnealingWarmRestarts(optimizer, T_0 = 2, T_mult = 2),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}