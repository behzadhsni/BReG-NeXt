
import torch
import pytorch_lightning

from .BReGNeXt import BReG_NeXt

class BReGNeXtPTLDriver(pytorch_lightning.LightningModule):

    def __init__(self,):
        self._model = BReG_NeXt()

    def training_step(self, batch):
        pass

    def validation_step(self, batch):
        pass

    def configure_optimizers(self):

        optimizer = torch.optim.Adam([{'params': self._model.conv0_params(), 'weight_decay': 0.0001},
                                      {'params': self._model.model_params()}], lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.80)
        return [optimizer], [scheduler]
