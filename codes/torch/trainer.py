# Copyright (c) 2021 Regents of the University of California
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from argparse import ArgumentParser

import cv2
import torch
import torchvision
import tfrecord
import pytorch_lightning
import random

from BReGNeXt import BReGNeXt
from utils import ShuffleDataset


def focal_loss2(input_tensor, target_tensor, weight=None, gamma=2, reduction='mean'):
    log_prob = torch.nn.functional.log_softmax(input_tensor, dim=-1)
    probs = torch.exp(log_prob)
    return torch.nn.functional.nll_loss(((1 - probs) ** gamma) * log_prob,
                                        target_tensor, weight=weight, reduction=reduction)


_image_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ColorJitter(brightness=(0, 0.05), contrast=(0.7,1.3), saturation=(0.6, 1.6), hue=0.08),
    torchvision.transforms.RandomResizedCrop((64,64)),
    torchvision.transforms.ToTensor(),
])


def decode_and_preprocess_image(features):
    # Decode the image
    features['image_raw'] = features['image_raw'].reshape(64, 64, 3)
    features['image_raw'] = _image_transform(features['image_raw'])
    features['image_raw'] = features['image_raw'] - torch.FloatTensor([0.5727663, 0.44812188, 0.39362228]).unsqueeze(-1).unsqueeze(-1)
    features['label'] = torch.LongTensor(features['label'])
    return features


class BReGNeXtPTLDriver(pytorch_lightning.LightningModule):

    def __init__(self, use_focal_loss = False):

        super(BReGNeXtPTLDriver, self).__init__()

        self._use_focal_loss = use_focal_loss
        self._model = BReGNeXt()

    def training_step(self, batch, batch_idx):
        logits = self._model(batch['image_raw'])
        batch['label'] = batch['label'].reshape(-1)
        loss = focal_loss2(logits, batch['label']) if self._use_focal_loss else torch.nn.functional.cross_entropy(logits, batch['label'])
        accuracy = (logits.argmax(dim=-1) == batch['label']).float().mean()

        self.log('train/accuracy', accuracy, prog_bar=True)
        self.log('train/loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        logits = self._model(batch['image_raw'])
        batch['label'] = batch['label'].reshape(-1)
        loss = focal_loss2(logits, batch['label']) if self._use_focal_loss else torch.nn.functional.cross_entropy(logits, batch['label'])
        accuracy = (logits.argmax(dim=-1) == batch['label']).float().mean()

        self.log('val/accuracy', accuracy, prog_bar=True)
        self.log('val/loss', loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self._model.parameters(), lr=0.0001, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.80)
        return [optimizer], [scheduler]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--use_focal_loss', action='store_true', help='Use focal loss when training.')
    parser = pytorch_lightning.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    train_dataset = ShuffleDataset(tfrecord.torch.dataset.TFRecordDataset(
        data_path='/home/david/Projects/BReG-NeXt/tfrecords/training_FER2013_sample.tfrecords',
        index_path=None,
        description={'image_raw': 'byte', 'label': 'int'},
        transform=decode_and_preprocess_image,
    ), 1024)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, num_workers=4)

    valid_dataset = tfrecord.torch.dataset.TFRecordDataset(
        data_path='/home/david/Projects/BReG-NeXt/tfrecords/validation_FER2013_sample.tfrecords',
        index_path=None,
        description={'image_raw': 'byte', 'label': 'int'},
        transform=decode_and_preprocess_image,
    )
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, num_workers=4)

    # Fit the model to the trainer.
    model = BReGNeXtPTLDriver(use_focal_loss=args.use_focal_loss)
    trainer = pytorch_lightning.Trainer.from_argparse_args(args)
    trainer.fit(model, train_dataloader, valid_dataloader)
