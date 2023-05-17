import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

class SyncNet(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.wI = nn.Parameter(torch.tensor(config["syncnet_initw"]))
        self.bI = nn.Parameter(torch.tensor(config["syncnet_initb"]))

        self.stride = config["syncnet_stride"]
        self.nOut = config["vid_emb_dim"]

        self.netcnnaud = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1)),
            nn.Conv2d(96, 256, kernel_size=(5, 5), stride=(2, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            nn.Conv2d(256, 384, kernel_size=(5, 3), stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1)),
            nn.Conv2d(384, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(0, 1)),
            nn.Conv2d(
                256, 512, kernel_size=(3, 1), padding=(0, 0), stride=(1, self.stride)
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.netfcaud = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, self.nOut, kernel_size=1),
        )

        self.netcnnimg = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=(7, 7), stride=(2, 2), padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            nn.Conv2d(96, 256, kernel_size=(5, 5), stride=(2, 2), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            nn.Conv2d(256, 512, kernel_size=(6, 6), padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        self.netfcimg = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, self.nOut, kernel_size=1),
        )


        if not isinstance(config["syncnet_ckpt"], type(None)):
            print("Load syncnet pretrained model from " + config["syncnet_ckpt"])
            self.loadparameters(config["syncnet_ckpt"])

    def loadparameters(self, ckpt_path):
        loaded_state = torch.load(ckpt_path, map_location=lambda loc, storage: loc)
        self_state = self.state_dict()
        for name, param in loaded_state["state_dict"].items():
            if name in self_state:
                print("load: ", name)
                self_state[name].copy_(param)

        return

    def forward_aud(self, aud):

        # Audio stream
        audmid = self.netcnnaud(aud)
        audout = self.netfcaud(audmid.squeeze(-2))
        return audout

    def forward_vid(self, vid):

        # Image stream
        vidmid = self.netcnnimg(vid)
        vidout = self.netfcimg(vidmid.squeeze(-1))

        return vidout

    def forward(self, vid, aud):

        # Image stream
        vidmid = self.netcnnimg(vid)
        vidout = self.netfcimg(vidmid.squeeze(-1))

        # Audio stream
        audmid = self.netcnnaud(aud)
        audout = self.netfcaud(audmid.squeeze(-2))
        return vidout, audout

    def forward_perceptual(self, aud):
        out = []
        for i, layer in enumerate(self.netcnnaud):
            if i == 0:
                mid = layer(aud)
            else:
                mid = layer(mid)
            if isinstance(layer, nn.ReLU):
                out.append(mid)

        mid = mid.squeeze(-2)
        for i, layer in enumerate(self.netfcaud):
            mid = layer(mid)
            if isinstance(layer, nn.ReLU):
                out.append(mid)
        out.append(mid)
        return out