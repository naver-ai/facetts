import os

from config import ex
import torch
import torchaudio as ta
import pytorch_lightning as pl

from model.syncnet_hifigan import SyncNet
from utils.mel_spectrogram import mel_spectrogram

from tqdm import tqdm
import cv2
import numpy as np
import random

N = 5
FOLDER = "./test/results/"

@ex.automain
def main(_config):

    pl.seed_everything(_config["seed"])
    files = [file for file in os.listdir(FOLDER) if file.endswith(".png")]

    speakers = {}
    for f in files:
        spk = f.split("_")[0]
        if spk not in speakers.keys():
            speakers[spk] = []
        speakers[spk].append(f)

    spk_list = list(speakers.keys())
    model = SyncNet(_config).eval().cuda()

    accs = []
    with torch.no_grad():
        for _ in tqdm(range(100)):

            idxs = random.sample(range(len(speakers.keys())), N)
            v, a = [], []
            
            for i in idxs:
                spk = spk_list[i]
                f = random.choice(speakers[spk])
                img = cv2.imread(os.path.join(FOLDER, f))
                audf = f.replace("_face.png", ".wav")
                aud, _ = ta.load(os.path.join(FOLDER, audf))
                aud = mel_spectrogram(
                    aud, 1024, 128, 16000, 160, 1024, 0.0, 8000.0, center=False
                ).squeeze()

                img = np.transpose(img, (2, 0, 1))
                img = torch.FloatTensor(img)
                zv, za = model(
                    img.unsqueeze(0).cuda(), aud.unsqueeze(0).unsqueeze(0).cuda()
                )
                za = torch.mean(za, 2, keepdim=True)

                v.append(zv.squeeze(0))
                a.append(za.squeeze(0))

            v = torch.stack(v, 0)
            a = torch.stack(a, 0)

            _, acc = model.compute_loss(v, a)

            accs.append(acc.cpu().detach().numpy())

    print(f"######## Done measurement.")
    print(f"######## Dir: ", FOLDER)
    acc = np.mean(accs)
    print(f"######## {N}-way ACC: ", acc)
