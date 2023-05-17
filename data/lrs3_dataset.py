import torch
import torchaudio
import os
import random
from model.utils import fix_len_compatibility
from utils.tts_util import intersperse, parse_filelist
from text import cmudict
from utils.mel_spectrogram import mel_spectrogram
import cv2
import numpy as np


class LRS3Dataset(torch.utils.data.Dataset):
    def __init__(self, split: str = "", config=None):
        assert split in ["train", "val", "test"]
        super().__init__()

        self.split = split
        self.config = config

        self.cmudict = cmudict.CMUDict(self.config["cmudict_path"])

        if self.split == "train":
            self.filelist = self.config["lrs3_train"]
            self.video_dir = os.path.join(self.config["lrs3_path"], "trainval")
            self.audio_dir = os.path.join(self.config["lrs3_path"], "wav/trainval")
        elif self.split == "val":
            self.filelist = self.config["lrs3_val"]
            self.video_dir = os.path.join(self.config["lrs3_path"], "trainval")
            self.audio_dir = os.path.join(self.config["lrs3_path"], "wav/trainval")
        elif self.split == "test":
            self.filelist = self.config["lrs3_test"]
            self.video_dir = os.path.join(self.config["lrs3_path"], "test")
            self.audio_dir = os.path.join(self.config["lrs3_path"], "wav/test")

        # Load datalist
        with open(self.filelist) as listfile:
            self.data_list = listfile.readlines()

        print(f"{split} set: ", len(self.data_list))

        spk_list = [data.split("\n")[0].split("/")[0] for data in self.data_list]
        spk_list = set(spk_list)
        print(f"{len(spk_list)=}")
        
        self.spk_list = dict()
        for i, spk in enumerate(spk_list):
            self.spk_list[spk] = i

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        name = self.data_list[index].split("\n")[0]
        vidname = name + ".mp4"
        textpath = name + ".txt"
        aud, sr = torchaudio.load(os.path.join(self.audio_dir, name + ".wav"))
        
        assert (sr == self.config["sample_rate"]), "sampling rate should be 16k."
        
        aud = mel_spectrogram(
            aud,
            self.config["n_fft"],
            self.config["n_mels"],
            self.config["sample_rate"],
            self.config["hop_len"],
            self.config["win_len"],
            self.config["f_min"],
            self.config["f_max"],
            center=False,
        )
        
        text = (
            open(os.path.join(self.video_dir, textpath))
            .readlines()[0]
            .split(":")[1]
            .strip()
        )
        
        if isinstance(text, type(None)):
            print(text)
            print(name)
        else:
            text += "."

        img = self.load_random_frame(self.video_dir, f"{name}.mp4", 1)
        txt = self.loadtext(text, self.cmudict, self.config["add_blank"])
        spk = self.spk_list[name.split("/")[0]]

        return {
            "spk_id": torch.LongTensor([int(spk)]),
            "spk": img,
            "y": aud.squeeze(),
            "x": txt,
            "name": name,
        }

    def loadtext(self, text, cmudict, add_blank=True):
        text_norm = text_to_sequence(text, dictionary=cmudict)
        if add_blank:
            text_norm = intersperse(text_norm, len(symbols))
        text_norm = torch.IntTensor(text_norm)
        return text_norm


    def load_random_frame(self, datadir, filename, len_frame=1):
        # len_frame == -1: load all frames
        # else: load random index frame with len_frames
        cap = cv2.VideoCapture(os.path.join(datadir, filename))
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if len_frame == -1:
            ridx = 0
            loadframe = nframes
        else:
            ridx = random.randint(2, nframes - len_frame)
            loadframe = len_frame
            cap.set(1, ridx)

        imgs = []
        for i in range(0, loadframe):
            _, img = cap.read()
            imgs.append(img)

        cap.release()
        imgs = np.stack(imgs, axis=3)
        imgs = np.transpose(imgs, (2, 3, 0, 1))

        return imgs


class TextMelVideoBatchCollate(object):
    def __call__(self, batch):
        B = len(batch)
        y_max_length = max([item["y"].shape[-1] for item in batch])
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item["x"].shape[-1] for item in batch])
        n_feats = batch[0]["y"].shape[-2]

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        y_lengths, x_lengths = [], []
        spk = []

        for i, item in enumerate(batch):
            y_, x_, spk_ = item["y"], item["x"], item["spk"]
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            y[i, :, : y_.shape[-1]] = y_
            x[i, : x_.shape[-1]] = x_
            spk.append(spk_)

        y_lengths = torch.LongTensor(y_lengths)
        x_lengths = torch.LongTensor(x_lengths)
        spk = torch.cat(spk, dim=0)
        return {
            "x": x,
            "x_len": x_lengths,
            "y": y,
            "y_len": y_lengths,
            "spk": spk,
        }
