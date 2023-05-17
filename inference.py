import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import os
import copy

from config import ex
from model.face_tts import FaceTTS
from data import _datamodules

import numpy as np
from scipy.io.wavfile import write

from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils.tts_util import intersperse
import cv2

from tqdm import tqdm


@ex.automain
def main(_config):

    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    print("######## Initializing TTS model")
    model = FaceTTS(_config).cuda()

    if _config['use_custom']:      
        print(f"######## Load {_config['test_faceimg']}")
        # use custom face image to synthesize the speech
        spk = cv2.imread(os.path.join(f"{_config['test_faceimg']}"))
        spk = cv2.resize(spk, (224, 224))
        spk = np.transpose(spk, (2, 0, 1))
        spk = torch.FloatTensor(spk).unsqueeze(0).to(model.device)
    else:
        # use LRS3 image 
        print(f"######## Load {_config['dataset']}")
        dm = _datamodules[f"dataset_{_config['dataset']}"](_config)
        dm.set_test_dataset()
        sample = dm.test_dataset[0] # you can adjust the index of test sample
        spk = sample['spk'].to(model.device)

    print(f"######## Load checkpoint from {_config['resume_from']}")
    _config['enc_dropout'] = 0.0
    model.load_state_dict(torch.load(_config['resume_from'])['state_dict'])
        
    model.eval()
    model.zero_grad()

    print("######## Initializing HiFi-GAN")
    vocoder = torch.hub.load('bshall/hifigan:main', 'hifigan').eval().cuda()

    print(f"######## Load text description from {_config['test_txt']}")
    with open(_config['test_txt'], 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines()]

    cmu = cmudict.CMUDict(_config['cmudict_path'])

    with torch.no_grad():
        
        for i, text in tqdm(enumerate(texts)):
            x = torch.LongTensor(
                intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))
            ).to(model.device)[None]
            
            x_len = torch.LongTensor([x.size(-1)]).to(model.device)
            y_enc, y_dec, attn = model.forward(
                x,
                x_len,
                n_timesteps=_config["timesteps"],
                temperature=1.5,
                stoc=False,
                spk=spk,
                length_scale=0.91,
            )

            audio = (
                vocoder.forward(y_dec[-1]).cpu().squeeze().clamp(-1, 1).numpy()
                * 32768
            ).astype(np.int16)
            
            if not os.path.exists(_config["output_dir"]):
                os.makedirs(_config["output_dir"])

            write(
                f"{_config['output_dir']}/sample_{i}.wav",
                _config["sample_rate"],
                audio,
            )

    print(f"######## Done inference. Check '{_config['output_dir']}' folder")
