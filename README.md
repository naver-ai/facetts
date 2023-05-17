# [Face-TTS] Imaginary Voice: Face-styled Diffusion Model for Text-to-Speech (ICASSP'2023)


<a href="https://arxiv.org/abs/2302.13700"><img src="https://img.shields.io/badge/arXiv-2302.13700-%23B31B1B"></a>
<a href="https://facetts.github.io/"><img src="https://img.shields.io/badge/Project%20Page-online-brightgreen"></a>


This repository is the official implementation of Face-TTS.

---
## Installation

1. Install python packages
```
pip install -r requirements.txt
```

2. Build monotonic align module
```
cd model/monotonic_align; python setup.py build_ext --inplace; cd ../..
```

---
## Preparation
1. Download trained model weights from <a href="https://drive.google.com/file/d/18ERr-91Z1Mnc2Aq9n1nBPijzb5gSymLq/view?usp=sharing">here</a>

2. Download <a href="https://mmai.io/datasets/lip_reading/">LRS3</a> into `'data/lrs3/'`

3. Extract and save audio as '*.wav' files in `'data/lrs3/wav'`
   ```
   python data/extract_audio.py
   ```

---
## Test

:exclamation: Face should be cropped and aligned for LRS3 distribution. You can use <a href="https://github.com/joonson/syncnet_python/tree/master/detectors">'syncnet_python/detectors'</a>.

1. Prepare text description in txt file.
```
echo "This is test" > test/text.txt
```


2. Inference Face-TTS.
```
python inference.py
```

3. Result will be saved in `'test/'`. 

:zap: To make MOS test set, we use 'test/ljspeech_text.txt' to randomly select text description.

--- 
## Training

1. Check config.py 

2. Run
```
python run.py
```

---
## Reference
This repo is based on 
<a href="https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS">Grad-TTS</a>, 
<a href="https://github.com/bshall/hifigan">HiFi-GAN-16k</a>, 
<a href="https://github.com/joonson/syncnet_trainer">SyncNet</a>.  Thanks!

---
## Citation

```
@inproceedings{lee2023imaginary,
  author    = {Lee, Jiyoung and Chung, Joon Son and Chung, Soo-Whan},
  title     = {Imaginary Voice: Face-styled Diffusion Model for Text-to-Speech},
  booktitle = {IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP)},
  year      = {2023},
}
```

---
## License

```
Face-TTS
Copyright (c) 2023-present NAVER Cloud Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
