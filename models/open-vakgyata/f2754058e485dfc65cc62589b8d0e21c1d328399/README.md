---
language:
- en
- hi
- or
- bn
- ta
- te
- kn
- ml
- mr
- gu
license: cc-by-nc-4.0
pipeline_tag: audio-classification
library_name: transformers
tags:
- language-identification
- indian-languages
- multilingual
- speech
- asr-preprocessing
- callcenter-ai
- speech-analytics
- audio-classification
- wav2vec2
- transformers
- pytorch
- huggingface
---
**Model Name:** open-vakgyata

**Model Overview:**
open-vakgyata is an open-source language identification model capable of detecting and classifying indian languages from speech inputs.

**Supported Languages:**
| Language             | Code  |
|----------------------|-------|
| English (India)      | en-IN |
| Hindi                | hi-IN |
| Odia                 | or-IN |
| Bengali              | bn-IN |
| Tamil                | ta-IN |
| Telugu               | te-IN |
| Kannada              | kn-IN |
| Malayalam            | ml-IN |
| Marathi              | mr-IN |
| Gujarati             | gu-IN |

**Specification**
- Supported Sampling Rate: 16000
- Recomonded Audio Format: 16kHz, 16bit PCM

**Usage:**

```py
from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
import torch

device = "cpu" # "cuda"

model_id = "onecxi/open-vakgyata"

processor = AutoFeatureExtractor.from_pretrained(model_id)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id).to(device)

```

**Inference:**

```py
import torchaudio

audio, sr = torchaudio.load("path/to/audio.wav")

# Process the waveform and move to the appropriate device
inputs = processor(audio.flatten(), sampling_rate=sr, return_tensors="pt").to(device)

# Perform inference
with torch.no_grad():
    logits = model(**inputs).logits

# Get language probabilities
probs = logits.softmax(dim=-1).cpu().numpy()
language = model.config.id2label.get(probs.argmax())

print(language)
```

---

## **Citation**

If you use this model in your research or application, please consider citing the model and its base source:

```
@misc{vakgyata2024,
  title={vakgyata: Language Identification for Indian Speech},
  author={OneCXI},
  year={2024},
  url={https://huggingface.co/onecxi/open-vakgyata}
}
```

---