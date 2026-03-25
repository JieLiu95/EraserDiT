# EraserDiT: Fast Video Inpainting with Diffusion Transformer Model

## Install dependencies

```
pip install -r requirements.txt
```

## Local demo
Currently requires single gpu memory over 60GB.
```
python inference.py --vid_path data/10268234.mp4 --mask_path data/10268234_mask.mp4 --prompt "There is a bridge over the lake." 
```

## 📦 Pre-trained Models

Pre-trained models are available on OSS. After downloading, place the model into the converted_ckpts folder.
