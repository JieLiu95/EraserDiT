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

## Reference
```
@article{liu2025eraserdit,
  title={EraserDiT: Fast Video Inpainting with Diffusion Transformer Model},
  author={Liu, Jie and Hui, Zheng},
  journal={arXiv preprint arXiv:2506.12853},
  year={2025}
}
```
