import time
from pathlib import Path
import moviepy
import os
import cv2
import numpy as np
from moviepy import ImageSequenceClip
import copy
import time
import inspect
import torch

class TimeStamp():
    def __init__(self):
        self.file=""
        self.function_name=""
        self.line=0
        self.time=None
        self.tip=""
    
    def __call__(self, reset=False, tip=None, detail=False, *args, **kwds):
        current_time=time.time()
        stack = inspect.stack()
        caller=stack[1]
        current_file=caller.filename
        current_function_name=caller.function
        current_line=caller.lineno
        if tip is not None:
            self.tip=f"<<{tip}>> "

        if self.time is not None and not reset:
            if not detail:
                print(f"{self.tip if self.tip is not None else ''}{self.function_name}@{self.line}->{current_function_name}@{current_line}: {time.time()-self.time:.3f}s")
            else:
                print(f"{self.tip if self.tip is not None else ''}{self.file}@{self.function_name}: {self.line}->{current_file}@{current_function_name}:{current_line}:{(current_time-self.time):.3f}s")
        self.file=current_file
        self.function_name=current_function_name
        self.line=current_line
        self.time=current_time

    def reset(self, remain_tip=False):
        self.time=None
        if not remain_tip:
            self.tip=None

    def exit(self):
        stack = inspect.stack()
        caller=stack[1]
        current_file=caller.filename
        current_function_name=caller.function
        current_line=caller.lineno
        print(f"<exit> {caller.filename}@{caller.function}: {caller.lineno}")
        exit(0)
        
class GlobalValues:
    ENABLE_PER=False
    DEBUG=False

gtime_stamp=TimeStamp()

def test_time(enable=True):
    def wrapper0(func):
        def wrapper(*args, **kwargs):
            if enable:
                start=time.time()
                output=func(*args, **kwargs)
                print(f"<@{func.__name__}>: {time.time()-start:.3f}s")
                return output
            else:
                return func(*args, **kwargs)
        return wrapper
    return wrapper0

class TensorSaveImage():
    @staticmethod
    def save_torch_tensor_jpg_nhwc(tensor:torch.Tensor,save_path):
        TensorSaveImage.save_torch_tensor_jpg_nchw(tensor.contiguous().permute(0,3,1,2),save_path=save_path)

    @staticmethod
    def save_torch_tensor_jpg_nchw(tensor:torch.Tensor,save_path):
        import torch
        from torchvision import transforms
        from PIL import Image

        if torch.max(tensor).item()>2:
            tensor=(tensor/255)
        
        tensor=tensor.clamp(0,1)
        to_pil_image = transforms.ToPILImage()
        image = to_pil_image(tensor[0].cpu())

        base_dir=os.path.dirname(save_path)
        if len(base_dir)>0:
            os.makedirs(base_dir, exist_ok=True)
        image.save(save_path)

    @staticmethod
    def save_numpy_tensor_jpg_nchw(tensor:np.ndarray,save_path):
        TensorSaveImage.save_numpy_tensor_jpg_nhwc(tensor.transpose(0,2,3,1),save_path=save_path)

    @staticmethod
    def save_numpy_tensor_jpg_nhwc(tensor:np.ndarray,save_path):
        import numpy as np
        from torchvision import transforms
        from PIL import Image

        if np.max(tensor)<=1:
            tensor=tensor*255
        
        to_pil_image = transforms.ToPILImage()
        image = to_pil_image(tensor[0])

        base_dir=os.path.dirname(save_path)
        if len(base_dir)>0:
            os.makedirs(base_dir, exist_ok=True)
        image.save(save_path)

def save_video_with_audio_mask(video, mask, video_ori, path: str, fps):
    frames = []
    for idx in range(len(mask)):
        img_temp = np.array(video[idx].permute(1, 2, 0))
        ori_temp = np.array(video_ori[idx].permute(1, 2, 0) * 255)
        mask_gaussian = cv2.dilate(mask[idx], cv2.getStructuringElement(cv2.MORPH_CROSS, (11, 11)), iterations=11)
        mask_gaussian = cv2.GaussianBlur(mask_gaussian, (9, 9), 0)
        mask_gaussian = cv2.GaussianBlur(mask_gaussian, (9, 9), 0)
        mask_gaussian = cv2.GaussianBlur(mask_gaussian, (9, 9), 0)
        img = img_temp * mask_gaussian + ori_temp * (1-mask_gaussian)
        frames.append(img)
    clip = ImageSequenceClip(frames, fps=fps)
    # final_clip = clip.with_audio(audio)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # final_clip.write_videofile(path, codec="libx264", audio_codec="libmp3lame", bitrate="10M")
    clip.write_videofile(path, codec="libx264", bitrate="10M")

def save_video_with_audio(video, path: str, fps):
    frames = []
    for img in video:
        frames.append(np.array(img.permute(1, 2, 0)))
    clip = ImageSequenceClip(frames, fps=fps)
    # final_clip = clip.with_audio(audio)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # final_clip.write_videofile(path, codec="libx264", audio_codec="libmp3lame", bitrate="10M")
    clip.write_videofile(path, codec="libx264", bitrate="10M")


def desc_tensor(tensor: np.ndarray | torch.Tensor):
    print(f"@type: {type(tensor)}")
    if isinstance(tensor,torch.Tensor):
        print(f"@> {tensor.shape}, {tensor.device}, {tensor.dtype}")
    else:
        print(f"@> {tensor.shape}")

def save_video(video, path: str, fps=15):
    frames = []
    for img in video:
        frames.append(np.array(img.permute(1, 2, 0)))
    clip = ImageSequenceClip(frames, fps=fps)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    clip.write_videofile(path, codec="libx264", bitrate="10M")


def resize_mp4(input_path:str, resize_to:tuple, output_path:str=None)->int:
    if output_path is None:
        tmp = Path(input_path)
        output_path=os.path.join(tmp.parent, tmp.stem+f"_{resize_to[-2]}_{resize_to[-1]}.mp4")

    tmp = Path(output_path).parent.resolve()
    os.makedirs(tmp, exist_ok=True)

    video = moviepy.VideoFileClip(input_path)
    output_video = video.resized(resize_to)
    output_video.write_videofile(output_path, codec="libx264", fps=video.fps)
    return output_path, video.n_frames

def repeate_mp4(input_path:str, repeate_to:int, output_path:str=None):
    if output_path is None:
        tmp = Path(input_path)
        output_path=os.path.join(tmp.parent, tmp.stem+f"_{resize_to[-2]}_{resize_to[-1]}.mp4")

    tmp = Path(output_path).parent.resolve()
    os.makedirs(tmp, exist_ok=True)

    video = moviepy.VideoFileClip(input_path)
    frames=[]
    index=0
    direct=1

    assert video.n_frames>0, "repeate_mp4 get empty video"
    for idx in range(repeate_to):
        frames.append(np.array(video.get_frame(index/video.fps)))
        if index<=0:
            direct=1
            
        elif index>=(video.n_frames-1):
            direct=-1

        index+=direct

        if video.n_frames==1:
            index=0


    clip = ImageSequenceClip(frames, fps=video.fps)
    clip.write_videofile(output_path, codec="libx264", bitrate="10M")
    

def ori_size_mapping_hwc(imgs, height_v_pad, width_v_pad):
    temp_imgs = copy.deepcopy(imgs)
    if height_v_pad>0:
        temp_imgs = temp_imgs[0:-height_v_pad, :, :]
    if width_v_pad>0:
        temp_imgs = temp_imgs[:, 0: -width_v_pad, :]
    return temp_imgs

def ori_size_mapping_chw(imgs, height_v_pad, width_v_pad):
    temp_imgs = copy.deepcopy(imgs)
    if len(temp_imgs.shape)>3:
        if height_v_pad>0:
            temp_imgs = temp_imgs[:, :, 0:-height_v_pad, :]
        if width_v_pad>0:
            temp_imgs = temp_imgs[:, :, :, 0: -width_v_pad]
    else:
        if height_v_pad>0:
            temp_imgs = temp_imgs[:, 0:-height_v_pad, :]
        if width_v_pad>0:
            temp_imgs = temp_imgs[:, :, 0: -width_v_pad]
    return temp_imgs