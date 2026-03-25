from .colorfix_wmask import adaptive_instance_normalization_mask
from .common import GlobalValues
from .common import test_time
import torch
from .common import GlobalValues
from moviepy import ImageSequenceClip,VideoFileClip,concatenate_videoclips
import os
from pathlib import Path
import shutil
import ffmpeg
import numpy as np


@test_time(enable=GlobalValues.ENABLE_PER)
def torch_nchw_to_video(tensor: torch.Tensor, path: str, fps, write_to=True, append=False):
    return torch_nhwc_to_video(tensor=tensor.contiguous().permute(0,2,3,1),path=path,fps=fps,write_to=write_to,append=append)

@test_time(enable=GlobalValues.ENABLE_PER)
def torch_nhwc_to_video(tensor: torch.Tensor, path: str, fps, write_to=True, append=False):
    tmp = Path(path).parent.resolve()
    os.makedirs(tmp, exist_ok=True)

    clips=[]
    raw_clip=None
    if os.path.exists(path) and append:
        raw_clip = VideoFileClip(path)
        clips.append(raw_clip)
        fps=clips[0].fps
        if GlobalValues.DEBUG:
            print(f"append data to raw video:{clips[0].reader.nframes}")
    tensor_numpy_nhwc=tensor.cpu().numpy()
    new_clip = ImageSequenceClip(list(tensor_numpy_nhwc), fps=fps)
    clips.append(new_clip)

    clip = concatenate_videoclips(clips)
    if write_to:   
        clip.write_videofile("_cache_.mp4" if raw_clip is not None else path, codec="libx264")
    
        if raw_clip is not None:
            raw_clip.close()
            clip.close()
            shutil.move("_cache_.mp4", path)
    return None

class FFmpegWriter:
    def __init__(self, path, width,height,fps, input_fmt='rgb24',vcodec='libx264',output_fmt='yuv420p',bitrate='10M', loglevel='warning'):
        self.process=(ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt=input_fmt,
                        s='{}x{}'.format(width, height), r=fps)
                .output(path, vcodec=vcodec, pix_fmt=output_fmt,loglevel=loglevel,
                        video_bitrate=bitrate, r=fps)
                .overwrite_output()
                .run_async(pipe_stdin=True))
        self.is_close = False

    def Write(self, data: bytes)->bool:
        if self.is_close:
            return False
        
        try:
            self.process.stdin.write(data)
        except Exception as ex:
            return False
        return True

    def Close(self):
        if not self.is_close:
            self.process.stdin.close()
            self.process.wait()
            self.is_close=True
    
    def __del__(self):
        self.Close()

@test_time(enable=GlobalValues.ENABLE_PER)
def torch_nchw_to_video_stream(tensor: torch.Tensor, writer: FFmpegWriter, 
                               crop_flag, videos_input_ori, video_ori, mask_ori, output_bbox,
                               write_to=True):
    return torch_nhwc_to_video_stream(tensor=tensor.contiguous().permute(0,2,3,1),writer=writer, 
                                      crop_flag=crop_flag, videos_input_ori=videos_input_ori, video_ori=video_ori, mask_ori=mask_ori,
                                      output_bbox=output_bbox,
                                      write_to=write_to)

@test_time(enable=GlobalValues.ENABLE_PER)
def torch_nhwc_to_video_stream(tensor: torch.Tensor, writer: FFmpegWriter, 
                               crop_flag, videos_input_ori, video_ori, mask_ori, output_bbox,
                               write_to=True):
    if not write_to:
        return
    tensor = tensor.cpu()
    output_frames_np = []
    
    # color fix and merge into the original video,ref original video without mask area
    output_frames_np = adaptive_instance_normalization_mask(
        content_feat=tensor.permute(0,3,1,2)/255.0,
        style_feat=video_ori.permute(0,3,1,2) / 255.0,
        refer_mask=(1 - mask_ori.permute(0,3,1,2)), # refer_mask=mask_gaussian,
        valid_mask=None, # (mask_gaussian + mask_ori),
        type="RGB",
        per_channel=True,
        )
    output_frames_np = (output_frames_np.permute(0,2,3,1)*255.).numpy().astype(np.uint8)
    output_frames_np = np.clip(output_frames_np,0,255)
    # tensor_numpy_nhwc=tensor.cpu().numpy().astype(np.uint8)
    if crop_flag:
        x1, x2, y1, y2 = output_bbox
        videos_input_ori = videos_input_ori.numpy()
        videos_input_ori[:, y1:y2, x1:x2, :] = np.array(output_frames_np)
        for idx in range(videos_input_ori.shape[0]):
            writer.Write(videos_input_ori[idx].tobytes())
    else:
        for idx in range(len(output_frames_np)):
            writer.Write(np.ascontiguousarray(output_frames_np[idx]).tobytes())
            # writer.Write(output_frames_np[idx].tobytes())
    return None
