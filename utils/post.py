from .common import GlobalValues
from .common import test_time
from .post_pkg import torch_nchw_to_video, torch_nchw_to_video_stream, FFmpegWriter
import torch
from .common import GlobalValues

@test_time(enable=GlobalValues.ENABLE_PER)
def post_cuda(output_frames: torch.Tensor, ori_shape, model_video_shape, save_path="output.mp4", fps=30,write_to=True, append=False):
    output_frames = output_frames[:ori_shape[0],:,:ori_shape[1],:ori_shape[2]]
    torch_nchw_to_video(output_frames,save_path,fps,write_to=write_to,append=append)
    return None

@test_time(enable=GlobalValues.ENABLE_PER)
def post_cuda_normalized(output_frames: torch.Tensor, ori_shape, model_video_shape, save_path="output.mp4", fps=30,write_to=True, append=False):
    output_frames = (output_frames[:ori_shape[0],:,:ori_shape[1],:ori_shape[2]]*255).to(torch.uint8)
    torch_nchw_to_video(output_frames,save_path,fps,write_to=write_to,append=append)
    return None

@test_time(enable=GlobalValues.ENABLE_PER)
def post_stream(output_frames: torch.Tensor, ori_shape, model_video_shape,writer: FFmpegWriter, write_to=True):
    output_frames = output_frames[:ori_shape[0],:,:ori_shape[1],:ori_shape[2]]
    torch_nchw_to_video_stream(output_frames,writer=writer,write_to=write_to)
    return None

@test_time(enable=GlobalValues.ENABLE_PER)
def post_stream_normalized(output_frames: torch.Tensor, ori_shape, model_video_shape, 
                           crop_flag, videos_input_ori, video_ori, mask_ori, output_bbox,
                           writer: FFmpegWriter,write_to=True):
    output_frames = (output_frames[:ori_shape[0],:,:ori_shape[1],:ori_shape[2]]*255).to(torch.uint8)
    torch_nchw_to_video_stream(output_frames,writer=writer,
                               crop_flag=crop_flag, videos_input_ori=videos_input_ori, video_ori=video_ori, mask_ori=mask_ori,
                               output_bbox=output_bbox,
                               write_to=write_to)
    return None