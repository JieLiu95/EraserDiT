import csv
import decord
import math
import numpy as np
import torch
from typing import Tuple
from torchvision.transforms.v2 import Pad
from torch.utils import dlpack
from utils.common import GlobalValues
from utils.common import test_time

class VideoInpaintPre():
    def __init__(self, device=None,align_w=32,align_h=32,ksize=(7,7),dilate_iter=7, threshold=0.039, shift_alpha=9, TEMP_INFER_LEN=121, enable_approximate=True, crop_flag=False):
        super().__init__()
        self.device=device
        self.align_w=align_w
        self.align_h=align_h
        self.ksize=ksize
        self.dilate_iter=dilate_iter
        self.threshold=threshold
        self.shift_alpha=shift_alpha
        self.TEMP_INFER_LEN=TEMP_INFER_LEN
        self.enable_approximate=enable_approximate
        self.crop_flag=crop_flag

    @test_time(enable=GlobalValues.ENABLE_PER)
    def __call__(self, videos:torch.Tensor, masks:torch.Tensor,batch_idx=0, format="nchw")->Tuple[torch.Tensor, torch.Tensor, int]:
        '''
        return videos [NCHW], mask_input [NCHW], batch_size
        '''

        video_num_frame= videos.shape[0]
        assert video_num_frame>0, "VideoInpaintPre.complete_videos get empty video!"
        
        if format=="nchw":
            video_ori_align= self.align_video_nchw2nchw(videos)
            masks_ori_align= self.align_video_nchw2nchw(masks)
        elif format=="nhwc":
            video_ori_align= self.align_video_nhwc2nchw(videos)
            masks_ori_align= self.align_video_nhwc2nchw(masks)
        else:
            print("bad format for VideoInpaintPre.__call__, only support nchw and nhwc")
        
        wrapping_size=self.TEMP_INFER_LEN-self.shift_alpha
        head_batch=(batch_idx==0)
        if not head_batch:
            batch_size=math.ceil(video_num_frame/wrapping_size)
            num_frames_padded=batch_size*wrapping_size-video_num_frame
        else:
            batch_size = max(1, math.ceil((video_num_frame-self.shift_alpha)/(self.TEMP_INFER_LEN-self.shift_alpha)))
            num_frames_padded=(batch_size*wrapping_size+self.shift_alpha)-video_num_frame

        # repeat frame
        if num_frames_padded>0:
            video_ori_align_temp = video_ori_align.flip(0)[1:len(video_ori_align)-1]
            masks_ori_align_temp = masks_ori_align.flip(0)[1:len(masks_ori_align)-1]
            if num_frames_padded<=len(video_ori_align)-2:
                video_ori_align = torch.cat([video_ori_align, video_ori_align_temp[:num_frames_padded]], dim=0)
                masks_ori_align = torch.cat([masks_ori_align, masks_ori_align_temp[:num_frames_padded]], dim=0)
            else:
                video_ori_align = torch.cat([video_ori_align, video_ori_align_temp], dim=0)
                masks_ori_align = torch.cat([masks_ori_align, masks_ori_align_temp], dim=0)
                if head_batch:
                    inf_len = self.TEMP_INFER_LEN
                else:
                    inf_len = self.TEMP_INFER_LEN-self.shift_alpha
                repeat_num = max(1, math.ceil(inf_len/len(video_ori_align)))
                video_ori_align = video_ori_align.repeat(repeat_num, 1, 1, 1)[:inf_len]
                masks_ori_align = masks_ori_align.repeat(repeat_num, 1, 1, 1)[:inf_len]
        
        video_align_normalized, video_align_normalized_masked, mask_align_input = self.mask_video_nchw(video_ori_align,masks_ori_align,head_batch=head_batch)
        # videos_align_normalized_completed, mask_input_completed, batch_size = self.complete_videos(video_align_normalized_masked, mask_align_input,head_batch=head_batch)
        return video_align_normalized_masked, mask_align_input, batch_size
    
    def bbox_video(self, height_input_ori, width_input_ori, bbox_path, inference_idx):
        if height_input_ori > width_input_ori:
            crop_h = 1920
            crop_w = 1088
        else:
            crop_w = 1920
            crop_h = 1088
        xxyy = bbox_cal(bbox_path, crop_w, crop_h, width_input_ori, height_input_ori, inference_idx)
        if xxyy is not None:
            self.x1, self.x2, self.y1, self.y2 = xxyy
        return xxyy
    
    @staticmethod
    def TranslateShape(shape, src="nhwc", dst="nchw"):
        if set(src) != set(dst) or len(src) != len(dst):
            return None

        idxs=[]
        for c in dst:
            for i in range(len(src)):
                if dst[i]==c:
                    idxs.append(i)
        
        dsc_shape=[]
        for idx in idxs:
            dsc_shape.append(shape[idx])
        return dsc_shape

    @test_time(enable=GlobalValues.ENABLE_PER)
    def load_videos(self, video_path, mask_path, bbox_path=None, decord_device=decord.gpu(0), sample_rate=1,batch_idx=0) -> Tuple[np.ndarray|torch.Tensor, np.ndarray|torch.Tensor, int,int]:
        '''
        return video_sampled [NHWC], masks_sampled [NHWC], ori_fps
        ________________________
        videos data all in device same as decord_device. if device is None, means CPU
        '''
        if decord_device is None:
            decord_device = decord.cpu()
            # decord.bridge.set_bridge('torch')

        video_reader = decord.VideoReader(video_path,ctx=decord_device)
        video_fps = video_reader.get_avg_fps()
        video_num_frames_ori = len(video_reader)

        mask_reader = decord.VideoReader(mask_path,ctx=decord_device)
        mask_num_frames_ori = len(mask_reader)
        if video_num_frames_ori != mask_num_frames_ori:
            raise Exception(f"video-len({video_num_frames_ori}) different from mask-len({mask_num_frames_ori})")

        if batch_idx==0:
            start_index=0
            end_index=self.TEMP_INFER_LEN        # unuse
        else:
            start_index=sample_rate*(batch_idx*(self.TEMP_INFER_LEN-self.shift_alpha)+self.shift_alpha)
            end_index=sample_rate*((batch_idx+1)*(self.TEMP_INFER_LEN-self.shift_alpha)+self.shift_alpha)

        sample_index = list(range(start_index, min(video_num_frames_ori,end_index), sample_rate))
        print("sample_index",sample_index)
        if len(sample_index)<1:
            return None,None,None,None,None

        videos_input_ori = video_reader.get_batch(sample_index)
        masks_input_ori = mask_reader.get_batch(sample_index)
        videos_input_ori = dlpack.from_dlpack(videos_input_ori.to_dlpack())           # type: torch.Tensor
        masks_input_ori = dlpack.from_dlpack(masks_input_ori.to_dlpack())
        if self.crop_flag:
            videos = videos_input_ori[:, self.y1:self.y2, self.x1:self.x2, :]
            masks = masks_input_ori[:, self.y1:self.y2, self.x1:self.x2, :]
            return videos, masks, video_fps, videos_input_ori, masks_input_ori
        else:
            videos = videos_input_ori
            masks = masks_input_ori

            
            # return videos, masks, video_fps, videos_input_ori, masks_input_ori
            return videos, masks, video_fps, None, None
        
    @test_time(enable=GlobalValues.ENABLE_PER)
    def load_videos_by_scene(self, video_path, mask_path, bbox_path=None, decord_device=decord.gpu(0), sample_rate=1,batch_idx=0,scene = None) -> Tuple[np.ndarray|torch.Tensor, np.ndarray|torch.Tensor, int,int]:
        '''
        return video_sampled [NHWC], masks_sampled [NHWC], ori_fps
        ________________________
        videos data all in device same as decord_device. if device is None, means CPU
        '''
        scene_start_indx,scene_end_indx = scene
        if decord_device is None:
            decord_device = decord.cpu()
            # decord.bridge.set_bridge('torch')

        video_reader = decord.VideoReader(video_path,ctx=decord_device)
        video_fps = video_reader.get_avg_fps()
        video_num_frames_ori = len(video_reader)

        mask_reader = decord.VideoReader(mask_path,ctx=decord_device)
        mask_num_frames_ori = len(mask_reader)
        if video_num_frames_ori != mask_num_frames_ori:
            raise Exception(f"video-len({video_num_frames_ori}) different from mask-len({mask_num_frames_ori})")

        if batch_idx==0:
            start_index=0 + scene_start_indx
            end_index=self.TEMP_INFER_LEN + scene_start_indx     # unuse
        else:
            start_index=sample_rate*(batch_idx*(self.TEMP_INFER_LEN-self.shift_alpha)+self.shift_alpha+ scene_start_indx)
            end_index=sample_rate*((batch_idx+1)*(self.TEMP_INFER_LEN-self.shift_alpha)+self.shift_alpha+ scene_start_indx)

        sample_index = list(range(start_index, min(scene_end_indx,end_index), sample_rate))
        print("sample_index",sample_index)
        if len(sample_index)<1:
            return None,None,None,None,None

        videos_input_ori = video_reader.get_batch(sample_index)
        masks_input_ori = mask_reader.get_batch(sample_index)
        videos_input_ori = dlpack.from_dlpack(videos_input_ori.to_dlpack())           # type: torch.Tensor
        masks_input_ori = dlpack.from_dlpack(masks_input_ori.to_dlpack())
        if self.crop_flag:
            videos = videos_input_ori[:, self.y1:self.y2, self.x1:self.x2, :]
            masks = masks_input_ori[:, self.y1:self.y2, self.x1:self.x2, :]
            return videos, masks, video_fps, videos_input_ori, masks_input_ori
        else:
            videos = videos_input_ori
            masks = masks_input_ori

            
            # return videos, masks, video_fps, videos_input_ori, masks_input_ori
            return videos, masks, video_fps, None, None

    @test_time(enable=GlobalValues.ENABLE_PER)
    def align_video_nhwc2nchw(self, video: torch.Tensor)->Tuple[torch.Tensor,int,int]:
        '''
        can data-parallel
        '''
        if video is None:
            return None,0,0

        video.to(device=self.device)

        _, height, width, _ = video.shape
        height_new = math.ceil(height / self.align_h) * self.align_h    # 736
        width_new = math.ceil(width / self.align_w) * self.align_w      # 1280

        height_pad = height_new - height          # 16
        width_pad = width_new - width             # 0

        pad = Pad(padding=[0,0,width_pad,height_pad],fill=0,padding_mode="edge")
        result = video.contiguous().permute(0,3,1,2)
        result =pad(result)
        return pad(video.contiguous().permute(0,3,1,2))

    @test_time(enable=GlobalValues.ENABLE_PER)
    def align_video_nchw2nchw(self, video: torch.Tensor)->Tuple[torch.Tensor,int,int]:
        '''
        can data-parallel
        '''
        if video is None:
            return None,0,0
        
        video.to(device=self.device)

        n_frams, _ ,height, width = video.shape
        height_new = math.ceil(height / self.align_h) * self.align_h    # 736
        width_new = math.ceil(width / self.align_w) * self.align_w      # 1280

        height_pad = height_new - height          # 16
        width_pad = width_new - width             # 0

        pad = Pad(padding=[0,0,width_pad,height_pad],fill=0,padding_mode="edge")
        return pad(video)
    
    @test_time(enable=GlobalValues.ENABLE_PER)
    def mask_video_nchw(self, video_align: torch.Tensor, mask_align: torch.Tensor, head_batch=True) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        '''
        no data-parallel
        '''
        if video_align is None or mask_align is None:
            return None,None,None
        video_align.to(self.device)
        mask_align.to(self.device)

        # create MORPH_CROSS block， a litter different from cv with 
        dilate_conv_kernel= torch.zeros((mask_align.shape[1],1,*self.ksize),dtype=torch.float16,device=mask_align.device,requires_grad=False)
        dilate_conv_kernel[:,:,self.ksize[0]//2,:]=1
        dilate_conv_kernel[:,:,:,self.ksize[1]//2]=1

        mask_align = torch.where(mask_align>(255/2*self.threshold),1,0).to(torch.float16)         # to 1， avoid out-of-memory
        max_approximate=1
        for iter in range(self.dilate_iter):
            mask_align=torch.nn.functional.conv2d(mask_align,dilate_conv_kernel,stride=1,padding=(self.ksize[0]//2,self.ksize[1]//2),groups=mask_align.shape[1])
            max_approximate*=self.ksize[0]+self.ksize[1]-1
            if self.enable_approximate:
                if iter==(self.dilate_iter-1) or ((self.ksize[0]+self.ksize[1]-1)*max_approximate)>64000:
                    # to 1， avoid out-of-memory
                    mask_align = torch.where(mask_align>(max_approximate*self.threshold),1,0)
                    max_approximate=1
            else:
                mask_align = torch.where(mask_align>((self.ksize[0]+self.ksize[1]-1)*self.threshold),1,0)
            
            if iter==(self.dilate_iter-1):
                mask_align=mask_align.to(torch.uint8)
            else:
                mask_align=mask_align.to(torch.float16)
            
        video_align_normalized = video_align/255
        video_align_normalized_masked = video_align_normalized*(1-mask_align)     # mask to be white

        n_frame,channel,height,width=mask_align.shape
        shift_n_frame=1
        if head_batch:
            shift_n_frame=1
            wrapping_batch_size= (n_frame-1)//8      # shift batch size=1
            left_batch = (n_frame-1)%8
        else:
            shift_n_frame=0
            wrapping_batch_size= (n_frame)//8        # shift batch size=math.ceil(shift_alpha/8), example: 2
            left_batch = (n_frame)%8                 # 

        mask_align_compress_list = []
        # head batch
        if head_batch:
            mask_align_compress_list.append(mask_align[:1,:,:,:])

        # wrapping batch
        mask_wrapping_batch=torch.sum(mask_align[shift_n_frame:shift_n_frame+wrapping_batch_size*8,:,:,:].reshape(wrapping_batch_size,8,channel,height,width),dim=1,keepdim=False)
        mask_wrapping_batch = torch.where(mask_wrapping_batch>=1,255,0)
        mask_align_compress_list.append(mask_wrapping_batch)

        if left_batch>0:
            mask_left_batch=torch.sum(mask_align[-left_batch:,:,:,:],dim=0,keepdim=True)
            mask_left_batch = torch.where(mask_left_batch>=1,255,0)
            mask_align_compress_list.append(mask_left_batch)

        mask_align_compress = torch.cat(mask_align_compress_list,dim=0,out=None).to(torch.float32)

        # to gray image
        gray_conv_kernel= torch.tensor([[[[0.299/255]],[[0.587/255]],[[0.0004]]]],dtype=torch.float32,device=mask_align.device,requires_grad=False)   # fuse to-gray and normalize, shape:(1,3,1,1)
        mask_align_compress_gray_normalized=torch.nn.functional.conv2d(mask_align_compress,gray_conv_kernel,stride=1,padding=0)

        return video_align_normalized, video_align_normalized_masked, mask_align_compress_gray_normalized


    @test_time(enable=GlobalValues.ENABLE_PER)
    def complete_videos(self,videos: torch.Tensor, mask_input: torch.Tensor,head_batch=True)->Tuple[torch.Tensor,torch.Tensor,int]:  
        '''
        NO data-parallel
        if head_batch is True, videos.shape[0]==batch_size*(TEMP_INFER_LEN-shift_alpha)+shift_alpha, else videos.shape[0]==batch_size*(TEMP_INFER_LEN-shift_alpha)
        '''
        if videos is None or mask_input is None:
            return None,None,0

        videos.to(self.device)
        mask_input.to(self.device)

        video_num_frame= videos.shape[0]
        assert video_num_frame>0, "VideoInpaintPre.complete_videos get empty video!"

        wrapping_size=self.TEMP_INFER_LEN-self.shift_alpha
        if not head_batch:
            batch_size=math.ceil(video_num_frame/wrapping_size)
            num_frames_padded=batch_size*wrapping_size-video_num_frame
        else:
            batch_size = max(1, math.ceil((video_num_frame-self.shift_alpha)/(self.TEMP_INFER_LEN-self.shift_alpha)))
            num_frames_padded=(batch_size*wrapping_size+self.shift_alpha)-video_num_frame
        num_frames_padded_mask = num_frames_padded//8

        # repeat frame
        if num_frames_padded>0:
            videos_temp = videos.flip(0)[1:len(videos)-1]
            videos = torch.cat([videos, videos_temp[:num_frames_padded]], dim=0)
            if num_frames_padded_mask>0:
                mask_input_temp  = mask_input.flip(0)[1:len(mask_input)-1]
                mask_input = torch.cat([mask_input, mask_input_temp[:num_frames_padded_mask]], dim=0)
        return videos, mask_input, batch_size

def bbox_cal(bbox_path, crop_w, crop_h, input_w, input_h, inference_idx):
    bboxs = list(csv.DictReader(open(bbox_path, "r", encoding="utf-8-sig")))
    x1_list = []
    x2_list = []
    y1_list = []
    y2_list = []
    for item in bboxs:
        bbox_temp = item["bboxes"]    
        # bbox_temp = bbox_temp.split("[[")[1].split("]]")[0].split(", ")
        bbox_temp = bbox_temp.split("[[")[1].split("]]")[0].split("], [")[inference_idx].split(", ")
        x1, y1, x2, y2 = int(bbox_temp[0]), int(bbox_temp[1]), int(bbox_temp[2]), int(bbox_temp[3]), 
        if x1+x2+y1+y2 ==0:
            continue
        else:
            x1_list.append(x1)
            x2_list.append(x2)
            y1_list.append(y1)
            y2_list.append(y2)
    x1_min = np.min(x1_list)
    x2_max = np.max(x2_list)
    y1_min = np.min(y1_list)
    y2_max = np.max(y2_list)
    if x2_max - x1_min>crop_w or y2_max - y1_min >crop_h:
        return None
    else:
        x1_r = max((x2_max + x1_min - crop_w)//2, 0)
        if crop_w + x1_r > input_w:
            x2_r = input_w
            x1_r = input_w - crop_w
        else:
            x2_r = crop_w + x1_r
        y1_r = max((y2_max + y1_min - crop_h)//2, 0)
        if crop_h + y1_r > input_h:
            y2_r = input_h
            y1_r = input_h - crop_h
        else:
            y2_r = crop_h + y1_r

    return x1_r, x2_r, y1_r, y2_r