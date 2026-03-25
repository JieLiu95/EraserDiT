import torch
from PIL import Image
from decord import VideoReader, cpu
from transformers import AutoModel, AutoTokenizer

torch.manual_seed(100)

model = AutoModel.from_pretrained('/mnt/afs/liujie/ckpts/MiniCPM-o-2_6', trust_remote_code=True, init_audio=False,
    attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained('/mnt/afs/liujie/ckpts/MiniCPM-o-2_6', trust_remote_code=True)

video_reader = VideoReader("/mnt/afs/liujie/datasets/dance/bilibili_dance_cut_V/4441160/4441160-2023.07.01-BV1HV411M7q7-122-1502.mp4", ctx=cpu(0))
image = Image.fromarray(video_reader[0].asnumpy())

# First round chat 
question = "List the names of objects present in the image, briefly describe the scene in the image"

msgs = [{'role': 'user', 'content': [image, question]}]

answer = model.chat(
    msgs=msgs,
    tokenizer=tokenizer
)
print(answer + "\n\n")

# Second round chat, pass history context of multi-turn conversation
prompt2 = "Remove the woman, and briefly explain which objects are included in the image and answer in the form of object names and words. And what is the scene in briefly."

msgs.append({"role": "assistant", "content": [answer]})
msgs.append({"role": "user", "content": [prompt2]})

answer = model.chat(
    msgs=msgs,
    tokenizer=tokenizer
)
print(answer)