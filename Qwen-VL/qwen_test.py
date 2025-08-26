#from modelscope import snapshot_download
import os
# 在导入任何其他库之前设置这些环境变量
os.environ['DISABLE_TE'] = '1'
os.environ['USE_TRANSFORMER_ENGINE'] = '0'

from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
#import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Downloading model checkpoint to a local dir model_dir
# model_dir = snapshot_download('qwen/Qwen-VL')
# model_dir = snapshot_download(repo_id='qwen/Qwen-VL-Chat')
model_dir = "/mnt/ali-sh-1/usr/yujing1/workspaces/MLLM/Qwen-VL-from-scratch/Qwen-VL/models/Qwen-VL"
# 确保本地模型代码能被 import
sys.path.append(model_dir)
# 可选：显式指定 endpoint 避免联网
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# Loading local checkpoints
# trust_remote_code is still set as True since we still load codes from local dir instead of transformers
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="cuda",
    trust_remote_code=True
).eval()

print("Model and tokenizer loaded successfully from local path.")