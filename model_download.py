from huggingface_hub import snapshot_download
import os

# 说明：用于下载Qwen3-1.7B的模型权重，保存在~/huggingface/Qwen3-1.7B/目录下
os.makedirs('~/huggingface/Qwen3-1.7B/'.replace('~', os.path.expanduser('~')), exist_ok=True)
snapshot_download(
    repo_id='Qwen/Qwen3-1.7B',
    local_dir='~/huggingface/Qwen3-1.7B/'.replace('~', os.path.expanduser('~')),
    local_dir_use_symlinks=False
)