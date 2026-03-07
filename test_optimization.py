import os
import time
import torch
from nanovllm import LLM, SamplingParams

# 说明：此脚本用于测试nano-vllm的优化效果
def main():
    # Qwen3-1.7B模型路径
    model_path = os.path.expanduser("~/huggingface/Qwen3-1.7B")
    # 加载模型
    llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)
    
    # 测试用例：20个长短混合Prompt
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "介绍一下人工智能的发展历史",
        "列出100以内的所有质数",
        "什么是大语言模型？",
        "写一篇关于春天的散文",
        "解释一下操作系统的进程调度算法",
        "介绍深度学习的核心概念",
        "写一段Python代码实现快速排序",
        "什么是自然语言处理？",
        "解释一下计算机网络的TCP/IP协议",
        "介绍一下量子计算的基本原理",
        "什么是Transformer架构？",
        "解释一下KV缓存的作用和原理",
        "写一篇关于考研复试的准备攻略",
        "介绍一下CUDA编程的核心概念",
        "什么是PagedAttention？",
        "解释一下Python list和numpy数组的区别",
        "写一段C++代码实现单链表",
        "介绍一下大模型推理的核心瓶颈",
        "解释一下短作业优先调度算法的优缺点",
        "什么是混合精度推理？",
    ]

    # 执行测试
    print("开始Qwen3-1.7B模型优化效果测试（适配3080 20G）...")
    torch.cuda.reset_peak_memory_stats()  # 重置显存统计
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    total_time = time.time() - start_time
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # 峰值显存（GB）

    # 统计指标
    total_decode_tokens = sum(len(output["token_ids"]) for output in outputs)
    decode_throughput = total_decode_tokens / total_time

    # 输出结果
    print("\n===== Qwen3-1.7B模型测试结果（3080 20G） =====")
    print(f"GPU峰值显存占用：{peak_memory:.2f}GB")
    print(f"总推理耗时：{total_time:.2f}s")
    print(f"Decode总Token数：{total_decode_tokens}，吞吐量：{decode_throughput:.2f} tok/s")
    print("\n===== 生成结果示例（前3个） =====")
    for i in range(3):  # 只打印前3个结果，避免终端刷屏
        print(f"\nPrompt：{prompts[i][:50]}...")
        print(f"生成结果：{outputs[i]['text'][:100]}...")

if __name__ == "__main__":
    main()