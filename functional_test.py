import os
import time
import torch
from nanovllm import LLM, SamplingParams

# 功能测试脚本：验证真实Prompt的推理性能+生成质量（适配RTX3080 20G）
def main():
    # ===================== 基础配置 =====================
    MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3-1.7B")
    TEST_TYPE = "优化前"  # 优化后修改为"优化后"
    OUTPUT_FILE = f"nano_vllm_功能测试_{TEST_TYPE}_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    
    # ===================== 初始化模型 =====================
    print(f"【{TEST_TYPE}】开始Qwen3-1.7B模型功能测试（RTX3080 20G）...")
    llm = LLM(MODEL_PATH, enforce_eager=True, tensor_parallel_size=1)
    
    # ===================== 测试用例 =====================
    # 采样参数
    sampling_params = SamplingParams(
        temperature=0.6,
        max_tokens=512,
    )
    # Qwen模型标准prompt格式：让模型明确用户指令，输出简洁回答
    def format_qwen_prompt(prompt):
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    raw_prompts = [
        "介绍一下人工智能的发展历史。",
        "列出100以内的所有质数。",
        "什么是大语言模型？",
        "写一篇关于春天的散文。",
        "解释一下操作系统的进程调度算法。",
        "介绍深度学习的核心概念。",
        "写一段Python代码实现快速排序。",
        "什么是自然语言处理？",
        "解释一下计算机网络的TCP/IP协议。",
        "介绍一下量子计算的基本原理。",
        "什么是Transformer架构？",
        "解释一下KV缓存的作用和原理。",
        "写一篇关于考研复试的准备攻略。",
        "介绍一下CUDA编程的核心概念。",
        "什么是PagedAttention？",
        "解释一下Python list和numpy数组的区别。",
        "写一段C++代码实现单链表。",
        "介绍一下大模型推理的核心瓶颈。",
        "解释一下短作业优先调度算法的优缺点。",
        "什么是混合精度推理？",
    ]
    # 格式化所有prompt
    prompts = [format_qwen_prompt(p) for p in raw_prompts]

    # ===================== 执行测试 =====================
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    total_time = time.time() - start_time
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3

    # ===================== 计算核心性能指标 =====================
    total_decode_tokens = sum(len(output["token_ids"]) for output in outputs)
    decode_throughput = total_decode_tokens / total_time
    gpu_name = torch.cuda.get_device_name(0)

    # ===================== 终端输出（只展示纯回答内容） =====================
    print("\n" + "="*50)
    print(f"【{TEST_TYPE}】Qwen3-1.7B 测试结果（RTX3080 20G）")
    print("="*50)
    print(f"GPU型号          : {gpu_name}")
    print(f"GPU峰值显存占用   : {peak_memory:.2f} GB")
    print(f"总推理耗时        : {total_time:.2f} s")
    print(f"Decode总Token数  : {total_decode_tokens}")
    print(f"Decode吞吐量     : {decode_throughput:.2f} tok/s")
    print("="*50)
    print(f"【前三个Prompt&完整输出】")
    for i in range(3):
        print(f"Prompt  : {raw_prompts[i]}")  # 展示原始prompt（非格式化后的）
        # 提取纯回答内容（去掉Qwen格式前缀）
        pure_output = outputs[i]['text'].replace("<|im_end|>", "").strip()
        print(f"Output  : {pure_output}")
        print("\n" + "-"*40 + "\n")
    print("="*50)

    # ===================== 写入TXT文件（优化格式） =====================
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        # 1. 测试元信息
        f.write(f"# nano-vllm 功能测试报告（单次运行）\n")
        f.write(f"测试类型        : {TEST_TYPE}\n")
        f.write(f"测试时间        : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"GPU型号        : {gpu_name}\n")
        f.write(f"模型路径        : {MODEL_PATH}\n")
        f.write(f"测试Prompt数量  : {len(prompts)}\n")
        f.write("\n\n")
        
        # 2. 核心性能指标
        f.write(f"# 核心性能指标（用于优化前后对比）\n")
        f.write(f"GPU峰值显存占用    : {peak_memory:.2f} GB\n")
        f.write(f"总推理耗时         : {total_time:.2f} s\n")
        f.write(f"Decode总Token数   : {total_decode_tokens}\n")
        f.write(f"Decode吞吐量      : {decode_throughput:.2f} tok/s\n")
        f.write("\n\n")
        
        # 3. 前三个Prompt&完整输出（优化格式）
        f.write(f"# 前三个Prompt&完整输出（验证生成质量）\n")
        for i in range(3):
            f.write(f"Prompt: {raw_prompts[i]}\n")
            pure_output = outputs[i]['text'].replace("<|im_end|>", "").strip()
            f.write(f"Output: {pure_output}\n")
            f.write("\n" + "-"*40 + "\n\n")

    print(f"\n✅ 测试报告已保存至：{OUTPUT_FILE}")
    print("⚠️  优化后测试时，仅需修改 TEST_TYPE = '优化后' 即可")


if __name__ == "__main__":
    main()