import os
import sys
import time
import torch
from random import randint, seed
from nanovllm import LLM, SamplingParams

# nano-vllm 极限性能基准测试脚本，测引擎的速度天花板，输入是随机 Token，只关注吞吐量
def main():
    # ===================== 基础配置 =====================
    seed(0)
    num_seqs = 1024
    max_input_len = 1024
    max_output_len = 1024
    model_path = os.path.expanduser("~/huggingface/Qwen3-1.7B/")
    test_type = "CUDAGraph"  # 优化后改这里
    output_file = f"nano_vllm_基准性能测试_{test_type}_{time.strftime('%Y%m%d_%H%M%S')}.txt"

    # ===================== 环境信息 =====================
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
    # 核心软件栈版本
    python_version = sys.version.split()[0]
    torch_version = torch.__version__
    cuda_version = torch.version.cuda if torch.cuda.is_available() else "N/A"
    # 操作系统信息
    os_info = os.popen("uname -sr").read().strip()  # Linux/Mac：内核版本；Windows可替换为"ver"
    # ===================== 初始化模型 =====================
    llm = LLM(model_path, enforce_eager=False, max_model_len=4096)

    # ===================== Warmup（预热，消除首次开销） =====================
    print("Warmup中...")
    llm.generate(["Benchmark: "], SamplingParams())

    # ===================== 生成测试数据 =====================
    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_output_len)) for _ in range(num_seqs)]

    # ===================== 核心性能测试 =====================
    print("正式测试中...")
    t_start = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t_total = time.time() - t_start

    # ===================== 计算指标 =====================
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t_total
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3  # 新增：记录峰值显存

    # ===================== 终端输出 =====================
    print("\n" + "="*60)
    print(f"【{test_type}】nano-vllm基准测试结果")
    print("="*60)
    print(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"操作系统: {os_info}")
    print(f"软件栈: Python={python_version}, PyTorch={torch_version}, CUDA={cuda_version}")
    print(f"GPU型号: {gpu_name}, 总显存: {gpu_mem_total:.2f}GB, 峰值显存: {peak_mem:.2f}GB")
    print(f"测试配置: 序列数={num_seqs}, 输入长度=100~{max_input_len}, 输出长度=100~{max_output_len}")
    print(f"总生成Token数: {total_tokens} tok")
    print(f"总耗时: {t_total:.2f} s")
    print(f"吞吐量: {throughput:.2f} tok/s")
    print("="*60)

    # ===================== 写入文件（结构化，新增软件栈） =====================
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# nano-vllm 官方基准性能测试报告\n\n")
        f.write(f"测试类型: {test_type}\n")
        f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"操作系统: {os_info}\n")
        f.write(f"Python版本: {python_version}\n")
        f.write(f"PyTorch版本: {torch_version}\n")
        f.write(f"CUDA版本: {cuda_version}\n")
        f.write(f"GPU型号: {gpu_name}\n")
        f.write(f"GPU总显存: {gpu_mem_total:.2f}GB\n")
        f.write(f"GPU峰值显存: {peak_mem:.2f}GB\n")
        f.write(f"测试配置: 序列数={num_seqs}, 输入长度=100~{max_input_len}, 输出长度=100~{max_output_len}\n")
        f.write(f"总生成Token数: {total_tokens}\n")
        f.write(f"总耗时: {t_total:.2f}s\n")
        f.write(f"吞吐量: {throughput:.2f}tok/s\n")

    print(f"\n✅ 基准测试报告已保存至：{output_file}")

if __name__ == "__main__":
    main()
