import argparse
import json
import time
import torch
from transformers import AutoTokenizer

import llama

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text generation with a Llama model.")

    parser.add_argument("--model", type=str, required=True, help="Path to the model.")
    parser.add_argument(
        "--prompts", type=str, nargs="+", required=True, help="List of prompts for text generation."
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=64, help="Maximum number of new tokens to generate."
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help='Device to use for inference (e.g., "cuda", "cpu").'
    )
    parser.add_argument(
        "--num-warmup-iterations", type=int, default=1, help="For torch.compile, warmup must be >= 1."
    )
    parser.add_argument(
        "--num-profiling-iterations", type=int, default=3, help="Number of iterations for performance measurement."
    )
    parser.add_argument(
        "--use-triton", action="store_true", help="Enable hand-written Triton kernels."
    )
    parser.add_argument(
        "--use-compile", action="store_true", help="Enable torch.compile optimization."
    )

    args = parser.parse_args()

    use_triton = args.use_triton
    use_compile = args.use_compile
    
    if use_compile and use_triton:
        print("⚠️ Warning: compile 和 triton 不能同时开启！为了公平 PK，已自动关闭手写 Triton，纯测 torch.compile。")
        use_triton = False

    if use_compile:
        mode_name = "torch.compile"
    elif use_triton:
        mode_name = "Hand-written Triton"
    else:
        mode_name = "PyTorch Native"

    print(f"🔥 PK Mode: {mode_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  

    inputs = tokenizer(args.prompts, padding=True, return_tensors="pt").to(args.device)
    
    # 加载模型
    model = llama.ModelForCausalLM.from_pretrained(args.model).to(args.device)

    # 🚀 正确调用模型内部的 compile 方法！
    if use_compile:
        model.compile()

    texts = []

    # Warmup 阶段 (对于 compile 模式，这一步至关重要，是真正的编译发生时刻)
    print("⏳ Starting Warmup...")
    for i in range(args.num_warmup_iterations):
        outputs = model.generate(inputs.input_ids, max_new_tokens=args.max_new_tokens, use_triton=use_triton)

    if args.device == "cuda":
        torch.cuda.synchronize()

    elapsed_time = 0

    # Profiling 阶段
    print(f"🚀 Profiling {args.num_profiling_iterations} iterations...")
    for _ in range(args.num_profiling_iterations):
        start_time = time.time()

        outputs = model.generate(
            inputs.input_ids, 
            max_new_tokens=args.max_new_tokens,
            use_triton=use_triton
        )

        if args.device == "cuda":
            torch.cuda.synchronize()

        end_time = time.time()
        elapsed_time += end_time - start_time
        texts.append(tokenizer.batch_decode(outputs, skip_special_tokens=True))

    average_time = elapsed_time / args.num_profiling_iterations
    num_input_tokens = inputs["input_ids"].size(-1)
    num_output_tokens = outputs.size(-1) - num_input_tokens
    num_tokens_per_second = num_output_tokens / average_time

    print(
        json.dumps(
            {
                "mode": mode_name,
                "average_time": average_time,
                "num_tokens_per_second": num_tokens_per_second,
            },
            indent=4 
        )
    )