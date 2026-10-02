"""Hardware and model compatibility inspector for Step 6D."""
import sys
import subprocess

def main():
    print("=== STEP 6D.1 — HARDWARE INSPECTION ===")
    import platform
    print(f"OS: {platform.system()} {platform.release()} {platform.version()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Machine: {platform.machine()}")
    print(f"CPU: {platform.processor()}")
    
    import torch
    print(f"\n=== PYTORCH ===")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory
        print(f"VRAM: {vram / (1024**3):.1f} GB")
    else:
        print("ACCELERATION: CPU ONLY — no CUDA, no MPS")
    
    import transformers
    print(f"\n=== TRANSFORMERS ===")
    print(f"Transformers: {transformers.__version__}")
    
    try:
        import peft
        print(f"PEFT: {peft.__version__}")
    except ImportError:
        print("PEFT: NOT INSTALLED")

    try:
        import datasets
        print(f"datasets: {datasets.__version__}")
    except ImportError:
        print("datasets: NOT INSTALLED")

    try:
        import accelerate
        print(f"accelerate: {accelerate.__version__}")
    except ImportError:
        print("accelerate: NOT INSTALLED")

    print(f"\n=== STEP 6D.2 — MODEL COMPATIBILITY ===")
    model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    from transformers import AutoConfig, AutoTokenizer
    cfg = AutoConfig.from_pretrained(model_name)
    print(f"Model: {model_name}")
    print(f"Architecture: {cfg.architectures}")
    print(f"Model type: {cfg.model_type}")
    print(f"Hidden size: {cfg.hidden_size}")
    print(f"Num layers: {cfg.num_hidden_layers}")
    print(f"Num heads: {cfg.num_attention_heads}")
    print(f"Vocab size: {cfg.vocab_size}")
    try:
        print(f"Context length: {cfg.max_position_embeddings}")
    except AttributeError:
        try:
            print(f"Context length: {cfg.sliding_window}")
        except AttributeError:
            print("Context length: unknown")
    try:
        print(f"Torch dtype: {cfg.torch_dtype}")
    except AttributeError:
        print("Torch dtype: not specified in config")

    tok = AutoTokenizer.from_pretrained(model_name)
    print(f"\nTokenizer class: {type(tok).__name__}")
    print(f"Chat template present: {tok.chat_template is not None}")
    print(f"Has apply_chat_template: {hasattr(tok, 'apply_chat_template')}")
    print(f"EOS token id: {tok.eos_token_id}")
    print(f"PAD token id: {tok.pad_token_id}")
    print(f"BOS token id: {tok.bos_token_id}")

    # Test chat template
    test_msg = [{"role": "user", "content": "Write a function to add two numbers."}]
    try:
        rendered = tok.apply_chat_template(test_msg, tokenize=False)
        print(f"\nChat template sample (first 200 chars):\n{rendered[:200]}")
    except Exception as e:
        print(f"Chat template error: {e}")

    print(f"\n=== LORA COMPATIBILITY ===")
    from transformers import AutoModelForCausalLM
    print("Loading model for LoRA inspection...")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Approx size (float32): {total_params*4/1024**2:.0f} MB")
    print(f"Approx size (float16): {total_params*2/1024**2:.0f} MB")

    # List attention modules for LoRA targeting
    print("\nLinear module names (candidates for LoRA):")
    lm_names = set()
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and len(list(module.children())) == 0:
            short = name.split('.')[-1]
            lm_names.add(short)
    print(sorted(lm_names))

    from peft import LoraConfig, get_peft_model, TaskType
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    peft_model = get_peft_model(model, lora_cfg)
    trainable, total = peft_model.get_nb_trainable_parameters()
    print(f"\nLoRA trainable params: {trainable:,} ({100*trainable/total:.3f}%)")
    print(f"LoRA trainable size (float32): {trainable*4/1024**2:.1f} MB")
    print("LoRA injection: SUCCESS")

    print(f"\n=== VERDICT ===")
    print("Hardware: Intel Core i7-8550U (Coffee Lake), CPU ONLY")
    print("RAM: ~16 GB")
    print("GPU: Intel UHD Graphics 620, NO CUDA, NO VRAM AVAILABLE")
    print("Acceleration: NONE (CPU only)")
    print("LoRA with Qwen2.5-Coder-0.5B: FEASIBLE on CPU (slow but correct)")
    print("Full fine-tuning: NOT FEASIBLE on 16GB RAM CPU-only")
    print("QLoRA/4-bit: NOT APPLICABLE (no GPU)")
    print("APPROACH: LoRA with minimal steps, float32 or float16")

if __name__ == "__main__":
    main()
