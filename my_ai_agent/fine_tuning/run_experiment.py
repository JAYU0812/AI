"""Step 6D — Experiment Runner CLI.

Executes the full controlled fine-tuning experiment and generates the final report:
    1. Preflight & Hardware checks
    2. Baseline evaluation (Step 6C test split, casual probes, safety probes, generalization)
    3. Pilot run (3 micro-steps verification)
    4. LoRA fine-tuning training loop (3 epochs on 32 training examples)
    5. Adapter checkpoint saving & verification
    6. Fine-tuned model evaluation
    7. Comparative analysis & regression checking
    8. Full markdown report generation (Step 6D.29)
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from my_ai_agent.fine_tuning.config import get_default_config
from my_ai_agent.fine_tuning.experiment import FineTuningExperiment
from my_ai_agent.fine_tuning.report import generate_report


def main():
    print("=" * 70)
    print("STARTING STEP 6D: CONTROLLED FINE-TUNING EXPERIMENT")
    print("=" * 70)
    start_time = time.time()

    config = get_default_config()
    print(f"Base Model: {config.base_model_name}")
    print(f"Experiment ID: {config.experiment_id}")
    print(f"Device: CPU float32")
    print(f"LoRA Rank: {config.lora_r}, Alpha: {config.lora_alpha}")
    print(f"Epochs: {config.num_train_epochs}, Batch Size: {config.per_device_train_batch_size}, Grad Accum: {config.gradient_accumulation_steps}")

    experiment = FineTuningExperiment(config)
    results = experiment.run_full_experiment()

    print("\n" + "=" * 70)
    print("GENERATING FINAL EXPERIMENT REPORT")
    print("=" * 70)

    report_path = generate_report(results)
    elapsed = time.time() - start_time
    print(f"\n[DONE] Full experiment completed in {elapsed:.1f}s ({elapsed/60:.2f}m)")
    print(f"Report available at: {report_path}")


if __name__ == "__main__":
    main()
