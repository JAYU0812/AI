"""Step 6D — End-to-End Experiment Orchestrator.

Runs the complete controlled fine-tuning experiment:

    6D.1  Hardware inspection
    6D.2  Model compatibility verification
    6D.8  Dataset preflight
    6D.9  Token length analysis
    6D.10 Baseline evaluation on Step 6C test split
    6D.12 Pilot run
    6D training loop (LoRA)
    6D.16 Checkpoint verification
    6D.17 Fine-tuned evaluation on Step 6C test split
    6D.24 Generalization probes
    6D.25 Casual conversation smoke test
    6D.26 Resource analysis
    6D.27 Reproducibility artifacts

All results are saved to:
    my_ai_agent/fine_tuning/results/experiment_001/

IMPORTANT:
    - The Step 6A held-out benchmark is NOT run here.
      It requires the full evaluation framework (my_ai_agent/evaluation/).
    - We document the previously recorded baseline (62.5%) as the reference.
    - Fine-tuned model evaluation on the Step 6A benchmark should be run
      separately after loading the adapter.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .config import ExperimentConfig, get_default_config, get_pilot_config
from .dataset import FineTuningDataset, run_preflight
from .tokenizer import load_tokenizer, compute_token_stats, FineTuningTokenizedDataset
from .resource_monitor import ResourceMonitor, get_environment_info
from .trainer import LoRATrainer, build_lora_model
from .checkpoint import save_final_adapter, load_adapter_for_inference, verify_adapter_inference
from .evaluator import Step6CTestEvaluator
from .safety import run_safety_probes
from .comparison import (
    build_comparison_dict,
    save_comparison,
    BASELINE_6A_RESULTS,
)
from .inference import BaselineInference, FineTunedInference


class FineTuningExperiment:
    """Step 6D: Controlled fine-tuning experiment orchestrator."""

    def __init__(self, config: Optional[ExperimentConfig] = None):
        self.config = config or get_default_config()
        self.config.create_dirs()
        self.results: Dict[str, Any] = {"experiment_id": self.config.experiment_id}
        self._monitor = ResourceMonitor(
            interval_sec=self.config.monitor_interval_sec,
            output_path=str(Path(self.config.metrics_dir) / "resource_monitor.json"),
        )

    def run_hardware_check(self) -> Dict[str, Any]:
        """Step 6D.1 + 6D.2: Hardware inspection and model compatibility."""
        print("\n" + "="*60)
        print("STEP 6D.1-6D.2: Hardware & Model Inspection")
        print("="*60)

        env = get_environment_info()
        print(f"OS: {env['os']}")
        print(f"CPU: {env['cpu']}")
        print(f"RAM: {env.get('ram_gb', 'unknown')} GB")
        print(f"CUDA Available: {env['cuda_available']}")
        print(f"Acceleration: {env['acceleration']}")
        print(f"PyTorch: {env['torch']}")
        print(f"Transformers: {env['transformers']}")
        print(f"PEFT: {env['peft']}")

        hw_verdict = {
            "cuda_available": env["cuda_available"],
            "acceleration": env["acceleration"],
            "approach": "LoRA on CPU (float32)" if not env["cuda_available"] else "LoRA on GPU",
            "full_fine_tuning_feasible": False,
            "qlora_applicable": False,
            "lora_feasible": True,
        }

        env["hardware_verdict"] = hw_verdict
        self.results["environment"] = env

        # Save environment info
        env_path = Path(self.config.experiment_dir) / "environment_info.json"
        with open(env_path, "w") as f:
            json.dump(env, f, indent=2, default=str)
        print(f"Environment info saved: {env_path}")

        return env

    def run_dataset_preflight(self) -> Dict[str, Any]:
        """Step 6D.8: Dataset preflight and Step 6D.9: Token length analysis."""
        print("\n" + "="*60)
        print("STEP 6D.8-6D.9: Dataset Preflight & Token Analysis")
        print("="*60)

        dataset = FineTuningDataset(self.config)
        dataset.load_all(verify_manifest=True)
        print(f"Loaded: {len(dataset.train_records)} train, "
              f"{len(dataset.val_records)} val, "
              f"{len(dataset.test_records)} test")

        tokenizer = load_tokenizer(self.config.base_model_name)

        # Preflight
        preflight = run_preflight(
            train_records=dataset.train_records,
            val_records=dataset.val_records,
            tokenizer=tokenizer,
            max_seq_length=self.config.max_seq_length,
        )
        print(f"Preflight status: {preflight['status']}")
        if preflight["critical_errors"]:
            print(f"CRITICAL ERRORS: {preflight['critical_errors']}")
            raise RuntimeError(f"Dataset preflight FAILED: {preflight['critical_errors']}")
        if preflight["warnings"]:
            print(f"Warnings: {len(preflight['warnings'])}")

        # Token stats
        all_texts = dataset.get_formatted_train() + dataset.get_formatted_val()
        token_stats = compute_token_stats(all_texts, tokenizer)
        print(f"Token stats: min={token_stats['min']}, max={token_stats['max']}, "
              f"mean={token_stats['mean']:.0f}, p95={token_stats['p95']}, "
              f"max_seq_length={self.config.max_seq_length}")
        print(f"Examples needing truncation: {preflight['truncation_count']}")

        preflight["token_stats"] = token_stats
        self.results["dataset_preflight"] = preflight
        self.results["dataset_summary"] = dataset.summary()

        # Save
        preflight_path = Path(self.config.experiment_dir) / "dataset_preflight_report.json"
        with open(preflight_path, "w") as f:
            json.dump(preflight, f, indent=2, default=str)
        print(f"Preflight report saved: {preflight_path}")

        return preflight, dataset, tokenizer

    def run_baseline_evaluation(self, dataset: Any, tokenizer: Any) -> Dict[str, Any]:
        """Step 6D.10: Evaluate baseline on Step 6C test split."""
        print("\n" + "="*60)
        print("STEP 6D.10: Baseline Evaluation on Step 6C Test Split")
        print("="*60)

        evaluator = Step6CTestEvaluator()
        baseline = BaselineInference(self.config.base_model_name, self.config)
        baseline.load()

        self._monitor.record_event("baseline_eval_start")

        baseline_test = evaluator.evaluate_on_test_split(
            inference_fn=baseline.generate,
            test_records=dataset.test_records,
            label="baseline",
        )
        print(f"Baseline test split score: {baseline_test['average_score']:.3f}")

        # Casual conversation baseline
        baseline_casual = evaluator.run_casual_conversation(baseline.generate, "baseline")
        print(f"Baseline casual conversation: {baseline_casual['verdict']} "
              f"({baseline_casual['coherent_count']}/{baseline_casual['total_probes']})")

        # Safety probes baseline
        baseline_safety = run_safety_probes(baseline.generate)
        print(f"Baseline safety: {baseline_safety['pass_count']}/{baseline_safety['total_probes']} probes safe")

        # Generalization baseline
        baseline_gen = evaluator.run_generalization_probes(baseline.generate, "baseline")
        print(f"Baseline generalization: {baseline_gen['average_score']:.3f}")

        baseline.unload()
        gc.collect()
        self._monitor.record_event("baseline_eval_complete")

        baseline_results = {
            "test_split": baseline_test,
            "casual_conversation": baseline_casual,
            "safety_probes": baseline_safety,
            "generalization": baseline_gen,
        }
        self.results["baseline_evaluation"] = baseline_results
        return baseline_results

    def run_pilot(self, dataset: Any, tokenizer: Any) -> Dict[str, Any]:
        """Step 6D.12: Pilot run to verify training pipeline."""
        print("\n" + "="*60)
        print("STEP 6D.12: Pilot Run (3 micro-steps)")
        print("="*60)

        pilot_config = get_pilot_config()
        pilot_config.create_dirs()
        peft_model, _ = build_lora_model(pilot_config)
        trainer = LoRATrainer(pilot_config)
        train_texts = dataset.get_formatted_train()

        pilot_result = trainer.run_pilot(peft_model, tokenizer, train_texts)
        self.results["pilot"] = pilot_result

        # Clean up
        del peft_model
        gc.collect()

        if pilot_result["status"] != "PASS":
            raise RuntimeError(f"Pilot FAILED — aborting training: {pilot_result['errors']}")

        print("[Pilot] PASS — training pipeline verified")
        return pilot_result

    def run_training(self, dataset: Any, tokenizer: Any) -> Dict[str, Any]:
        """Run full LoRA training (Step 6D trainer)."""
        print("\n" + "="*60)
        print("STEP 6D: Fine-Tuning Training")
        print("="*60)

        peft_model, _ = build_lora_model(self.config)
        trainer = LoRATrainer(self.config, resource_monitor=self._monitor)

        train_texts = dataset.get_formatted_train()
        val_texts = dataset.get_formatted_val()

        self._monitor.record_event("training_start")
        training_metrics = trainer.train(peft_model, tokenizer, train_texts, val_texts)
        self._monitor.record_event("training_end")

        # Save final adapter
        save_final_adapter(
            peft_model=peft_model,
            tokenizer=tokenizer,
            adapter_dir=self.config.adapter_dir,
            config_dict=self.config.to_dict(),
        )

        # Save metrics
        trainer.save_metrics()
        self.results["training_metrics"] = training_metrics
        self.results["adapter_path"] = self.config.adapter_dir
        self.results["best_checkpoint"] = trainer.best_checkpoint_path

        del peft_model
        gc.collect()

        return training_metrics

    def run_finetuned_evaluation(self, dataset: Any) -> Dict[str, Any]:
        """Step 6D.16-6D.17: Verify adapter and evaluate fine-tuned model."""
        print("\n" + "="*60)
        print("STEP 6D.16-6D.17: Fine-Tuned Model Evaluation")
        print("="*60)

        # Use best checkpoint if available, otherwise use adapter dir
        adapter_path = self.results.get("best_checkpoint") or self.config.adapter_dir

        # Step 6D.16: Verify inference works
        print("[Step 6D.16] Verifying adapter loads and inference works...")
        from .checkpoint import load_adapter_for_inference, verify_adapter_inference
        ft_model, ft_tokenizer = load_adapter_for_inference(
            self.config.base_model_name, adapter_path
        )
        inference_verify = verify_adapter_inference(ft_model, ft_tokenizer)
        print(f"Inference verification: {inference_verify['status']}")
        self.results["inference_verification"] = inference_verify

        if inference_verify["status"] == "FAIL":
            raise RuntimeError("Adapter inference verification FAILED")

        # Build inference fn
        def ft_inference_fn(instruction: str) -> Dict[str, Any]:
            from .inference import _run_inference
            return _run_inference(ft_model, ft_tokenizer, instruction,
                                  self.config.max_new_tokens, self.config.do_sample)

        evaluator = Step6CTestEvaluator()
        self._monitor.record_event("finetuned_eval_start")

        ft_test = evaluator.evaluate_on_test_split(
            inference_fn=ft_inference_fn,
            test_records=dataset.test_records,
            label="fine_tuned",
        )
        print(f"Fine-tuned test split score: {ft_test['average_score']:.3f}")

        ft_casual = evaluator.run_casual_conversation(ft_inference_fn, "fine_tuned")
        print(f"Fine-tuned casual: {ft_casual['verdict']}")

        ft_safety = run_safety_probes(ft_inference_fn)
        print(f"Fine-tuned safety: {ft_safety['pass_count']}/{ft_safety['total_probes']}")

        ft_gen = evaluator.run_generalization_probes(ft_inference_fn, "fine_tuned")
        print(f"Fine-tuned generalization: {ft_gen['average_score']:.3f}")

        del ft_model, ft_tokenizer
        gc.collect()
        self._monitor.record_event("finetuned_eval_complete")

        ft_results = {
            "test_split": ft_test,
            "casual_conversation": ft_casual,
            "safety_probes": ft_safety,
            "generalization": ft_gen,
        }
        self.results["finetuned_evaluation"] = ft_results
        return ft_results

    def run_comparison(self) -> Dict[str, Any]:
        """Build full comparison between baseline and fine-tuned."""
        print("\n" + "="*60)
        print("STEP 6D: Comparison Analysis")
        print("="*60)

        b_eval = self.results.get("baseline_evaluation", {})
        ft_eval = self.results.get("finetuned_evaluation", {})

        # For 6C test split comparison
        b_test_score = b_eval.get("test_split", {}).get("average_score", 0.0)
        ft_test_score = ft_eval.get("test_split", {}).get("average_score", 0.0)
        delta = round(ft_test_score - b_test_score, 3)

        print(f"Step 6C test split: baseline={b_test_score:.3f}, fine_tuned={ft_test_score:.3f}, delta={delta:+.3f}")

        b_gen = b_eval.get("generalization", {}).get("average_score", 0.0)
        ft_gen = ft_eval.get("generalization", {}).get("average_score", 0.0)
        print(f"Generalization: baseline={b_gen:.3f}, fine_tuned={ft_gen:.3f}, delta={ft_gen-b_gen:+.3f}")

        # NOTE: Step 6A benchmark comparison uses the pre-recorded baseline
        # The fine-tuned score on Step 6A is not automatically run here
        # (requires full evaluation framework). We note this limitation.
        comparison = {
            "experiment_id": self.config.experiment_id,
            "step_6c_test_comparison": {
                "baseline_score": b_test_score,
                "finetuned_score": ft_test_score,
                "delta": delta,
            },
            "generalization_comparison": {
                "baseline_score": b_gen,
                "finetuned_score": ft_gen,
                "delta": round(ft_gen - b_gen, 3),
            },
            "casual_conversation": {
                "baseline_verdict": b_eval.get("casual_conversation", {}).get("verdict", "UNKNOWN"),
                "finetuned_verdict": ft_eval.get("casual_conversation", {}).get("verdict", "UNKNOWN"),
            },
            "safety": {
                "baseline_violations": b_eval.get("safety_probes", {}).get("violation_count", 0),
                "finetuned_violations": ft_eval.get("safety_probes", {}).get("violation_count", 0),
            },
            "step_6a_baseline_reference": BASELINE_6A_RESULTS["overall_score"],
            "step_6a_finetuned": "NOT YET EVALUATED — requires running full benchmark with adapter",
            "note": (
                "Full Step 6A benchmark evaluation with the fine-tuned model requires loading "
                "the LoRA adapter into the LocalInferenceBackend and running the 104-task benchmark. "
                "See run_step6a_with_adapter() in the report for instructions."
            ),
        }

        # Safety comparison
        if b_eval.get("safety_probes") and ft_eval.get("safety_probes"):
            from .safety import compare_safety
            safety_cmp = compare_safety(b_eval["safety_probes"], ft_eval["safety_probes"])
            comparison["safety_comparison"] = safety_cmp
            print(f"Safety comparison: {safety_cmp['verdict']}")
            if safety_cmp.get("experiment_fail"):
                print("WARNING: SAFETY REGRESSION DETECTED")

        self.results["comparison"] = comparison

        cmp_path = Path(self.config.experiment_dir) / "comparison.json"
        save_comparison(comparison, str(cmp_path))
        print(f"Comparison saved: {cmp_path}")

        return comparison

    def finalize(self) -> Dict[str, Any]:
        """Save all reproducibility artifacts and stop monitoring."""
        print("\n" + "="*60)
        print("STEP 6D: Finalizing Experiment")
        print("="*60)

        # Stop resource monitor
        self._monitor.stop()
        resource_summary = self._monitor.get_summary()
        self._monitor.save()
        self.results["resource_summary"] = resource_summary

        # Save config
        config_path = self.config.save()
        print(f"Config saved: {config_path}")

        # Save full results
        results_path = Path(self.config.experiment_dir) / "experiment_results.json"
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Full results saved: {results_path}")

        print(f"\nTotal training time: {self.results.get('training_metrics', {}).get('training_time_hms', 'N/A')}")
        print(f"Resource peak RAM: {resource_summary.get('peak_ram_mb', 0):.0f} MB")

        return self.results

    def run_full_experiment(self) -> Dict[str, Any]:
        """Run the complete Step 6D experiment end-to-end."""
        self._monitor.start()

        try:
            # 1. Hardware check
            self.run_hardware_check()
            self.config.save()

            # 2. Dataset preflight
            preflight, dataset, tokenizer = self.run_dataset_preflight()

            # 3. Baseline evaluation
            baseline_results = self.run_baseline_evaluation(dataset, tokenizer)

            # 4. Pilot run
            pilot_result = self.run_pilot(dataset, tokenizer)

            # 5. Training
            training_metrics = self.run_training(dataset, tokenizer)

            # 6. Fine-tuned evaluation
            ft_results = self.run_finetuned_evaluation(dataset)

            # 7. Comparison
            comparison = self.run_comparison()

        except Exception as e:
            self.results["fatal_error"] = str(e)
            print(f"\nEXPERIMENT ERROR: {e}")
            raise

        finally:
            final = self.finalize()

        return final
