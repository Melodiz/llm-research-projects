import os
import csv
from transformers import TrainerCallback


class CollapseDetector(TrainerCallback):
    """Monitors NOT-ISO collapse."""

    def __init__(self, run_name, output_dir="outputs"):
        self.run_name = run_name
        self.output_dir = output_dir
        self.csv_path = os.path.join(output_dir, run_name, "collapse_monitor.csv")
        
        # Lazy import to avoid loading graph env during config parsing
        from graph_isomorphism_env import GraphIsomorphismVerifier
        self.verifier = GraphIsomorphismVerifier()
        
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "class_prediction_ratio", "frac_zero_std"])

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        step = state.global_step
        frac_zero = logs.get("frac_reward_zero_std", 0.0)
        completions = logs.get("completions", [])
        
        ratio = -1.0 # Default if no completions found
        
        if completions:
            not_iso_count = 0
            for comp in completions:
                # completions in trl logs might be strings or dicts
                if isinstance(comp, list) and len(comp) > 0 and isinstance(comp[0], dict):
                     text = comp[0].get("content", "")
                else:
                     text = str(comp)
                
                ans = self.verifier.extract_answer(text)
                if not ans:
                    ans = text  # Fallback to scanning the whole text if tags are missing
                    
                if self.verifier._is_not_isomorphic_declaration(ans):
                    not_iso_count += 1
                    
            ratio = not_iso_count / len(completions)

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([step, ratio, frac_zero])

        if ratio > 0.95:
            print(f"\n[COLLAPSE] Step {step}: COLLAPSE DETECTED — model is degenerate ({ratio:.0%} NOT ISO)\n")
        elif ratio > 0.8:
            print(f"\n[WARNING] Step {step}: COLLAPSE WARNING: {ratio:.0%} NOT ISO predictions\n")
            
        if frac_zero > 0.6:
            print(f"\n[WARNING] Step {step}: {frac_zero:.0%} dead groups — reward variance collapse risk HIGH\n")
