from config import SEED
from utils import set_seed, ensure_dir, save_json
from training.train_text_generation import run_text_generation
from training.train_translation import run_translation

def main():
    set_seed(SEED)
    ensure_dir("results")

    print("Running Task 1: Text Generation")
    textgen_results = run_text_generation(output_dir="results")

    print("\nRunning Task 2: Machine Translation")
    translation_results = run_translation(output_dir="results")

    final_results = {
        "task_1_text_generation": textgen_results,
        "task_2_machine_translation": translation_results,
    }

    save_json(final_results, "results/final_results.json")
    print("\nAll results saved to results/final_results.json")

if __name__ == "__main__":
    main()