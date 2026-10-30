"""Evaluates untuned base model on clean and perturbed test sets."""

import json
from pathlib import Path
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_DIR = Path("data")
RESULTS_DIR = Path("results/logs")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def parse_prediction(text: str) -> int:
    """Parses model response to extract binary label (1=True/Yes, 0=False/No)."""
    cleaned = text.strip().lower()
    if cleaned.startswith("yes") or "true" in cleaned:
        return 1
    if cleaned.startswith("no") or "false" in cleaned:
        return 0
    # Fallback to loose check
    if "yes" in cleaned:
        return 1
    if "no" in cleaned:
        return 0
    return -1  # Unparseable


def evaluate_dataset(model, tokenizer, records: list[dict], device: str) -> dict:
    correct = 0
    unparseable = 0
    predictions = []

    for item in tqdm(records, leave=False):
        passage = item["passage"]
        question = item["question"]
        label = item["label"]  # 1 or 0

        # Chat format for instruction tuning
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Read the passage and answer the question with only 'Yes' or 'No'.",
            },
            {
                "role": "user",
                "content": f"Passage: {passage}\nQuestion: {question}\nAnswer:",
            },
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=4,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        gen_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        response = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        pred_label = parse_prediction(response)

        if pred_label == label:
            correct += 1
        elif pred_label == -1:
            unparseable += 1

        predictions.append(
            {
                "question": question,
                "label": label,
                "response": response,
                "prediction": pred_label,
            }
        )

    acc = correct / len(records)
    return {
        "accuracy": acc,
        "correct": correct,
        "total": len(records),
        "unparseable": unparseable,
        "predictions": predictions,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading model and tokenizer: {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    eval_manifest = {
        "clean": DATA_DIR / "splits/test.jsonl",
        "typo": DATA_DIR / "perturbed/test_typo.jsonl",
        "formatting": DATA_DIR / "perturbed/test_formatting.jsonl",
        "deletion": DATA_DIR / "perturbed/test_deletion.jsonl",
        "shuffle": DATA_DIR / "perturbed/test_shuffle.jsonl",
        "distractor": DATA_DIR / "perturbed/test_distractor.jsonl",
        "synonym": DATA_DIR / "perturbed/test_synonym.jsonl",
    }

    summary_metrics = {}

    for name, path in eval_manifest.items():
        print(f"\nEvaluating condition: {name} ({path.name})...")
        records = load_jsonl(path)
        # Running on first 200 items for a fast baseline smoke test
        eval_subset = records[:200]
        res = evaluate_dataset(model, tokenizer, eval_subset, device)
        summary_metrics[name] = {
            "accuracy": round(res["accuracy"] * 100, 2),
            "unparseable": res["unparseable"],
            "total_tested": res["total"],
        }
        print(f"[{name.upper()}] Accuracy: {summary_metrics[name]['accuracy']}% | Unparseable: {res['unparseable']}")

    # Save summary log
    out_file = RESULTS_DIR / "untuned_baseline_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    print(f"\nAll baseline evaluations completed. Saved to {out_file}")


if __name__ == "__main__":
    main()