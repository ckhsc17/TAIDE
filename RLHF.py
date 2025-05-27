import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import os

# === Configuration ===
REWARD_MODEL_NAME = "distilbert-base-uncased"  # 可替換成你喜歡的模型
DATA_FILE = "logs/reward_data.json"            # 請將你的資料放在這裡
OUTPUT_DIR = "./reward_model_checkpoint"

# === Step 1: Load data ===
def load_scored_data(path):
    with open(path, 'r') as f:
        raw = json.load(f)

    samples = []

    for entry in raw:
        question = entry["question"]
        for mode in ["rag", "without-rag"]:
            answer = entry["answers"].get(mode)
            score = entry["feedback"].get(mode)
            if answer and score is not None:
                samples.append({
                    "prompt": question,
                    "response": answer,
                    "score": score,
                })
    return samples


# === Step 2: Format into HF Dataset ===
def build_reward_dataset(samples, tokenizer):
    prompts = [s["prompt"] for s in samples]
    responses = [s["response"] for s in samples]
    scores = [s["score"] for s in samples]

    texts = [f"{p}\n\n{r}" for p, r in zip(prompts, responses)]

    tokenized = tokenizer(texts, truncation=True, padding=True, max_length=512)
    tokenized["labels"] = scores
    return Dataset.from_dict(tokenized)


# === Step 3: Train reward model ===
def train():
    tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        REWARD_MODEL_NAME,
        num_labels=1  # Reward score is scalar
    )

    data = load_scored_data(DATA_FILE)
    dataset = build_reward_dataset(data, tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,
        num_train_epochs=8,
        learning_rate=5e-5,
        logging_dir='./logs',
        eval_strategy="no",
        save_strategy="epoch",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )

    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    train()
