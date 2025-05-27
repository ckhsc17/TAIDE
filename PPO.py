from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from trl import PPOTrainer, PPOConfig
from datasets import load_dataset, Dataset
import torch
import json

# === Step 1: 設定基礎模型與 Tokenizer ===
MODEL_NAME = "TAIDE/a.2.0.0"  # 或其他可 fine-tune 的 base 模型（需可用 transformers）
REWARD_MODEL_PATH = "./reward_model_checkpoint"  # reward model 預訓練位置

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, return_dict=True)

# === Step 2: 載入 Reward Model ===
from transformers import AutoModelForSequenceClassification

reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_PATH)
reward_model = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL_PATH)
reward_model.eval()


# === Step 3: 載入訓練資料 ===
def load_prompt_dataset(path):
    with open(path, 'r') as f:
        raw_data = json.load(f)

    samples = []
    for d in raw_data:
        q = d["question"]
        if d["feedback"]["rag"] is not None:  # 只用有標註的回應
            samples.append({"query": q})
    return Dataset.from_list(samples)


dataset = load_prompt_dataset("data/reward_data.json")

# === Step 4: 設定 PPO 參數 ===
ppo_config = PPOConfig(
    model_name=MODEL_NAME,
    learning_rate=1e-5,
    batch_size=4,
    log_with=None,
    mini_batch_size=1,
    optimize_cuda_cache=True,
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    tokenizer=tokenizer,
    dataset=dataset
)

# === Step 5: 開始訓練 ===

for batch in ppo_trainer.dataloader:
    queries = batch["query"]

    # Step 1: Generate responses with current model
    responses = []
    for q in queries:
        input_ids = tokenizer(q, return_tensors="pt").input_ids.to(model.device)
        outputs = model.generate(input_ids, max_new_tokens=100)
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        responses.append(response_text)

    # Step 2: Compute reward using reward model
    texts = [f"{q}\n\n{a}" for q, a in zip(queries, responses)]
    reward_inputs = reward_tokenizer(texts, return_tensors="pt", truncation=True, padding=True).to(model.device)
    with torch.no_grad():
        reward_scores = reward_model(**reward_inputs).logits.squeeze().cpu().tolist()

    # Step 3: Apply PPO Step
    stats = ppo_trainer.step(queries, responses, reward_scores)
    print(stats)

# === Step 6: 儲存 PPO 微調後的模型 ===
ppo_trainer.model.save_pretrained("ppo_taide_model")
tokenizer.save_pretrained("ppo_taide_model")
