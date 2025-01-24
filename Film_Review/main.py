import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import zipfile

# Check GPU availability
print("[INFO] Checking GPU availability...")
if not torch.cuda.is_available():
    raise RuntimeError(
        "[ERROR] GPU not available. Please go to Runtime > Change runtime type > GPU."
    )
print("[INFO] GPU is available!")

# Install required libraries (uncomment if running in Colab)
# !pip install -U torch transformers datasets accelerate peft bitsandbytes

# Step 1: Load DeepSeek LLM with LoRA
print("[INFO] Loading DeepSeek LLM with LoRA and 4-bit precision...")

model_name = "deepseek-ai/deepseek-llm-7b-base"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

# Apply LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print("[INFO] DeepSeek LLM Loaded Successfully!")

# Step 2: Load and preprocess the IMDB dataset
print("[INFO] Loading and tokenizing IMDB dataset...")

dataset = load_dataset("imdb")


def tokenize_function(examples):
    inputs = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    inputs["labels"] = inputs["input_ids"].copy()
    return inputs


tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Prepare smaller datasets for quick training
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(500))
small_test_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(100))

print("[INFO] Dataset ready for training!")

# Step 3: Set training arguments
print("[INFO] Setting up training parameters...")
os.environ["WANDB_DISABLED"] = "true"

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=3e-4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir="./logs",
    fp16=True,
)
print("[INFO] Training parameters set!")

# Step 4: Initialize Trainer
print("[INFO] Initializing Trainer...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset.remove_columns(["text"]),
    eval_dataset=small_test_dataset.remove_columns(["text"]),
)

print("[INFO] Trainer Initialized!")
torch.cuda.empty_cache()
print("[INFO] Cleared CUDA Cache!")

# Step 5: Fine-tune the model
print("[INFO] Starting Fine-Tuning...")
trainer.train()
print("[INFO] Fine-Tuning Complete!")

# Step 6: Generate predictions with the fine-tuned model
def generate_prediction(review_text):
    inputs = tokenizer(review_text, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_length=100, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


print("[INFO] Generating predictions...")
reviews = [
    "The movie was absolutely fantastic! I loved the cinematography and the acting was superb.",
    "This was the worst movie I've ever seen. The plot made no sense and the dialogue was terrible.",
    "It was an okay movie. Some parts were really good, but overall it was just average.",
]

for review in reviews:
    print(f"[INPUT REVIEW]: {review}")
    print(f"[PREDICTED SENTIMENT]: {generate_prediction(review)}")
    print("=" * 80)

# Step 7: Compress output folder into a ZIP file
def compress_folder_to_zip(source_folder: str, output_zip_file: str) -> None:
    """
    Compress a folder and its contents into a ZIP file.

    Args:
        source_folder (str): Path to the folder to be compressed.
        output_zip_file (str): Path to the resulting ZIP file.
    """
    with zipfile.ZipFile(output_zip_file, "w", zipfile.ZIP_DEFLATED) as zip_archive:
        for root_dir, _, filenames in os.walk(source_folder):
            for filename in filenames:
                full_file_path = os.path.join(root_dir, filename)
                archive_name = os.path.relpath(full_file_path, source_folder)
                zip_archive.write(full_file_path, archive_name)
    print(f"[INFO] Folder '{source_folder}' compressed into '{output_zip_file}' successfully!")


compress_folder_to_zip("./results", "./results_archive.zip")
print("[INFO] All steps completed successfully!")
