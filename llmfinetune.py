from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, get_cosine_schedule_with_warmup
import torch
import torch.nn as nn
import pandas as pd
import time
import bitsandbytes as bnb
from peft import prepare_model_for_kbit_training
import transformers

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset
dataset = load_dataset("cfilt/iitb-english-hindi")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("LingoIITGN/ganga-1b")

# Update the tokenizer to handle context length 2048
tokenizer.model_max_length = 2048  # Set maximum sequence length for the tokenizer

# Load the model with 4-bit quantization using bitsandbytes
model = AutoModelForCausalLM.from_pretrained(
    "LingoIITGN/ganga-1b",
    load_in_4bit=True,  # Enable 4-bit quantization
    device_map="auto",  # Automatically distribute model across available GPUs
    torch_dtype=torch.float32  # Use float32 precision for training
).to(device)

# Enable gradient checkpointing to reduce memory usage
model.gradient_checkpointing_enable()

# Prepare model for low-bit training (4-bit quantization)
model = prepare_model_for_kbit_training(model)

# Function to print trainable parameters
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

# Print the trainable parameters in the model
print_trainable_parameters(model)

# Prepare dataset
class TranslationDataset(Dataset):
    """
    Custom Dataset for Translation
    """
    def __init__(self, df, tokenizer, max_length=2048):  # Update max_length to 2048
        self.data = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        source_text = self.data[idx]['translation']['en']
        target_text = self.data[idx]['translation']['hi']
        inputs = self.tokenizer(
            source_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        targets = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs["labels"] = targets["input_ids"]
        return {key: tensor.squeeze(0) for key, tensor in inputs.items()}

# Convert dataset to DataFrame
data = dataset["train"]
df = data.to_pandas()
df = df[['translation']]

# Create TranslationDataset
translation_dataset = TranslationDataset(data, tokenizer)

# Split into train and validation sets
train_size = int(0.8 * len(translation_dataset))
valid_size = len(translation_dataset) - train_size
train_data, valid_data = random_split(translation_dataset, [train_size, valid_size])

# Training parameters
num_epochs = 5
learning_rate = 4e-4  # Updated learning rate
batch_size = 1  # Set to 1, gradient accumulation will simulate larger batch size
gradient_accumulation_steps = 4  # Number of steps to accumulate gradients

# Set tokenizer pad_token to eos_token
tokenizer.pad_token = tokenizer.eos_token

# Trainer arguments (using gradient accumulation, fp16 precision, and paged adam optimizer)
training_args = transformers.TrainingArguments(
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    warmup_steps=2,
    max_steps=10,
    learning_rate=learning_rate,
    fp16=False,  # Using float32 precision, so fp16 is set to False
    logging_steps=1,
    output_dir="outputs",
    optim="adamw",  # AdamW optimizer
    lr_scheduler_type="cosine",  # Use cosine learning rate scheduler
)

# Trainer setup with data collator for LM
trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),  # MLM for causal language models
)

# Disable cache for training
model.config.use_cache = False  # Silence the warnings. Please re-enable for inference!

# Start training
trainer.train()

# Save the model
model.save_pretrained("fine_tuned_ganga")
tokenizer.save_pretrained("fine_tuned_ganga")

# Test the fine-tuned model
def translate(input_text):
    input_ids = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
    outputs = model.generate(input_ids, max_length=50, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example translation
input_sentence = "How are you?"
print("Input:", input_sentence)
print("Translation:", translate(input_sentence))
