# train_model.py
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
import random
import warnings
import sys
import os

# Filter warnings untuk output yang lebih bersih
warnings.filterwarnings('ignore')

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForTokenClassification,
        TrainingArguments,
        Trainer,
        DataCollatorForTokenClassification
    )
    from datasets import Dataset, DatasetDict
    from seqeval.metrics import f1_score, precision_score, recall_score
except ImportError:
    print("❌ Error: Pastikan packages 'transformers', 'datasets', 'seqeval' sudah terinstal.")
    sys.exit(1)

# --- 1. HYPERPARAMETERS UTAMA ---
MODEL_NAME = "xlm-roberta-base"
DATA_DIR = "data" # Sesuaikan path ini jika data Anda tidak di folder 'data'
MODEL_SAVE_PATH = "./model_ner_improved_final"
AUGMENTATION_FACTOR = 3 

# --- 2. CUSTOM CLASSES (FocalLoss & ImprovedTrainer) ---

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=3, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=self.ignore_index)(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class ImprovedTrainer(Trainer):
    def __init__(self, *args, focal_alpha=1, focal_gamma=3, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(
                weight=self.class_weights.to(model.device),
                ignore_index=-100
            )
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss = self.focal_loss(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

# --- 3. DATA UTILS ---

def load_nergrit_data(file_path: str) -> Tuple[List[List[str]], List[List[str]]]:
    sentences = []
    labels = []
    current_tokens = []
    current_labels = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_tokens:
                    sentences.append(current_tokens)
                    labels.append(current_labels)
                    current_tokens = []
                    current_labels = []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    current_tokens.append(parts[0])
                    current_labels.append(parts[1])
        if current_tokens:
            sentences.append(current_tokens)
            labels.append(current_labels)
            
    print(f"✅ Loaded {len(sentences)} sentences from {file_path}")
    return sentences, labels

def augment_ner_data(tokens_list: List[List[str]], labels_list: List[List[str]], augmentation_factor: int):
    # Logika augmentasi yang sama dari notebook Anda
    augmented_tokens = list(tokens_list)
    augmented_labels = list(labels_list)
    for _ in range(augmentation_factor - 1):
        for tokens, labels in zip(tokens_list, labels_list):
            new_tokens = tokens.copy()
            new_labels = labels.copy()
            non_entity_indices = [i for i, label in enumerate(labels) if label == "O"]
            if len(non_entity_indices) >= 2 and random.random() > 0.5:
                idx1, idx2 = random.sample(non_entity_indices, 2)
                new_tokens[idx1], new_tokens[idx2] = new_tokens[idx2], new_tokens[idx1]
            augmented_tokens.append(new_tokens)
            augmented_labels.append(new_labels)
    print(f"📊 Augmentation: {len(tokens_list)} -> {len(augmented_tokens)} samples")
    return augmented_tokens, augmented_labels

def compute_class_weights(labels_list: List[List[str]], label2id: dict):
    all_labels = [label for sent_labels in labels_list for label in sent_labels]
    label_counts = Counter(all_labels)
    total = sum(label_counts.values())
    class_weights = {}
    
    for label, count in label_counts.items():
        weight = total / (len(label_counts) * count)
        class_weights[label] = weight

    weights_tensor = torch.zeros(len(label2id))
    for label, weight in class_weights.items():
        weights_tensor[label2id[label]] = weight
    
    return weights_tensor

def tokenize_and_align_labels(examples, tokenizer, label_all_tokens=True):
    # Logika tokenisasi yang sama dari notebook Anda
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding=False,
        max_length=512
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics(eval_pred, id2label):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    true_labels = []
    true_predictions = []

    for prediction, label in zip(predictions, labels):
        true_label = []
        true_prediction = []

        for pred, lab in zip(prediction, label):
            if lab != -100:
                true_label.append(id2label[lab])
                true_prediction.append(id2label[pred])

        true_labels.append(true_label)
        true_predictions.append(true_prediction)

    results = {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }
    return results

# --- 4. MAIN FUNCTION ---

def main():
    print("="*60)
    print("🚀 Starting NER Model Training (XLM-R Base)")
    print("="*60)

    # Load Data
    train_tokens_raw, train_labels_raw = load_nergrit_data(f"{DATA_DIR}/train_preprocess.txt")
    valid_tokens, valid_labels = load_nergrit_data(f"{DATA_DIR}/valid_preprocess.txt")
    test_tokens, test_labels = load_nergrit_data(f"{DATA_DIR}/test_preprocess.txt")

    # Data Augmentation (Factor 3)
    train_tokens, train_labels = augment_ner_data(train_tokens_raw, train_labels_raw, AUGMENTATION_FACTOR)
    
    # Create Label Mappings
    all_labels = sorted(list(set([label for sent_labels in train_labels for label in sent_labels])))
    label2id = {label: idx for idx, label in enumerate(all_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    print(f"✅ Total labels: {len(label2id)}")

    # Compute Class Weights
    class_weights = compute_class_weights(train_labels, label2id)
    print(f"✅ Class weights computed for {len(class_weights)} classes.")
    
    # Create Datasets
    def create_dataset_dict(tokens, labels, label2id):
        label_ids = [[label2id[label] for label in sent_labels] for sent_labels in labels]
        data_dict = {"tokens": tokens, "ner_tags": label_ids}
        return Dataset.from_dict(data_dict)
    
    train_dataset = create_dataset_dict(train_tokens, train_labels, label2id)
    valid_dataset = create_dataset_dict(valid_tokens, valid_labels, label2id)
    test_dataset = create_dataset_dict(test_tokens, test_labels, label2id)
    dataset_dict = DatasetDict({"train": train_dataset, "validation": valid_dataset, "test": test_dataset})
    
    # Load Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    print(f"✅ Model {MODEL_NAME} loaded.")

    # Tokenization
    tokenized_datasets = dataset_dict.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer),
        batched=True,
        remove_columns=dataset_dict["train"].column_names
    )
    print("✅ Datasets tokenized.")

    # Setup Trainer
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)
    training_args = TrainingArguments(
        output_dir="./results_ner_improved",
        learning_rate=3e-5,
        warmup_ratio=0.1,
        num_train_epochs=10, 
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,
        weight_decay=0.01,
        max_grad_norm=1.0,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir="./logs_improved",
        logging_steps=50,
        logging_first_step=True,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        report_to="none",
        push_to_hub=False,
    )
    
    trainer = ImprovedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, id2label),
        focal_alpha=1,
        focal_gamma=3, 
        class_weights=class_weights,
    )

    # Start Training
    print("\n🚀 Starting Training...")
    train_result = trainer.train()

    # Save Model (Trainer akan otomatis menyimpan BEST model dari Step 1400)
    trainer.save_model(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print(f"\n✅ TRAINING COMPLETE! Model saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()