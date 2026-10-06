import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from config import DistillConfig

def prepare_absa_dataset(config: DistillConfig):
    dataset = load_dataset(config.dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(config.teacher_model_name)

    def preprocess_function(examples):
        tokenized = tokenizer(
            text=examples["text"],
            text_pair=examples["span"],
            padding="max_length",
            truncation=True,
            max_length=config.max_length,
            return_tensors="pt"
        )

        float_labels = []
        for label in examples["label"]:
            label_str = str(label).lower()
            if "pos" in label_str:
                float_labels.append(1.0)
            elif "neg" in label_str:
                float_labels.append(0.0)
            else:
                float_labels.append(0.5)
        
        tokenized["labels"] = torch.tensor(float_labels, dtype=torch.float32)
        return tokenized

    tokenized_datasets = dataset.map(
        preprocess_function, 
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    tokenized_datasets.set_format("torch")
    return tokenized_datasets, tokenizer

if __name__ == "__main__":
    config = DistillConfig()
    dataset, tokenizer = prepare_absa_dataset(config)
    print(f"Train size: {len(dataset['train'])}")
    
    