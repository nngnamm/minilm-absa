import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

from config import DistillConfig
from dataset import prepare_absa_dataset

def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) / 1_000_000

def evaluate_accuracy(model, dataloader, device):
    """Runs the model over the test set and calculates regression metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Accuracy", leave=False):
            # Move inputs to GPU
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            if "token_type_ids" in batch:
                token_type_ids = batch["token_type_ids"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Extract predictions and labels
            preds = outputs.logits.squeeze(-1).cpu().numpy()
            labels = batch["labels"].numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
            
    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    pearson_corr, _ = pearsonr(all_labels, all_preds)
    
    return mse, mae, pearson_corr

def measure_latency(model, dataloader, device, num_batches=50):
    """Measures inference speed using a real batch of data."""
    model.eval()
    batch = next(iter(dataloader))
    
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
    if "token_type_ids" in batch:
        kwargs["token_type_ids"] = batch["token_type_ids"].to(device)
    
    # Warmup to avoid GPU cold-start lag
    for _ in range(10):
        with torch.no_grad():
            model(**kwargs)
            
    # Benchmark
    start_time = time.perf_counter()
    for _ in range(num_batches):
        with torch.no_grad():
            model(**kwargs)
    end_time = time.perf_counter()
    
    return ((end_time - start_time) / num_batches) * 1000

def main():
    config = DistillConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking on device: {device}\n")
    
    # 1. Load the Test Dataset
    print("Loading test dataset...")
    dataset, _ = prepare_absa_dataset(config)
    test_dataloader = DataLoader(dataset["test"], batch_size=32)
    
    # 2. Load Models
    teacher_path = "./outputs/teacher_absa_expert"
    student_path = "./outputs/minilm_absa_student_4L"
    
    print("Loading Teacher Model...")
    teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_path).to(device)
    
    print("Loading Student Model...\n")
    student_model = AutoModelForSequenceClassification.from_pretrained(student_path).to(device)
    
    # 3. Gather Metrics
    print("--- Evaluating Teacher ---")
    t_params = count_parameters(teacher_model)
    t_mse, t_mae, t_pearson = evaluate_accuracy(teacher_model, test_dataloader, device)
    t_latency = measure_latency(teacher_model, test_dataloader, device)
    
    print("\n--- Evaluating Student ---")
    s_params = count_parameters(student_model)
    s_mse, s_mae, s_pearson = evaluate_accuracy(student_model, test_dataloader, device)
    s_latency = measure_latency(student_model, test_dataloader, device)
    
    # 4. Print Final Report
    print("\n============================================================")
    print("                 FINAL BENCHMARK REPORT                     ")
    print("============================================================")
    print(f"Metric                 | Teacher (12L) | Student (4L) | Improvement")
    print(f"------------------------------------------------------------")
    print(f"Parameters             | {t_params:^13.1f} | {s_params:^12.1f} | {((t_params-s_params)/t_params)*100:.1f}% Smaller")
    print(f"Latency (ms/batch)     | {t_latency:^13.1f} | {s_latency:^12.1f} | {t_latency/s_latency:.1f}x Faster")
    print(f"------------------------------------------------------------")
    print(f"Test MSE (Lower=Better)| {t_mse:^13.4f} | {s_mse:^12.4f} | Retained {(t_mse/s_mse)*100:.1f}%")
    print(f"Test MAE               | {t_mae:^13.4f} | {s_mae:^12.4f} | -")
    print(f"Pearson Correlation    | {t_pearson:^13.4f} | {s_pearson:^12.4f} | Retained {(s_pearson/t_pearson)*100:.1f}%")
    print("============================================================\n")

if __name__ == "__main__":
    main()