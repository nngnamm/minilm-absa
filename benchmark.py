import time
import torch
from transformers import AutoModelForSequenceClassification

def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) / 1_000_000

def measure_latency(model, dummy_inputs, num_runs=100):
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    dummy_inputs = {k: v.to(device) for k, v in dummy_inputs.items()}
    
    model.eval()
    
    # Warm up (avoids cold-start initialization lag)
    for _ in range(10):
        with torch.no_grad():
            model(**dummy_inputs)
            
    # Benchmark
    start_time = time.perf_counter()
    for _ in range(num_runs):
        with torch.no_grad():
            model(**dummy_inputs)
    end_time = time.perf_counter()
    
    return ((end_time - start_time) / num_runs) * 1000

if __name__ == "__main__":
    print("Loading models for benchmarking...\n")
    teacher_path = "./outputs/teacher_absa_expert"
    student_path = "./outputs/minilm_absa_student_4L"
    
    teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_path)
    student_model = AutoModelForSequenceClassification.from_pretrained(student_path)
    
    # Create a dummy batch of 32 sequences, length 128
    dummy_inputs = {
        "input_ids": torch.randint(0, 30000, (32, 128)),
        "attention_mask": torch.ones(32, 128)
    }
    
    print("--- Parameter Size ---")
    t_params = count_parameters(teacher_model)
    s_params = count_parameters(student_model)
    print(f"Teacher (12L/768d): {t_params:.1f}M parameters")
    print(f"Student (4L/384d):  {s_params:.1f}M parameters")
    print(f"Reduction:          {((t_params - s_params) / t_params) * 100:.1f}% smaller\n")
    
    print("--- Inference Speed (Batch Size 32) ---")
    t_latency = measure_latency(teacher_model, dummy_inputs)
    s_latency = measure_latency(student_model, dummy_inputs)
    print(f"Teacher Latency:    {t_latency:.1f} ms / batch")
    print(f"Student Latency:    {s_latency:.1f} ms / batch")
    print(f"Speedup:            {t_latency / s_latency:.2f}x faster")