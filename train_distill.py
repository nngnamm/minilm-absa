import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_scheduler
from accelerate import Accelerator
from tqdm.auto import tqdm

from config import DistillConfig
from dataset import prepare_absa_dataset
from model_teacher import get_teacher_model
from model_student import get_student_model
from minilm import calculate_minilm_loss

def main():
    config = DistillConfig()
    accelerator = Accelerator()
    
    accelerator.print("=== Starting ABSA MiniLM Distillation (Regression) ===")
    
    dataset, tokenizer = prepare_absa_dataset(config)
    train_dataloader = DataLoader(dataset["train"], shuffle=True, batch_size=config.batch_size)

    teacher_model = get_teacher_model(config)
    student_model = get_student_model(config)

    mse_loss_fn = nn.MSELoss()

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    num_training_steps = config.num_train_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=int(num_training_steps * config.warmup_ratio), num_training_steps=num_training_steps
    )

    student_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        student_model, optimizer, train_dataloader, lr_scheduler
    )
    teacher_model = accelerator.prepare(teacher_model)

    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process)
    
    student_model.train()
    for epoch in range(config.num_train_epochs):
        total_loss = 0
        for batch in train_dataloader:
            batch["output_attentions"] = True
            batch["output_hidden_states"] = True

            with torch.no_grad():
                teacher_outputs = teacher_model(**batch)

            student_outputs = student_model(**batch)
            
            task_loss = mse_loss_fn(student_outputs.logits.squeeze(-1), batch["labels"])

            minilm_loss = calculate_minilm_loss(
                teacher_model=accelerator.unwrap_model(teacher_model),
                student_model=accelerator.unwrap_model(student_model),
                teacher_outputs=teacher_outputs,
                student_outputs=student_outputs,
                temperature=config.temperature
            )

            loss = (config.alpha * minilm_loss) + ((1.0 - config.alpha) * task_loss)

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            progress_bar.update(1)
            total_loss += loss.item()
            
        accelerator.print(f"Epoch {epoch+1}/{config.num_train_epochs} - Avg Loss: {total_loss / len(train_dataloader):.4f}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(student_model)
        unwrapped_model.save_pretrained(config.output_dir)
        tokenizer.save_pretrained(config.output_dir)
        accelerator.print(f"Model safely saved to {config.output_dir}")

if __name__ == "__main__":
    main()