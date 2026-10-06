import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, get_scheduler
from accelerate import Accelerator
from tqdm.auto import tqdm

from config import DistillConfig
from dataset import prepare_absa_dataset

def main():
    config = DistillConfig()
    accelerator = Accelerator()
    
    accelerator.print("=== Stage 1: Training the ABSA Regression Teacher ===")

    dataset, tokenizer = prepare_absa_dataset(config)
    train_dataloader = DataLoader(dataset["train"], shuffle=True, batch_size=config.batch_size)

    accelerator.print(f"Initializing raw {config.teacher_model_name}...")
    teacher_model = AutoModelForSequenceClassification.from_pretrained(
        config.teacher_model_name,
        num_labels=config.num_labels,
        problem_type=config.problem_type,
        attn_implementation="eager"
    )

    mse_loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(teacher_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    num_training_steps = config.num_train_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=int(num_training_steps * config.warmup_ratio), num_training_steps=num_training_steps
    )

    teacher_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        teacher_model, optimizer, train_dataloader, lr_scheduler
    )

    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process)
    teacher_model.train()
    
    for epoch in range(config.num_train_epochs):
        total_loss = 0
        for batch in train_dataloader:
            outputs = teacher_model(**batch)

            loss = mse_loss_fn(outputs.logits.squeeze(-1), batch["labels"])
            
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            progress_bar.update(1)
            total_loss += loss.item()
            
        accelerator.print(f"Epoch {epoch+1}/{config.num_train_epochs} - Avg MSE Loss: {total_loss / len(train_dataloader):.4f}")

    expert_dir = "./outputs/teacher_absa_expert"
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(teacher_model)
        unwrapped_model.save_pretrained(expert_dir)
        tokenizer.save_pretrained(expert_dir)
        accelerator.print(f"Expert Teacher safely saved to {expert_dir}")

if __name__ == "__main__":
    main()