from transformers import AutoModelForSequenceClassification
from config import DistillConfig

def get_teacher_model(config: DistillConfig):
    print(f"Loading Teacher Model: {config.teacher_model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        config.teacher_model_name,
        num_labels=config.num_labels,
        problem_type=config.problem_type,
        attn_implementation="eager"  
    )
    
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
        
    return model