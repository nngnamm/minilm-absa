from transformers import AutoModelForSequenceClassification
from config import DistillConfig

def get_teacher_model(config: DistillConfig):
    model = AutoModelForSequenceClassification.from_pretrained(
        config.teacher_model_name,
        num_labels=config.num_labels,
        problem_type=config.problem_type
    )
    
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
        
    return model

