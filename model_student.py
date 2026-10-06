from transformers import AutoConfig, AutoModelForSequenceClassification
from config import DistillConfig

def get_student_model(config: DistillConfig):
    print(f"Initializing {config.student_num_hidden_layers}-Layer Student Model...")
    
    student_config = AutoConfig.from_pretrained(
        config.teacher_model_name,
        num_labels=config.num_labels,
        problem_type=config.problem_type,
        attn_implementation="eager"  
    )
    
    student_config.num_hidden_layers = config.student_num_hidden_layers
    student_config.hidden_size = config.student_hidden_size
    student_config.intermediate_size = config.student_intermediate_size
    student_config.num_attention_heads = config.student_num_attention_heads
    
    model = AutoModelForSequenceClassification.from_config(student_config)
    return model