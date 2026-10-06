from dataclasses import dataclass

@dataclass
class DistillConfig:
    teacher_model_name: str = "bert-base-uncased"
    dataset_name: str = "tomaarsen/setfit-absa-semeval-restaurants"
    output_dir: str = "./outputs/minilm_absa_student_4L"

    student_num_hidden_layers: int = 4
    student_hidden_size: int = 384
    student_intermediate_size: int = 1536
    student_num_attention_heads: int = 12
    num_labels: int = 1
    problem_type: str = "regression"

    max_length: int = 128
    batch_size: int = 32
    learning_rate: float = 5e-5
    num_train_epochs: int = 5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

    alpha: float = 0.5
    temperature: float = 1.0

    