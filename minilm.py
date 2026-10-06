import torch
import torch.nn.functional as F

def calculate_minilm_loss(teacher_attentions, student_attentions, temperature=1.0):
    loss = 0.0

    t_att = teacher_attentions[-1]
    s_att = student_attentions[-1]

    s_log_probs = F.log_softmax(s_att / temperature, dim=-1)
    t_probs = F.softmax(t_att / temperature, dim=-1)

    loss += F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (temperature ** 2)
    
    return loss