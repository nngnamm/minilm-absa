import math
import torch
import torch.nn.functional as F

def compute_value_relation(hidden_states, value_layer, num_heads):
    v = value_layer(hidden_states)
    batch_size, seq_len, _ = v.size()
    head_dim = v.size(-1) // num_heads
    v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    vv_t = torch.matmul(v, v.transpose(-1, -2)) / math.sqrt(head_dim)
    return vv_t

def calculate_minilm_loss(teacher_model, student_model, teacher_outputs, student_outputs, temperature=1.0):
    loss = 0.0

    t_att_probs = teacher_outputs.attentions[-1]
    s_att_probs = student_outputs.attentions[-1]
   
    s_log_probs = torch.log(s_att_probs + 1e-12)
    loss_A = F.kl_div(s_log_probs, t_att_probs, reduction="batchmean")
    loss += loss_A

    t_hidden_in = teacher_outputs.hidden_states[-2]
    s_hidden_in = student_outputs.hidden_states[-2]
 
    t_v_layer = teacher_model.bert.encoder.layer[-1].attention.self.value
    s_v_layer = student_model.bert.encoder.layer[-1].attention.self.value

    t_vv_logits = compute_value_relation(t_hidden_in, t_v_layer, teacher_model.config.num_attention_heads)
    s_vv_logits = compute_value_relation(s_hidden_in, s_v_layer, student_model.config.num_attention_heads)

    t_vr_probs = F.softmax(t_vv_logits / temperature, dim=-1)
    s_vr_log_probs = F.log_softmax(s_vv_logits / temperature, dim=-1)
    
    loss_VR = F.kl_div(s_vr_log_probs, t_vr_probs, reduction="batchmean") * (temperature ** 2)
    loss += loss_VR
    
    return loss