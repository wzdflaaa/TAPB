import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer import TransformerEncoder, TransformerDecoder

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def freeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False

class MLMHead(nn.Module):
    def __init__(self, n_embd, vocab_size):
        super().__init__()
        self.linear1 = nn.Linear(n_embd, n_embd)
        self.activation = nn.GELU()
        self.ln = nn.LayerNorm(n_embd)
        self.linear2 = nn.Linear(n_embd, vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.linear2.bias = self.bias

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.ln(x)
        logits = self.linear2(x)
        return logits

class TAPB(nn.Module):
    def __init__(self, model_configs):
        super().__init__()
        # Params
        d_model = model_configs['DrugEncoder']['d_model'] #drug encoder的输出维度
        n_heads = model_configs['DrugEncoder']['n_head']  #drug encoder的head数量-注意力头数 
        vocab_size = model_configs['DrugEncoder']['vocab_size'] #drug encoder的词表大小(用于MLM任务）)
        fusion_n_heads = model_configs['TransformerDeocder']['n_head'] #fusion解码器的head数量（融合解码器的注意力头数）
        self.d_model = d_model 
        self.encoder_n_heads = n_heads 
        self.fusion_n_heads = fusion_n_heads

        
        # Drug
        self.precompute_freqs_cis = precompute_freqs_cis(d_model // n_heads, 4000)
        self.drug_encoder = TransformerEncoder(config=model_configs['DrugEncoder'])
        self.MLMHead = MLMHead(d_model, vocab_size)

        # Fusion
        self.aggregator = TransformerDecoder(config=model_configs['TransformerDeocder'])
        self.classifier = nn.Linear(self.d_model // self.fusion_n_heads, 2)

        # classifier2 is used for TAPB ablation study  消融实验部分__可以用来实现baseline
        self.classifier2 = nn.Linear(self.d_model, 2)
        
        #Protein
        self.pr_linear = nn.Linear(model_configs)
        
        
        
    def encode_drug(self, input_drugs, freqs_cis):
        drug_id = input_drugs['input_ids']
        drug_padding_mask = ~input_drugs['attention_mask'].bool()
        bz, len_d = drug_id.size()
        drug_attn_mask = drug_padding_mask.unsqueeze(1).unsqueeze(1).expand(-1, self.encoder_n_heads, len_d, -1)
        drug_f = self.drug_encoder(drug_id, drug_attn_mask, freqs_cis[:len_d, :].to(drug_id.device))
        return drug_f

    def encode_protein(self, pr_f, pr_mask):
        #ablation: 直接使用预训练的蛋白质特征 投影到fusion所需维度d——model
        pr_f = self.pr_linear(pr_f)
        return pr_f, pr_mask


    def fusion(self, drug_f, pr_f, drug_padding_mask, protein_padding_mask):
        bz, len_d, _ = drug_f.size()
        drug_attn_mask = ~drug_padding_mask.bool().unsqueeze(1).unsqueeze(1).expand(-1, self.fusion_n_heads, len_d, -1)
        cross_attn_mask = ~protein_padding_mask.bool().unsqueeze(1).unsqueeze(1).expand(-1, self.fusion_n_heads, len_d, -1)
        fusion_f, attention_map = self.aggregator(src=pr_f, tgt=drug_f, self_attn_mask=drug_attn_mask,
                                                cross_attn_mask=cross_attn_mask)
        return fusion_f, attention_map

    
    def forward(self, input_drugs, input_proteins, pr_mask=None, masked_drugs=None):
            # encode
            drug_f = self.encode_drug(input_drugs, self.precompute_freqs_cis)
            pr_f, pr_mask = self.encode_protein(input_proteins, pr_mask)

            # fusion
            fusion_f, attn_map = self.fusion(drug_f, pr_f, input_drugs['attention_mask'], pr_mask)

            # mlm
            drug_mlm_logits = None
            if masked_drugs is not None:
                drug_f_mlm = self.encode_drug(masked_drugs, self.precompute_freqs_cis)
                drug_mlm_logits = self.MLMHead(drug_f_mlm).permute(0, 2, 1)

            #baseline：直接对融合后的特征进行池化，并分类
            pool_f = fusion_f.mean(dim=1)
            logits = self.classifier2(pool_f)
            logits = F.softmax(logits, dim=-1)

            return {'logits': logits, 'fusion_f': fusion_f, 'attn_map': attn_map, 'drug_mlm_logits': drug_mlm_logits}
    
    """confounder alignment module 和 backdoor adjustment 的实现
            已从模型中摘除，不再依赖 confounder / backdoor 参数
    def confounder_aligment_module(self, pr_f, ci):
        device = pr_f.device
        bz = pr_f.size(0)
        Q = self.linear_q(pr_f).view(bz, -1, self.c_center, 1280 // self.c_center).permute(0, 2, 1, 3)
        K = self.linear_k(ci).unsqueeze(1).permute(0, 2, 1, 3)
        V = self.linear_v(ci).unsqueeze(1).permute(0, 2, 1, 3)
        A = torch.matmul(Q, K.permute(0, 1, 3, 2))
        A = F.softmax(A / torch.sqrt(torch.tensor(K.shape[1], dtype=torch.float32, device=device)), dim=-1)
        Q = torch.matmul(A, V)
        Q = Q.permute(0, 2, 1, 3).contiguous()
        Q = Q.view(bz, -1, 1280)
        Q = Q + pr_f
        Q = self.pr_linear(Q)
        return Q
    def backdoor_adjustmet(self, logits):
        p_ci = self.p_ci.unsqueeze(0).unsqueeze(-1)
        logits = logits * p_ci
        return logits.sum(1)
    """