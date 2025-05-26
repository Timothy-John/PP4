import torch
from torch import nn
import torchvision
import torch.nn.functional as F
from collections import OrderedDict
import math
from .basic_module import BasicModule

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Define linear layers for Q, K, V transformations
        self.query_layer = nn.Linear(embed_dim, embed_dim)
        self.key_layer = nn.Linear(embed_dim, embed_dim)
        self.value_layer = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
    
    def split_heads(self, x):
        # Reshape and split into heads
        batch_size, seq_length, embed_dim = x.size()
        x = x.view(batch_size, seq_length, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # Rearrange to [batch, num_heads, seq_length, head_dim]
    
    def combine_heads(self, x):
        # Concatenate heads back together
        batch_size, num_heads, seq_length, head_dim = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()  # Rearrange to [batch, seq_length, num_heads, head_dim]
        return x.view(batch_size, seq_length, num_heads * head_dim)
    
    def forward(self, x):
        Q = self.split_heads(self.query_layer(x))
        K = self.split_heads(self.key_layer(x))
        V = self.split_heads(self.value_layer(x))
        # Scaled Dot-Product Attention for each head
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.head_dim ** 0.5
        attention_weights = F.softmax(scores, dim=-1)
        multihead_output = torch.matmul(attention_weights, V)
        # Combine heads and apply output projection
        multihead_output = self.combine_heads(multihead_output)
        return self.out_proj(multihead_output), attention_weights

class ScaledDotAttention(nn.Module):
    def __init__(self, embed_dim):
        super(ScaledDotAttention, self).__init__()
        self.query_layer = nn.Linear(embed_dim, embed_dim)
        self.key_layer = nn.Linear(embed_dim, embed_dim)
        self.value_layer = nn.Linear(embed_dim, embed_dim)
        self.scale_factor = embed_dim ** 0.5  # Square root of embed dimension for scaling
        
    def forward(self, x):
        Q = self.query_layer(x)
        K = self.key_layer(x)
        V = self.value_layer(x)
        
        # Compute the scaled dot-product attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale_factor
        attention_weights = F.softmax(scores, dim=-1)
        
        # Output weighted sum of values
        attention_output = torch.matmul(attention_weights, V)

        return attention_output, attention_weights

class Final_FF(BasicModule):
    def __init__(self):
        super().__init__()
        self.attention = ScaledDotAttention(embed_dim=400)
        #self.attention = MultiHeadAttention(embed_dim=400, num_heads=1)
        #self.fc0 = nn.Linear(400, 300)
        self.fc1 = nn.Linear(400, 126)

    def forward(self, x, MERT_FF_out):
        x = torch.cat([x.squeeze(),MERT_FF_out.squeeze()]).unsqueeze(0)
        #feat = self.fc0(x)
        feat_attn_out, feat_attn_weight = self.attention(x.unsqueeze(-2))
        x = self.fc1(feat_attn_out[:,-1,:])
        return x,feat_attn_out.squeeze(1)
