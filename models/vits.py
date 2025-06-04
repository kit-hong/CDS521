import torch
import torch.nn as nn
from einops import rearrange, repeat

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by the patch size."

        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2

        self.projection = nn.Conv2d(in_channels,embed_dim,kernel_size=patch_size,stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(self.num_patches + 1, embed_dim))

    def forward(self, x):
        B = x.shape[0] # Batch size
        x = self.projection(x)
        x = rearrange(x, 'b c h w -> b (h w) c')

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embedding

        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x
    
class VisionTransformer(nn.Module):
    def __init__(self, 
                 img_size = 224, 
                 patch_size = 16, 
                 in_channels = 3, 
                 num_classes = 5,
                 embed_dim = 768, 
                 num_heads = 12, 
                 depth = 12, 
                 mlp_ratio=4.0, 
                 dropout=0.1):
        super().__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.transformer_encoder_blocks = nn.Sequential(
            *[TransformerEncoder(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
            )
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer_encoder_blocks(x)
        x = self.norm(x)
        cls_token = x[:, 0]
        x = self.classifier_head(cls_token)
        return x
    
if __name__ == "__main__":
    print("This is a Vision Transformer model implementation.") 