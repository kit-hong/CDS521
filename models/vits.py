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
    
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise
        self.norm = nn.LayerNorm(dim) 
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)  
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)  # [B, C, H, W]
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C] for LayerNorm
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # back to [B, C, H, W]
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma.view(1, -1, 1, 1) * x
        x = self.drop_path(x)
        return shortcut + x
    
class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x / keep_prob * random_tensor
       
class DepthwiseConvTokenizer(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.in_channels = in_channels

        self.depthwise_conv = nn.Conv2d(
            in_channels, in_channels, 
            kernel_size=3, 
            padding=1,
            groups=in_channels
        )
        self.pointwise_conv = nn.Conv2d(
            in_channels, in_channels * embed_dim, 
            kernel_size=1, 
            groups=in_channels 
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        x = self.depthwise_conv(x)  # (B, C, H, W)
        
        x = self.pointwise_conv(x)  # (B, C * embed_dim, H, W)
        
        x = x.view(B, C, self.embed_dim, H, W)
        tokens = torch.mean(x, dim=(3, 4))  # (B, C, embed_dim)
        
        return self.norm(tokens)
        
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
    
class CustomCNNVit(nn.Module):
    def __init__(self, 
                 img_size = 224, 
                 in_channels = 3,
                 num_features = 6, 
                 num_classes = 5,
                 embed_dim = 768, 
                 num_heads = 12, 
                 depth = 12, 
                 mlp_ratio=4.0, 
                 dropout=0.1):
        super().__init__()

        self.img_size = img_size
        self.embed_dim = embed_dim
        self.num_features = num_features
        self.num_classes = num_classes
 
        self.CNNEncoder = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features),  
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(num_features, 2*num_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2*num_features),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(2*num_features, 4*num_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4*num_features),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(4*num_features, 8*num_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8*num_features),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.tokenizer = DepthwiseConvTokenizer(8*num_features, embed_dim)
        
        # hardcode to use cls token 
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        max_seq_len = 8*num_features + 1
        #self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) * 0.02)
        #self.pos_dropout = nn.Dropout(dropout)

        self.transformer_encoder_blocks = nn.Sequential(
            *[TransformerEncoder(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
            )
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
            )
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02) # glorot for linear layers
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # he initialization
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # get cnn features
        cnn_features = self.CNNEncoder(x)
        # tokenize feature maps
        tokens = self.tokenizer(cnn_features)
        
        batch_size, seq_len, embed_dimension = tokens.shape

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, embed_dim)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # (B, seq_len+1, embed_dim)

        #tokens = tokens + self.pos_encoding[:, :seq_len+1, :]
        #tokens = self.pos_dropout(tokens)  

        # transformer encoder
        tokens = self.transformer_encoder_blocks(tokens)
        tokens = self.norm(tokens)

        cls_token = tokens[:, 0]  # (B, embed_dim)
        logits = self.classifier_head(cls_token)

        return logits
    
class CustomCNNVitV2(nn.Module):
    def __init__(self, 
                 img_size = 224, 
                 in_channels = 3,
                 cnn_depth = [3,3,3,3],
                 num_features = [12, 24, 48, 96], 
                 num_classes = 5,
                 embed_dim = 768, 
                 num_heads = 12, 
                 depth = 12, 
                 mlp_ratio=4.0, 
                 dropout=0.1):
        super().__init__()

        self.img_size = img_size
        self.embed_dim = embed_dim
        self.num_features = num_features
        self.num_classes = num_classes

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, num_features[0], kernel_size=4, stride=4),
            nn.GroupNorm(1, num_features[0])  # replaces incorrect LayerNorm
        )

        self.cnn_block1 = nn.Sequential(*[
            ConvNeXtBlock(num_features[0], drop_path=0.05)
            for i in range(cnn_depth[0])
        ])
        self.downsample1 = nn.Conv2d(num_features[0], num_features[1], kernel_size=2, stride=2)

        self.cnn_block2 = nn.Sequential(*[
            ConvNeXtBlock(num_features[1], drop_path=0.05)
            for i in range(cnn_depth[1])
        ])
        self.downsample2 = nn.Conv2d(num_features[1], num_features[2], kernel_size=2, stride=2)

        self.cnn_block3 = nn.Sequential(*[
            ConvNeXtBlock(num_features[2], drop_path=0.05)
            for i in range(cnn_depth[2])
        ])
        self.downsample3 = nn.Conv2d(num_features[2], num_features[3], kernel_size=2, stride=2)

        self.cnn_block4 = nn.Sequential(*[
            ConvNeXtBlock(num_features[3], drop_path=0.05)
            for i in range(cnn_depth[3])
        ])

        self.tokenizer1 = DepthwiseConvTokenizer(num_features[0], embed_dim)
        self.tokenizer2 = DepthwiseConvTokenizer(num_features[1], embed_dim)
        self.tokenizer3 = DepthwiseConvTokenizer(num_features[2], embed_dim)
        self.tokenizer4 = DepthwiseConvTokenizer(num_features[3], embed_dim)

        # hardcode to use cls token and pos encoding
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        max_seq_len = (img_size // 4) ** 2 + 1  # stem downsamples 4×

        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) * 0.02)
        self.pos_dropout = nn.Dropout(dropout)

        self.transformer_encoder_blocks = nn.Sequential(
            *[TransformerEncoder(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
            )
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
            )
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02) # glorot for linear layers
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # he initialization
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # get cnn features
        x = self.stem(x)
        cnn_features_1 = self.cnn_block1(x)

        x = self.downsample1(cnn_features_1)
        cnn_features_2 = self.cnn_block2(x)

        x = self.downsample2(cnn_features_2)
        cnn_features_3 = self.cnn_block3(x)

        x = self.downsample3(cnn_features_3)
        cnn_features_4 = self.cnn_block4(x) 

        # tokenize feature maps
        tokens_stage_1 = self.tokenizer1(cnn_features_1)
        tokens_stage_2 = self.tokenizer2(cnn_features_2)
        tokens_stage_3 = self.tokenizer3(cnn_features_3)
        tokens_stage_4 = self.tokenizer4(cnn_features_4)

        # concatenate tokens from all stages
        tokens = torch.cat([tokens_stage_1, tokens_stage_2, tokens_stage_3, tokens_stage_4], dim=1)  # (B, seq_len, embed_dim)

        # prepraring for transformer encoder
        batch_size, seq_len, embed_dimension = tokens.shape

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, embed_dim)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # (B, seq_len+1, embed_dim)

        tokens = tokens + self.pos_encoding[:, :seq_len+1, :]
        tokens = self.pos_dropout(tokens)  

        # transformer encoder
        tokens = self.transformer_encoder_blocks(tokens)
        tokens = self.norm(tokens)

        cls_token = tokens[:, 0]  # (B, embed_dim)
        logits = self.classifier_head(cls_token)

        return logits
    
def testing(flag = True):
    if flag:
        model = CustomCNNVitV2(
        img_size = 224, 
        in_channels = 3,
        cnn_depth = [3,3,3,3],
        num_features = [12, 24, 48, 96], 
        num_classes = 5,
        embed_dim = 768, 
        num_heads = 12, 
        depth = 12, 
        mlp_ratio=4.0, 
        dropout=0.1
    )

        # Test input
        batch_size = 1
        x = torch.randn(batch_size, 3, 224, 224)

        print(f"Input shape: {x.shape}")

        model.eval()
        with torch.no_grad():
            logits = model(x)
            print(f"Output shape: {logits.shape}")
            print(f"Output values: {logits}")
        
if __name__ == "__main__":
    print("This is a Vision Transformer model implementation.") 
    testing(True)