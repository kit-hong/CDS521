import torch
import numpy as np

def np2th(weights, conv=False):
    if conv:
        # HWIO (TF) to OIHW (PyTorch)
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

def load_vits16_weights(model, npz_weights):
    # Patch embedding
    model.patch_embedding.projection.weight.data.copy_(
        np2th(npz_weights['embedding/kernel'], conv=True)
    )
    model.patch_embedding.projection.bias.data.copy_(
        np2th(npz_weights['embedding/bias'])
    )
    model.patch_embedding.cls_token.data.copy_(
        np2th(npz_weights['cls'])
    )
    # Remove batch dim for pos_embedding if needed
    pos_embed = np2th(npz_weights['Transformer/posembed_input/pos_embedding'])
    if len(pos_embed.shape) == 3 and pos_embed.shape[0] == 1:
        pos_embed = pos_embed.squeeze(0)
    model.patch_embedding.pos_embedding.data.copy_(pos_embed)
    
    # Transformer blocks
    for i, block in enumerate(model.transformer_encoder_blocks):
        prefix = f'Transformer/encoderblock_{i}'

        # LayerNorms
        block.norm1.weight.data.copy_(
            np2th(npz_weights[f'{prefix}/LayerNorm_0/scale'])
        )
        block.norm1.bias.data.copy_(
            np2th(npz_weights[f'{prefix}/LayerNorm_0/bias'])
        )
        block.norm2.weight.data.copy_(
            np2th(npz_weights[f'{prefix}/LayerNorm_2/scale'])
        )
        block.norm2.bias.data.copy_(
            np2th(npz_weights[f'{prefix}/LayerNorm_2/bias'])
        )

        # Attention - convert (384, 6, 64) -> (384, 384) for weights and (6, 64) -> (384,) for bias
        def merge_heads(w):
            # [in, num_heads, head_dim] -> [in, in]
            return w.reshape(w.shape[0], -1)
        def merge_bias(b):
            return b.reshape(-1)
        
        # Query
        q_weight = np2th(merge_heads(npz_weights[f'{prefix}/MultiHeadDotProductAttention_1/query/kernel']))
        k_weight = np2th(merge_heads(npz_weights[f'{prefix}/MultiHeadDotProductAttention_1/key/kernel']))
        v_weight = np2th(merge_heads(npz_weights[f'{prefix}/MultiHeadDotProductAttention_1/value/kernel']))
        # PyTorch MultiheadAttention expects [embed_dim, 3*embed_dim] for in_proj_weight
        # Concatenate qkv
        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=1)
        block.attn.in_proj_weight.data.copy_(qkv_weight.t())  # needs [3*embed_dim, embed_dim]

        # Bias
        q_bias = np2th(merge_bias(npz_weights[f'{prefix}/MultiHeadDotProductAttention_1/query/bias']))
        k_bias = np2th(merge_bias(npz_weights[f'{prefix}/MultiHeadDotProductAttention_1/key/bias']))
        v_bias = np2th(merge_bias(npz_weights[f'{prefix}/MultiHeadDotProductAttention_1/value/bias']))
        qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
        block.attn.in_proj_bias.data.copy_(qkv_bias)

        # Out proj
        # out/kernel: (6, 64, 384) → (384, 384) for PyTorch (out_features, in_features)
        out_weight = np2th(npz_weights[f'{prefix}/MultiHeadDotProductAttention_1/out/kernel'])
        out_weight = out_weight.reshape(-1, out_weight.shape[-1])  # (384, 384)
        block.attn.out_proj.weight.data.copy_(out_weight)
        block.attn.out_proj.bias.data.copy_(
            np2th(npz_weights[f'{prefix}/MultiHeadDotProductAttention_1/out/bias'])
        )

        # MLP
        block.mlp[0].weight.data.copy_(
            np2th(npz_weights[f'{prefix}/MlpBlock_3/Dense_0/kernel']).t()
        )
        block.mlp[0].bias.data.copy_(
            np2th(npz_weights[f'{prefix}/MlpBlock_3/Dense_0/bias'])
        )
        block.mlp[3].weight.data.copy_(
            np2th(npz_weights[f'{prefix}/MlpBlock_3/Dense_1/kernel']).t()
        )
        block.mlp[3].bias.data.copy_(
            np2th(npz_weights[f'{prefix}/MlpBlock_3/Dense_1/bias'])
        )
    
    # Final LayerNorm
    model.norm.weight.data.copy_(
        np2th(npz_weights['Transformer/encoder_norm/scale'])
    )
    model.norm.bias.data.copy_(
        np2th(npz_weights['Transformer/encoder_norm/bias'])
    )

    # model.classifier_head.weight.data.copy_(np2th(npz_weights['head/kernel']).t())
    # model.classifier_head.bias.data.copy_(np2th(npz_weights['head/bias']))

    print("Loaded weights from npz into ViT-S/16!")

'''
example usage:

model = VisionTransformer(img_size=224, patch_size=16, in_channels=3, num_classes=5, embed_dim=384, num_heads=6, depth=12)
npz = np.load(path)
load_vits16_weights(model, npz)
'''

if __name__ == "__main__":
    print("This module is not meant to be run directly. Use it to load weights into a VisionTransformer model.")