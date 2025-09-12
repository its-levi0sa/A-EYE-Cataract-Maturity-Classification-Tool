import torch.nn as nn
from .utils import conv_3x3_bn, MV2Block
from .modified_mobilevit import ModifiedMobileViT
from .radial_tokenizer import RadialTokenizer

class AEyeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        dims = config['dims']
        embed_dim = config['embed_dim']
        num_rings = config['num_rings']
        
        # --- Transformer depths ---
        transformer_depths = [2, 4, 3]

        self.tokenizer = RadialTokenizer(num_rings=num_rings)
        self.num_rings = self.tokenizer.num_rings

        # Model Architecture Stages
        self.stage1 = conv_3x3_bn(3, dims[0], stride=2)
        self.stage2 = MV2Block(dims[0], dims[1], stride=2)
        
        self.stage3 = ModifiedMobileViT(
            in_channels=dims[1], embed_dim=embed_dim, depth=transformer_depths[0], num_rings=self.num_rings
        )
        self.stage4 = MV2Block(dims[1], dims[2], stride=2)
        
        self.stage5 = ModifiedMobileViT(
            in_channels=dims[2], embed_dim=embed_dim, depth=transformer_depths[1], num_rings=self.num_rings
        )
        self.stage6 = MV2Block(dims[2], dims[3], stride=2)
        
        self.stage7 = ModifiedMobileViT(
            in_channels=dims[3], embed_dim=embed_dim, depth=transformer_depths[2], num_rings=self.num_rings
        )
        
        # Classification Head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Sequential(
            nn.Linear(dims[3], dims[3] // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dims[3] // 2, 1)
        )

    def forward(self, x_img, return_tokens=False):
        tokens = self.tokenizer(x_img)
        x = self.stage1(x_img)
        x = self.stage2(x)
        x = self.stage3(x, tokens)
        x = self.stage4(x)
        x = self.stage5(x, tokens)
        x = self.stage6(x)
        x = self.stage7(x, tokens)
        x = self.pool(x).view(x.size(0), -1)
        output = self.fc(x)

        if return_tokens:
            return output, tokens
        return output