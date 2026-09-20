import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import cast
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.models.feature_extraction import create_feature_extractor

class Routing(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2, temperature=30):
        super().__init__()
        self.temperature = temperature
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(in_channels, out_channels)
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return F.softmax(self.fc(x) / self.temperature, dim=1)

class CondConv2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=1, bias: bool = True, num_experts: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.stride = stride
        self.padding = padding
        self.use_bias = bias
        self.num_experts = num_experts
        self.routing = Routing(in_channels, out_channels=num_experts)

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
            for _ in range(num_experts)
        ])
        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        routing_weights = self.routing(x)
        expert_weights = torch.stack([cast(torch.Tensor, c.weight) for c in self.convs])
        expert_biases = torch.stack([cast(torch.Tensor, c.bias) for c in self.convs]) if self.use_bias else None

        mixed_weights = torch.einsum('be,eoikl->boikl', routing_weights, expert_weights)
        mixed_weights = mixed_weights.reshape(B * self.out_channels, C, self.kernel_size[0], self.kernel_size[1])
        x_reshaped = x.reshape(1, B * C, H, W)
        mixed_biases = torch.einsum('be,eo->bo', routing_weights, expert_biases).reshape(-1) if expert_biases is not None else None

        out = F.conv2d(x_reshaped, mixed_weights, bias=mixed_biases, stride=self.stride, padding=self.padding, groups=B)
        return out.reshape(B, self.out_channels, out.shape[2], out.shape[3])

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        pad = kernel_size // 2 if isinstance(kernel_size, int) else (kernel_size[0] // 2, kernel_size[1] // 2)
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=pad, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, nb_filter):
        super().__init__()
        self.branch1x1 = SeparableConv2d(in_channels, nb_filter, kernel_size=1)
        self.branch3x3_base = SeparableConv2d(in_channels, nb_filter, kernel_size=1)
        self.branch3x3_1 = SeparableConv2d(nb_filter, nb_filter, kernel_size=(3, 1))
        self.branch3x3_2 = SeparableConv2d(nb_filter, nb_filter, kernel_size=(1, 3))
        self.branch5x5_base = SeparableConv2d(in_channels, nb_filter, kernel_size=1)
        self.branch5x5_1 = SeparableConv2d(nb_filter, nb_filter, kernel_size=(3, 1))
        self.branch5x5_2 = SeparableConv2d(nb_filter, nb_filter, kernel_size=(1, 3))
        self.branch5x5_final1 = SeparableConv2d(nb_filter, nb_filter, kernel_size=(3, 1))
        self.branch5x5_final2 = SeparableConv2d(nb_filter, nb_filter, kernel_size=(1, 3))
        self.branchpool_max = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branchpool_conv = SeparableConv2d(in_channels, nb_filter, kernel_size=1)

    def forward(self, x):
        b1 = self.branch1x1(x)
        b3_b = self.branch3x3_base(x)
        b3 = self.branch3x3_1(b3_b) + self.branch3x3_2(b3_b)
        b5_b = self.branch5x5_base(x)
        b5_m = self.branch5x5_1(b5_b) + self.branch5x5_2(b5_b)
        b5 = self.branch5x5_final1(b5_m) + self.branch5x5_final2(b5_m)
        bp = self.branchpool_conv(self.branchpool_max(x))
        return torch.cat([b1, b3, b5, bp], dim=1)

class PatchTokenizer(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, img_size):
        super().__init__()
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, x):
        return self.projection(x).flatten(2).transpose(1, 2) + self.position_embedding

class SSEBlock(nn.Module):
    def __init__(self, in_channels, ratio=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // ratio)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // ratio, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        se = self.sigmoid(self.fc2(self.relu(self.fc1(self.avg_pool(x).view(b, c))))).view(b, c, 1, 1)
        se_out = x * se
        s_mean = torch.mean(x, dim=1, keepdim=True)
        s_std = torch.std(x, dim=1, keepdim=True)
        s_max = torch.max(x, dim=1, keepdim=True)[0]
        # Output channels = in_channels + 3
        return torch.cat([se_out, s_mean, s_std, s_max], dim=1)

class MangiferaNet(nn.Module):
    def __init__(self, num_classes=6, dropout_rate=0.4):
        super().__init__()

        # --- 1. THE BACKBONE ---
        base_model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        return_nodes = {'features.1': 'x1', 'features.3': 'x2', 'features.6': 'x_inc'}
        self.backbone = create_feature_extractor(base_model, return_nodes=return_nodes)

        # Unfreeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = True

        # --- 2. MULTI-SCALE BRANCHES ---
        # Branch 1
        self.cond1_1 = CondConv2D(in_channels=16, out_channels=32, kernel_size=3, stride=2)
        self.cond1_2 = CondConv2D(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.cond1_3 = CondConv2D(in_channels=64, out_channels=64, kernel_size=3, stride=2)
        self.conv1_final = nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=0)
        self.sse1 = SSEBlock(in_channels=128, ratio=4) # Outputs 128 + 3 = 131 channels
        self.p1_proj = nn.Conv2d(131, 128, kernel_size=1) # Projects 131 -> 128

        # Branch 2
        self.cond2_1 = CondConv2D(in_channels=24, out_channels=32, kernel_size=3, stride=2)
        self.cond2_2 = CondConv2D(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.conv2_final = nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=0)
        self.sse2 = SSEBlock(in_channels=128, ratio=4) # Outputs 128 + 3 = 131 channels
        self.p2_proj = nn.Conv2d(131, 128, kernel_size=1) # Projects 131 -> 128

        # Branch 3 (Inception)
        self.inception = InceptionBlock(in_channels=32, nb_filter=64) # Outputs 4 * 64 = 256 channels
        self.cond3_1 = CondConv2D(in_channels=256, out_channels=128, kernel_size=3, stride=2)
        self.conv3_final = nn.Conv2d(128, 128, kernel_size=4, stride=1, padding=0)
        self.sse3 = SSEBlock(in_channels=128, ratio=4) # Outputs 128 + 3 = 131 channels
        self.p3_proj = nn.Conv2d(131, 128, kernel_size=1) # Projects 131 -> 128

        # --- 3. VISION TRANSFORMER ---
        self.tokenizer = PatchTokenizer(in_channels=256, patch_size=7, embed_dim=128, img_size=28)
        self.vit_dropout = nn.Dropout(dropout_rate)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256, dropout=dropout_rate, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.final_norm = nn.LayerNorm(128)
        self.vit_projection = nn.Conv2d(128, 128, kernel_size=1)

        # --- 4. FUSION & CLASSIFICATION ---
        self.fusion_sse = SSEBlock(in_channels=128, ratio=4) # Outputs 128 + 3 = 131 channels
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(131, num_classes) # Correctly takes 131 input features

    def forward(self, x):
        features = self.backbone(x)
        x1, x2, x_inc = features['x1'], features['x2'], features['x_inc']

        # Branch 1
        p1 = self.p1_proj(self.sse1(self.conv1_final(self.cond1_3(self.cond1_2(self.cond1_1(x1))))))

        # Branch 2
        p2 = self.p2_proj(self.sse2(self.conv2_final(self.cond2_2(self.cond2_1(x2)))))

        # Branch 3
        inc_out = self.inception(x_inc)
        p3 = self.p3_proj(self.sse3(self.conv3_final(self.cond3_1(inc_out))))

        # ViT Branch
        tokens = self.tokenizer(inc_out)
        tokens = self.vit_dropout(tokens)
        vit_out = self.transformer(tokens)
        vit_out = self.final_norm(vit_out)

        b = vit_out.size(0)
        vit_spatial = vit_out.transpose(1, 2).view(b, 128, 4, 4)
        vit_spatial = F.interpolate(vit_spatial, size=(11, 11), mode='bilinear', align_corners=False)
        vit_spatial = self.vit_projection(vit_spatial)

        # Merge all branches (all are 128 channels at 11x11)
        merged = p1 + p2 + p3 + vit_spatial
        merged = self.fusion_sse(merged) # Becomes 131 channels

        out = self.global_pool(merged)
        out = torch.flatten(out, 1)

        return self.classifier(out)

