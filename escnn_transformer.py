import torch
import torch.nn as nn
from escnn import gspaces
from torch.amp import autocast
from escnn.nn import (
    FieldType, GeometricTensor, Linear,
    R2Conv, InnerBatchNorm, ReLU, GroupPooling,PointwiseDropout,
    init
)


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    return total_params, trainable_params


class Rot2DLifting(nn.Module):
    def __init__(self, gspace, out_type, in_channels=3, bias=False, downsize=1):
        super().__init__()

        self.in_type = FieldType(
            gspace,
            in_channels * [gspace.trivial_repr]
        )
        self.out_type = out_type
        self.downsize = downsize
        
        # Handle both integer and tuple downsize
        # If tuple, use first element for kernel_size calculation
        if isinstance(downsize, (tuple, list)):
            downsize_int = downsize[0] if len(downsize) > 0 else 1
        else:
            downsize_int = int(downsize)
        
        self.kernel_size = downsize_int * 2 + 1
        self.padding = (self.kernel_size - 1) // 2

        self.conv = R2Conv(
            self.in_type,
            self.out_type,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.downsize,  # Can be int or tuple
            bias=False
        )
        self.bn = InnerBatchNorm(self.out_type)
        self.act = ReLU(self.out_type)

    def forward(self, x):
        x = GeometricTensor(x, self.in_type)
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)

class Rot2DMultiHeadAttention(nn.Module):
    def __init__(self, gspace, in_type, num_heads, dropout=0.1, attn_dropout=0.1):
        super().__init__()

        assert len(in_type.representations) % num_heads == 0

        self.gspace = gspace
        self.in_type = in_type
        self.num_heads = num_heads

        self.head_size = len(in_type.representations) // num_heads
        self.repr = in_type.representations[0]

        self.head_type = FieldType(
            gspace,
            self.head_size * [self.repr]
        )

        self.scalar_type = FieldType(
            gspace,
            [gspace.trivial_repr]
        )

        self.to_q = R2Conv(in_type, in_type, 1, bias=False)
        self.to_k = R2Conv(in_type, in_type, 1, bias=False)
        self.to_v = R2Conv(in_type, in_type, 1, bias=False)
        self.to_out = R2Conv(in_type, in_type, 1, bias=False)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.dropout = PointwiseDropout(in_type, dropout)

    def _split_heads(self, x):
        chunks = x.tensor.chunk(self.num_heads, dim=1)
        return [GeometricTensor(t, self.head_type) for t in chunks]

    def forward(self, x):
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        qh = self._split_heads(q)
        kh = self._split_heads(k)
        vh = self._split_heads(v)

        out_heads = []

        for qi, ki, vi in zip(qh, kh, vh):
            head_dim = qi.tensor.shape[1]
            
            B, C, H, W = qi.tensor.shape
            
            # Equivariant Spatial Attention
            # Compute attention scores: inner product over channels
            qi_flat = qi.tensor.view(B, C, H * W)  # [B, C, HW]
            ki_flat = ki.tensor.view(B, C, H * W)  # [B, C, HW]
            vi_flat = vi.tensor.view(B, C, H * W)  # [B, C, HW]
            
            # Compute attention scores: [B, HW, HW]
            scores = torch.bmm(qi_flat.transpose(1, 2), ki_flat) / (head_dim ** 0.5)
            attn_weights = torch.softmax(scores, dim=-1)  # [B, HW, HW]
            attn_weights = self.attn_dropout(attn_weights)
            
            # Apply attention to values: [B, HW, C]
            out_flat = torch.bmm(attn_weights, vi_flat.transpose(1, 2))  # [B, HW, C]
            out_flat = out_flat.transpose(1, 2)  # [B, C, HW]
            
            out_tensor = out_flat.view(B, C, H, W)
            
            out_heads.append(GeometricTensor(out_tensor, self.head_type))

        out = torch.cat([h.tensor for h in out_heads], dim=1)
        out = GeometricTensor(out, self.in_type)
        return self.dropout(self.to_out(out))


class Rot2DConvMultiHeadAttention(nn.Module):
    """
    Multi-head convolutional self-attention using larger kernel sizes.
    Unlike Rot2DMultiHeadAttention which uses 1x1 convolutions, this uses
    larger kernels (e.g., 3x3, 5x5) to capture local spatial patterns.
    """
    def __init__(self, gspace, in_type, num_heads, kernel_size=3, dropout=0.1, attn_dropout=0.1):
        super().__init__()

        assert len(in_type.representations) % num_heads == 0
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        
        self.gspace = gspace
        self.in_type = in_type
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2

        self.head_size = len(in_type.representations) // num_heads
        self.repr = in_type.representations[0]

        self.head_type = FieldType(
            gspace,
            self.head_size * [self.repr]
        )

        self.scalar_type = FieldType(
            gspace,
            [gspace.trivial_repr]
        )

        # Use larger kernel convolutions for Q, K, V projections
        self.to_q = R2Conv(
            in_type, in_type, 
            kernel_size=kernel_size, 
            padding=self.padding, 
            bias=False
        )
        self.to_k = R2Conv(
            in_type, in_type, 
            kernel_size=kernel_size, 
            padding=self.padding, 
            bias=False
        )
        self.to_v = R2Conv(
            in_type, in_type, 
            kernel_size=kernel_size, 
            padding=self.padding, 
            bias=False
        )
        self.to_out = R2Conv(in_type, in_type, 1, bias=False)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.dropout = PointwiseDropout(in_type, dropout)

    def _split_heads(self, x):
        """Split the channel dimension into multiple heads"""
        chunks = x.tensor.chunk(self.num_heads, dim=1)
        return [GeometricTensor(t, self.head_type) for t in chunks]

    def forward(self, x):
        """
        Forward pass with convolutional multi-head attention.
        
        Args:
            x: GeometricTensor with shape [B, C, H, W]
            
        Returns:
            GeometricTensor with same shape as input
        """
        # with autocast(device_type="cuda", enabled=False):
        # Convolutional projections for Q, K, V
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # Split into multiple heads
        qh = self._split_heads(q)
        kh = self._split_heads(k)
        vh = self._split_heads(v)

        out_heads = []

        for qi, ki, vi in zip(qh, kh, vh):
            head_dim = qi.tensor.shape[1]
            B, C, H, W = qi.tensor.shape
            
            # Equivariant Spatial Attention with convolutional features
            qi_flat = qi.tensor.view(B, C, H * W)  # [B, C, HW]
            ki_flat = ki.tensor.view(B, C, H * W)  # [B, C, HW]
            vi_flat = vi.tensor.view(B, C, H * W)  # [B, C, HW]
            
            # Compute attention scores: [B, HW, HW]
            scores = torch.bmm(qi_flat.transpose(1, 2), ki_flat) / (head_dim ** 0.5)
            scores = scores - scores.amax(dim=-1, keepdim=True)

            # Apply softmax to get attention weights
            attn_weights = torch.softmax(scores, dim=-1)  # [B, HW, HW]
            attn_weights = self.attn_dropout(attn_weights)
            
            # Apply attention to values: [B, HW, C]
            out_flat = torch.bmm(attn_weights, vi_flat.transpose(1, 2))  # [B, HW, C]
            out_flat = out_flat.transpose(1, 2)  # [B, C, HW]
            
            out_tensor = out_flat.view(B, C, H, W)
            
            out_heads.append(GeometricTensor(out_tensor, self.head_type))

        # Concatenate all heads
        out = torch.cat([h.tensor for h in out_heads], dim=1)
        out = GeometricTensor(out, self.in_type)
        
        # Final 1x1 projection and dropout
        return self.dropout(self.to_out(out))


class Rot2DMLP(nn.Module):
    def __init__(self, in_type, hidden_mult=4, dropout=0.1):
        super().__init__()

        gspace = in_type.gspace
        reprs = in_type.representations
        hidden_type = FieldType(
            gspace,
            hidden_mult * reprs
        )

        self.net = nn.Sequential(
            R2Conv(in_type, hidden_type, 1, bias=False),
            InnerBatchNorm(hidden_type),
            ReLU(hidden_type),
            PointwiseDropout(hidden_type, dropout),
            R2Conv(hidden_type, in_type, 1, bias=False),
            InnerBatchNorm(in_type),
            # ReLU(in_type),
            PointwiseDropout(in_type, dropout)
        )

    def forward(self, x):
        return self.net(x)

class Rot2DTransformerBlock(nn.Module):
    def __init__(self, gspace, in_type, num_heads, use_conv_attn=False, conv_kernel_size=3):
        super().__init__()

        if use_conv_attn:
            self.attn = Rot2DConvMultiHeadAttention(
                gspace, in_type, num_heads, kernel_size=conv_kernel_size
            )
        else:
            self.attn = Rot2DMultiHeadAttention(
                gspace, in_type, num_heads
            )
        self.mlp = Rot2DMLP(in_type)

        self.norm1 = InnerBatchNorm(in_type)
        self.norm2 = InnerBatchNorm(in_type)

        self.alpha1 = nn.Parameter(torch.ones(1))
        self.alpha2 = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        x = x + self.alpha1 * self.attn(self.norm1(x))
        x = x + self.alpha2 * self.mlp(self.norm2(x))
        return x

class Rot2DClassificationHead(nn.Module):
    def __init__(self, in_type, gspace, num_classes=10, dropout=0.1):
        super().__init__()

        self.in_type = in_type
        self.pool = GroupPooling(in_type)
        
        self.out_type = FieldType(
            gspace,
            num_classes * [gspace.trivial_repr]
        )

        N = gspace.fibergroup.order() if gspace.fibergroup.order() > 0 else 1
        linear_input_size = in_type.size // N 
        hidden_size = linear_input_size // 2
        self.fc1 = nn.Linear(linear_input_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout2 = nn.Dropout(dropout * 0.5)

    def forward(self, x, return_probs=False):
        x = self.pool(x)
        x = x.tensor
        x = x.mean(dim=(2, 3))
        
        x = self.fc1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.dropout2(x)
        
        if return_probs:
            return nn.functional.softmax(x, dim=1)
        return x


class Rot2DTransformer(nn.Module):
    def __init__(self, depth=4, in_channels=3, channels=8, heads=4, num_classes=10, 
                 gspace=gspaces.rot2dOnR2(N=4), downsize=1, use_conv_attn=False, conv_kernel_size=3):
        super().__init__()

        self.gspace = gspace
        self.in_channels = in_channels
        self.hidden_type = FieldType(
            self.gspace,
            channels * [self.gspace.regular_repr]
        )

        self.lift = Rot2DLifting(self.gspace, self.hidden_type, self.in_channels, downsize=downsize)

        self.blocks = nn.ModuleList([
            Rot2DTransformerBlock(
                self.gspace,
                self.hidden_type,
                heads,
                use_conv_attn=use_conv_attn,
                conv_kernel_size=conv_kernel_size
            )
            for _ in range(depth)
        ])

        self.head = Rot2DClassificationHead(
            self.hidden_type,
            self.gspace,
            num_classes
        )
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights properly for equivariant layers"""
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, R2Conv):
                    try:
                        init.general_orthogonal_init(m.weights, m.basisexpansion)
                    except (AttributeError, TypeError):
                        # Fallback to deltaorthonormal
                        init.deltaorthonormal_init(m.weights, m.basisexpansion)
                
                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    m.weight.data.mul_(0.1)
                    
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                
                elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, InnerBatchNorm)):
                    if hasattr(m, 'weight') and m.weight is not None:
                        nn.init.constant_(m.weight, 1.0)
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)

    def forward(self, x, return_probs=False):
        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        return self.head(x, return_probs)



if __name__ == "__main__":
    
    gspace = gspaces.rot2dOnR2(N=4)  # C4
    model = Rot2DTransformer(
    depth=4,
    channels=8,
    heads=4,
    gspace=gspace,
    downsize=2
    )

    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(y.shape)
    print(y.type())
    print("blah blah blah")  
