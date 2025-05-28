import torch
import torch.nn as nn
import torch.nn.functional as F

class ColorAwareConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
        
        self.color_pool = nn.AdaptiveAvgPool2d(8)
        self.color_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        
        color_stats = self.color_pool(x)
        color_stats = self.color_conv(color_stats)
        color_features = F.adaptive_avg_pool2d(color_stats, 1)
        color_features = F.interpolate(color_features, size=out.shape[2:])
        
        return out + 0.2 * color_features

class MobileViTCoBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim):
        super().__init__()
        self.ph, self.pw = patch_size
        
        self.local_rep = nn.Sequential(
            ColorAwareConv(channel, channel, stride=1),
            nn.Conv2d(channel, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU()
        )
        
        actual_depth = max(1, depth // 2)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=n_head, 
                dim_feedforward=mlp_dim,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ),
            num_layers=actual_depth
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 2, channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.SiLU()
        )
        
    def forward(self, x):
        local_rep = self.local_rep(x)
        
        #reshape for transformer
        y = local_rep.permute(0, 2, 3, 1)
        y = y.reshape(y.shape[0], -1, y.shape[-1])
        y = self.transformer(y)
        
        #reshape back to spatial dimensions
        y = y.reshape(local_rep.shape[0], local_rep.shape[2], local_rep.shape[3], -1)
        y = y.permute(0, 3, 1, 2)
        
        out = torch.cat([local_rep, y], dim=1)
        out = self.fusion(out)
        
        return out

class ColorHistogramLayer(nn.Module):
    def __init__(self, bins=16, output_dim=64):
        super().__init__()
        self.bins = bins
        self.fc = nn.Sequential(
            nn.Linear(bins * 3, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        r_hist = self._create_hist(x[:, 0].view(batch_size, -1), 0.0, 1.0)
        g_hist = self._create_hist(x[:, 1].view(batch_size, -1), 0.0, 1.0)
        b_hist = self._create_hist(x[:, 2].view(batch_size, -1), 0.0, 1.0)
        
        hist_features = torch.cat([r_hist, g_hist, b_hist], dim=1)
        return self.fc(hist_features)
    
    def _create_hist(self, x, min_val, max_val):
        bin_edges = torch.linspace(min_val, max_val, self.bins + 1, device=x.device)
        hist = torch.zeros(x.shape[0], self.bins, device=x.device)
        
        for i in range(self.bins):
            mask = (x >= bin_edges[i]) & (x < bin_edges[i + 1])
            if i == self.bins - 1:  
                mask = mask | (x == bin_edges[i + 1])
            hist[:, i] = mask.float().mean(dim=1) 
        
        return hist

class MobileViTCo(nn.Module):
    def __init__(self, image_size=128, num_classes=4, variant='XXS'):
        super().__init__()
        
        if variant == 'XXS':
            channels = [16, 24, 48, 64, 80]
            transformer_dims = [64, 80, 96]
            depths = [2, 2, 2]
            mlp_dims = [128, 160, 192]
        elif variant == 'XS':
            channels = [16, 32, 48, 80, 160]
            transformer_dims = [96, 120, 160]
            depths = [2, 4, 4]
            mlp_dims = [192, 240, 320]
        else: 
            raise ValueError(f"Variant {variant} not supported. Choose from 'XXS' or 'XS'")
        
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU()
        )
        
        #mobilenet blocks
        
        self.mv2_1 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.SiLU()
        )
        
        self.mv2_2 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[2]),
            nn.SiLU()
        )
        
        # MobileViTCo blocks
        self.mvit1 = MobileViTCoBlock(
            dim=transformer_dims[0],
            depth=depths[0],
            channel=channels[2],
            kernel_size=3,
            patch_size=(2, 2),
            mlp_dim=mlp_dims[0]
        )
        
        self.mv2_3 = nn.Sequential(
            nn.Conv2d(channels[2], channels[3], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[3]),
            nn.SiLU()
        )

        self.mvit2 = MobileViTCoBlock(
            dim=transformer_dims[1],
            depth=depths[1],
            channel=channels[3],
            kernel_size=3,
            patch_size=(2, 2),
            mlp_dim=mlp_dims[1]
        )
        
        self.mv2_4 = nn.Sequential(
            nn.Conv2d(channels[3], channels[4], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[4]),
            nn.SiLU()
        )
        
        self.mvit3 = SimplifiedMobileViTBlock(
            dim=transformer_dims[2],
            depth=depths[2],
            channel=channels[4],
            kernel_size=3,
            patch_size=(2, 2),
            mlp_dim=mlp_dims[2]
        )
        
        self.color_histogram = ColorHistogramLayer(bins=16, output_dim=64)
        
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(channels[4] + 64, num_classes)
        
    def forward(self, x):
        color_hist = self.color_histogram(x)
        
        x = self.conv1(x)
        x = self.mv2_1(x)
        x = self.mv2_2(x)
        x = self.mvit1(x)
        x = self.mv2_3(x)
        x = self.mvit2(x)
        x = self.mv2_4(x)
        x = self.mvit3(x)
        
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        
        combined = torch.cat([x, color_hist], dim=1)
        combined = self.dropout(combined)
        return self.classifier(combined)