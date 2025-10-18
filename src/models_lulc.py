""" Models definitions archictures """
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LULCNet(nn.Module):
    """
    LULC-Net: Land Use Land Cover Network
    Arquitetura otimizada para segmentação semântica de uso e cobertura da terra
    """
    
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight.data)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight.data)
    
    def __init__(self, in_channels=3, out_channels=6, features=64, dropout_rate=0.2):
        super(LULCNet, self).__init__()
        
        self.features = features
        self.dropout_rate = dropout_rate
        
        # Encoder - Downsampling Path
        self.enc1 = self._conv_block(in_channels, features, "enc1")
        self.enc2 = self._conv_block(features, features * 2, "enc2")
        self.enc3 = self._conv_block(features * 2, features * 4, "enc3")
        self.enc4 = self._conv_block(features * 4, features * 8, "enc4")
        
        # Bridge
        self.bridge = self._conv_block(features * 8, features * 16, "bridge")
        
        # Decoder - Upsampling Path com camadas de concatenação corretas
        self.upconv4 = nn.Conv2d(features * 16, features * 8, kernel_size=1)
        self.dec4 = self._conv_block(features * 16, features * 8, "dec4")  # 512 + 512 = 1024 -> 512
        
        self.upconv3 = nn.Conv2d(features * 8, features * 4, kernel_size=1)
        self.dec3 = self._conv_block(features * 8, features * 4, "dec3")   # 256 + 256 = 512 -> 256
        
        self.upconv2 = nn.Conv2d(features * 4, features * 2, kernel_size=1)
        self.dec2 = self._conv_block(features * 4, features * 2, "dec2")   # 128 + 128 = 256 -> 128
        
        self.upconv1 = nn.Conv2d(features * 2, features, kernel_size=1)
        self.dec1 = self._conv_block(features * 2, features, "dec1")       # 64 + 64 = 128 -> 64
        
        # Final classification layer
        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Dropout
        self.dropout = nn.Dropout2d(dropout_rate)
        
        self.apply(self.weight_init)
    
    def _conv_block(self, in_ch, out_ch, name):
        """Bloco convolucional com batch normalization e ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)                    # [B, 64, H, W]
        e2 = self.enc2(self.pool(e1))        # [B, 128, H/2, W/2]
        e3 = self.enc3(self.pool(e2))        # [B, 256, H/4, W/4]
        e4 = self.enc4(self.pool(e3))        # [B, 512, H/8, W/8]
        
        # Bridge with dropout
        bridge = self.bridge(self.pool(e4))  # [B, 1024, H/16, W/16]
        bridge = self.dropout(bridge)
        
        # Decoder with skip connections
        # Nível 4
        d4_up = self.upsample(bridge)        # [B, 1024, H/8, W/8]
        d4_up = self.upconv4(d4_up)          # [B, 512, H/8, W/8]
        d4 = torch.cat([d4_up, e4], dim=1)   # [B, 1024, H/8, W/8]
        d4 = self.dec4(d4)                   # [B, 512, H/8, W/8]
        d4 = self.dropout(d4)
        
        # Nível 3
        d3_up = self.upsample(d4)            # [B, 512, H/4, W/4]
        d3_up = self.upconv3(d3_up)          # [B, 256, H/4, W/4]
        d3 = torch.cat([d3_up, e3], dim=1)   # [B, 512, H/4, W/4]
        d3 = self.dec3(d3)                   # [B, 256, H/4, W/4]
        d3 = self.dropout(d3)
        
        # Nível 2
        d2_up = self.upsample(d3)            # [B, 256, H/2, W/2]
        d2_up = self.upconv2(d2_up)          # [B, 128, H/2, W/2]
        d2 = torch.cat([d2_up, e2], dim=1)   # [B, 256, H/2, W/2]
        d2 = self.dec2(d2)                   # [B, 128, H/2, W/2]
        d2 = self.dropout(d2)
        
        # Nível 1
        d1_up = self.upsample(d2)            # [B, 128, H, W]
        d1_up = self.upconv1(d1_up)          # [B, 64, H, W]
        d1 = torch.cat([d1_up, e1], dim=1)   # [B, 128, H, W]
        d1 = self.dec1(d1)                   # [B, 64, H, W]
        
        # Final classification
        out = self.final_conv(d1)            # [B, n_classes, H, W]
        out = F.log_softmax(out, dim=1)
        
        return out

def build_lulc_model(params):
    """
    Constrói o modelo LULC-Net com os parâmetros fornecidos
    """
    model = LULCNet(
        in_channels=3,
        out_channels=params['n_classes'],
        features=params.get('features', 64),
        dropout_rate=params.get('dropout_rate', 0.2)
    )
    
    model.to(params['device'])
    return model

# Função compatível com a interface original do models.py
def build_model(model_name: str, params: dict):
    """
    Função compatível com a interface original do models.py
    """
    if model_name == 'lulcnet':
        return build_lulc_model(params)
    else:
        raise Exception(f"{model_name} -> invalid model name.")