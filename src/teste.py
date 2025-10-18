# test_lulc.py
import torch
from models_lulc import build_lulc_model

# Parâmetros simples (igual ao seu código original)
params = {
    'n_classes': 6,
    'features': 64,
    'dropout_rate': 0.2,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

# Criar modelo
model = build_lulc_model(params)

# Testar forward pass
x = torch.randn(2, 3, 256, 256)  # batch_size=2, channels=3, height=256, width=256
output = model(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Output min/max: {output.min():.4f}/{output.max():.4f}")