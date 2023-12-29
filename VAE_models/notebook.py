import torch

noise_scale_during_inference = 1.4
z = torch.randn(16)*noise_scale_during_inference
z_unsqueeze = torch.randn(16).unsqueeze(0)*noise_scale_during_inference

print('z:', z, z.shape)
print('z(unsqueeze):', z_unsqueeze, z_unsqueeze.shape)
