import torch
import torch.nn as nn

class RadialTokenizer(nn.Module):
    def __init__(self, image_size=256, num_rings=16):
        super().__init__()
        self.image_size = image_size
        self.center = (image_size // 2, image_size // 2)
        self.num_rings = num_rings

        if self.num_rings == 4: ring_width = 32
        elif self.num_rings == 8: ring_width = 16
        elif self.num_rings == 16: ring_width = 8
        else: raise ValueError("Unsupported number of rings. Must be 4, 8, or 16.")
            
        self.rings = [(i * ring_width, (i + 1) * ring_width) for i in range(self.num_rings)]

        # Pre-compute ring masks for GPU efficiency
        y, x = torch.meshgrid(torch.arange(0, image_size), torch.arange(0, image_size), indexing='ij')
        distance_grid = torch.sqrt((x - self.center[0])**2 + (y - self.center[1])**2)
        
        ring_masks = [(distance_grid >= r_inner) & (distance_grid < r_outer) for r_inner, r_outer in self.rings]
        self.register_buffer('ring_masks', torch.stack(ring_masks, dim=0).float())

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        B, C, H, W = image_tensor.shape
        original_device = image_tensor.device
        
        masks = self.ring_masks.to(original_device)
        masked_pixels = masks.unsqueeze(0).unsqueeze(2) * image_tensor.unsqueeze(1)
        num_pixels_per_ring = masks.sum(dim=[1, 2]) + 1e-6

        # Calculate statistics
        mean_vals = masked_pixels.sum(dim=[3, 4]) / num_pixels_per_ring.view(1, self.num_rings, 1)
        mean_sq_vals = (masked_pixels**2).sum(dim=[3, 4]) / num_pixels_per_ring.view(1, self.num_rings, 1)
        std_vals = torch.sqrt(torch.clamp(mean_sq_vals - mean_vals**2, min=0))
        
        # --- DETERMINISM ---
        flat_pixels = masked_pixels.view(B, self.num_rings, C, -1)
        flat_pixels_cpu = flat_pixels.cpu()
        flat_pixels_cpu[flat_pixels_cpu == 0] = float('nan')
        
        # Calculate median on the CPU
        median_vals_cpu = torch.nanmedian(flat_pixels_cpu, dim=3).values
        
        # Move the result back to the original device
        median_vals = median_vals_cpu.to(original_device)

        # Concatenate features: (mean, std, median) for each channel
        tokens = torch.cat([mean_vals, std_vals, median_vals], dim=2)
        return tokens.to(device=original_device, dtype=torch.float32)