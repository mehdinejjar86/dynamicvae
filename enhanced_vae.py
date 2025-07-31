import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights
import math

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, H, W = x.size()
       
        proj_query = self.query(x).reshape(batch_size, -1, W*H).permute(0, 2, 1) # B x N x C'
        proj_key = self.key(x).reshape(batch_size, -1, W*H) # B x C' x N
        energy = torch.bmm(proj_query, proj_key) # B x N x N
        attention = F.softmax(energy, dim=-1) # B x N x N
        proj_value = self.value(x).reshape(batch_size, -1, W*H) # B x C x N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1)) # B x C x N
        out = out.reshape(batch_size, C, H, W) 

        out = self.gamma * out + x
        return out


class DynamicVAE(nn.Module):
    def __init__(
        self,
        input_channels=1,
        input_height=128,
        input_width=128,
        encoder_channels=[32, 64, 128, 256],
        decoder_channels=[256, 128, 64, 32],
        latent_dim=512,
        use_attention=True,
        use_residual=True
    ):
        super().__init__()
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.latent_dim = latent_dim
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels

        # Build Encoder
        self.encoder_blocks, self.skip_connection_dims, self.final_dim = self._build_encoder()

        # Latent space mapping
        # Calculate the size after the last encoder block
        linear_input_size = encoder_channels[-1] * self.final_dim[0] * self.final_dim[1]
        self.fc_mu = nn.Linear(linear_input_size, latent_dim)
        self.fc_var = nn.Linear(linear_input_size, latent_dim)

        # Build Decoder parts
        self.decoder_input = nn.Linear(
            latent_dim,
            decoder_channels[0] * self.final_dim[0] * self.final_dim[1]
        )
        # Decoder blocks
        self.decoder_blocks, self.final_conv = self._build_decoder()


        # Perceptual network
        self.perceptual_net = self._build_perceptual_net()
        for param in self.perceptual_net.parameters():
            param.requires_grad = False # Freeze VGG weights
        self.perceptual_net.eval()

    def _build_encoder(self):
        encoder_layers = nn.ModuleList()
        skip_connection_dims = []
        current_h, current_w = self.input_height, self.input_width
        in_channels = self.input_channels

        # Store initial dimensions (sometimes needed for final upsample in decoder)
        skip_connection_dims.append((current_h, current_w))

        for out_channels in self.encoder_channels:
            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True) # Use inplace=True for efficiency
            )
            encoder_layers.append(block)

            # Conv output size: floor((Input + 2*Padding - Kernel) / Stride) + 1
            current_h = math.floor((current_h + 2*1 - 3) / 2) + 1
            current_w = math.floor((current_w + 2*1 - 3) / 2) + 1
            skip_connection_dims.append((current_h, current_w))
            in_channels = out_channels

        final_dim = (current_h, current_w)
        return encoder_layers, skip_connection_dims, final_dim

    def _build_decoder(self):
        decoder_blocks = nn.ModuleList()

        in_channels = self.decoder_channels[0]

        decoder_blocks.append(nn.Sequential(
             nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
             nn.BatchNorm2d(in_channels), # Added BN here
             nn.LeakyReLU(0.2, inplace=True)
        ))

        # Build blocks for each upsampling step
        # Loop through the target output channels for each block
        for i, out_channels in enumerate(self.decoder_channels[1:]):
            # Determine the number of channels coming from the skip connection
            # Stage 'i' in the decoder build loop corresponds to concatenating features[i+1] in the decode forward pass.
            # features[i+1] comes from encoder block len(encoder_channels) - 2 - i
            skip_encoder_channel_idx = len(self.encoder_channels) - 2 - i 

            # Add boundary check for safety
            if skip_encoder_channel_idx < 0:
                 raise IndexError(f"Calculated invalid skip channel index {skip_encoder_channel_idx} for decoder build stage {i}")

            skip_channels = self.encoder_channels[skip_encoder_channel_idx] 

            # Input channels for this block's main convolution
            conv_in_channels = in_channels + skip_channels

            block_layers = [
                nn.Conv2d(conv_in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            ]

            if self.use_residual:
                block_layers.append(ResidualBlock(out_channels))
            if self.use_attention:
                block_layers.append(AttentionBlock(out_channels))

            decoder_blocks.append(nn.Sequential(*block_layers))
            in_channels = out_channels 

        
        final_conv = nn.Sequential(
            # Input channels are the output channels of the last decoder block
            nn.Conv2d(in_channels, self.input_channels, kernel_size=3, padding=1),
            nn.Sigmoid() 
            # If your input isn't normalized to [0,1], use Tanh or linear activation
        )

        return decoder_blocks, final_conv


    def _build_perceptual_net(self):
        # Use VGG16 up to a certain layer for perceptual features
        weights = VGG16_Weights.IMAGENET1K_V1
        vgg = vgg16(weights=weights).features[:16].eval() # Use layers up to relu3_3

        # Adapt first layer if input is grayscale
        if self.input_channels == 1:
            # Get existing weights
            existing_layer = vgg[0]
            new_layer = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
            # Average weights across the input channel dimension
            new_layer.weight.data = existing_layer.weight.data.sum(dim=1, keepdim=True)
            # Copy bias
            new_layer.bias.data = existing_layer.bias.data
            vgg[0] = new_layer # Replace the first conv layer
        elif self.input_channels != 3:
             print(f"Warning: Perceptual network (VGG16) expects 3 input channels, but got {self.input_channels}. Using first conv layer as is.")

        return vgg

    def encode(self, x):
        # Pass input through encoder blocks, collecting outputs for skip connections
        features = []
        for i, block in enumerate(self.encoder_blocks):
            x = block(x)
            features.append(x) # Store output of each block

        # Flatten the output of the last encoder block
        x_flat = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x_flat)
        logvar = self.fc_var(x_flat)

        # Return features ordered from deepest (smallest spatial dim) to shallowest
        return mu, logvar, features[::-1]

    def decode(self, z, features):
        batch_size = z.size(0)
        # Project and reshape latent vector
        x = self.decoder_input(z)
        # Reshape to match dimensions expected by the first decoder block (after initial conv)
        x = x.reshape(batch_size, self.decoder_channels[0], self.final_dim[0], self.final_dim[1])

        # Apply the initial decoder block (before first upsample)
        x = self.decoder_blocks[0](x)

        # Loop through the main decoder stages (corresponding to upsampling steps)
        # Number of upsampling stages = len(self.decoder_blocks) - 1
        num_upsample_stages = len(self.decoder_blocks) - 1

        for i in range(num_upsample_stages):
            # Determine the target spatial size for this upsampling step
            # This is the size stored *before* the corresponding encoder block's downsampling conv
            # skip_connection_dims = [(128,130), (64,65), (32,33), (16,17), (8,9)] (len=5)
            # Target size index: len - 2 - i
            target_size_idx = len(self.skip_connection_dims) - 2 - i
            target_size = self.skip_connection_dims[target_size_idx]

            # 1. Upsample x
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)

            # 2. Get the corresponding skip connection feature
            # features = [Feat(8,9,C=256), Feat(16,17,C=128), Feat(32,33,C=64), Feat(64,65,C=32)] (len=4)
            # Skip feature index: i + 1 (features[0] is the deepest, not used for concat)
            feature_idx = i + 1
            if feature_idx < len(features):
                skip_feature = features[feature_idx]

                # Ensure skip feature spatial dimensions match target size (should ideally)
                if skip_feature.shape[2:] != target_size:
                     # This might happen with slightly different calculation methods for output size
                     print(f"Warning: Interpolating skip connection {feature_idx} from {skip_feature.shape[2:]} to {target_size}")
                     skip_feature = F.interpolate(skip_feature, size=target_size, mode='bilinear', align_corners=False)

                # 3. Concatenate along channel dimension
                # print(f"Decode stage {i}: Concat x {x.shape} with skip {skip_feature.shape}")
                x = torch.cat([x, skip_feature], dim=1)
            else:
                 # This case should ideally not happen if encoder/decoder lists match
                 print(f"Warning: Missing skip feature for decoder stage {i} (expected index {feature_idx})")


            # 4. Pass through the corresponding decoder block
            # Block index: i + 1 (since decoder_blocks[0] was the initial block)
            decoder_block = self.decoder_blocks[i + 1]
            x = decoder_block(x)

        # Final upsampling to original input size (if necessary)
        # Check if the last block's output size matches the original input size
        original_size = self.skip_connection_dims[0]
        if x.shape[2:] != original_size:
             x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=False)

        # Apply the final convolution layer
        x = self.final_conv(x)

        return x


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar, features = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, features)
        return recon, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, perceptual_weight=0.1, kld_weight=0.5): 

        bce_loss = F.binary_cross_entropy(recon_x, x, reduction='mean')
        # Or mse_loss = F.mse_loss(recon_x, x, reduction='mean')

        perceptual_loss_calculated = False 
        perceptual_loss = torch.tensor(0.0, device=x.device) # Default value

        if self.input_channels == 1:
            if x.shape[1] == 1:
                x_vgg = x             
                recon_x_vgg = recon_x 
                perceptual_loss_calculated = True
            else:
                print(f"Warning: VAE configured for 1 channel, but loss_function received input with {x.shape[1]} channels. Skipping perceptual loss.")

        elif self.input_channels == 3:
            if x.shape[1] == 3:
                x_vgg = x             
                recon_x_vgg = recon_x 
                perceptual_loss_calculated = True
            elif x.shape[1] == 1:
                print("Info: Replicating 1-channel input to 3 channels for 3-channel perceptual net.")
                x_vgg = x.repeat(1, 3, 1, 1)
                recon_x_vgg = recon_x.repeat(1, 3, 1, 1)
                perceptual_loss_calculated = True
            else:
                print(f"Warning: VAE configured for 3 channels, but loss_function received input with {x.shape[1]} channels. Skipping perceptual loss.")
        else:
             print(f"Warning: VAE configured for {self.input_channels} channels. Skipping perceptual loss.")

        if perceptual_loss_calculated:
            try:
                 with torch.no_grad(): 
                     real_features = self.perceptual_net(x_vgg)

                 recon_features = self.perceptual_net(recon_x_vgg)
                 perceptual_loss = F.mse_loss(recon_features, real_features, reduction='mean')
            except Exception as e:
                 print(f"Error during perceptual network forward pass: {e}. Setting perceptual loss to 0.")
                 perceptual_loss = torch.tensor(0.0, device=x.device)
                 # Ensure weight multiplication doesn't cause issues if loss becomes NaN/Inf later
                 if not torch.isfinite(perceptual_loss):
                     perceptual_loss = torch.tensor(0.0, device=x.device)

        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        total_loss = bce_loss + perceptual_weight * perceptual_loss + kld_weight * kld_loss

        if not torch.isfinite(total_loss):
            print("Warning: Loss became NaN or Inf.")
            print(f"  BCE Loss: {bce_loss.item()}")
            print(f"  Perceptual Loss: {perceptual_loss.item() if isinstance(perceptual_loss, torch.Tensor) else perceptual_loss}")
            print(f"  KLD Loss: {kld_loss.item()}")

        return total_loss


if __name__ == "__main__":
    input_h, input_w = 128, 130 
    vae = DynamicVAE(
        input_height=input_h,
        input_width=input_w,
        input_channels=1, # Example: Grayscale
        encoder_channels=[32, 64, 128, 256],
        decoder_channels=[256, 128, 64, 32],
        latent_dim=512,
        use_attention=True,
        use_residual=True
    )

    sample_input = torch.rand(4, 1, input_h, input_w) # Batch size 4

    recon, mu, logvar = vae(sample_input)

    loss = vae.loss_function(recon, sample_input, mu, logvar)

    print("Input shape:", sample_input.shape)
    print("Reconstructed shape:", recon.shape)
    print("Mu shape:", mu.shape)
    print("Logvar shape:", logvar.shape)
    print("Calculated Loss:", loss.item())

    assert recon.shape == sample_input.shape, "Output shape mismatch!"

    print("\nModel Summary:")
    print(vae)
    num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {num_params:,}")