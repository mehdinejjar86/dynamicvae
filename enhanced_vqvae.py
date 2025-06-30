import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights
import math

# --- Reusing components from your original code ---

class ResidualBlock(nn.Module):
    """
    A standard Residual Block for convolutional networks.
    Adds the input to the output of a sequential block of layers.
    """
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
    """
    Self-Attention Block for convolutional features.
    Calculates attention weights and applies them to the feature map.
    Based on Non-local Neural Networks.
    """
    def __init__(self, in_channels):
        super().__init__()
        # Query, Key, Value convolutions for attention mechanism
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        
        # Learnable parameter for scaling the attention output
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, H, W = x.size()
       
        # Reshape query, key, value for matrix multiplication
        # Query: (B, C', H*W) -> (B, H*W, C')
        proj_query = self.query(x).view(batch_size, -1, W*H).permute(0, 2, 1)
        # Key: (B, C', H*W)
        proj_key = self.key(x).view(batch_size, -1, W*H)
        
        # Calculate energy (dot product attention)
        energy = torch.bmm(proj_query, proj_key) # (B, H*W, H*W)
        
        # Apply softmax to get attention weights
        attention = F.softmax(energy, dim=-1) # (B, H*W, H*W)
        
        # Reshape value
        proj_value = self.value(x).view(batch_size, -1, W*H) # (B, C, H*W)

        # Apply attention to value
        out = torch.bmm(proj_value, attention.permute(0, 2, 1)) # (B, C, H*W)
        
        # Reshape back to original spatial dimensions
        out = out.view(batch_size, C, H, W) 

        # Scale and add residual connection
        out = self.gamma * out + x
        return out

class VectorQuantizer(nn.Module):
    """
    Vector Quantizer layer for VQ-VAE.
    Maps continuous encoder outputs to discrete codebook vectors.
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.num_embeddings = num_embeddings # Number of vectors in the codebook
        self.embedding_dim = embedding_dim   # Dimension of each codebook vector
        self.commitment_cost = commitment_cost # Weight for the commitment loss

        # Learnable codebook embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # Initialize embeddings uniformly
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, inputs):
        # inputs shape: (B, C, H, W) where C is embedding_dim
        input_shape = inputs.shape
        
        # Flatten input for distance calculation: (B*H*W, embedding_dim)
        # Permute to (B, H, W, C) then flatten to (B*H*W, C)
        flat_input = inputs.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)

        # Calculate L2 distances between flattened input vectors and all codebook embeddings
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2x^T y
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True) # ||x||^2
            + torch.sum(self.embedding.weight ** 2, dim=1) # ||y||^2 for all y in codebook
            - 2 * torch.matmul(flat_input, self.embedding.weight.t()) # -2x^T y
        )

        # Find the closest codebook embedding for each input vector
        encoding_indices = torch.argmin(distances, dim=1)

        # Convert indices to one-hot encodings: (B*H*W, num_embeddings)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()

        # Quantize the input: map each input vector to its closest codebook embedding
        # This is a matrix multiplication: (B*H*W, num_embeddings) x (num_embeddings, embedding_dim)
        quantized = torch.matmul(encodings, self.embedding.weight)
        
        # Reshape quantized output back to original spatial dimensions: (B, H, W, C)
        quantized = quantized.view(input_shape[0], input_shape[2], input_shape[3], -1)
        # Permute back to (B, C, H, W)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        # Calculate VQ losses:
        # 1. Embedding loss (e_latent_loss): ensures the encoder output moves towards the codebook vectors
        #    (quantized.detach() means gradients don't flow through the codebook)
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        
        # 2. Commitment loss (q_latent_loss): ensures the codebook vectors move towards the encoder output
        #    (inputs.detach() means gradients don't flow through the encoder)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        
        # Total VQ loss
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-Through Estimator:
        # During backward pass, gradients for 'quantized' are passed directly to 'inputs'
        # This allows gradients to flow through the discrete quantization step.
        quantized = inputs + (quantized - inputs).detach()

        # Return quantized output, VQ loss, and the encoding indices (reshaped to spatial)
        return quantized, loss, encoding_indices.view(input_shape[0], input_shape[2], input_shape[3])


class DynamicVQVAE(nn.Module):
    """
    An enhanced VQ-VAE model incorporating Residual Blocks, Attention Blocks,
    and Perceptual Loss (using VGG16) for improved image generation.
    """
    def __init__(
        self,
        input_channels=1,
        input_height=128,
        input_width=128,
        encoder_channels=[32, 64, 128, 256],
        decoder_channels=[256, 128, 64, 32],
        num_embeddings=512,      # Number of embeddings in the VQ codebook
        embedding_dim=256,       # Dimension of each codebook vector (latent space dimension for VQ)
        commitment_cost=0.25,    # Weight for the commitment loss in VQ
        use_attention=True,      # Whether to use Attention Blocks
        use_residual=True        # Whether to use Residual Blocks
    ):
        super().__init__()
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels

        # Build Encoder: Outputs features for VQ layer and skip connections
        self.encoder_blocks, self.skip_connection_dims, self.final_dim, encoder_out_channels = \
            self._build_encoder_enhanced(encoder_channels)

        # Projection layer to match encoder output channels to embedding_dim
        # This is crucial if the last encoder channel count doesn't equal embedding_dim
        if encoder_out_channels != self.embedding_dim:
            self.proj_to_embedding = nn.Conv2d(encoder_out_channels, self.embedding_dim, kernel_size=1)
        else:
            self.proj_to_embedding = nn.Identity() # No-op if dimensions already match

        # Vector Quantizer layer
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

        # Build Decoder: Takes quantized output and reconstructs the image
        self.decoder_blocks, self.final_conv = self._build_decoder_enhanced(decoder_channels)

        # Perceptual network (VGG16 features)
        self.perceptual_net = self._build_perceptual_net()
        # Freeze VGG weights as it's used as a fixed feature extractor
        for param in self.perceptual_net.parameters():
            param.requires_grad = False
        self.perceptual_net.eval()

    def _build_encoder_enhanced(self, encoder_channels):
        """
        Constructs the encoder network.
        Includes convolutional layers with downsampling, BatchNorm, and LeakyReLU.
        Optionally adds Residual and Attention blocks.
        Collects spatial dimensions at each stage for skip connections.
        """
        encoder_layers = nn.ModuleList()
        skip_connection_dims = [] # Stores (H, W) dimensions for skip connections
        current_h, current_w = self.input_height, self.input_width
        in_channels = self.input_channels

        # Store initial input dimensions for the first skip connection (original image size)
        skip_connection_dims.append((current_h, current_w))

        # Build convolutional blocks for downsampling
        for out_channels in encoder_channels:
            block_layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            if self.use_residual:
                block_layers.append(ResidualBlock(out_channels))
            if self.use_attention:
                block_layers.append(AttentionBlock(out_channels))
            
            encoder_layers.append(nn.Sequential(*block_layers))

            # Calculate new spatial dimensions after convolution with stride 2
            current_h = math.floor((current_h + 2*1 - 3) / 2) + 1
            current_w = math.floor((current_w + 2*1 - 3) / 2) + 1
            
            # Store dimensions for skip connections
            skip_connection_dims.append((current_h, current_w))
            in_channels = out_channels # Update input channels for the next block

        final_dim = (current_h, current_w) # Final spatial dimensions of the latent representation
        return encoder_layers, skip_connection_dims, final_dim, in_channels # Return last output channels

    def _build_decoder_enhanced(self, decoder_channels):
        """
        Constructs the decoder network.
        Starts with the quantized latent representation and upsamples to reconstruct the image.
        Includes ConvTranspose2d for upsampling, BatchNorm, LeakyReLU, and optional Residual/Attention blocks.
        Handles skip connections from the encoder.
        """
        decoder_blocks = nn.ModuleList()

        # The first block processes the quantized latent (z_q) directly.
        # z_q has self.embedding_dim channels. This block maps it to decoder_channels[0].
        initial_decoder_channels = decoder_channels[0]
        
        first_block_layers = [
            nn.Conv2d(self.embedding_dim, initial_decoder_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(initial_decoder_channels),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        if self.use_residual:
            first_block_layers.append(ResidualBlock(initial_decoder_channels))
        if self.use_attention:
            first_block_layers.append(AttentionBlock(initial_decoder_channels))
        decoder_blocks.append(nn.Sequential(*first_block_layers))

        in_channels = initial_decoder_channels

        # Build blocks for each upsampling step
        # Loop through the target output channels for each block (starting from the second element in decoder_channels)
        for i, out_channels in enumerate(decoder_channels[1:]):
            # Determine the number of channels coming from the corresponding encoder skip connection.
            # `self.encoder_channels` lists the output channels of each encoder block.
            # `features[::-1]` from encoder gives `[deepest, next_deepest, ..., shallowest]`.
            # `features[i+1]` in the decode loop corresponds to `encoder_channels[-(i+2)]`.
            skip_channels = self.encoder_channels[-(i + 2)] 

            # Input channels for this decoder block: current channels + skip connection channels
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
            in_channels = out_channels # Update input channels for the next block

        # Final convolution layer to map to the desired output channels (e.g., 1 for grayscale, 3 for RGB)
        final_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.input_channels, kernel_size=3, padding=1),
            nn.Sigmoid() # Sigmoid for output normalized to [0, 1]
        )

        return decoder_blocks, final_conv

    def _build_perceptual_net(self):
        """
        Builds a perceptual network using a pre-trained VGG16 model.
        Used to calculate perceptual loss by comparing feature maps.
        """
        # Use VGG16 up to a certain layer (e.g., relu3_3, which is index 16 in .features)
        weights = VGG16_Weights.IMAGENET1K_V1
        vgg = vgg16(weights=weights).features[:16].eval() 

        # Adapt the first layer if the input is grayscale (VGG expects 3 channels)
        if self.input_channels == 1:
            existing_layer = vgg[0] # Get the first convolutional layer of VGG
            # Create a new Conv2d layer with 1 input channel
            new_layer = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
            # Average the existing 3-channel weights across the input channel dimension
            new_layer.weight.data = existing_layer.weight.data.sum(dim=1, keepdim=True)
            new_layer.bias.data = existing_layer.bias.data # Copy bias
            vgg[0] = new_layer # Replace the first conv layer in VGG
        elif self.input_channels != 3:
             print(f"Warning: Perceptual network (VGG16) expects 3 input channels, but got {self.input_channels}. Using first conv layer as is.")

        return vgg

    def forward(self, x):
        """
        Forward pass through the Enhanced VQ-VAE.
        Encodes input, quantizes the latent representation, and decodes to reconstruct.
        """
        # Encoder pass: Collect features for skip connections
        features = []
        current_x = x
        for block in self.encoder_blocks:
            current_x = block(current_x)
            features.append(current_x) # Store output of each encoder block

        # Project encoder output to the embedding dimension
        z_e = self.proj_to_embedding(current_x)

        # VQ Layer: Quantize the encoder output
        z_q, vq_loss, encoding_indices = self.vq_layer(z_e)

        # Decoder pass: Reconstruct the image from the quantized latent
        # The `z_q` is already spatial and has `embedding_dim` channels.
        # The first decoder block (decoder_blocks[0]) is designed to take this directly.
        decoded_x = self.decoder_blocks[0](z_q)

        # Reverse encoder features for proper skip connection order in the decoder
        # `features[::-1]` makes `features[0]` the deepest feature, `features[1]` the next deepest, etc.
        encoder_features_reversed = features[::-1]

        # Loop through the remaining decoder stages (upsampling steps)
        num_upsample_stages = len(self.decoder_channels) - 1
        for i in range(num_upsample_stages):
            # Determine the target spatial size for this upsampling step.
            # `skip_connection_dims` contains sizes from original input to deepest encoder output.
            # `target_size_idx` calculates the size before the corresponding encoder downsampling.
            target_size_idx = len(self.skip_connection_dims) - 2 - i
            target_size = self.skip_connection_dims[target_size_idx]

            # 1. Upsample the current decoded feature map
            decoded_x = F.interpolate(decoded_x, size=target_size, mode='bilinear', align_corners=False)

            # 2. Get the corresponding skip connection feature from the encoder
            feature_idx = i + 1 # `encoder_features_reversed[0]` is deepest, `[1]` is next
            if feature_idx < len(encoder_features_reversed):
                skip_feature = encoder_features_reversed[feature_idx]

                # Ensure skip feature dimensions match target size (important for concatenation)
                if skip_feature.shape[2:] != target_size:
                     skip_feature = F.interpolate(skip_feature, size=target_size, mode='bilinear', align_corners=False)

                # 3. Concatenate skip feature along the channel dimension
                decoded_x = torch.cat([decoded_x, skip_feature], dim=1)
            else:
                print(f"Warning: Missing skip feature for decoder stage {i} (expected index {feature_idx})")

            # 4. Pass through the corresponding decoder block
            decoder_block = self.decoder_blocks[i + 1] # `decoder_blocks[0]` was the initial block
            decoded_x = decoder_block(decoded_x)

        # Final upsampling to match the original input image size if needed
        original_size = self.skip_connection_dims[0] # This is (input_height, input_width)
        if decoded_x.shape[2:] != original_size:
            decoded_x = F.interpolate(decoded_x, size=original_size, mode='bilinear', align_corners=False)

        # Apply the final convolution layer to get the reconstructed image
        recon_x = self.final_conv(decoded_x)

        # Return reconstructed image, VQ loss, encoder output (pre-quantization), and encoding indices
        return recon_x, vq_loss, z_e, encoding_indices

    def loss_function(self, recon_x, x, vq_loss, perceptual_weight=0.1):
        """
        Calculates the total loss for the Enhanced VQ-VAE.
        Includes Binary Cross-Entropy (or MSE), Perceptual Loss, and the VQ loss.
        Note: KLD (Kullback-Leibler Divergence) is typically not used in VQ-VAE.
        """
        # Reconstruction loss (e.g., Binary Cross-Entropy for images normalized to [0,1])
        bce_loss = F.binary_cross_entropy(recon_x, x, reduction='mean')
        # Alternatively, for continuous pixel values, F.mse_loss(recon_x, x, reduction='mean') could be used.

        # Perceptual loss calculation
        perceptual_loss_calculated = False
        perceptual_loss = torch.tensor(0.0, device=x.device) # Initialize to 0.0

        # Prepare images for VGG (perceptual network) based on input channels
        if self.input_channels == 1: # Grayscale input
            if x.shape[1] == 1:
                x_vgg = x
                recon_x_vgg = recon_x
                perceptual_loss_calculated = True
            else:
                print(f"Warning: VAE configured for 1 channel, but loss_function received input with {x.shape[1]} channels. Skipping perceptual loss.")
        elif self.input_channels == 3: # RGB input
            if x.shape[1] == 3:
                x_vgg = x
                recon_x_vgg = recon_x
                perceptual_loss_calculated = True
            elif x.shape[1] == 1:
                # Replicate 1-channel input to 3 channels for VGG if VAE is configured for 3 channels
                x_vgg = x.repeat(1, 3, 1, 1)
                recon_x_vgg = recon_x.repeat(1, 3, 1, 1)
                perceptual_loss_calculated = True
            else:
                print(f"Warning: VAE configured for 3 channels, but loss_function received input with {x.shape[1]} channels. Skipping perceptual loss.")
        else:
             print(f"Warning: VAE configured for {self.input_channels} channels. Skipping perceptual loss.")

        if perceptual_loss_calculated:
            try:
                 with torch.no_grad(): # Do not calculate gradients for VGG features
                     real_features = self.perceptual_net(x_vgg)
                 recon_features = self.perceptual_net(recon_x_vgg)
                 perceptual_loss = F.mse_loss(recon_features, real_features, reduction='mean')
            except Exception as e:
                 print(f"Error during perceptual network forward pass: {e}. Setting perceptual loss to 0.")
                 perceptual_loss = torch.tensor(0.0, device=x.device)
                 # Ensure loss is finite in case of rare errors
                 if not torch.isfinite(perceptual_loss):
                     perceptual_loss = torch.tensor(0.0, device=x.device)

        # Total loss is a weighted sum of reconstruction, perceptual, and VQ losses
        total_loss = bce_loss + perceptual_weight * perceptual_loss + vq_loss

        # Check for NaN/Inf in total loss (useful for debugging training stability)
        if not torch.isfinite(total_loss):
            print("Warning: Loss became NaN or Inf.")
            print(f"  BCE Loss: {bce_loss.item()}")
            print(f"  Perceptual Loss: {perceptual_loss.item() if isinstance(perceptual_loss, torch.Tensor) else perceptual_loss}")
            print(f"  VQ Loss: {vq_loss.item()}")

        return total_loss


if __name__ == "__main__":
    # Example usage of the EnhancedVQVAE
    input_h, input_w = 128, 130 # Example non-square input dimensions
    
    # Initialize the EnhancedVQVAE model
    enhanced_vqvae = DynamicVQVAE(
        input_height=input_h,
        input_width=input_w,
        input_channels=1, # Grayscale image
        encoder_channels=[32, 64, 128, 256], # Channels for encoder blocks
        decoder_channels=[256, 128, 64, 32], # Channels for decoder blocks
        num_embeddings=512,      # Number of codebook vectors
        embedding_dim=256,       # Dimension of each codebook vector
        commitment_cost=0.25,    # Commitment cost for VQ loss
        use_attention=True,      # Enable Attention Blocks
        use_residual=True        # Enable Residual Blocks
    )

    # Create a sample input tensor (batch_size, channels, height, width)
    sample_input = torch.rand(4, 1, input_h, input_w) # Batch size 4

    # Perform a forward pass
    recon, vq_loss, z_e, encoding_indices = enhanced_vqvae(sample_input)

    # Calculate the total loss
    # Note: loss_function now takes vq_loss directly from the forward pass
    loss = enhanced_vqvae.loss_function(recon, sample_input, vq_loss, perceptual_weight=0.1)

    # Print shapes and loss values for verification
    print("Input shape:", sample_input.shape)
    print("Reconstructed shape:", recon.shape)
    print("Encoder (pre-VQ) output shape:", z_e.shape)
    print("Encoding indices shape:", encoding_indices.shape)
    print("VQ loss:", vq_loss.item())
    print("Calculated Total Loss:", loss.item())

    # Assert that the output shape matches the input shape
    assert recon.shape == sample_input.shape, "Output shape mismatch!"

    print("\nModel Summary:")
    print(enhanced_vqvae)
    
    # Calculate and print total trainable parameters
    num_params = sum(p.numel() for p in enhanced_vqvae.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {num_params:,}")

