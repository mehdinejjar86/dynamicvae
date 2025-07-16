import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, inputs):
        input_shape = inputs.shape  # (B, C, H, W)
        flat_input = inputs.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)

        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_input, self.embedding.weight.t())
        )

        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()

        quantized = torch.matmul(encodings, self.embedding.weight)
        quantized = quantized.view(input_shape[0], input_shape[2], input_shape[3], -1)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        quantized = inputs + (quantized - inputs).detach()

        return quantized, loss, encoding_indices.view(input_shape[0], input_shape[2], input_shape[3])

class DynamicVQVAE(nn.Module):
    def __init__(
        self,
        input_channels=1,
        input_height=28,
        input_width=28,
        encoder_channels=[32, 64, 128, 256, 512],
        decoder_channels=[512, 256, 128, 64, 32],
        num_embeddings=512,
        commitment_cost=0.25,
        embedding_dim=256
    ):
        super().__init__()
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.encoder_channels = [input_channels] + encoder_channels
        self.decoder_channels = decoder_channels + [input_channels]
        self.embedding_dim = embedding_dim

        self.encoder, self.final_dim, encoder_out_channels = self._build_encoder()

        if encoder_out_channels != self.embedding_dim:
            self.encoder.add_module("proj_to_embedding", nn.Conv2d(encoder_out_channels, self.embedding_dim, kernel_size=1))

        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = self._build_decoder(self.final_dim)

    def _build_encoder(self):
        layers = []
        current_h, current_w = self.input_height, self.input_width
        for i in range(len(self.encoder_channels) - 1):
            layers.append(nn.Conv2d(self.encoder_channels[i], self.encoder_channels[i + 1], kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(self.encoder_channels[i + 1]))
            layers.append(nn.LeakyReLU(0.2))
            current_h = (current_h + 1) // 2
            current_w = (current_w + 1) // 2
        return nn.Sequential(*layers), (current_h, current_w), self.encoder_channels[-1]

    def _build_decoder(self, final_dim):
        layers = []
        current_h, current_w = final_dim
        if self.decoder_channels[0] != self.embedding_dim:
            layers.append(nn.Conv2d(self.embedding_dim, self.decoder_channels[0], kernel_size=1))
        for i in range(len(self.decoder_channels) - 1):
            layers.append(nn.ConvTranspose2d(
                self.decoder_channels[i],
                self.decoder_channels[i + 1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ))
            if i < len(self.decoder_channels) - 2:
                layers.append(nn.BatchNorm2d(self.decoder_channels[i + 1]))
                layers.append(nn.LeakyReLU(0.2))
            else:
                layers.append(nn.Sigmoid())
            current_h *= 2
            current_w *= 2
        if current_h != self.input_height or current_w != self.input_width:
            pad_h = self.input_height - current_h
            pad_w = self.input_width - current_w
            layers.append(nn.ConstantPad2d((0, pad_w, 0, pad_h), 0))
        return nn.Sequential(*layers)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, encoding_indices = self.vq_layer(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss, z_e, encoding_indices

    def loss_function(self, recon_x, x, vq_loss):
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        return recon_loss + vq_loss

# Example usage:
if __name__ == "__main__":
    # Create a DynamicVQVAE instance for a non-square image (e.g., 28x30)
    model = DynamicVQVAE(
        input_channels=1,
        input_height=28,
        input_width=30,
        encoder_channels=[32, 64, 128],
        decoder_channels=[128, 64, 32],
        num_embeddings=512,
        commitment_cost=0.25
    )
    
    sample_input = torch.randn(1, 1, 28, 30)
    recon, vq_loss, z_e, encoding_indices = model(sample_input)
    
    print("Input shape:", sample_input.shape)
    print("Reconstructed shape:", recon.shape)
    print("Encoder (latent) output shape:", z_e.shape)
    print("Encoding indices shape:", encoding_indices.shape)
    print("VQ loss:", vq_loss.item())
