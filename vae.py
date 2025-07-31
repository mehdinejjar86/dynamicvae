import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicVAE(nn.Module):
    def __init__(
        self,
        input_channels=1,
        input_height=28,
        input_width=28,
        encoder_channels=[32, 64, 128],
        decoder_channels=[128, 64, 32],
        latent_dim=20
    ):
        super(DynamicVAE, self).__init__()

        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.encoder_channels = [input_channels] + encoder_channels
        self.decoder_channels = decoder_channels + [input_channels]
        self.latent_dim = latent_dim

        # Build Encoder with dimension tracking
        self.encoder, self.final_dim = self._build_encoder()
        self.decoder = self._build_decoder(self.final_dim)

        # Latent space mapping
        self.fc_mu = nn.Linear(
            self.encoder_channels[-1] * self.final_dim[0] * self.final_dim[1],
            latent_dim
        )
        self.fc_var = nn.Linear(
            self.encoder_channels[-1] * self.final_dim[0] * self.final_dim[1],
            latent_dim
        )
        self.decoder_input = nn.Linear(
            latent_dim,
            self.decoder_channels[0] * self.final_dim[0] * self.final_dim[1]
        )

    def _build_encoder(self):
        layers = []
        current_h = self.input_height
        current_w = self.input_width
        
        for i in range(len(self.encoder_channels)-1):
            layers.append(nn.Conv2d(
                self.encoder_channels[i],
                self.encoder_channels[i+1],
                kernel_size=3,
                stride=2,
                padding=1
            ))
            layers.append(nn.BatchNorm2d(self.encoder_channels[i+1]))
            layers.append(nn.LeakyReLU(0.2))
            
            # Update dimensions
            current_h = (current_h + 1) // 2
            current_w = (current_w + 1) // 2

        return nn.Sequential(*layers), (current_h, current_w)

    def _build_decoder(self, final_dim):
        layers = []
        current_h, current_w = final_dim
        
        for i in range(len(self.decoder_channels)-1):
            layers.append(nn.ConvTranspose2d(
                self.decoder_channels[i],
                self.decoder_channels[i+1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ))
            if i < len(self.decoder_channels)-2:
                layers.append(nn.BatchNorm2d(self.decoder_channels[i+1]))
                layers.append(nn.LeakyReLU(0.2))
            else:
                layers.append(nn.Sigmoid())
            
            # Update dimensions
            current_h = current_h * 2
            current_w = current_w * 2

        # Final adjustment layer if needed
        if current_h != self.input_height or current_w != self.input_width:
            pad_h = self.input_height - current_h
            pad_w = self.input_width - current_w
            layers.append(nn.ConstantPad2d((0, pad_w, 0, pad_h), 0))

        return nn.Sequential(*layers)

    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.reshape(-1, self.decoder_channels[0], *self.final_dim)
        x = self.decoder(x)
        return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD


if __name__ == "__main__":
    # Example with non-square image
    # 28x30 input example
    vae = DynamicVAE(
        input_height=28,
        input_width=30,
        encoder_channels=[32, 64, 128],
        decoder_channels=[128, 64, 32],
        latent_dim=20
    )

    sample_input = torch.randn(1, 1, 28, 30)
    recon, mu, logvar = vae(sample_input)
    
    print("Input shape:", sample_input.shape)
    print("Reconstructed shape:", recon.shape)
    print("Mu shape:", mu.shape)
    print("Logvar shape:", logvar.shape)
