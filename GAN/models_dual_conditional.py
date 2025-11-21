import torch
import torch.nn as nn

class DualConditionalGenerator(nn.Module):
    """Generator with two conditional inputs: tank type and view angle"""
    def __init__(self, latent_dim=100, num_tanks=10, num_views=4, embed_dim=50):
        super().__init__()
        self.latent_dim = latent_dim

        # Embeddings for tank type and view
        self.tank_embedding = nn.Embedding(num_tanks, embed_dim)
        self.view_embedding = nn.Embedding(num_views, embed_dim)

        # Combined input size: noise + tank_embed + view_embed
        input_dim = latent_dim + embed_dim * 2

        # Generator network (for 200x200 output)
        self.model = nn.Sequential(
            # Input: (latent_dim + embed_dim*2) x 1 x 1
            nn.ConvTranspose2d(input_dim, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            # State: 1024 x 4 x 4

            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # State: 512 x 8 x 8

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # State: 256 x 16 x 16

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # State: 128 x 32 x 32

            nn.ConvTranspose2d(128, 64, 4, 2, 8, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # State: 64 x 50 x 50

            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # State: 32 x 100 x 100

            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output: 3 x 200 x 200
        )

    def forward(self, noise, tank_labels, view_labels):
        """
        Args:
            noise: (batch, latent_dim) random noise
            tank_labels: (batch,) tank type indices
            view_labels: (batch,) view angle indices
        """
        # Get embeddings
        tank_embed = self.tank_embedding(tank_labels)  # (batch, embed_dim)
        view_embed = self.view_embedding(view_labels)  # (batch, embed_dim)

        # Concatenate noise with both embeddings
        gen_input = torch.cat([noise, tank_embed, view_embed], dim=1)

        # Reshape for conv layers: (batch, channels, 1, 1)
        gen_input = gen_input.unsqueeze(-1).unsqueeze(-1)

        return self.model(gen_input)


class DualConditionalDiscriminator(nn.Module):
    """Discriminator with two conditional inputs: tank type and view angle"""
    def __init__(self, num_tanks=10, num_views=4, embed_dim=50):
        super().__init__()

        # Embeddings for tank type and view
        self.tank_embedding = nn.Embedding(num_tanks, embed_dim)
        self.view_embedding = nn.Embedding(num_views, embed_dim)

        # Image encoder (for 200x200 input)
        self.image_encoder = nn.Sequential(
            # Input: 3 x 200 x 200
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 32 x 100 x 100

            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 64 x 50 x 50

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 128 x 25 x 25

            nn.Conv2d(128, 256, 5, 2, 2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 256 x 13 x 13

            nn.Conv2d(256, 512, 5, 2, 2, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 512 x 7 x 7

            nn.Conv2d(512, 1024, 5, 2, 2, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 1024 x 4 x 4
        )

        # Flatten to 1024 * 4 * 4 = 16384
        img_features = 1024 * 4 * 4

        # Classifier combines image features with embeddings
        self.classifier = nn.Sequential(
            nn.Linear(img_features + embed_dim * 2, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, images, tank_labels, view_labels):
        """
        Args:
            images: (batch, 3, 200, 200) input images
            tank_labels: (batch,) tank type indices
            view_labels: (batch,) view angle indices
        """
        # Encode image
        img_features = self.image_encoder(images)
        img_features = img_features.view(img_features.size(0), -1)

        # Get embeddings
        tank_embed = self.tank_embedding(tank_labels)
        view_embed = self.view_embedding(view_labels)

        # Concatenate all features
        combined = torch.cat([img_features, tank_embed, view_embed], dim=1)

        return self.classifier(combined)


def weights_init(m):
    """Initialize network weights"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
