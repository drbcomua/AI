"""
05. Conditional GAN (CGAN)
==========================

A Conditional GAN using class embedding overlays to guide image synthesis.

Architecture Diagram / Layout:
    Generator:
        Input z [Batch, Latent_Dim], Class Label [Batch]
            -> Label Embedding -> Concatenate with z [Batch, Latent_Dim + Embed_Dim]
            -> Project to [Batch, Latent_Dim + Embed_Dim, 1, 1]
            -> ConvTranspose2d (k=7, s=1, p=0) -> BatchNorm2d -> ReLU [Batch, 128, 7, 7]
            -> ConvTranspose2d (k=4, s=2, p=1) -> BatchNorm2d -> ReLU [Batch, 64, 14, 14]
            -> ConvTranspose2d (k=4, s=2, p=1) -> Tanh [Batch, 1, 28, 28]
    Discriminator:
        Input Image [Batch, 1, 28, 28], Class Label [Batch]
            -> Label Embedding -> Project to spatial shape [Batch, 1, 28, 28]
            -> Concatenate along channel dim [Batch, 2, 28, 28]
            -> Conv2d (k=4, s=2, p=1) -> LeakyReLU [Batch, 64, 14, 14]
            -> Conv2d (k=4, s=2, p=1) -> BatchNorm2d -> LeakyReLU [Batch, 128, 7, 7]
            -> Conv2d (k=7, s=1, p=0) -> Flat logit [Batch, 1]

Key insights / educational takeaways:
    * Conditioning allows directed generation, mapping latent coordinates to specific class manifolds.
    * Concat of projected spatial label representations provides the discriminator with class context.

Run:
    python "05.cgan.py" --epochs 5
    python "05.cgan.py" --limit 2000 --epochs 2
"""

import os
import torch
import torch.nn as nn
import gen_common as mc


class CGANGenerator(nn.Module):
    """Conditional Generator using label embeddings concatenated with latent noise."""
    def __init__(self, latent_dim: int = 64, num_classes: int = 10, embed_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.label_emb = nn.Embedding(num_classes, embed_dim)

        combined_dim = latent_dim + embed_dim

        self.net = nn.Sequential(
            nn.ConvTranspose2d(combined_dim, 128, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, labels):
        # z: [B, latent_dim], labels: [B]
        l_emb = self.label_emb(labels) # [B, embed_dim]
        # Concatenate along latent dim
        joined = torch.cat([z, l_emb], dim=1) # [B, combined_dim]
        joined = joined.unsqueeze(-1).unsqueeze(-1) # [B, combined_dim, 1, 1]
        return self.net(joined)


class CGANDiscriminator(nn.Module):
    """Conditional Discriminator concatenating spatial label maps with image channels."""
    def __init__(self, num_classes: int = 10, embed_dim: int = 16):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, embed_dim)
        # Project label embedding to a 28x28 spatial map channel
        self.label_projector = nn.Linear(embed_dim, 28 * 28)

        self.net = nn.Sequential(
            # Input shape: 2 channels (1 image channel + 1 projected label channel) x 28 x 28
            nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, kernel_size=7, stride=1, padding=0, bias=False)
        )

    def forward(self, x, labels):
        # x: [B, 1, 28, 28], labels: [B]
        l_emb = self.label_emb(labels) # [B, embed_dim]
        l_map = self.label_projector(l_emb).view(-1, 1, 28, 28) # [B, 1, 28, 28]

        # Concatenate along channel dimension
        joined = torch.cat([x, l_map], dim=1) # [B, 2, 28, 28]
        return self.net(joined).view(-1, 1)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def main():
    p = mc.build_argparser("Conditional GAN")
    args = p.parse_args()

    # Recommended learning rates for adversarial stabilization
    lr = 2e-4 if args.lr == 1e-3 else args.lr
    device = mc.get_device(args.device)

    print(f"Loading {args.dataset} dataset (limit={args.limit})...")
    train_loader, _, num_classes = mc.get_dataloaders(
        name=args.dataset, batch_size=args.batch_size, limit=args.limit
    )

    generator = CGANGenerator(latent_dim=args.latent_dim, num_classes=num_classes).to(device)
    discriminator = CGANDiscriminator(num_classes=num_classes).to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    n_params = sum(p.numel() for p in generator.parameters()) + sum(p.numel() for p in discriminator.parameters())
    print(f"Device: {device} | trainable params: {n_params:,}")
    print("-" * 64)

    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()
        loss_d_accum = 0.0
        loss_g_accum = 0.0
        total_batches = 0

        for real_imgs, labels in train_loader:
            b_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            labels = labels.to(device)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            opt_d.zero_grad()

            # Real loss
            real_labels = torch.ones(b_size, 1, device=device)
            out_real = discriminator(real_imgs, labels)
            loss_real = criterion(out_real, real_labels)

            # Fake loss
            z = torch.randn(b_size, args.latent_dim, device=device)
            fake_labels = torch.randint(0, num_classes, (b_size,), device=device)
            fake_imgs = generator(z, fake_labels)

            out_fake = discriminator(fake_imgs.detach(), fake_labels)
            fake_target = torch.zeros(b_size, 1, device=device)
            loss_fake = criterion(out_fake, fake_target)

            loss_d = loss_real + loss_fake
            loss_d.backward()
            opt_d.step()

            # ---------------------
            #  Train Generator
            # ---------------------
            opt_g.zero_grad()

            # Generator wants discriminator to accept fakes as real
            out_fake_g = discriminator(fake_imgs, fake_labels)
            loss_g = criterion(out_fake_g, real_labels)

            loss_g.backward()
            opt_g.step()

            loss_d_accum += loss_d.item()
            loss_g_accum += loss_g.item()
            total_batches += 1

        epoch_d_loss = loss_d_accum / total_batches
        epoch_g_loss = loss_g_accum / total_batches

        print(f"Epoch {epoch:2d}/{args.epochs} | loss_D {epoch_d_loss:.4f} | loss_G {epoch_g_loss:.4f}")

    print("-" * 64)

    # Save visual outputs
    if not args.no_figure:
        save_dir = os.path.dirname(os.path.abspath(__file__))

        # Generate conditional grid: 10 rows (classes 0-9), 8 columns (random seeds)
        generator.eval()
        with torch.no_grad():
            grid_imgs = []
            for c in range(num_classes):
                # 8 samples of class c
                z_c = torch.randn(8, args.latent_dim, device=device)
                labels_c = torch.full((8,), c, dtype=torch.long, device=device)
                gen_c = generator(z_c, labels_c)
                grid_imgs.append(gen_c)

            # Concatenate to shape [80, 1, 28, 28]
            grid_imgs = torch.cat(grid_imgs, dim=0)

            gen_path = os.path.join(save_dir, "cgan_generated_samples.png")
            mc.save_grid_png(grid_imgs, gen_path, nrows=num_classes, ncols=8)

        # Save latent space walk of class 0 (T-shirt / digit 0)
        walk_path = os.path.join(save_dir, "cgan_latent_walk.png")
        mc.save_latent_walk_png(generator, walk_path, args.latent_dim, device=device)


if __name__ == "__main__":
    main()
