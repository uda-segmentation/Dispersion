# ---------------------------------------------------------------
# Anonymous submission for AAAI 2026.
# Paper title: "Unsupervised Domain Adaptation for Semantic Segmentation Based on Instance Directional Dispersion"
# No author or affiliation information included in this version.
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torchvision.models as models


class AuxiliaryResNet50(nn.Module):
    """
    Auxiliary semantic segmentation network pre-trained on concatenated
    depth and dispersion maps. Designed to assess pseudo-label reliability
    via geometric cues.

    Architecture:
        - ResNet-50 backbone (first conv adapted for 2-channel input) as encoder.
        - Lightweight decoder: sequential conv blocks for progressive refinement.
        - Final 1x1 conv to produce class logits, followed by bilinear upsampling.

    Notes:
        - Input is expected to be a fixed resolution (e.g., 512x512) 2-channel tensor
          encoding depth and dispersion.
        - Output is upsampled to approximate original image resolution.
    """

    def __init__(self, num_classes=19, pretrained=False):
        super(AuxiliaryResNet50, self).__init__()
        self.num_classes = num_classes

        # Load ResNet-50 and modify first conv to accept 2-channel input (dispersion + depth).
        self.encoder = models.resnet50(pretrained=pretrained)
        self.encoder.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])

        # Decoder: reduce channels progressively.
        self.decoder1 = self.conv_block(1024, 512)
        self.decoder2 = self.conv_block(512, 256)
        self.decoder3 = self.conv_block(256, 128)
        self.decoder4 = self.conv_block(128, 64)

        # Logits projection.
        self.final_conv = nn.Conv2d(64, self.num_classes, kernel_size=1)

        self.pool = nn.MaxPool2d(2, 2)

        # Bilinear upsampling to recover spatial resolution.
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def conv_block(self, in_channels, out_channels):
        """Basic conv -> BN -> ReLU block used in decoder."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward pass.
        x: [B, 2, H, W] tensor (dispersion + depth), expected fixed resolution.
        Returns:
            logits upsampled to higher resolution: [B, num_classes, H_out, W_out]
        """
        # Encoder
        enc1 = self.encoder[0](x)
        enc2 = self.encoder[1](enc1)
        enc3 = self.encoder[4](enc2)
        enc4 = self.encoder[5](enc3)
        enc5 = self.encoder[6](enc4)

        # Decoder
        dec1 = self.decoder1(enc5)
        dec2 = self.decoder2(dec1)
        dec3 = self.decoder3(dec2)
        dec4 = self.decoder4(dec3)

        # Class logits and upsample
        out = self.final_conv(dec4)
        out = self.upsample(out)
        return out
