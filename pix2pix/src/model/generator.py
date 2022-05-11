import torch

"""
Архитектура описана в:
https://arxiv.org/pdf/1611.07004.pdf

BatchNorm2d заменен на InstanceNorm2d
ConvTranspose2d местами заменен на Upsample + Conv2d
"""


class ConvBlock(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels, type, dropout_state=False, use_upscale=False
    ):
        assert type == "encoder" or type == "decoder"
        super(ConvBlock, self).__init__()

        if type == "encoder":
            self.transform = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                    padding_mode="reflect",
                ),
                torch.nn.InstanceNorm2d(out_channels, affine=True),
                torch.nn.LeakyReLU(0.2),
            )

        if type == "decoder":
            if use_upscale:
                self.transform = [
                    torch.nn.Upsample(
                        scale_factor=2, mode="bilinear", align_corners=True
                    ),
                    torch.nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                        padding_mode="reflect",
                    ),
                ]
            else:
                self.transform = [
                    torch.nn.ConvTranspose2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                ]
            self.transform += [
                torch.nn.InstanceNorm2d(out_channels, affine=True),
                torch.nn.ReLU(),
            ]
            if dropout_state:
                self.transform += [torch.nn.Dropout(0.5)]
            self.transform = torch.nn.Sequential(*self.transform)

    def forward(self, x):
        return self.transform(x)


class GetGenerator(torch.nn.Module):
    def __init__(self, in_channels=3, enc_fmaps=[64, 128, 256, 512, 512, 512, 512]):
        super(GetGenerator, self).__init__()

        # Encoder related part
        self.enc_blocs = [
            torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=enc_fmaps[0],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=True,
                    padding_mode="reflect",
                ),
                torch.nn.LeakyReLU(0.2),
            )
        ]
        for ind, maps_amount in enumerate(enc_fmaps[1:]):
            self.enc_blocs += [
                ConvBlock(
                    in_channels=enc_fmaps[ind],
                    out_channels=maps_amount,
                    type="encoder",
                )
            ]
        self.enc_blocs = torch.nn.ModuleList(self.enc_blocs)

        # Bottleneck related part
        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=enc_fmaps[-1],
                out_channels=enc_fmaps[-1],
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            torch.nn.ReLU(),
        )

        # Decoder related part
        dec_fmaps = [enc_fmaps[-1]] + list(reversed(enc_fmaps))
        dropouts = [True] * 3 + [False] * (len(dec_fmaps) - 3)
        multiply = [1] + [2] * (len(dec_fmaps) - 1)
        use_upscale = [False] * 4 + [True] * (len(dec_fmaps) - 4)
        self.dec_blocs = []
        for ind, maps_amount in enumerate(dec_fmaps[1:]):
            self.dec_blocs += [
                ConvBlock(
                    in_channels=dec_fmaps[ind] * multiply[ind],
                    out_channels=maps_amount,
                    type="decoder",
                    dropout_state=dropouts[ind],
                    use_upscale=use_upscale[ind],
                )
            ]
        self.dec_blocs = torch.nn.ModuleList(self.dec_blocs)
        self.dec_final_block = torch.nn.Sequential(
            # torch.nn.ConvTranspose2d(
            #     in_channels=dec_fmaps[-1] * 2,
            #     out_channels=in_channels,
            #     kernel_size=4,
            #     stride=2,
            #     padding=1,
            # ),
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            torch.nn.Conv2d(
                in_channels=dec_fmaps[-1] * 2,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="reflect",
            ),
            torch.nn.Tanh(),
        )

    def forward(self, x):
        enc_output = []
        for transformation in self.enc_blocs:
            x = transformation(x)
            enc_output += [x]
        enc_output.reverse()

        x = self.bottleneck(x)

        x = self.dec_blocs[0](x)
        for ind, transformation in enumerate(self.dec_blocs[1:]):
            x = transformation(torch.cat([x, enc_output[ind]], dim=1))
        x = self.dec_final_block(torch.cat([x, enc_output[-1]], dim=1))

        return x


if __name__ == "__main__":
    x = torch.randn([1, 3, 512, 512])
    model = GetGenerator()
    print("MODEL: ")
    print(model)
    print("INPUT: ", x.size())
    print("OUTPUT: ", model(x).shape)
