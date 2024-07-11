



class Postnet(nn.Module):
    """The Tacotron postnet consists of a number of 1-d convolutional layers
    with Xavier initialization and a tanh activation, with batch normalization.
    Depending on configuration, the postnet may either refine the MEL spectrogram
    or upsample it to a linear spectrogram

    Arguments
    ---------
    n_mel_channels: int
        the number of MEL spectrogram channels
    postnet_embedding_dim: int
        the postnet embedding dimension
    postnet_kernel_size: int
        the kernel size of the convolutions within the decoders
    postnet_n_convolutions: int
        the number of convolutions in the postnet

    Example
    -------
    >>> import torch
    >>> from speechbrain.lobes.models.Tacotron2 import Postnet
    >>> layer = Postnet()
    >>> x = torch.randn(2, 80, 861)
    >>> output = layer(x)
    >>> output.shape
    torch.Size([2, 80, 861])
    """

    def __init__(
        self,
        n_mel_channels=80,
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,
    ):
        super().__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    n_mel_channels,
                    postnet_embedding_dim,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=int((postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="tanh",
                ),
                nn.BatchNorm1d(postnet_embedding_dim),
            )
        )

        for i in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(
                        postnet_embedding_dim,
                        postnet_embedding_dim,
                        kernel_size=postnet_kernel_size,
                        stride=1,
                        padding=int((postnet_kernel_size - 1) / 2),
                        dilation=1,
                        w_init_gain="tanh",
                    ),
                    nn.BatchNorm1d(postnet_embedding_dim),
                )
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    postnet_embedding_dim,
                    n_mel_channels,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=int((postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="linear",
                ),
                nn.BatchNorm1d(n_mel_channels),
            )
        )
        self.n_convs = len(self.convolutions)

    def forward(self, x):
        """Computes the forward pass of the postnet

        Arguments
        ---------
        x: torch.Tensor
            the postnet input (usually a MEL spectrogram)

        Returns
        -------
        output: torch.Tensor
            the postnet output (a refined MEL spectrogram or a
            linear spectrogram depending on how the model is
            configured)
        """
        i = 0
        for conv in self.convolutions:
            if i < self.n_convs - 1:
                x = F.dropout(torch.tanh(conv(x)), 0.5, training=self.training)
            else:
                x = F.dropout(conv(x), 0.5, training=self.training)
            i += 1

        return x
