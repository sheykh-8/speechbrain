





class Encoder(nn.Module):
    """The Tacotron2 encoder module, consisting of a sequence of  1-d convolution banks (3 by default)
    and a bidirectional LSTM

    Arguments
    ---------
    encoder_n_convolutions: int
        the number of encoder convolutions
    encoder_embedding_dim: int
        the dimension of the encoder embedding
    encoder_kernel_size: int
        the kernel size of the 1-D convolutional layers within
        the encoder

    Example
    -------
    >>> import torch
    >>> from speechbrain.lobes.models.Tacotron2 import Encoder
    >>> layer = Encoder()
    >>> x = torch.randn(2, 512, 128)
    >>> input_lengths = torch.tensor([128, 83])
    >>> outputs = layer(x, input_lengths)
    >>> outputs.shape
    torch.Size([2, 128, 512])

    """

    def __init__(
        self,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,
        encoder_kernel_size=5,
    ):
        super().__init__()

        convolutions = []
        for _ in range(encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(
                    encoder_embedding_dim,
                    encoder_embedding_dim,
                    kernel_size=encoder_kernel_size,
                    stride=1,
                    padding=int((encoder_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="relu",
                ),
                nn.BatchNorm1d(encoder_embedding_dim),
            )
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(
            encoder_embedding_dim,
            int(encoder_embedding_dim / 2),
            1,
            batch_first=True,
            bidirectional=True,
        )

    @torch.jit.ignore
    def forward(self, x, input_lengths):
        """Computes the encoder forward pass

        Arguments
        ---------
        x: torch.Tensor
            a batch of inputs (sequence embeddings)

        input_lengths: torch.Tensor
            a tensor of input lengths

        Returns
        -------
        outputs: torch.Tensor
            the encoder output
        """
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True
        )

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        return outputs

    @torch.jit.export
    def infer(self, x, input_lengths):
        """Performs a forward step in the inference context

        Arguments
        ---------
        x: torch.Tensor
            a batch of inputs (sequence embeddings)

        input_lengths: torch.Tensor
            a tensor of input lengths

        Returns
        -------
        outputs: torch.Tensor
            the encoder output
        """
        device = x.device
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x.to(device))), 0.5, self.training)

        x = x.transpose(1, 2)

        input_lengths = input_lengths.cpu()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True
        )
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        return outputs

