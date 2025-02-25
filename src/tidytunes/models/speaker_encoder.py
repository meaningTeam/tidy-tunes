import torch

from tidytunes.models.external.resnet_speaker_encoder import ResNetSpeakerEncoder


class SpeakerEncoder(torch.nn.Module):

    def __init__(
        self,
        model: ResNetSpeakerEncoder,
        num_input_frames: int,
        hop_length: int,
        sampling_rate: int,
    ):
        """
        Speaker embedding encoder using a ResNet-based x-vector extractor.

        Args:
            model (ResNetSpeakerEncoder): Pre-trained speaker encoder model.
            num_input_frames (int): Number of frames per input chunk.
            hop_length (int): Hop length (samples per frame).
            sampling_rate (int): Sampling rate of the input audio.
        """
        super().__init__()
        self.model = model
        self.hop_length = hop_length
        self.num_input_frames = num_input_frames
        self.sampling_rate = sampling_rate

    def forward(self, x: torch.Tensor, x_lens: torch.Tensor):
        """
        Computes speaker embeddings for input audio sequences.

        Args:
            x (torch.Tensor): Input batch of audio waveforms (B, T).
            x_lens (torch.Tensor): Lengths of valid audio in each batch (B,).

        Returns:
            List[torch.Tensor]: Speaker embeddings for each input in the batch.
        """
        chunks_flat, lens = [], []

        for a, al in zip(x, x_lens):

            a = a[:al].unsqueeze(0)
            chunks = self.split_to_chunks(a)

            # Repeat-pad to num_frames
            chunk_len = chunks.shape[-1]
            pad_size = self.num_input_frames * self.hop_length - chunk_len

            # Stupid workaround for long circular paddings
            idx = 0
            while pad_size > 0:
                p = min(chunk_len, pad_size)
                chunks = torch.cat([chunks, chunks[..., idx : idx + p]], dim=-1)
                idx += p
                pad_size -= p

            chunks_flat.append(chunks)
            lens.append(chunks.shape[0])

        chunks_flat = torch.cat(chunks_flat, dim=0)
        embeddings_flat = self.model(chunks_flat)
        embeddings = torch.split(embeddings_flat, lens)

        return embeddings

    def split_to_chunks(self, x: torch.Tensor):
        num_samples = min(self.num_input_frames * self.hop_length, x.shape[1])
        num_full_chunks = (
            x.shape[1] // num_samples
        )  # Calculate the number of full chunks

        frames_flat = []
        for i in range(num_full_chunks):
            s = i * num_samples
            e = s + num_samples
            frames_flat.append(x[:, s:e])

        frames_flat = torch.cat(frames_flat, dim=0)
        return frames_flat
