from ConvaiDataset import ConvaiDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import torch
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler

# quantization
def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)


def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)




def train(num_epochs=3):
    model.train()
    for epoch in range(num_epochs):
        model.train()
        for audio_batch, text_batch in dataloader:
            # Move data to the appropriate device (CPU or GPU)
            audio_batch = audio_batch.to(device)
            text_batch = [text for text in text_batch]  # Ensure text_batch is a list of strings
            print(text_batch)
            # Forward pass with internal loss computation
            loss = model(
                audio_batch,
                text=text_batch,  # Text conditioning, one element per batch
                embedding_mask_proba=0.1  # Probability of masking text for classifier-free guidance
            )

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

if __name__ == "__main__":
    print('*Starting Training Script*')
    model = DiffusionModel(
        net_t=UNetV0,  # The model type used for diffusion (U-Net V0 in this case)
        in_channels=1,  # U-Net: number of input/output (audio) channels
        channels=[8, 32, 64, 128, 256, 512, 512, 1024, 1024],  # U-Net: channels at each layer
        factors=[1, 4, 4, 4, 2, 2, 2, 2, 2],  # U-Net: downsampling and upsampling factors at each layer
        items=[1, 2, 2, 2, 2, 2, 2, 4, 4],  # U-Net: number of repeating items at each layer
        attentions=[0, 0, 0, 0, 0, 1, 1, 1, 1],  # U-Net: attention enabled/disabled at each layer
        attention_heads=8,  # U-Net: number of attention heads per attention item
        attention_features=64,  # U-Net: number of attention features per attention item
        diffusion_t=VDiffusion,  # The diffusion method used
        sampler_t=VSampler,  # The diffusion sampler used
        use_text_conditioning=True,  # U-Net: enables text conditioning (default T5-base)
        use_embedding_cfg=True,  # U-Net: enables classifier free guidance
        embedding_max_length=64,  # U-Net: text embedding maximum length (default for T5-base)
        embedding_features=768,  # U-Net: text embedding features (default for T5-base)
        cross_attentions=[0, 0, 0, 1, 1, 1, 1, 1, 1],  # U-Net: cross-attention enabled/disabled at each layer
    )
    print('*Loaded Diffusion Model*')
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()  # Mean Squared Error Loss

    print('*Loading Dataset*')
    dataset = ConvaiDataset('dataset/preprocessed_data.h5')
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    print('*Dataset Loaded*')

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)
    print('using device: %s' % device)
    num_epochs=3
    train(num_epochs)
    torch.save(model.state_dict(), 'audio_diffusion_model.pth')


