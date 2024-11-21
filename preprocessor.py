import os
import torch
import torchaudio
import laion_clap
import h5py
from ConvaiDataset import ConvaiDataset


def load_text_descriptions(description_dir):
    descriptions = {}
    for root, _, files in os.walk(description_dir):
        for filename in files:
            if filename.endswith('.txt'):
                filepath = os.path.join(root, filename)
                with open(filepath, 'r') as file:
                    relative_path = os.path.relpath(filepath, description_dir)
                    key = os.path.splitext(relative_path)[0]  # Remove extension
                    descriptions[key] = file.read().strip()
    return descriptions


def load_and_pad_audio(audio_dir, target_length=15, target_sample_rate=48000):
    audio_data = {}
    for root, _, files in os.walk(audio_dir):
        for filename in files:
            if filename.endswith('.wav'):
                filepath = os.path.join(root, filename)
                waveform, sample_rate = torchaudio.load(filepath)

                # Resample if the sample rate is different from the target
                if sample_rate != target_sample_rate:
                    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
                    waveform = resampler(waveform)
                    sample_rate = target_sample_rate

                # Convert to mono by selecting the first channel if it's stereo
                if waveform.size(0) > 1:
                    waveform = waveform[0, :].unsqueeze(0)

                # Calculate the number of samples for the target length
                num_samples = target_length * sample_rate

                # Pad or truncate the waveform
                if waveform.size(1) < num_samples:
                    padding = num_samples - waveform.size(1)
                    waveform = torch.nn.functional.pad(waveform, (0, padding))
                else:
                    waveform = waveform[:, :num_samples]

                # Use relative path from the base directory as the key
                relative_path = os.path.relpath(filepath, audio_dir)
                key = os.path.splitext(relative_path)[0]  # Remove extension
                audio_data[key] = waveform
    return audio_data


def generate_text_embeddings(descriptions, clap_model):
    text_embeddings = {}
    for key, text in descriptions.items():
        embedding = clap_model.get_text_embedding([text], use_tensor=True).squeeze(0)
        embedding = embedding.detach()
        text_embeddings[key] = embedding
    return text_embeddings

def save_to_hdf5(audio_data, text_descriptions, output_file='dataset/preprocessed_data.h5'):
    with h5py.File(output_file, 'w') as h5f:
        for key in audio_data:
            group = h5f.create_group(key)
            group.create_dataset('audio', data=audio_data[key].numpy())
            dt = h5py.string_dtype(encoding='ascii', length=len(text_descriptions[key]))
            group.create_dataset('text', data=text_descriptions[key], dtype=dt)


def verify_dataset(text_descriptions, audio_data):
    print("*verifying dataset*")
    common_keys = set(audio_data.keys()).intersection(set(text_descriptions.keys()))

    audio_keys = set(audio_data.keys())
    description_keys = set(text_descriptions.keys())

    missing_audio = description_keys - audio_keys
    missing_descriptions = audio_keys - description_keys

    if missing_audio:
        print(f"Missing audio files for: {missing_audio}")
        return False
    if missing_descriptions:
        print(f"Missing descriptions for: {missing_descriptions}")
        return False

    print(f"Number of aligned item-pairs: {len(common_keys)}")
    return True


def main():
    # Directories
    audio_dir = 'dataset/audio'
    description_dir = 'dataset/label'

    # Load data
    descriptions = load_text_descriptions(description_dir)
    audio_data = load_and_pad_audio(audio_dir)

    # Ensure keys match
    if not verify_dataset(descriptions, audio_data):
        print("ERROR: dataset corrupt, text & audio files are not matching")

    # Generate text embeddings
    # Save to HDF5
    save_to_hdf5(audio_data, descriptions)


if __name__ == "__main__":
   main()