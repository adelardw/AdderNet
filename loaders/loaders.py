import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torchaudio.transforms import Resample
import glob
import numpy as np


def load_speech_files(directories):
    """
    Loads all speech files from the specified directories.

    Parameters:
        directories (list(str)): List of directory paths
            to search for speech files.

    Returns:
        list(str): List of file paths to the found speech files.

    Example:
    >>> load_speech_files(["data/speech"])
    """
    speech_files = []
    for directory in directories:
        speech_files.extend(glob.glob(f"{directory}/**/*.flac",
                                      recursive=True))
    return speech_files


def load_files(directory):
    """
    Loads all noise files from the specified directory.

    Parameters:
        directory (str): Path to the directory containing noise files.

    Returns:
        list(str): List of file paths to the noise files.
    """
    return [os.path.join(directory, f)
            for f in os.listdir(directory) if f.endswith(".wav")]


class SpeechNoiseDataset(Dataset):
    """
    A PyTorch Dataset for mixing speech and noise
    audio files with optional transformations.

    Attributes:
        speech_files (list(str)): List of file paths
            to the speech audio files.
        noise_files (list(str)): List of file paths to the noise audio files.
        sample_rate (int): Target sample rate for audio data.
        transform (callable, optional): A function/transform
            to apply to the audio data.
        padding_strategy (str): Strategy for padding audio files
            ('longest', 'average', or 'median').
        audio_length (int): Target length for audio padding/trimming
            based on the specified strategy.
    """

    def __init__(self, speech_files,
                 noise_dir,
                 sample_rate=16000,
                 transform=None,
                 padding_strategy="longest"):
        self.speech_files = speech_files
        self.noise_files = load_files(noise_dir)
        self.sample_rate = sample_rate
        self.transform = transform
        self.padding_strategy = padding_strategy

        if self.padding_strategy == "average":
            self.audio_length = self.calculate_audio_length(method="average")
        elif self.padding_strategy == "median":
            self.audio_length = self.calculate_audio_length(method="median")
        elif self.padding_strategy == "longest":
            self.audio_length = self.calculate_audio_length(method="longest")

    def calculate_audio_length(self, method="longest"):
        """
        Calculates the target length for audio files
        based on the specified method.

        Parameters:
            method (str): Method for calculating target length.
                Choices are 'longest', 'average', and 'median'.

        Returns:
            int: The target length in samples based on the selected method.
        """
        lengths = []
        for file_path in self.speech_files:
            waveform, sr = torchaudio.load(file_path)
            if sr != self.sample_rate:
                resampler = Resample(orig_freq=sr, new_freq=self.sample_rate)
                waveform = resampler(waveform)
            lengths.append(waveform.size(1))

        if method == "longest":
            return max(lengths)
        elif method == "average":
            return int(np.mean(lengths))
        elif method == "median":
            return int(np.median(lengths))

    def pad_or_trim(self, waveform, audio_length):
        """
        Pads or trims the waveform to the specified target length.

        Parameters:
            waveform (torch.Tensor): The audio waveform
                to be padded or trimmed.
            audio_length (int): The target length in samples.

        Returns:
            torch.Tensor: The padded or trimmed waveform.
        """
        current_length = waveform.size(1)
        if current_length < audio_length:
            padding = audio_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif current_length > audio_length:
            waveform = waveform[:, :audio_length]
        return waveform

    def __len__(self):
        """
        Returns the number of speech files in the dataset.

        Returns:
            int: Number of speech files.
        """
        return len(self.speech_files)

    def __getitem__(self, idx):
        """
        Retrieves a mixed audio sample of speech with noise.

        Parameters:
            idx (int): Index of the speech file to retrieve.

        Returns:
            tuple of torch.Tensor: A tuple containing
                the mixed waveform and the original speech waveform.
        """
        speech_path = self.speech_files[idx]
        noise_path = random.choice(self.noise_files)

        speech_waveform, speech_sr = torchaudio.load(speech_path)
        noise_waveform, noise_sr = torchaudio.load(noise_path)

        if speech_sr != self.sample_rate:
            resampler = Resample(orig_freq=speech_sr,
                                 new_freq=self.sample_rate)
            speech_waveform = resampler(speech_waveform)
        if noise_sr != self.sample_rate:
            resampler = Resample(orig_freq=noise_sr,
                                 new_freq=self.sample_rate)
            noise_waveform = resampler(noise_waveform)

        if noise_waveform.size(1) < speech_waveform.size(1):
            repeat_ctr = speech_waveform.size(1) // noise_waveform.size(1) + 1
            noise_waveform = noise_waveform.repeat(1, repeat_ctr)
        noise_waveform = noise_waveform[:, :speech_waveform.size(1)]

        mixed_waveform = speech_waveform + noise_waveform
        mixed_waveform = self.pad_or_trim(mixed_waveform, self.audio_length)
        speech_waveform = self.pad_or_trim(speech_waveform, self.audio_length)

        if self.transform:
            mixed_waveform = self.transform(mixed_waveform)
            speech_waveform = self.transform(speech_waveform)

        return mixed_waveform, speech_waveform


def get_loaders(speech_dirs,
                noise_dir,
                sample_rate=16000,
                batch_size=4,
                split_ratios=(0.8, 0.1, 0.1),
                padding_strategy="longest"):
    """
    Creates DataLoader objects for training, validation, and testing datasets.

    Parameters:
        speech_dirs (list(str)): Directory paths containing speech audio files.
        noise_dir (str): Directory paths containing noise audio files.
        sample_rate (int, optional): Target sample rate for audio data.
            Defaults to 16000.
        batch_size (int, optional): Number of samples per batch. Defaults to 4.
        split_ratios (tuple of float, optional): Ratios for splitting data into
            training, validation, and testing. Defaults to (0.8, 0.1, 0.1).
        padding_strategy (str, optional): Strategy for padding audio files
            ('longest', 'average', or 'median'). Defaults to 'longest'.

    Returns:
        tuple of DataLoader: DataLoader objects for
        training, validation, and testing sets.

    Example:
    >>> get_loaders(["data/speech"],
                    "data/noise",
                    sample_rate=16000,
                    batch_size=4,
                    padding_strategy="longest")
    """
    all_speech_files = load_speech_files(speech_dirs)

    random.shuffle(all_speech_files)
    num_train = int(split_ratios[0] * len(all_speech_files))
    num_val = int(split_ratios[1] * len(all_speech_files))

    train_speech_files = all_speech_files[:num_train]
    val_speech_files = all_speech_files[num_train:num_train + num_val]
    test_speech_files = all_speech_files[num_train + num_val:]

    train_dataset = SpeechNoiseDataset(train_speech_files,
                                       os.path.join(noise_dir, "tr"),
                                       sample_rate=sample_rate,
                                       padding_strategy=padding_strategy)
    val_dataset = SpeechNoiseDataset(val_speech_files,
                                     os.path.join(noise_dir, "cv"),
                                     sample_rate=sample_rate,
                                     padding_strategy=padding_strategy)
    test_dataset = SpeechNoiseDataset(test_speech_files,
                                      os.path.join(noise_dir, "tt"),
                                      sample_rate=sample_rate,
                                      padding_strategy=padding_strategy)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False)

    return train_loader, val_loader, test_loader
