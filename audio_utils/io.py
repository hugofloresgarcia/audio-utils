from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

#######
# I/O #
#######

def load_audio_file(path_to_audio, sample_rate=48000):
    """ wrapper for loading monophonic audio with librosa
    Args:
        path_to_audio (str): path to audio file
        sample_rate (int): target sample rate
    returns:
        audio (np.ndarray): monophonic audio with shape (samples,)
    """
    audio, sr = librosa.load(path_to_audio, mono=True, sr=sample_rate)
    # add channel dimension
    audio = np.expand_dims(audio, axis=-2)
    return audio

def _add_file_format_to_filename(path: str, file_format: str):
    if Path(path).suffix != file_format:
        path = path + file_format
    return path

def write_audio_file(audio: np.ndarray, path_to_audio: str, sample_rate: int, 
                     audio_format='flac', exist_ok=False):
    """write audio file to disk, raises an error if file exists

    Args:
        audio (np.ndarray): audio array shape (channels, samples)
        path_to_audio (str): save path
        sample_rate (int, optional): Sample rate corresponding to the audio array. 
                                    This function does not resample
        audio_format (str, optional): save format

    """
    ok_audio_formats = ('flac', 'ogg', 'wav')
    assert audio_format in ok_audio_formats, f'expected one of {ok_audio_formats} but got {audio_format}'

    path_to_audio = _add_file_format_to_filename(path_to_audio, audio_format)
    path_to_audio = Path(path_to_audio)

    if path_to_audio.exists() and not exist_ok:
        raise FileExistsError(f'{path_to_audio} exists. cant save audio')

    _check_audio_types(audio)
    assert audio.ndim == 2
    
    # reshape array to (samples, channels for sf)
    audio = np.reshape(audio, (audio.shape[1], audio.shape[0]))
    sf.write(path_to_audio, audio, sample_rate)
