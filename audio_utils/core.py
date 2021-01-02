import warnings

import numpy as np
import librosa

############
# CHECKING #
############
def _check_audio_types(audio: np.ndarray):
    assert isinstance(audio, np.ndarray), f'expected np.ndarray but got {type(audio)} as input.'
    assert audio.ndim == 2, f'audio must be shape (channels, time), got shape {audio.shape}'
    if audio.shape[-1] < audio.shape[-2]:
        warnings.warn(f'got audio shape {audio.shape}. Audio should be (channels, time). \
                        typically, the number of samples is much larger than the number of channels. ')
    if _is_zero(audio):
        warnings.warn(f'provided audio array is all zeros')

def _is_mono(audio: np.ndarray):
    _check_audio_types(audio)
    num_channels = audio.shape[-2]
    return num_channels == 1

def _is_zero(audio: np.ndarray):
    return np.all(audio == 0);

def librosa_input_wrap(audio: np.ndarray):
    _check_audio_types(audio)
    if _is_mono(audio): 
        audio = audio[0]
    return audio

def librosa_output_wrap(audio: np.ndarray):
    if audio.ndim == 1:
        audio = np.expand_dims(audio, axis=0)
    return  audio

def _coalesce_timestamps(timestamps: np.ndarray, condition: callable):
    """ coalesce timestamps in a timestamp matrix if a callable returns true
    """
    coalesced_timestamps = []

    last_start_time = timestamps[0][0]
    last_end_time = timestamps[0][1]
    for i in range(1, len(timestamps)+1):
        start_time, end_time = timestamps[i]
        if condition(last_end_time, start_time):
            last_end_time = end_time
        else:
            ts = [last_start_time, end_time]
            coalesced_timestamps.append(ts)
            last_start_time = start_time
            last_end_time = end_time

    coalesced_timestamps = np.array(coalesced_timestamps)

    print(timestamps)
    print(coalesced_timestamps)

    return coalesced_timestamps

#########
# UTILS #
######### 

def split_on_silence(audio: np.ndarray, sr: int, top_db: int = 45, min_silence_duration: int = 0.3):
    """ Wrapper for librosa.effects.split, with an added min_silence_duration parameter. 
    That is, it will coalesce timestamps if the silence in between them is less than min_silence_duration
    """
    audio = librosa_input_wrap(audio)
    timestamps = librosa.effects.split(audio, top_db=top_db)

    timestamps = _coalesce_timestamps(timestamps, condition=lambda e, s: abs((s-e)*sr) < min_silence_duration )

    return timestamps

def get_audio_from_timestamp(audio: np.ndarray, sr: int, timestamp: tuple) -> np.ndarray:
    """get audio subarray from a timestamp

    Args:
        audio ([np.ndarray]): audio array with shape (channels, samples)
        sr ([int]): sample rate
        timestamp ([np.ndarray]): np.ndarray with shape (start_time, end_time) (in seconds)
    """
    _check_audio_types(audio)
    idxs = timestamp * sr
    return audio[idxs[0], idxs[1]]

def window(audio: np.ndarray, window_len: int = 48000, hop_len: int = 4800):
    """split audio into overlapping windows

    note: this is not a memory efficient view like librosa.util.frame. 
    It will return a new copy of the array

    Args:
        audio (np.ndarray): audio array with shape (channels, samples)
        window_len (int, optional): [description]. Defaults to 48000.
        hop_len (int, optional): [description]. Defaults to 4800.
    Returns:
        audio_windowed (np.ndarray): windowed audio array with shape (window, channels, samples)
    """
    _check_audio_types(audio)
    # determine how many window_len windows we can get out of the audio array
    # use ceil because we can zero pad
    n_chunks = int(np.ceil(len(audio)/(window_len))) 
    start_idxs = np.arange(0, n_chunks * window_len, hop_len)

    windows = []
    for start_idx in start_idxs:
        # end index should be start index + window length
        end_idx = start_idx + window_len
        # BUT, if we have reached the end of the audio, stop there
        end_idx = min([end_idx, len(audio)])
        # create audio window
        win = np.array(audio[:, start_idx:end_idx])
        # zero pad window if needed
        win = zero_pad(win, required_len=window_len)
        windows.append(win)
    
    audio_windowed = np.stack(windows)
    return audio_windowed

def downmix(audio: np.ndarray):
    """ downmix an audio array.
    must be shape (channels, mono)

    Args:
        audio ([np.ndarray]): array to downmix
    """
    _check_audio_types(audio)
    audio = audio.mean(axis=-2, keepdims=True)
    return audio

def resample(audio: np.ndarray, old_sr: int, new_sr: int = 48000) -> np.ndarray:
    """wrapper around librosa for resampling

    Args:
        audio (np.ndarray): audio array shape (channels, time)
        old_sr (int): old sample rate
        new_sr (int, optional): target sample rate.  Defaults to 48000.

    Returns:
        np.darray: resampled audio. shape (channels, time)
    """
    _check_audio_types(audio)

    if _is_mono(audio):
        audio = audio[0]
        audio = librosa.resample(audio, old_sr, new_sr)
        audio = np.expand_dims(audio, axis=-2)
    else:
        audio = librosa.resample(audio, old_sr, new_sr)
    return audio

def zero_pad(audio: np.ndarray, required_len: int = 48000) -> np.ndarray:
    """zero pad audio array to meet a multiple of required_len
    all padding is done at the end of the array (no centering)

    Args:
        audio (np.ndarray): audio array w shape (channels, sample)
        required_len (int, optional): target length in samples. Defaults to 48000.

    Returns:
        np.ndarray: zero padded audio
    """
    _check_audio_types(audio)

    num_frames = audio.shape[-1]

    before = 0
    after = required_len - num_frames%required_len
    if after == required_len:
        return audio
    audio = np.pad(audio, pad_width=((0, 0), (before, after)), mode='constant', constant_values=0)
    return audio