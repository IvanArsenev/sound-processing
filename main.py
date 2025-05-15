"""
Module for hot word recognition using Porcupine and subsequent speech recognition using Vosk.
"""

import struct
import os
import json
import logging
import argparse

import pvporcupine
import pyaudio
import vosk
import sounddevice as sd
import wave

from config import ACCESS_KEY

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

CUSTOM_KEYWORD_PATH = './stones.ppn'
VOSK_MODEL_PATH = 'vosk-model-small-ru-0.22'


def listen_for_hotword(porcupine, audio_stream):
    """
    Listen to the audio stream and process audio frames to detect hotword.

    Args:
        porcupine: Porcupine hotword detection object.
        audio_stream: Audio input stream.

    Returns:
        int: Index of detected hotword or -1 if none detected.
    """
    pcm = audio_stream.read(porcupine.frame_length)
    pcm_data = struct.unpack_from('h' * porcupine.frame_length, pcm)
    return porcupine.process(pcm_data)


def recognize_next_word(model):
    """Recognizes speech after detecting a hot word in live mode."""
    logging.info('Listening for words in live mode...')
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16', channels=1) as stream:
        rec = vosk.KaldiRecognizer(model, 16000)
        while True:
            data = stream.read(8000)
            data = bytes(data[0])
            if not rec.AcceptWaveform(data):
                partial = json.loads(rec.PartialResult())
                partial_text = partial.get('partial', '')
                if partial_text:
                    logging.info('Partial: %s', partial_text)
                if len(partial_text) > 1:
                    break


def recognize_next_word_from_position(model, file_path, start_frame):
    """Recognizes speech starting from a specific frame in a file."""
    if not os.path.isfile(file_path):
        logging.error('File not found: %s', file_path)
        return

    try:
        with wave.open(file_path, 'rb') as wav_file:
            if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2 or wav_file.getframerate() != 16000:
                logging.error('Audio file must be mono, 16-bit, and 16000 Hz')
                return

            wav_file.setpos(start_frame)
            rec = vosk.KaldiRecognizer(model, 16000)

            while True:
                pcm = wav_file.readframes(8000)
                if len(pcm) == 0:
                    break
                if rec.AcceptWaveform(pcm):
                    result = json.loads(rec.Result())
                    text = result.get('text', '')
                    if text:
                        logging.info('Recognized: %s', text)
                        break
    except Exception as e:
        logging.error('Error processing file: %s', e)


def run_live_mode():
    """Run the original live hotword listening and recognition loop."""
    logging.info('Check for files...')
    if not os.path.exists(CUSTOM_KEYWORD_PATH):
        logging.error('The %s file was not found.', CUSTOM_KEYWORD_PATH)
        return
    if not os.path.exists(VOSK_MODEL_PATH):
        logging.error('Vosk model not found on path %s', VOSK_MODEL_PATH)
        return
    logging.info('Check for files...Done!')
    logging.info('Porcupine Initialization...')
    porcupine = pvporcupine.create(
        access_key=ACCESS_KEY,
        keyword_paths=[CUSTOM_KEYWORD_PATH]
    )
    logging.info('Porcupine Initialization...Done!')
    logging.info('PyAudio Initialization...')
    audio_stream = pyaudio.PyAudio().open(
        rate=porcupine.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine.frame_length
    )
    logging.info('PyAudio Initialization...Done!')
    logging.info('Vosk Initialization...')
    model = vosk.Model(VOSK_MODEL_PATH)
    logging.info('Vosk Initialization...Done!')
    logging.info('Listening for hotword...')
    try:
        while True:
            pcm = audio_stream.read(porcupine.frame_length)
            pcm_data = struct.unpack_from("h" * porcupine.frame_length, pcm)
            result = porcupine.process(pcm_data)
            if result >= 0:
                logging.info('Hotword detected! Next word recognition...')
                recognize_next_word(model)
                logging.info('Listening for hotword...')
    except KeyboardInterrupt:
        logging.warning('Stopping...')
    finally:
        audio_stream.close()
        porcupine.delete()


def run_file_mode(path):
    """Handle the file mode: search for hotword and recognize next word."""
    if not os.path.isfile(path):
        logging.error('File not found: %s', path)
        return

    logging.info('File found: %s', path)

    porcupine = pvporcupine.create(
        access_key=ACCESS_KEY,
        keyword_paths=[CUSTOM_KEYWORD_PATH]
    )
    model = vosk.Model(VOSK_MODEL_PATH)

    try:
        with wave.open(path, 'rb') as wav_file:
            if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2 or wav_file.getframerate() != porcupine.sample_rate:
                logging.error('Audio file must be mono, 16-bit, and %d Hz', porcupine.sample_rate)
                return

            logging.info('Analyzing audio for hotword...')
            while True:
                pcm = wav_file.readframes(porcupine.frame_length)
                if len(pcm) == 0:
                    break
                pcm_data = struct.unpack_from("h" * porcupine.frame_length, pcm)
                result = porcupine.process(pcm_data)

                if result >= 0:
                    logging.info('Hotword detected in file: %s', path)
                    start_frame = wav_file.tell()
                    recognize_next_word_from_position(model, path, start_frame)
                    logging.info('Analyzing audio for hotword...')
    except Exception as e:
        logging.error('Error processing file: %s', e)
    finally:
        porcupine.delete()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Hotword recognition and speech recognition module.'
    )
    parser.add_argument(
        '--mode',
        choices=['live', 'file'],
        required=True,
        help='Operating mode: live (microphone) or file (audio file)'
    )
    parser.add_argument(
        '--path',
        type=str,
        help='Path to the audio file, required if mode is file'
    )

    args = parser.parse_args()

    if args.mode == 'live':
        run_live_mode()
    elif args.mode == 'file':
        if not args.path:
            parser.error('--path is required when --mode=file')
        run_file_mode(args.path)


if __name__ == '__main__':
    main()
