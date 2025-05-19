"""
Module for hot word recognition using Porcupine and subsequent speech recognition using Vosk.
"""

import argparse
import json
import logging
import os
import time
import struct
import wave
from io import BytesIO
import requests

import pvporcupine
import pyaudio
import sounddevice as sd
import vosk
from pydub import AudioSegment
from pydub.playback import play

from config import ACCESS_KEY

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

CUSTOM_KEYWORD_PATH = './stones.ppn'
VOSK_MODEL_PATH = 'vosk-model-small-ru-0.22'
BUFFER_SIZE = 1024 * 100
BUFFER_TIME = 5
RADIO_URL = 'https://radio.kotah.ru/soundcheck'


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


def recognize_next_word_microphone(model):
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
                recognize_next_word_microphone(model)
                logging.info('Listening for hotword...')
    except KeyboardInterrupt:
        logging.warning('Stopping...')
    finally:
        audio_stream.close()
        porcupine.delete()


def recognize_word_in_chunk(model, audio_chunk):
    """Recognizes the next word in a given audio chunk."""
    rec = vosk.KaldiRecognizer(model, 16000)
    if rec.AcceptWaveform(audio_chunk):
        result = json.loads(rec.Result())
        recognized_text = result.get('text', '')
        logging.info('Recognized: %s', recognized_text)
        return recognized_text
    partial = json.loads(rec.PartialResult())
    partial_text = partial.get('partial', '')
    logging.info('Partial: %s', partial_text)
    return partial_text


def run_radio_mode():
    """Run the original broadcast hotword listening and recognition loop."""
    porcupine = pvporcupine.create(
        access_key=ACCESS_KEY,
        keyword_paths=[CUSTOM_KEYWORD_PATH]
    )
    try:
        response = requests.get(RADIO_URL, stream=True)
        response.raise_for_status()
        logging.info('Streaming radio...')
        audio_buffer = BytesIO()
        start_time = time.time()
        vosk_model = vosk.Model(VOSK_MODEL_PATH)

        def process_audio_chunk(buffer):
            """Process a single audio chunk for hotword detection."""
            try:
                buffer.seek(0)
                audio_segment = AudioSegment.from_file(buffer, format='mp3')
                audio_data = audio_segment.raw_data

                pcm_data = struct.unpack_from(
                    'h' * porcupine.frame_length,
                    audio_data[:porcupine.frame_length * 2]
                )
                result = porcupine.process(pcm_data)
                logging.info('Listening for hotword in audio chunk...')
                if result >= 0:
                    logging.info('Hotword detected! Recognizing next word in chunk...')
                    recognize_word_in_chunk(vosk_model, audio_data)

                play(audio_segment)
            except (IOError, struct.error, ValueError) as err:
                logging.info('Error processing audio chunk: %s', err)

        for chunk in response.iter_content(chunk_size=1024):
            if not chunk:
                continue

            audio_buffer.write(chunk)
            if audio_buffer.tell() >= BUFFER_SIZE or time.time() - start_time >= BUFFER_TIME:
                process_audio_chunk(audio_buffer)
                audio_buffer = BytesIO()
                start_time = time.time()

    except requests.RequestException as e:
        logging.info('Error connecting to radio stream: %s', e)
    finally:
        if 'porcupine' in locals():
            porcupine.delete()


def recognize_next_word_from_position(model, file_path, start_frame):
    """Recognizes speech starting from a specific frame in a file."""
    if not os.path.isfile(file_path):
        logging.error('File not found: %s', file_path)
        return

    try:
        with wave.open(file_path, 'rb') as wav_file:
            if (wav_file.getnchannels() != 1 or
                    wav_file.getsampwidth() != 2 or
                    wav_file.getframerate() != 16000):
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
    except (IOError, wave.Error, ValueError) as e:
        logging.error('Error processing file: %s', e)


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
            if (wav_file.getnchannels() != 1 or
                    wav_file.getsampwidth() != 2 or
                    wav_file.getframerate() != porcupine.sample_rate):
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
                    sample_rate = wav_file.getframerate()
                    start_frame = wav_file.tell()
                    timing_minutes = int(start_frame / float(sample_rate)) // 60
                    timing_seconds = int(start_frame / float(sample_rate)) % 60
                    logging.info('Hotword detected at %s:%s!', timing_minutes, timing_seconds)
                    recognize_next_word_from_position(model, path, start_frame)
                    logging.info('Analyzing audio for hotword...')

    except struct.error:
        logging.info('File ended!')
    except (IOError, wave.Error, ValueError) as e:
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
        choices=['live', 'file', 'radio'],
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
    elif args.mode == 'radio':
        run_radio_mode()
    elif args.mode == 'file':
        if not args.path:
            parser.error('--path is required when --mode=file')
        run_file_mode(args.path)


if __name__ == '__main__':
    main()
