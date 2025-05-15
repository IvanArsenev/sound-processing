import struct
import os
import json
import logging
import argparse

import pvporcupine
import pyaudio
import vosk
import sounddevice as sd

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
    """Recognizes speech after detecting a hot word."""
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
                recognize_next_word(model)
                logging.info('Listening for hotword...')
    except KeyboardInterrupt:
        logging.warning('Stopping...')
    finally:
        audio_stream.close()
        porcupine.delete()


def run_file_mode(path):
    """Handle the file mode: check file existence and print file name."""
    if os.path.isfile(path):
        logging.info(f'File found: {path}')
        print(f'File to process: {path}')
    else:
        logging.error(f'File not found: {path}')


def main():
    parser = argparse.ArgumentParser(description='Hotword recognition and speech recognition module.')
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
