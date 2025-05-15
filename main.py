import pvporcupine
import pyaudio
import struct
import os
import vosk
import json
import sounddevice as sd
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

ACCESS_KEY = '6+HrlHyfpm+mwvSnz93sRH1oM+AfzOKrlsGVMlu5DgWYS6AErZa2lA=='
CUSTOM_KEYWORD_PATH = './stones.ppn'
VOSK_MODEL_PATH = 'vosk-model-small-ru-0.22'


def listen_for_hotword(porcupine, audio_stream):
    pcm = audio_stream.read(porcupine.frame_length)
    pcm_data = struct.unpack_from('h' * porcupine.frame_length, pcm)
    return porcupine.process(pcm_data)


def recognize_next_word(model):
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
                    logging.info(f'Partial: {partial_text}')
                if len(partial_text) > 1:
                    break


def main():
    logging.info('Check for files...')
    if not os.path.exists(CUSTOM_KEYWORD_PATH):
        logging.error(f'The {CUSTOM_KEYWORD_PATH} file was not found.')
        return
    if not os.path.exists(VOSK_MODEL_PATH):
        logging.error(f'Vosk model not found on path {VOSK_MODEL_PATH}')
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


if __name__ == '__main__':
    main()