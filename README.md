# 🔥 Hotword Recognition with Porcupine and Vosk

This project provides a robust solution for hotword detection using Porcupine followed by speech recognition using Vosk. It supports both live microphone input and pre-recorded audio file processing.

## 🌟 Features
- **Hotword Detection**: Uses Porcupine for accurate wake-word detection
- **Speech Recognition**: Integrates Vosk for speech-to-text conversion
- **Dual Mode Operation**: Supports both live microphone and audio file processing
- **Customizable Hotword**: Configure your own wake-word using Porcupine's format
- **Real-time Processing**: Low-latency audio processing pipeline

## 🛠️ Installation

0. Install FFmpeg:
```
https://ffmpeg.org/download.html
```

1. Clone the repository:
```
git clone https://github.com/IvanArsenev/sound-processing
cd sound-processing
```

2. Install dependencies:
```
pip install -r ./requirements.txt
```

3. Download required models:
- Get your Porcupine access key from [Picovoice Console](https://console.picovoice.ai/)
- Download Vosk language model (small Russian model included in example):
```
wget https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip
unzip vosk-model-small-ru-0.22.zip
```

## 📦 Project Structure
- `main.py`: Core implementation with hotword detection and speech recognition
- `config.py`: Configuration file for Porcupine access key
- `stones.ppn`: Example Porcupine hotword model file

## 🚀 Usage

### Live Mode (Microphone Input)
```
python main.py --mode=live
```

### File Mode (Audio File Processing)
```
python main.py --mode=file --path=./message.wav
```

### Radio Mode (Broadcast listening)
```
python main.py --mode=radio
```

## ⚙️ Parameters

| Parameter | Description                                 | Required |
|-----------|---------------------------------------------|----------|
| `--mode` | Operation mode: `live`, `radio` or `file`   | Yes |
| `--path` | Path to audio file (required for file mode) | Only in file mode |

## 🔧 Configuration
1. Edit `config.py` to add your Porcupine access key:
```python
ACCESS_KEY = "your-access-key-here"
```

2. Place your custom Porcupine hotword model (.ppn file) in the project root and update `CUSTOM_KEYWORD_PATH` in `main.py` if needed.

## 📝 Example Output
```
2023-11-15 14:30:45,123 - INFO - Porcupine Initialization...Done!
2023-11-15 14:30:45,456 - INFO - Listening for hotword...
2023-11-15 14:31:22,789 - INFO - Hotword detected! Next word recognition...
2023-11-15 14:31:23,012 - INFO - Partial: привет
2023-11-15 14:31:23,345 - INFO - Recognized: привет мир
2023-11-15 14:31:23,522 - INFO - Listening for hotword...
```

## 🖼️ Dataset for hot-word created on picovoice.ai
Open platform for datasets: `https://console.picovoice.ai/`

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements.

## 📜 License
[MIT License](LICENSE)