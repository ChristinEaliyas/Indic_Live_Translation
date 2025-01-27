# Overview

This project provides a real-time speech-to-text solution that listens to speech input, transcribes it, and translates it into a target language. The solution utilizes **Whisper** for speech recognition and **IndicTrans** for language translation. It is designed for dynamic transcription and translation with adjustable settings for audio recording.

## Features

- Real-time transcription: Converts speech into text in real time
- Translation: Translates the transcribed text into a target language (currently supports English to Tamil)
- Microphone Configuration: Choose microphone device and adjust energy threshold for better recording sensitivity
- Mock Audio: Option to use a pre-recorded mock audio file for testing purposes
- Real-time feedback: Displays transcription and translation output in real-time

## Requirements

- Python 3.x
- torch
- whisper
- speechrecognition
- transformers
- IndicTransToolkit
- numpy

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/speech-to-text-real-time-translation.git
cd speech-to-text-real-time-translation
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Alternatively, manually install dependencies:

```bash
pip install torch whisper speechrecognition transformers IndicTransToolkit numpy
```

Ensure you have a compatible microphone and (optional) a CUDA-enabled GPU for better performance.

## Usage

To run the application:

```bash
python speech_to_text_translation.py
```

### Command Line Arguments

- `--model`: Whisper model to use. Available options: "tiny", "base", "small", "medium", "large". Default is "medium"
- `--non_english`: If set, the model will not use the English variant for Whisper
- `--energy_threshold`: Energy level for microphone detection (default: 1000)
- `--record_timeout`: Duration in seconds for recording before stopping (default: 30 seconds)
- `--phrase_timeout`: Duration of silence in seconds before considering a phrase complete (default: 0.1)
- `--device_index`: Specify a microphone device index
- `--mock_audio`: Path to a mock audio file for testing
- `--list-devices`: List available microphone devices

### Example Usage

```bash
python speech_to_text_translation.py --model large --record_timeout 60 --phrase_timeout 0.2
```

### List Available Microphones

```bash
python speech_to_text_translation.py --list-devices
```

### Use Mock Audio File

```bash
python speech_to_text_translation.py --mock_audio "path_to_audio_file.wav"
```

## How It Works

- Microphone Setup: The script starts by either using a real-time microphone or a mock audio file
- Speech Recognition: The Whisper model transcribes the speech to text in real-time
- Translation: The transcribed text is then translated using the IndicTrans translation model (currently from English to Tamil)
- Output: The transcriptions and translations are displayed live in the terminal

## Example

```bash
Available microphones:
Device 0: Built-in Microphone
Device 1: External Microphone

Model loaded.

Hello, how are you?
தமிழ்: வணக்கம், எப்படி இருக்கிறீர்கள்?
```

## Troubleshooting

- No Microphone Detected: Ensure that your microphone is correctly connected and recognized by the system
- Model Issues: If you face issues with model loading, verify the versions of your dependencies, especially torch and the Hugging Face models
- Audio Issues: Ensure your microphone is functioning properly and the energy_threshold is set correctly

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgements

- [Whisper](https://github.com/openai/whisper) by OpenAI for speech-to-text transcription
- [IndicTransToolkit](https://github.com/ai4bharat/IndicTrans) for translation between Indian languages
