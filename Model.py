import argparse
import os
import sys
import numpy as np
import speech_recognition as sr
import whisper
import torch
from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor

# Device setup for translation model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Translation model parameters
src_lang, tgt_lang = "eng_Latn", "tam_Taml"
model_name = "ai4bharat/indictrans2-en-indic-dist-200M"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

translation_model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,  # performance might slightly vary for bfloat16
    attn_implementation="flash_attention_2"
).to(DEVICE)

ip = IndicProcessor(inference=True)

# Function to list microphones
def list_microphones():
    available_mics = sr.Microphone.list_microphone_names()
    print("Available microphones:")
    for index, name in enumerate(available_mics):
        print(f"Device {index}: {name}")
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=30,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=0.1,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    parser.add_argument("--device_index", type=int, default=None,
                        help="Specific device index for microphone")
    parser.add_argument("--mock_audio", type=str, default=None,
                        help="Path to mock audio file for testing")
    parser.add_argument("--list-devices", action="store_true",
                        help="List all available microphone devices")
    args = parser.parse_args()

    phrase_time = None
    data_queue = Queue()

    # Initialize speech recognizer
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False

    source = None
    
    if args.list_devices:
        list_microphones()

    # Check for mock audio file
    if args.mock_audio and os.path.exists(args.mock_audio):
        print(f"Using mock audio file: {args.mock_audio}")
        source = sr.AudioFile(args.mock_audio)
    else:
        # Find a working microphone
        available_mics = sr.Microphone.list_microphone_names()
        if not available_mics:
            print("No microphones detected.")
            sys.exit(1)
        
        print("Available microphones:")
        for index, name in enumerate(available_mics):
            print(f"Device {index}: {name}")
        
        # Try specified device index
        if args.device_index is not None:
            try:
                source = sr.Microphone(sample_rate=16000, device_index=args.device_index)
                print(f"Using specified device index: {args.device_index}")
            except Exception as e:
                print(f"Failed to use specified device: {e}")
        
        # Try first available device
        if source is None:
            try:
                source = sr.Microphone(sample_rate=16000, device_index=0)
                print("Using first available microphone")
            except Exception as e:
                print(f"Failed to initialize microphone: {e}")
                sys.exit(1)

    # Load Whisper model
    whisper_model = args.model
    if args.model != "large" and not args.non_english:
        whisper_model = whisper_model + ".en"
    whisper_model_instance = whisper.load_model(whisper_model)

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    transcription = ['']

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio: sr.AudioData) -> None:
        """ Callback to receive audio data when recordings finish. """
        data = audio.get_raw_data()
        data_queue.put(data)

    # Start background listening
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    print("Model loaded.\n")

    while True:
        try:
            now = datetime.utcnow()
            if not data_queue.empty():
                phrase_complete = False
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    phrase_complete = True
                phrase_time = now
                
                # Combine audio data
                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                # Get transcription from Whisper model
                result = whisper_model_instance.transcribe(audio_np, fp16=torch.cuda.is_available())
                text = result['text'].strip()

                # Update transcription list
                if phrase_complete:
                    transcription.append(text)
                else:
                    transcription[-1] = text

                # Only translate the current transcription
                input_sentences = [text]
                batch = ip.preprocess_batch(input_sentences, src_lang=src_lang, tgt_lang=tgt_lang)

                # Tokenize and generate input encodings
                inputs = tokenizer(
                    batch,
                    truncation=True,
                    padding="longest",
                    return_tensors="pt",
                    return_attention_mask=True,
                ).to(DEVICE)

                # Generate translations using the translation model
                with torch.no_grad():
                    generated_tokens = translation_model.generate(
                        **inputs,
                        use_cache=True,
                        min_length=0,
                        max_length=256,
                        num_beams=5,
                        num_return_sequences=1,
                    )

                # Decode translations
                with tokenizer.as_target_tokenizer():
                    generated_tokens = tokenizer.batch_decode(
                        generated_tokens.detach().cpu().tolist(),
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    )

                # Postprocess translations
                translations = ip.postprocess_batch(generated_tokens, lang=tgt_lang)

                # Clear console and print the translated text live
                os.system('cls' if os.name == 'nt' else 'clear')
                for translation in translations:
                    print(f"{tgt_lang}: {translation}")
                print('', end='', flush=True)

            else:
                sleep(0.25)
        except KeyboardInterrupt:
            break

  

if __name__ == "__main__":
    main()
