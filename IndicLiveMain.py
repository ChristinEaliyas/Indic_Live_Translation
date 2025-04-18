import sys
import numpy as np
import speech_recognition as sr
import whisper
import torch
from datetime import datetime
from queue import Queue
from time import sleep
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor
import streamlit as st

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

language_mapping = {
    "Assamese": "asm_Beng",
    "Bengali": "ben_Beng",
    "Bodo": "brx_Deva",
    "Dogri": "doi_Deva",
    "Gujarati": "guj_Gujr",
    "Hindi": "hin_Deva",
    "Kannada": "kan_Knda",
    "Kashmiri (Arabic)": "kas_Arab",
    "Kashmiri (Devanagari)": "kas_Deva",
    "Konkani": "gom_Deva",
    "Maithili": "mai_Deva",
    "Malayalam": "mal_Mlym",
    "Manipuri (Bengali)": "mni_Beng",
    "Manipuri (Meitei)": "mni_Mtei",
    "Marathi": "mar_Deva",
    "Nepali": "npi_Deva",
    "Odia": "ory_Orya",
    "Punjabi": "pan_Guru",
    "Sanskrit": "san_Deva",
    "Santali": "sat_Olck",
    "Sindhi (Arabic)": "snd_Arab",
    "Sindhi (Devanagari)": "snd_Deva",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Urdu (Arabic)": "urd_Arab"
}

model_name = "ai4bharat/indictrans2-en-indic-dist-200M"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

translation_model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
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
    src_lang = "eng_Latn" 

    st.title("Real-Time Transcription and Translation")
    st.write("Speak into the microphone to get transcription and translation.")

    tgt_lang = st.selectbox("Select Target Language", list(language_mapping.keys()))
    
    tgt_lang_code = language_mapping[tgt_lang]

    transcription_text = st.empty()
    translation_text = st.empty()

    recorder = sr.Recognizer()
    recorder.energy_threshold = 1500
    recorder.dynamic_energy_threshold = False

    available_mics = sr.Microphone.list_microphone_names()
    if not available_mics:
        print("No microphones detected.")
        sys.exit(1)
    
    source = sr.Microphone(sample_rate=16000)

    whisper_model = "base.en"
    whisper_model_instance = whisper.load_model(whisper_model)

    data_queue = Queue()

    def record_callback(_, audio: sr.AudioData) -> None:
        """ Callback to receive audio data when recordings finish. """
        data = audio.get_raw_data()
        data_queue.put(data)

    with source:
        recorder.adjust_for_ambient_noise(source)

    recorder.listen_in_background(source, record_callback, phrase_time_limit=30)

    print("Model loaded.\n")

    st.success("Model Loaded Successfully! You can start speaking now.", icon="✅")
    
    while True:
        try:
            if not data_queue.empty():
                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                result = whisper_model_instance.transcribe(audio_np, fp16=torch.cuda.is_available())
                text = result['text'].strip()

                transcription_text.markdown(f"### Transcription\n\n{text}")

                input_sentences = [text]
                batch = ip.preprocess_batch(input_sentences, src_lang=src_lang, tgt_lang=tgt_lang_code)

                inputs = tokenizer(
                    batch,
                    truncation=True,
                    padding="longest",
                    return_tensors="pt",
                    return_attention_mask=True,
                ).to(DEVICE)

                with torch.no_grad():
                    generated_tokens = translation_model.generate(
                        **inputs,
                        use_cache=True,
                        min_length=0,
                        max_length=256,
                        num_beams=5,
                        num_return_sequences=1,
                    )

                with tokenizer.as_target_tokenizer():
                    generated_tokens = tokenizer.batch_decode(
                        generated_tokens.detach().cpu().tolist(),
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    )

                translations = ip.postprocess_batch(generated_tokens, lang=tgt_lang_code)

                translation_text.markdown(f"### {tgt_lang}\n\n{translations[0]}")

            else:
                sleep(0.25)
        except KeyboardInterrupt:
            break
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
