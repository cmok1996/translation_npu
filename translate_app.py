import gradio as gr
from transformers import pipeline
from kokoro import KPipeline
import soundfile as sf
import tempfile
import requests
import json
import numpy as np
import os
import librosa
import whisper
from whisper.tokenizer import LANGUAGES
import openai
import openvino_genai
from pathlib import Path
from pydub import AudioSegment
import io
import os
from enum import IntEnum
import openvino_genai as ov_genai
import time
from transformers import AutoTokenizer

import threading
import queue


# Load Whisper locally for transcription
# whisper_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small")
whisper_pipeline = whisper.load_model("small")
DEVICE = "NPU"
MODEL_DIR = "C:/Users/aicoe/llm_openvino2025.2/models"
STT_MODEL = "openai/whisper-medium"
WHISPER_RUNTIME = 'openvino_genai' #'oga' #'dpais'
if DEVICE == "NPU":
    if WHISPER_RUNTIME == 'openvino_genai':
        model_dir = os.path.join(MODEL_DIR, STT_MODEL)
        whisper_pipe = openvino_genai.WhisperPipeline(model_dir, DEVICE, CACHE_DIR = ".npucache")
        # WHISPER_BASE_URL  = "http://127.0.0.1:8553/v1/openai" #"http://localhost:8000/v3" #

    elif WHISPER_RUNTIME == 'dpais':
        # WHISPER_BASE_URL  =  #"http://localhost:8000/v3" #
        whisper_client = openai.OpenAI(
            base_url = "http://127.0.0.1:8553/v1/openai",
            api_key = "dpais"
            # streaming = True
        )
    
else:
    whisper_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small")
    whisper_tokenizer = None

LLM_BASE_URL = "http://localhost:8000/v3"
API_KEY = 'dpais'

llm_client = openai.OpenAI(
    base_url = LLM_BASE_URL,
    api_key = API_KEY,
    # streaming = True
)
MODEL =  "aisingapore/Llama-SEA-LION-v3-8B-IT-4bit-group-sym-ov" # "microsoft/Phi-4-mini-instruct-4bit-group-sym-ov"" #   ###"Qwen/Qwen2.5-7B-Instruct-4bit-sym-ov" #"aisingapore/Llama-SEA-LION-v3-8B-IT-4bit-group-sym-ov" #"aisingapore/Llama-SEA-LION-v3.5-8B-R-4bit-sym-ov" # "aisingapore/Llama-SEA-LION-v3-8B-IT-4bit-group-sym-ov" #"Qwen\Qwen3-4B-4bit-sym-ov"  #'qwen2.5' #"csalab/sealion3.5:8b-Q4_K_M" #"aisingapore/Llama-SEA-LION-v3.5-8B-R"
LANGUAGES_TO_CODE = {v:k for k,v in LANGUAGES.items()}
CHUNK_LENGTH_MS = 30 * 1000
STEP_SIZE_MS = 30 * 1000

os.makedirs('tmp', exist_ok = True)



# --- Audio transcription ---
def load_audio_file(audio_file):
    if audio_file is None:
        print("Warning: audio_file is None")
        return None

    print(f"loading audio file: {audio_file}")

    if not os.path.exists(audio_file):
        print(f"Error: File does not exist: {audio_file}")
        return None
    
    try:
        audio, sr = librosa.load(audio_file, sr = 16000)
        print(f"Audio loaded successfully: {len(audio)} samples at {sr}Hz")

        with tempfile.NamedTemporaryFile(delete=False, suffix = ".wav") as temp_file:
            sf.write(temp_file.name, audio, sr)
            return temp_file.name
    except Exception as e:
        print(f"Error loading audio file: {str(e)}")
        return None

def detect_audio_language(audio_file):
    audio = whisper.load_audio(audio_file)
    audio = whisper.pad_or_trim(audio)  # make sure it's 30 seconds
    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio, n_mels=whisper_pipeline.dims.n_mels).to(whisper_pipeline.device)
    # detect the spoken language
    _, probs = whisper_pipeline.detect_language(mel)
    lang_code = next(iter({max(probs, key=probs.get)}))
    lang = LANGUAGES[lang_code].title()
    return lang

def chunk_audio(audio, chunk_length_ms, step_size_ms):
    chunks = []
    i=1
    for start_ms in range(0, len(audio), chunk_length_ms):
        if start_ms > len(audio):
            break
        end_ms = start_ms + chunk_length_ms
        chunk = audio[start_ms:end_ms]
        chunk_path = f'tmp/chunk_{i}.wav'
        chunk.export(chunk_path, format = "wav")
        # chunks.append((start_ms, chunk))
        chunks.append(chunk_path)
        i += 1
    return chunks

def transcribe_chunks(chunks):
    out = ''
    for chunk in chunks:
        audio_file = open(chunk, "rb")
        transcription = whisper_client.audio.transcriptions.create(model = "whisper", file = audio_file)
        out += transcription.text
        yield out

def audio_transcription_streaming(audio_file, source_lang):
    if audio_file is None:
        return "No audio file provided"
    if audio_file.split('.')[-1] == 'wav':
        audio = AudioSegment.from_wav(audio_file)
    elif audio_file.split('.')[-1] == 'mp3':
        audio_file = open(audio_file, "rb")
        audio = AudioSegment.from_file(io.BytesIO(audio_file.read()), format="mp3")

    chunks = chunk_audio(audio, chunk_length_ms = CHUNK_LENGTH_MS, step_size_ms=STEP_SIZE_MS)
    out = ''
    for chunk in chunks:
        if WHISPER_RUNTIME == 'openvino_genai':
            def read_wav(filepath):
                raw_speech, samplerate = librosa.load(chunk, sr=16000)
                return raw_speech.tolist()
            raw_speech = read_wav(chunk)
            lang_code = LANGUAGES_TO_CODE[source_lang.lower()]
            response = whisper_pipe.generate(raw_speech, language = f"<|{lang_code}|>")
            out += response.texts[0]
            yield out
        elif WHISPER_RUNTIME == 'dpais':
            # client_dpais = openai.OpenAI(api_key = "dpais", base_url = "http://127.0.0.1:8553/v1/openai")
            audio_file = open(chunk, "rb")
            transcription = whisper_client.audio.transcriptions.create(model = "whisper", file = audio_file)
            out += transcription.text
            yield out


def audio_transcription(audio_file, source_language):
    if audio_file is None:
        return "No audio file provided"

    # if isinstance(audio_file, tuple):
    #     sample_rate, audio_data = audio_file
    #     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
    #         sf.write(temp_file.name, audio_data, sample_rate)
    #         processed_audio = temp_file.name
    # else:
    #     processed_audio = load_audio_file(audio_file)

    # if processed_audio is None:
    #     return "Error processing audio file"
    
    try:
        # result = whisper_pipeline(processed_audio, return_timestamps = True)["text"]
        if DEVICE == "NPU":
            def read_wav(filepath):
                raw_speech, samplerate = librosa.load(filepath, sr=16000)
                return raw_speech.tolist()

            # if cache_dir is None:
            #     cache_dir = model_dir / "cache"
            # cache_dir = Path(cache_dir)
            # cache_dir.mkdir(parents=True, exist_ok=True)
            # pipeline_config = {'CACHE_DIR': ".npucache"}
           
            # Pipeline expects normalized audio with Sample Rate of 16kHz
            raw_speech = read_wav(audio_file)
            def streamer(subword):
                print(subword, end='', flush=True)
                yield subword
                # while True:
                #     x = yield  # wait to receive a value
                #     yield x  # yield it back to the caller
            # streamer = lambda x: print(x, end='', flush=True)


            response = whisper_pipe.generate(raw_speech)
            result = response.texts[0]

            # out = ''
            # for chunk in result:
            #     out += (chunk.choices[0].delta.content or "")
            #     yield out

            # audio_file_read = open(audio_file, "rb")
            # transcript = client.audio.translations.create(   model="whisper",   file=audio_file_read )
            # result = transcript.text
        else:
            lang_code = LANGUAGES_TO_CODE[source_language.lower()]
            result = whisper_pipe.transcribe(audio_file, language = lang_code)['text']

        return result
    except Exception as e:
        print(f"Transcription error: {str(e)}")
        return f"Transcription error: {str(e)}"

# --- Translation via Ollama ---
def text_translation(text, source_lang, target_lang, stream = True):
    prompt = f"""You are a translator. Translate the following text from {source_lang} to {target_lang}. Only return the translated sentence.
Text: {text}
"""

    start_time = time.time()

    response = llm_client.chat.completions.create(
        model=MODEL,  # Replace with any model
        messages=[
            {"role": "user", "content": f"{prompt}"}
        ],
        temperature=0.7,
        stream = stream
    )
    
    # response = requests.post(
    #     "http://localhost:11434/api/generate",
    #     headers={"Content-Type": "application/json"},
    #     data=json.dumps({
    #         "model": "aisingapore/Llama-SEA-LION-v3.5-8B-R",  # change this if you use mistral or another
    #         "prompt": prompt,
    #         "stream": False
    #     })
    # )

    # result = response.json()
    # raw_text = result["response"].strip()
    num_tokens = 0
    stream_times = []
    if stream == True:
        
        out = ''
        for chunk in response:
            out += (chunk.choices[0].delta.content or "")
            yield out
    else:
        raw_text = response.choices[0].message.content

        if '<think>' in raw_text:
            processed_text = raw_text.split("\n</think>\n\n")[-1]
        else:
            return raw_text
        return processed_text

    


# --- Text-to-speech with Kokoro ---
    
def text_to_speech(text, target_language):
    # Load Kokoro TTS locally
    lang_code_mapping = {"English": "a",
                         "Spanish": 'e', 
                         "French": 'f',
                         "Hindi": "h",
                         "Italian": "i",
                         "Japanese": "j",
                         "Brazilian": 'p',
                         "Chinese": 'z'}
    lang_voice_mapping = {"English": "af_heart",
                         "Spanish": 'ef_dora', 
                         "French": 'im_nicola',
                         "Tamil": "hf_alpha",
                         "Hindi": "hf_alpha",
                         "Italian": "if_sara",
                         "Japanese": "jm_kumo",
                         "Brazilian": 'pf_santa',
                         "Chinese": 'zm_yunjian'}
    lang_code = lang_code_mapping.get(target_language)
    if lang_code is None:
        return "Language not supported for voice output"
    tts_engine =  KPipeline(lang_code=lang_code)
    audio_chunks = []
    
    if target_language in ('Chinese', 'Japanese'):
        period = '。'
        num_sentences = 5
    else:
        period = '.'
        num_sentences = 8

    sentences = text.split(period)

    chunks = [period.join(sentences[i:i + num_sentences]) for i in range(0, len(sentences), num_sentences)]

    for text_chunk in chunks:
        
        generator = tts_engine(text_chunk, voice=lang_voice_mapping[target_language])
    
        # Collect all audio chunks
        for i, (gs, ps, audio) in enumerate(generator):
            audio_chunks.append(audio)

    # Concatenate all chunks into one waveform
    full_audio = np.concatenate(audio_chunks)

    # Save to temp WAV file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        sf.write(f.name, full_audio, samplerate=24000)
        return f.name

        
    # wav = tts_engine.tts(text)
    # with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
    #     sf.write(f.name, wav["audio"], samplerate=wav["sampling_rate"])
    #     return f.name

# --- Summarization function ---

def text_summarization_streaming(text, summary_length="medium",  target_lang = "English"):
    length_prompts = {
        "short": "Provide a brief 1-2 sentence summary.",
        "medium": "Provide a concise 3-4 sentence summary.",
        "long": "Provide a detailed summary with key points."
    }
    
    prompt = f"""You are a text summarizer. Your task is to summarize the text below to {target_lang} regardless of the language of the text. {length_prompts[summary_length]}
Text to summarize: {text}
Summary:"""
    
    response = llm_client.chat.completions.create(
        model=MODEL,  # Replace with any model you’ve pulled in Ollama
        messages=[
            {"role": "user", "content": f"{prompt}"}
        ],
        temperature=0.7,
        stream = True
    )

    out = ''
    for chunk in response:
        out += (chunk.choices[0].delta.content or "")
        yield out


# def summarize_audio(audio_file, source_lang, summary_length):
#     if source_lang == "Auto-Detect":
#         source_lang = detect_audio_language(audio_file)
#     transcription = audio_transcription_streaming(audio_file)
#     if transcription.startswith("No audio file") or transcription.startswith("Error"):
#         return transcription, ""
    
#     summary = text_summarization(transcription, summary_length)
#     return transcription, summary

# # --- Summarization  ---
# def text_summarization(text, summary_length="medium"):
#     length_prompts = {
#         "short": "Provide a brief 1-2 sentence summary.",
#         "medium": "Provide a concise 3-4 sentence summary.",
#         "long": "Provide a detailed summary with key points."
#     }
    
#     prompt = f"""You are a text summarizer. Your task is to summarize the text below to English regardless of the language of the text. {length_prompts[summary_length]}
# Text to summarize: {text}
# Summary:"""
    
#     response = llm_client.chat.completions.create(
#         model=MODEL,  # Replace with any model you’ve pulled in Ollama
#         messages=[
#             {"role": "user", "content": f"{prompt}"}
#         ],
#         temperature=0.7,
#     )

#     # response = requests.post(
#     #     "http://localhost:11434/api/generate",
#     #     headers={"Content-Type": "application/json"},
#     #     data=json.dumps({
#     #         "model": "aisingapore/Llama-SEA-LION-v3.5-8B-R",
#     #         "prompt": prompt,
#     #         "stream": False
#     #     })
#     # )
#     # result = response.json()
#     # raw_text = result["response"].strip()

#     raw_text = response.choices[0].message.content
#     if '<think>' in raw_text:
#         processed_text = raw_text.split("\n</think>\n\n")[-1]
#     else:
#         return raw_text
#     return processed_text


def get_source_language( audio_file, source_language,):
    if source_language == "Auto-Detect":
        source_language = detect_audio_language(audio_file)
        print(f"Detected language: {source_language}")
    return source_language
    
# --- Main pipeline ---
def voice_to_voice(audio_file, source_language, target_language):
    processed_audio = load_audio_file(audio_file)
    print(f'Done loading audio')
    source_language = get_source_language(processed_audio, source_language)
    # if source_language == "Auto-Detect":
    #     source_language = detect_audio_language(processed_audio)
    #     print(f"Detected language: {source_language}")

    transcription = audio_transcription_streaming(audio_file, source_language)
    print(f"Done transcribing in {source_language}")

    translated_text = text_translation(transcription, source_language, target_language)
    print(f"Done translating from {source_language} to {target_language}")

    speech_file = text_to_speech(translated_text, target_language)
    print("Done processing text to speech")
    return transcription, translated_text, speech_file

# Gradio UI

# Create multiple interfaces using Gradio Blocks
with gr.Blocks(title="Voice Translation & Meeting Transcription") as demo:
    gr.Markdown("# Voice Translation & Meeting Transcription App")
    gr.Markdown("Upload audio files (MP3, WAV, etc.) for translation or transcription")
    
    # with gr.Tabs():
    #     # Voice Translation Tab
    #     with gr.TabItem("Voice Translation"):
    #         gr.Markdown("### Translate speech from one language to another")
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                sources=["microphone", "upload"], 
                type="filepath", 
                label="Input Speech (Record or Upload MP3/WAV)"
            )
            source_lang = gr.Dropdown(
                # choices=["Auto-Detect", "English", "Malay", "Indonesian", "Spanish", "French", "Hindi", "Italian", "Japanese", "Chinese"],
                choices=["Auto-Detect", "English", "Chinese", "Indonesian", "Malay", "Tagalog", "Burmese", "Vietnamese", "Thai", "Lao", "Tamil", "Khmer"],
                value="Auto-Detect",
                label="Source Language"
            )
            target_lang = gr.Dropdown(
                # choices=["English", "Chinese", "Indonesian", "Malay", "Filipino", "Burmese", "Vietnamese", "Thai", "Lao", "Tamil", "Khmer"],
                choices=["Auto-Detect", "English", "Chinese", "Indonesian", "Malay", "Tagalog", "Burmese", "Vietnamese", "Thai", "Lao", "Tamil", "Khmer"],
                value="English",
                label="Target Language"
            )
            summary_length = gr.Dropdown(
                choices = ['short', 'medium', 'long'],
                value = "medium",
                label = "Summary length"
            )

            # Action buttons
            with gr.Row():
                overall_btn = gr.Button("GO", variant="primary")
            with gr.Row():
                transcribe_btn = gr.Button("Transcribe", variant="secondary")
                translate_btn = gr.Button("Translate", variant="secondary")
                summarize_btn = gr.Button("Summarize", variant="secondary")
                speech_btn = gr.Button("Speech", variant="secondary")
            with gr.Row():
                clear_btn = gr.Button("Clear", variant="secondary")
                

            # translate_btn = gr.Button("Translate", variant="primary")
        
        with gr.Column():
            # Output section
            transcription_output = gr.Textbox(
                label="Transcription", 
                lines=4,
                placeholder="Transcribed text will appear here..."
            )
            translation_output = gr.Textbox(
                label="Translation", 
                lines=4,
                placeholder="Translated text will appear here..."
            )
            voice_output = gr.Audio(label="Translated Voice Output")
            summary_output = gr.Textbox(
                label="Summary", 
                lines=4, 
                placeholder="Summary will appear here..."
            )
    
    overall_btn.click(
        fn=get_source_language,
        inputs=[audio_input, source_lang],
        outputs=[source_lang]
    ) \
    .then(fn=lambda: ("", "", None, ""), outputs=[transcription_output, translation_output, voice_output,  summary_output]) \
    .then(fn = audio_transcription_streaming, inputs = [audio_input, source_lang], outputs = [transcription_output]) \
    .then(fn = text_translation, inputs = [transcription_output, source_lang, target_lang], outputs = [translation_output]) \
    .then(fn = text_to_speech, inputs = [translation_output, target_lang], outputs = [voice_output]) \
    .then(fn = text_summarization_streaming, inputs = [transcription_output, summary_length, target_lang], outputs = [summary_output]) \

    transcribe_btn.click(
        fn=get_source_language,
        inputs=[audio_input, source_lang],
        outputs=[source_lang]
    ) \
    .then(
        fn=audio_transcription_streaming,
        inputs=[audio_input, source_lang],
        outputs=transcription_output
    )

    translate_btn.click(
        fn = text_translation, inputs = [transcription_output, source_lang, target_lang], outputs = [translation_output]
    ) 
    
    summarize_btn.click(
        fn=get_source_language,
        inputs=[audio_input, source_lang],
        outputs=[source_lang]
    ) \
    .then(fn = text_summarization_streaming, inputs = [transcription_output, summary_length, target_lang], outputs = [summary_output])

    speech_btn.click(
        fn=text_to_speech,
        inputs=[translation_output, target_lang],
        outputs=[voice_output]
    ) 

    clear_btn.click(
            fn=lambda: ("", "", None, ""),
            outputs=[transcription_output, translation_output, voice_output,  summary_output]
        )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True to create a public link
        debug=True
    )
