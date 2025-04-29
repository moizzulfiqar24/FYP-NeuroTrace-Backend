import os
import shutil
import glob
import re
import uuid
import pickle
from datetime import datetime
import faiss
import librosa
import soundfile as sf
import noisereduce as nr
import whisper
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from pydub import AudioSegment
import tempfile
import traceback
TEMP_DIR = tempfile.gettempdir()


nltk.download('punkt')
nltk.download('punkt_tab')

# ============================== CONFIG ===================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WHISPER_MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "../embedding_model/Speech To Text Model/tiny.en.pt"))
BASE_PATH = os.path.abspath(os.path.join(BASE_DIR, "../DB"))
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# ============================== PATH HELPERS ==============================

def ensure_dirs(*dirs):
    for d in dirs:
        try:
            os.makedirs(d, exist_ok=True)
        except Exception as e:
            print(f"❌ Failed to create directory '{d}': {e}")

def get_paths(user_id):
    todo = os.path.join(BASE_PATH, "ToDo", str(user_id))
    processed = os.path.join(BASE_PATH, "Processed", str(user_id))
    vector = os.path.join(BASE_PATH, "VectorStore", str(user_id))

    return {
        "todo_audio": os.path.join(todo, "AudioFiles"),
        "todo_objects": os.path.join(todo, "Objects"),
        "todo_transcripts": os.path.join(todo, "Transcripts"),
        "todo_chunks": os.path.join(todo, "Chunks"),

        "processed_audio": os.path.join(processed, "ProcessedAudioFiles"),
        "processed_transcripts": os.path.join(processed, "ProcessedTranscripts"),
        "processed_chunks": os.path.join(processed, "ProcessedChunks"),
        "processed_objects": os.path.join(processed, "ProcessedObjects"),

        "vector_path": vector,
        "faiss_index": os.path.join(vector, "chunk_index.faiss"),
        "meta_path": os.path.join(vector, "chunk_metadata.pkl"),

        "todo_root": todo,
        "processed_root": processed
    }

# ============================== AUDIO CLEANING ==============================

def clean_audio(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    y = bandpass_filter(y, sr)
    y = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.7, stationary=True)
    return y, sr

def bandpass_filter(data, sr, lowcut=300, highcut=3400):
    from scipy.signal import butter, lfilter
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(5, [low, high], btype='band')
    return lfilter(b, a, data)

# ============================== TEXT UTILITIES ==============================

def transcribe_audio(file_path, model):
    result = model.transcribe(file_path)
    return result['text'].strip().replace('\n', ' ')

def chunk_text(text, max_words=200):
    sentences = sent_tokenize(text)
    chunks, current, count = [], "", 0
    for sentence in sentences:
        words = word_tokenize(sentence)
        if count + len(words) <= max_words:
            current += ' ' + sentence
            count += len(words)
        else:
            chunks.append(current.strip())
            current = sentence
            count = len(words)
    if current:
        chunks.append(current.strip())
    return chunks

# ============================== OBJECT PARSING ==============================

def parse_object_output(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(path, 'r', encoding='latin-1') as f:
            content = f.read()
    summary_match = re.search(r"===== FINAL SUMMARY =====.*?Final Scene Prediction.*?: (.+)", content, re.S)
    objects = re.findall(r"\s+- (\w+): \d+ time\(s\)", content)
    location = summary_match.group(1).strip() if summary_match else "Unknown"
    obj_text = f"Detected in {location}: " + ', '.join(set(objects))
    return obj_text

# ============================== VECTOR DB ==============================

def update_vector_db(texts, paths, model):
    if os.path.isdir(paths["faiss_index"]):
        raise RuntimeError(f"Expected FAISS index file but found a directory: {paths['faiss_index']}")

    if os.path.exists(paths["faiss_index"]):
        index = faiss.read_index(paths["faiss_index"])
        with open(paths["meta_path"], 'rb') as f:
            metadata = pickle.load(f)
    else:
        index = faiss.IndexFlatL2(384)
        metadata = []

    for txt in texts:
        emb = model.encode([txt])[0]
        index.add(emb.reshape(1, -1))
        metadata.append({"text": txt, "timestamp": datetime.now().isoformat()})

    faiss.write_index(index, paths["faiss_index"])
    with open(paths["meta_path"], 'wb') as f:
        pickle.dump(metadata, f)

# ============================== CLEANUP HELPERS ==============================

def clear_files_in_folder(folder_path):
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

# ============================== INGESTION FUNCTIONS ==============================

def ingest_audio_file(user_id: str, audio_file_path: str):
    paths = get_paths(user_id)

    # Ensure all directories are created
    ensure_dirs(
        paths["todo_audio"],
        paths["todo_objects"],
        paths["todo_transcripts"],
        paths["todo_chunks"]
    )

    if not os.path.isfile(audio_file_path):
        raise FileNotFoundError(f"❌ Provided audio file not found: {audio_file_path}")

    filename = os.path.basename(audio_file_path)

    if not filename.lower().endswith(".wav"):
        raise ValueError("❌ Only .wav files are supported.")

    wav_path = os.path.join(paths["todo_audio"], filename)
    shutil.copy2(audio_file_path, wav_path)

    print(f"📥 WAV file '{filename}' ingested for user {user_id} at {wav_path}.")


def ingest_text_file(user_id: str, text_file_path: str):
    try:
        uuid.UUID(user_id)
    except ValueError:
        raise ValueError("Invalid UUID format for user_id.")

    paths = get_paths(user_id)
    ensure_dirs(paths["todo_objects"], paths["todo_audio"], paths["todo_transcripts"], paths["todo_chunks"])

    if not os.path.isfile(text_file_path):
        raise FileNotFoundError(f"Provided text file not found: {text_file_path}")

    destination = os.path.join(paths["todo_objects"], "Output.txt")  # standard expected name
    shutil.copy2(text_file_path, destination)
    print(f"📄 Text file ingested for user {user_id} at {destination}.")

# ============================== PIPELINE ==============================

def run_pipeline(user_id: str):
    paths = get_paths(user_id)

    ensure_dirs(
        paths["todo_audio"],
        paths["todo_objects"],
        paths["todo_transcripts"],
        paths["todo_chunks"],
        paths["processed_audio"],
        paths["processed_transcripts"],
        paths["processed_chunks"],
        paths["processed_objects"],
        paths["vector_path"]
    )

    whisper_model = whisper.load_model(WHISPER_MODEL_PATH)
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    audio_files = glob.glob(os.path.join(paths["todo_audio"], "*.wav"))
    object_txt_path = os.path.join(paths["todo_objects"], "Output.txt")

    audio_texts = []
    staged_audio = []
    staged_transcripts = []
    staged_chunks = []
    staged_object_txt = None

    TEMP_DIR = tempfile.gettempdir()  # Cross-platform temp folder

    try:
        # === AUDIO PROCESSING ===
        for file in tqdm(audio_files, desc="🔊 Processing audio"):
            print(f"🎧 Processing file: {file}")
            cleaned_audio, sr = clean_audio(file)

            cleaned_filename = os.path.basename(file).replace(".wav", "_cleaned.wav")
            temp_cleaned_path = os.path.join(TEMP_DIR, cleaned_filename)
            print(f"🔧 Writing cleaned audio to: {temp_cleaned_path}")

            try:
                sf.write(temp_cleaned_path, cleaned_audio, sr)
            except Exception as e:
                print(f"❌ Failed to write cleaned WAV file: {e}")
                traceback.print_exc()
                raise

            name = os.path.splitext(cleaned_filename)[0]
            transcript = transcribe_audio(temp_cleaned_path, whisper_model)
            transcript_filename = f"{name}.txt"
            temp_transcript_path = os.path.join(TEMP_DIR, transcript_filename)
            print(f"📝 Saving transcript to: {temp_transcript_path}")

            with open(temp_transcript_path, "w") as f:
                f.write(transcript)
            audio_texts.append(transcript)

            # === CHUNKING ===
            chunks = chunk_text(transcript)
            temp_chunk_dir = os.path.join(TEMP_DIR, f"{name}_chunks")
            os.makedirs(temp_chunk_dir, exist_ok=True)
            print(f"📦 Creating chunk directory: {temp_chunk_dir}")

            for i, ch in enumerate(chunks, 1):
                chunk_header = f"({name} chunk{i})"
                chunk_text_full = f"{chunk_header} {ch}"
                chunk_path = os.path.join(temp_chunk_dir, f"{name}_chunk{i}.txt")
                with open(chunk_path, "w") as f:
                    f.write(chunk_text_full)
                audio_texts.append(chunk_text_full)

            staged_audio.append((file, os.path.join(paths["processed_audio"], cleaned_filename)))
            staged_transcripts.append((temp_transcript_path, os.path.join(paths["processed_transcripts"], transcript_filename)))
            staged_chunks.append((temp_chunk_dir, os.path.join(paths["processed_chunks"], name)))

        # === OBJECT DETECTION ===
        if os.path.exists(object_txt_path):
            print("📦 Processing object detection file...")
            parsed_text = parse_object_output(object_txt_path)
            audio_texts.append(parsed_text)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_object_path = os.path.join(paths["processed_objects"], f"Output_{timestamp}.txt")
            staged_object_txt = (object_txt_path, final_object_path)

        # === VECTOR DB UPDATE ===
        print("🧠 Updating vector database...")
        if audio_texts:
            update_vector_db(audio_texts, paths, embed_model)

        # === FINAL COMMIT ===
        print("📁 Committing processed files...")

        for src, dst in staged_audio:
            print(f"📤 Moving cleaned audio: {src} → {dst}")
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                print(f"❌ Failed to copy cleaned audio file: {e}")
                traceback.print_exc()
                raise

        for src, dst in staged_transcripts:
            print(f"📤 Moving transcript: {src} → {dst}")
            shutil.copy2(src, dst)

        for src_dir, dst_dir in staged_chunks:
            print(f"📤 Copying chunk directory: {src_dir} → {dst_dir}")
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

        if staged_object_txt:
            print(f"📤 Moving object detection file: {staged_object_txt[0]} → {staged_object_txt[1]}")
            shutil.move(staged_object_txt[0], staged_object_txt[1])

        print("🧹 Cleaning up ToDo folder...")
        clear_files_in_folder(paths["todo_audio"])
        clear_files_in_folder(paths["todo_objects"])
        clear_files_in_folder(paths["todo_transcripts"])
        clear_files_in_folder(paths["todo_chunks"])

        print("✅ All processing completed successfully.")

    except Exception as e:
        print(f"❌ ERROR: {e}")
        traceback.print_exc()
        print("⚠️ Aborted. No data was moved or deleted.")

# ============================== ENTRYPOINTS ==============================

def main_audio(user_id: str, audio_path: str):
    ingest_audio_file(user_id, audio_path)
    run_pipeline(user_id)

def main_text(user_id: str, text_path: str):
    ingest_text_file(user_id, text_path)
    run_pipeline(user_id)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest and process an audio or text file for a user.")
    parser.add_argument("--user_id", required=True, help="UUID of the user")
    parser.add_argument("--file_path", required=True, help="Path to the audio or text file")
    parser.add_argument("--type", required=True, choices=["audio", "text"], help="Type of file to ingest")

    args = parser.parse_args()

    if args.type == "audio":
        main_audio(args.user_id, args.file_path)
    else:
        main_text(args.user_id, args.file_path)
