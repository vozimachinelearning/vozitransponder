from transformers import MarianMTModel, MarianTokenizer
import torch
import threading
import time
import requests
from huggingface_hub import HfFolder, hf_hub_download
from pathlib import Path

class MarianTranslator:
    _instance = None
    _models = {}
    _last_used = {}
    MAX_CACHED_MODELS = 2
    DOWNLOAD_TIMEOUT = 30  # 30 seconds timeout for model downloads
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.device = "cuda" if torch.cuda.is_available() else "cpu"
            cls._instance.initialized = False
            # Set up cache directory
            cls._instance.cache_dir = Path.home() / ".cache" / "huggingface"
            cls._instance.cache_dir.mkdir(parents=True, exist_ok=True)
        return cls._instance

    def __init__(self):
        if not self.initialized:
            self.initialized = True
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {self.device}")
            self.preload_models(['en', 'es'])

    def _get_model_name(self, source_lang, target_lang):
        if source_lang == target_lang:
            return None
            
        special_cases = {
            ('en', 'es'): 'Helsinki-NLP/opus-mt-en-es',
            ('es', 'en'): 'Helsinki-NLP/opus-mt-es-en',
        }
        
        lang_pair = (source_lang, target_lang)
        return special_cases.get(lang_pair, f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}')

    def _cleanup_old_models(self):
        if len(self._models) > self.MAX_CACHED_MODELS:
            oldest = min(self._last_used.items(), key=lambda x: x[1])[0]
            del self._models[oldest]
            del self._last_used[oldest]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("CUDA cache cleared")

    def _load_model(self, source_lang, target_lang):
        if source_lang == target_lang:
            return None, None
            
        model_key = f"{source_lang}-{target_lang}"
        current_time = time.time()
        
        if model_key in self._models:
            self._last_used[model_key] = current_time
            print(f"Using cached model for {model_key}")
            return self._models[model_key]['model'], self._models[model_key]['tokenizer']
        
        self._cleanup_old_models()
        model_name = self._get_model_name(source_lang, target_lang)
        print(f"Loading model: {model_name}")
        
        try:
            # First try to load from local cache
            try:
                print(f"Attempting to load from local cache: {model_name}")
                tokenizer = MarianTokenizer.from_pretrained(
                    model_name,
                    local_files_only=True,
                    cache_dir=str(self.cache_dir)
                )
                model = MarianMTModel.from_pretrained(
                    model_name,
                    local_files_only=True,
                    cache_dir=str(self.cache_dir)
                ).to(self.device)
                print("Successfully loaded from local cache")
            except Exception as cache_error:
                print(f"Cache load failed: {cache_error}. Attempting download...")
                # Set timeout for model download
                with requests.Session() as session:
                    session.request = lambda *args, **kwargs: requests.Session.request(
                        session, *args, **{**kwargs, 'timeout': self.DOWNLOAD_TIMEOUT}
                    )
                    
                    print(f"Downloading tokenizer for {model_name}...")
                    tokenizer = MarianTokenizer.from_pretrained(
                        model_name,
                        local_files_only=False,
                        cache_dir=str(self.cache_dir)
                    )
                    
                    print(f"Downloading model for {model_name}...")
                    model = MarianMTModel.from_pretrained(
                        model_name,
                        local_files_only=False,
                        cache_dir=str(self.cache_dir)
                    ).to(self.device)
                
            model.eval()
            print(f"Model loaded successfully: {model_name}")
            
            self._models[model_key] = {'model': model, 'tokenizer': tokenizer}
            self._last_used[model_key] = current_time
            return model, tokenizer
                
        except Exception as e:
            print(f"Model loading error for {model_name}: {str(e)}")
            return None, None

    def preload_models(self, lang_pairs):
        print("Preloading models...")
        for source_lang in lang_pairs:
            for target_lang in lang_pairs:
                if source_lang != target_lang:
                    print(f"Preloading model for {source_lang} -> {target_lang}")
                    self._load_model(source_lang, target_lang)
        print("Model preloading complete")

    def translate_async(self, text, target_lang, source_lang, callback):
        if source_lang == target_lang:
            return

        def run_translation():
            try:
                translated = self._translate(text, target_lang, source_lang)
                callback(translated)
            except Exception as e:
                print(f"Translation error: {e}")

        thread = threading.Thread(target=run_translation)
        thread.start()
        return thread

    def _translate(self, text, target_lang, source_lang):
        print(f"Starting translation: {source_lang} -> {target_lang}")
        print(f"Input text: {text}")
        try:
            model, tokenizer = self._load_model(source_lang, target_lang)
            if not model or not tokenizer:
                print(f"Translation model not available for {source_lang} to {target_lang}")
                return f"Translation model not available for {source_lang} to {target_lang}"

            with torch.no_grad():
                chunks = self._split_text_into_chunks(text, tokenizer, 128)
                translations = []
                for chunk in chunks:
                    inputs = tokenizer(chunk, return_tensors="pt", padding=True).to(self.device)
                    outputs = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)
                    translations.append(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
                result = " ".join(translations)
                print(f"Translation result: {result}")
                return result
        except Exception as e:
            print(f"Translation error: {e}")
            return f"Translation error: {str(e)}"

    def _split_text_into_chunks(self, text, tokenizer, max_length):
        tokens = tokenizer.tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for token in tokens:
            current_chunk.append(token)
            current_length += 1
            if current_length >= max_length:
                chunks.append(tokenizer.convert_tokens_to_string(current_chunk))
                current_chunk = []
                current_length = 0
        if current_chunk:
            chunks.append(tokenizer.convert_tokens_to_string(current_chunk))
        return chunks

def translate_text(text, target_lang, source_lang):
    translator = MarianTranslator()
    return translator._translate(text, target_lang, source_lang)