from dataclasses import dataclass

import torch

LANGUAGE_CODE_TO_NAME = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
}


@dataclass
class AlignedWord:
    word: str
    start: float
    end: float
    confidence: float = 1.0


@dataclass
class TranscriptionResult:
    text: str
    words: list[AlignedWord]
    language: str | None = None


class ASRModel(torch.nn.Module):
    """Base class for ASR models."""

    def forward(
        self,
        audio: torch.Tensor,
        language: str | None = None,
        return_timestamps: bool = True,
    ) -> list[TranscriptionResult]:
        """
        Transcribe batched audio and optionally get word-level timestamps.

        Args:
            audio (torch.Tensor): Audio tensor of shape (B, T) at the model's sampling rate
            language (str, optional): Language code (e.g., 'en', 'fr')
            return_timestamps (bool): If True, return word-level timestamps. If False,
                return only transcriptions without alignment (faster). Default: True.

        Returns:
            list[TranscriptionResult]: List of transcriptions, optionally with word-level
                alignment, one per audio sample in the batch. When return_timestamps is
                False, the words list will be empty.
        """
        raise NotImplementedError

    def _parse_timestamp_tokens(self, decoded_text: str) -> list[AlignedWord]:
        """
        Parse decoded output with timestamp tokens to extract word-level timestamps.

        The decoded text contains patterns like: <|0.00|> Hello world <|1.00|> How are you <|2.00|>
        Since these models output segment-level timestamps, we distribute time evenly among words
        within each segment as an approximation.

        Args:
            decoded_text: Decoded text with embedded timestamp tokens

        Returns:
            List of AlignedWord objects with word-level timestamps
        """
        import re

        timestamp_pattern = r"<\|(\d+\.\d+)\|>"

        timestamps = []
        for match in re.finditer(timestamp_pattern, decoded_text):
            timestamps.append((match.start(), match.end(), float(match.group(1))))

        if len(timestamps) < 2:
            clean_text = re.sub(r"<\|[^|]+\|>", "", decoded_text).strip()
            words_list = clean_text.split()
            return [AlignedWord(word=w, start=0.0, end=0.0) for w in words_list if w]

        words = []

        for i in range(len(timestamps) - 1):
            _, end_pos, start_time = timestamps[i]
            next_start_pos, _, end_time = timestamps[i + 1]

            segment_text = decoded_text[end_pos:next_start_pos].strip()
            segment_text = re.sub(r"<\|[^|]+\|>", "", segment_text).strip()

            if not segment_text:
                continue

            segment_words = segment_text.split()
            if not segment_words:
                continue

            # distribute time evenly among words within this segment
            duration_per_word = (end_time - start_time) / len(segment_words)
            for j, word in enumerate(segment_words):
                word_start = start_time + j * duration_per_word
                word_end = start_time + (j + 1) * duration_per_word
                words.append(
                    AlignedWord(
                        word=word,
                        start=round(word_start, 3),
                        end=round(word_end, 3),
                    )
                )

        return words


class WhisperASR(ASRModel):
    """Whisper ASR model wrapper with optional word-level timestamp extraction and batch processing."""

    def __init__(self, model_name: str = "openai/whisper-large-v3"):
        super().__init__()
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name, low_cpu_mem_usage=True, use_safetensors=True
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

    def forward(
        self,
        audio_16khz: torch.Tensor,
        language: str | None = None,
        return_timestamps: bool = True,
    ) -> list[TranscriptionResult]:
        """
        Transcribe batched audio and optionally get word-level timestamps.

        Args:
            audio_16khz (torch.Tensor): Audio tensor of shape (B, T) at 16kHz
            language (str, optional): Language code (e.g., 'en', 'fr')
            return_timestamps (bool): If True, return word-level timestamps. If False,
                return only transcriptions without alignment (faster). Default: True.

        Returns:
            list[TranscriptionResult]: List of transcriptions, optionally with word-level
                alignment, one per audio sample in the batch
        """

        B = audio_16khz.shape[0]
        device = audio_16khz.device

        inputs = self.processor(
            [sample.numpy() for sample in audio_16khz.cpu()],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        ).to(device)

        gen_kwargs = {}
        if return_timestamps:
            gen_kwargs["return_timestamps"] = True
        if language:
            gen_kwargs["language"] = LANGUAGE_CODE_TO_NAME.get(language, language)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        results = []
        for i in range(B):
            text = self.processor.decode(outputs[i], skip_special_tokens=True).strip()

            words = []
            if return_timestamps:
                decoded_with_timestamps = self.processor.decode(
                    outputs[i], skip_special_tokens=False, decode_with_timestamps=True
                )
                words = self._parse_timestamp_tokens(decoded_with_timestamps)

            results.append(
                TranscriptionResult(
                    text=text,
                    words=words,
                    language=language,
                )
            )

        return results


class VoxtralASR(ASRModel):
    """Voxtral ASR model wrapper with optional word-level timestamp extraction and batch processing.

    Voxtral is an encoder-decoder speech model from Mistral AI. It supports
    segment-level timestamps natively; word-level timestamps are approximated
    by distributing time evenly among words within each segment.
    """

    def __init__(
        self,
        model_name: str = "mistralai/Voxtral-Mini-3B-2507",
    ):
        super().__init__()
        from transformers import AutoProcessor, VoxtralForConditionalGeneration

        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = VoxtralForConditionalGeneration.from_pretrained(
            model_name, low_cpu_mem_usage=True
        )

    def forward(
        self,
        audio_16khz: torch.Tensor,
        language: str | None = None,
        return_timestamps: bool = True,
    ) -> list[TranscriptionResult]:
        """
        Transcribe batched audio and optionally get word-level timestamps.

        Args:
            audio_16khz (torch.Tensor): Audio tensor of shape (B, T) at 16kHz
            language (str, optional): Language code (e.g., 'en', 'fr')
            return_timestamps (bool): If True, return word-level timestamps. If False,
                return only transcriptions without alignment (faster). Default: True.

        Returns:
            list[TranscriptionResult]: List of transcriptions, optionally with word-level
                alignment, one per audio sample in the batch

        Note:
            Voxtral natively provides segment-level timestamps only. Word-level
            timestamps are approximated by distributing time evenly among words
            within each segment.
        """
        B = audio_16khz.shape[0]
        device = audio_16khz.device

        import transformers.models.voxtral.processing_voxtral as voxtral_processing

        inputs = self.processor.apply_transcription_request(
            language=language,
            audio=[sample.cpu().numpy() for sample in audio_16khz],
            model_id=self.model_name,
            format=["WAV"] * B,
        )
        inputs = inputs.to(device, dtype=torch.bfloat16)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=500)

        texts = self.processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

        results = []
        for i in range(B):
            text = texts[i].strip()

            # Voxtral doesn't support word-level timestamps in transcription mode
            words = []

            results.append(
                TranscriptionResult(
                    text=text,
                    words=words,
                    language=language,
                )
            )

        return results
