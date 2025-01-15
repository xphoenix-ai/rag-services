from utils.mapping.stt_languages import *
from utils.mapping.tts_languages import *
from utils.mapping.translator_languages import *


def get_language_maps(translator_class: str, stt_class: str, tts_class: str):
    translator_language_map = eval(f"{translator_class.upper()}_LANGUAGE_MAP_TRANSLATOR")
    stt_language_map = eval(f"{stt_class.upper()}_LANGUAGE_MAP_STT")
    tts_language_map = eval(f"{tts_class.upper()}_LANGUAGE_MAP_TTS")

    return translator_language_map, stt_language_map, tts_language_map


if __name__ == "__main__":
    # translator_language_map, stt_language_map, tts_language_map = get_language_maps("nllb", "whisper", "coqui")
    translator_language_map, stt_language_map, tts_language_map = get_language_maps("nllb", "hf", "coqui")
    print(translator_language_map)
    print(stt_language_map)
    print(tts_language_map)
