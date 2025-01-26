from utils.mapping.stt_languages import *
from utils.mapping.tts_languages import *
from utils.mapping.translator_languages import *

LANG_MAPS = {}


def get_language_maps(translator_class: str, stt_class: str, tts_class: str) -> tuple[dict, dict, dict]:
    """Get language mapping dictionaries for translator, STT, and TTS models.

    Args:
        translator_class (str): Name of translator class
        stt_class (str): Name of STT class
        tts_class (str): Name of TTS class

    Returns:
        tuple[dict, dict, dict]: Tuple containing (translator_map, stt_map, tts_map)
    """
    translator_language_map = eval(f"{translator_class.upper()}_LANGUAGE_MAP_TRANSLATOR")
    stt_language_map = eval(f"{stt_class.upper()}_LANGUAGE_MAP_STT")
    tts_language_map = eval(f"{tts_class.upper()}_LANGUAGE_MAP_TTS")

    translator_language_map = {k.lower(): v for (k, v) in translator_language_map.items()}
    stt_language_map = {k.lower(): v for (k, v) in stt_language_map.items()}
    tts_language_map = {k.lower(): v for (k, v) in tts_language_map.items()}

    return translator_language_map, stt_language_map, tts_language_map


def get_language_code(task: str, class_name: str, language: str) -> tuple[str, str]:
    """Get standardized language code for a given task and model class.

    Args:
        task (str): Task type ("translator", "stt", or "tts")
        class_name (str): Name of the model class
        language (str): Language name or code to standardize

    Returns:
        tuple[str, str]: Tuple containing (language_code, error_message)

    Raises:
        AssertionError: If task is not one of "translator", "stt", or "tts"
    """
    assert task.lower() in ["translator", "stt", "tts"]
    map_key = f"{class_name.upper()}_LANGUAGE_MAP_{task.upper()}"

    if map_key in LANG_MAPS:
        language_map = LANG_MAPS[map_key]
    else:
        language_map = eval(map_key)
        language_map = {k.lower(): v for (k, v) in language_map.items()}
        LANG_MAPS[map_key] = language_map

    lang_code = language_map.get(language.lower(), language)
    return lang_code, f"{language} not found!" if lang_code is None else ""


if __name__ == "__main__":
    # translator_language_map, stt_language_map, tts_language_map = get_language_maps("nllb", "whisper", "coqui")
    translator_language_map, stt_language_map, tts_language_map = get_language_maps("nllb", "hf", "coqui")
    print(translator_language_map)
    print(stt_language_map)
    print(tts_language_map)

    # tr_lang_code = get_language_code("translator", "nllb", "sinhala")
    tr_lang_code = get_language_code("translator", "marian_mt", "english")
    stt_lang_code = get_language_code("stt", "whisper", "sinhala")
    tts_lang_code = get_language_code("tts", "coqui", "english")

    print(tr_lang_code, stt_lang_code, tts_lang_code)
