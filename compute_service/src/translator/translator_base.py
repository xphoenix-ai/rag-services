import threading
from abc import ABC, abstractmethod
from typing import Callable, Tuple, List, Dict, Union, Any

from utils.unicode_converter import sinhala_to_singlish
from utils.mapping.language_utils import get_language_code


class TranslatorBase(ABC):
    """Base class for translators."""
    
    def __init__(self, class_name: str, *args: Any, **kwargs: Any) -> None:
        """Initialize the translator base class.

        Args:
            class_name (str): Name of the translator class
            *args (Any): Variable length argument list for model initialization
            **kwargs (Any): Arbitrary keyword arguments for model initialization
        """
        self.class_name = class_name
        self.tr_model = None
        self.init(*args, **kwargs)
        
    def init(self, *args: Any, **kwargs: Any) -> None:
        """Initialize model loading in background thread.

        Args:
            *args (Any): Variable length argument list passed to _load_model
            **kwargs (Any): Arbitrary keyword arguments passed to _load_model
        """
        thread = threading.Thread(target=self._load_model, args=(args), kwargs=kwargs)
        thread.start()
        
    @abstractmethod
    def _load_model(self, *args: Any, **kwargs: Any) -> None:
        """Initialize models - to be implemented by child classes.

        Args:
            *args (Any): Variable length argument list for model initialization
            **kwargs (Any): Arbitrary keyword arguments for model initialization

        Raises:
            NotImplementedError: Must be implemented by child classes
        """
        raise NotImplementedError

    @abstractmethod
    def translate(self, query: str, src_lang: str, tgt_lang: str, use_min_length: bool = False) -> Tuple[str, str]:
        """Translate text from source language to target language.

        Args:
            query (str): Text to translate
            src_lang (str): Source language code
            tgt_lang (str): Target language code
            use_min_length (bool, optional): Whether to use minimum length constraint. Defaults to False.

        Returns:
            Tuple[str, str]: Tuple containing (translated_text, error_message)
        """
        raise NotImplementedError
    
    def is_ready(self) -> bool:
        """Check if the model is loaded and ready for inference.

        Returns:
            bool: True if model is loaded and ready, False otherwise
        """
        if self.tr_model is not None:
            return True
        return False

    def get_lang_code(self, lang_name: str) -> Tuple[str, str]:
        """Get standardized language code for given language name.

        Args:
            lang_name (str): Name or code of the language

        Returns:
            Tuple[str, str]: Tuple containing (language_code, error_message)
        """
        return get_language_code("translator", self.class_name, lang_name)

    @staticmethod
    def preprocess(x: str) -> str:
        """Preprocess text before translation by encoding special characters.

        Args:
            x (str): Input text to preprocess

        Returns:
            str: Preprocessed text with encoded special characters
        """
        x, last = x[:-1], x[-1] 
        x = x.replace('.', ';;')
        x = x.replace('?', ';;;')
        x = x.replace('!', '^,')
        return f"{x}{last}"

    @staticmethod
    def postprocess(x: str) -> str:
        """Postprocess translated text by decoding special characters.

        Args:
            x (str): Translated text to postprocess

        Returns:
            str: Postprocessed text with decoded special characters
        """
        x = x.replace(';;;', '?')
        x = x.replace(';;', '.')
        x = x.replace('^,', '!')
        return x
        