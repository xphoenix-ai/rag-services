import threading
from abc import ABC, abstractmethod
from typing import Callable, Tuple, List, Dict, Union, Any

from utils.unicode_converter import sinhala_to_singlish
from utils.mapping.language_utils import get_language_code


class TranslatorBase(ABC):
    """base class for translators
    """
    def __init__(self, class_name, *args, **kwargs):
        self.class_name = class_name
        self.tr_model = None
        self.init(*args, **kwargs)
        
    def init(self, *args, **kwargs) -> None:
        """
        model loading is done in background
        """
        thread = threading.Thread(target=self._load_model, args=(args), kwargs=kwargs)
        thread.start()
        
    @abstractmethod
    def _load_model(self, *args, **kwargs) -> None:
        """initialize models
        """
        raise NotImplementedError

    @abstractmethod
    def translate(self, query: str, src_lang: str, tgt_lang: str, use_min_length: bool = False) -> Tuple[str, str]:
        raise NotImplementedError
    
    def is_ready(self) -> bool:
        if self.tr_model is not None:
            return True
        return False

    def get_lang_code(self, lang_name):
        return get_language_code("translator", self.class_name, lang_name)

    @staticmethod
    def preprocess(x: str) -> str:
        x, last = x[:-1], x[-1] 
        x = x.replace('.', ';;')
        x = x.replace('?', ';;;')
        x = x.replace('!', '^,')
        return f"{x}{last}"

    @staticmethod
    def postprocess(x: str) -> str:
        x = x.replace(';;;', '?')
        x = x.replace(';;', '.')
        x = x.replace('^,', '!')
        return x
        