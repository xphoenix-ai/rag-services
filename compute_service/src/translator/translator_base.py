import threading
from abc import ABC, abstractmethod
from typing import Callable, List, Dict, Union, Any

from utils.unicode_converter import sinhala_to_singlish


class TranslatorBase(ABC):
    """base class for translators
    """
    def __init__(self, *args, **kwargs):
        self.tr_model_singlish = None
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
    def singlish_to_sinhala(self, sing_query: str) -> str:
        """sing --> si"""
        raise NotImplementedError
    
    @abstractmethod
    def english_to_sinhala(self, en_query: str) -> str:
        """en --> si"""
        raise NotImplementedError
    
    @abstractmethod
    def sinhala_to_english(self, si_query: str) -> str:
        """si --> en"""
        raise NotImplementedError
    
    def sinhala_to_singlish(self, si_query: str) -> str:
        """si --> sing"""
        # si --> sing
        sing_response, _ = sinhala_to_singlish(si_query)
        
        return sing_response, ""
    
    def singlish_to_english(self, sing_query: str) -> str:
        """sing --> en"""
        # sing --> si
        si_response, _ = self.singlish_to_sinhala(sing_query)
        
        # si --> en
        en_response, _ = self.sinhala_to_english(si_response)
        
        return en_response, si_response
    
    def english_to_singlish(self, en_query: str) -> str:
        """sing --> en"""
        # en --> si
        si_response, _ = self.english_to_sinhala(en_query)
        
        # si --> sing
        sing_response, _ = self.sinhala_to_singlish(si_response)
        
        return sing_response, si_response
    
    def is_ready(self) -> bool:
        if (self.tr_model_singlish is not None) and (self.tr_model is not None):
            return True
        return False
    
    @staticmethod
    def preprocess(x: str) -> str:
        x, last = x[:-1], x[-1] 
        x = x.replace('.', '#,')
        x = x.replace('?', '`,')
        x = x.replace('!', '^,')
        return f"{x}{last}"

    @staticmethod
    def postprocess(x: str) -> str:
        x = x.replace('#,', '.')
        x = x.replace('`,', '?')
        x = x.replace('^,', '!')
        return x
        