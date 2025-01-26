import threading
from abc import ABC, abstractmethod
from typing import Callable, List, Dict, Union, Any


class LLMBase(ABC):
    """Base class for Large Language Models."""
    
    def __init__(self, class_name: str, *args: Any, **kwargs: Any) -> None:
        """Initialize the LLM base class.

        Args:
            class_name (str): Name of the LLM class
            *args (Any): Variable length argument list for model initialization
            **kwargs (Any): Arbitrary keyword arguments for model initialization
        """
        self.model = None
        self.class_name = class_name
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
    def generate(self, prompt: str, **generation_config: dict) -> str:
        """Generate text response from the LLM.

        Args:
            prompt (str): Input prompt for the LLM
            **generation_config: Generation configuration parameters

        Returns:
            str: Generated text response

        Raises:
            NotImplementedError: Must be implemented by child classes
        """
        raise NotImplementedError
    
    def is_ready(self) -> bool:
        """Check if the model is loaded and ready for inference.

        Returns:
            bool: True if model is loaded and ready, False otherwise
        """
        if self.model is not None:
            return True
        return False
