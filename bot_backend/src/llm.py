import os
import requests
from typing import Any, Dict, Iterator, List, Mapping, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk


class CustomLLM(LLM):
    """Custom LLM that communicates with an external language model server via HTTP.
    
    This class implements the LangChain LLM interface to interact with a custom
    language model service. It supports both regular calls and streaming responses.
    """
    
    url: str = os.getenv("LLM_URL")
    generation_config: dict = {
        "do_sample": True,
        "max_new_tokens": 200,
        "top_k": 20,
        "top_p": 0.95,
        "temperature": 0.1
    }
        
    def __get_response(self, prompt: str, **kwargs: Any) -> str:
        """Send prompt to the LLM server and get response.
        
        Args:
            prompt (str): The input prompt for the language model
            **kwargs: Additional generation parameters
            
        Returns:
            str: The model's response text
            
        Raises:
            requests.RequestException: If the HTTP request fails
            KeyError: If response doesn't contain expected "response" field
        """
        url = os.getenv("LLM_URL")
        generation_config = {
            "do_sample": True,
            "max_new_tokens": 200,
            "top_k": 20,
            "top_p": 0.95,
            "temperature": 0.1,
            "repetition_penalty": 1.0
        }
        prompt_json = {
            "prompt": prompt
        }
        # json_body = {**prompt_json, **self.generation_config}
        json_body = {**prompt_json, **generation_config}
        json_body.update(kwargs)
        
        # response = requests.post(self.url, json=json_body)
        response = requests.post(url, json=json_body)
        response = response.json()
        
        return response["response"]
        # return self.__postprocess(response["response"])

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given input.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of the stop substrings.
                If stop tokens are not supported consider raising NotImplementedError.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            The model output as a string. Actual completions SHOULD NOT include the prompt.
        """
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        
        return self.__get_response(prompt, **kwargs)

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Stream the LLM on the given prompt.

        This method should be overridden by subclasses that support streaming.

        If not implemented, the default behavior of calls to stream will be to
        fallback to the non-streaming version of the model and return
        the output as a single chunk.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of these substrings.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            An iterator of GenerationChunks.
        """
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        
        for char in self.__get_response(prompt, **kwargs):
            chunk = GenerationChunk(text=char)
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)

            yield chunk

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": "CustomLLM",
        }

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "custom"
    
    def is_ready(self):
        """Check if the LLM service is available and ready.
        
        Returns:
            bool: True if the LLM service is ready, False otherwise
        """
        try:
            response = requests.get(os.getenv("STATUS_URL"))
            status = response.json()["llm"]
        except:
            status = False
            
        return status
