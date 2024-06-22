import os
import requests
from typing import Any, Dict, Iterator, List, Mapping, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk


class CustomLLM(LLM):
    url: str = os.getenv("LLM_URL")
    # do_sample: Optional[bool] = True
    # temperature: Optional[float] = 0.1
    # top_p: Optional[float] = 0.95
    # top_k: Optional[int] = 20
    # max_new_tokens: Optional[int]   = 200
    generation_config: dict = {
        "do_sample": True,
        "max_new_tokens": 200,
        "top_k": 20,
        "top_p": 0.95,
        "temperature": 0.1
    }
    
    # def __init__(self, do_sample=True, max_new_tokens=200, top_k=20, top_p=0.95, temperature=0.1):
    # def __init__(self):
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

        # self.url = url
        # self.generation_config = {
        #     "do_sample": do_sample,
        #     "max_new_tokens": max_new_tokens,
        #     "top_k": top_k,
        #     "top_p": top_p,
        #     "temperature": temperature,
        # }
        # self.generation_config = self._get_model_default_parameters
        
        # self.do_sample = do_sample
        # self.max_new_tokens = max_new_tokens
        # self.top_k = top_k
        # self.top_p = top_p
        # self.temperature = temperature
        
        # self.do_sample = True
        # self.max_new_tokens = 200
        # self.top_k = 20
        # self.top_p = 0.95
        # self.temperature = 0.1
        
        # self.generation_config = {
        #     "do_sample": self.do_sample,
        #     "max_new_tokens": self.max_new_tokens,
        #     "top_k": self.top_k,
        #     "top_p": self.top_p,
        #     "temperature": self.temperature,
        # }
        
    # @property
    # def _get_model_default_parameters(self):
    #     return {
    #         "do_sample": self.do_sample,
    #         "max_new_tokens": self.max_new_tokens,
    #         "top_k": self.top_k,
    #         "top_p": self.top_p,
    #         "temperature": self.temperature,
    #     }

    def __postprocess(self, text):
        return text.split("\n", 1)[0]
        
    def __get_response(self, prompt: str, **kwargs: Any) -> str:
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
