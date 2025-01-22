"""Provides the LLM class."""
import os
from logging import Logger

from huggingface_hub import hf_hub_download
from langchain_community.llms import LlamaCpp

from tyrell.core import get_data_dir

# @TODO For nice typing, LLM should be base class, with this in LlamaLLM
class LLM:
    """Provides a class to build and interact with LLMs.

    Args:
        log (Logger): The logger for the LLM.
        config (dict): The configuration for the LLM.

    Attributes:
        HUGGINGFACE_MODEL_CACHE_PATH (str): The path to the Huggingface model cache.
        config (dict): The configuration for the LLM.
        log (Logger): The logger for the LLM.
        model_filepath (str): The path to the model file.
    """

    HUGGINGFACE_MODEL_CACHE_PATH = os.path.join(
        get_data_dir(),
        'models',
        'huggingface'
    )

    def __init__(self, log: Logger, config: dict) -> None:
        self.config = config
        self.log = log
        self.model_filepath = ''
        self.tokenizer = None

    def get(self) -> LlamaCpp:
        """Gets the LLM from the model details.

        Returns:
            LlamaCpp: The LLM.
        """
        self.model_filepath = hf_hub_download(
            repo_id=self.config['repo'],
            filename=self.config['filename'],
            cache_dir=self.HUGGINGFACE_MODEL_CACHE_PATH
        )

        match self.config['type']:
            case 'llama':
                return self._build_llama()
            case _:
                return None

    def _build_llama(self) -> LlamaCpp:
        """Builds the LlamaCpp from the model details.

        Returns:
            LlamaCpp: The LlamaCpp LLM.
        """
        # Print the config
        self.log.info(f"LLM config: {self.config}")
        self.log.info(f"LLM RoPE base: {self.config['rope_freq_base']}")
        self.log.info(f"LLM RoPE scale: {self.config['rope_freq_scale']}")
        return LlamaCpp(
            model_path=self.model_filepath,
            max_tokens=self.config['max_response_tokens'],
            n_batch=self.config['n_batch'],
            n_ctx=self.config['n_ctx'],
            rope_freq_base=float(self.config['rope_freq_base']),
            rope_freq_scale=float(self.config['rope_freq_scale']),
            n_gpu_layers=self.config['n_gpu_layers'],
            repeat_penalty=self.config['repeat_penalty'],
            temperature=self.config['temperature'],
            top_k=self.config['top_k'],
            min_p=self.config['min_p'],
            top_p=self.config['top_p'],
            verbose=self.config['verbose']
        )
