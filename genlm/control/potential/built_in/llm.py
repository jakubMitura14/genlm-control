import torch
import warnings
import numpy as np
from typing import NamedTuple, Optional
from arsenal.maths import logsumexp
from genlm.control.potential.base import Potential
from genlm.control.constant import EOS

# Try importing openai, but don't fail if it's not installed
# unless the user tries to use the OpenAI functionality.
_openai_client = None
_openai_library = None
try:
    import openai
    _openai_library = openai
except ImportError:
    openai = None # type: ignore


def load_model_by_name(name, backend, **kwargs):
    if backend == "vllm":
        from genlm.backend.llm import AsyncVirtualLM  # pragma: no cover

        model_cls = AsyncVirtualLM  # pragma: no cover
    elif backend == "hf":
        from genlm.backend.llm import AsyncTransformer

        model_cls = AsyncTransformer
    else:
        raise ValueError(
            f"Unknown backend: {backend}. Must be one of ['vllm', 'hf']"
        )  # pragma: no cover

    return model_cls.from_name(name, **kwargs)


class TokenMappings(NamedTuple):
    """
    Container for token mappings between bytes and tokens IDs in a language model.

    This mapping is generally different from the `decode` and `encode` mappings in the `PromptedLLM` class (see notes on EOS token handling).
    """

    decode: list[bytes]  # token_id -> bytes
    encode: dict[bytes, int]  # bytes -> token_id
    eos_idxs: list[int]  # IDs of EOS tokens

    @classmethod
    def create(cls, decode, eos_tokens):
        encode = {x: i for i, x in enumerate(decode)}
        if not all(eos in encode for eos in eos_tokens):
            raise ValueError("EOS token not in language model vocabulary")
        eos_idxs = [encode[eos] for eos in eos_tokens]
        return cls(decode=decode, encode=encode, eos_idxs=eos_idxs)


class PromptedLLM(Potential):
    """A potential representing a language model conditioned on a fixed prompt prefix.

    `PromptedLLM`s operate on byte sequences.

    Notes on EOS Token Handling:\n
    - Tokens to treat as end-of-sequence tokens are specified via the `eos_tokens` argument.\n
    - These tokens are excluded from the potential's vocabulary and as such do not appear in the `vocab` attribute.\n
        This means they cannot appear in any input contexts to the potential nor in the output of `logw_next`. They can be used in the prompt however.\n
    - The log probability assigned to the `genlm.control`'s reserved `EOS` token is the sum of the log probabilities of all the specified EOS tokens.\n

    This class wraps an `AsyncLM` instance.
    """

    def __init__(
        self,
        llm=None, # Make llm optional
        prompt_ids=None,
        eos_tokens=None,
        temperature=1,
        # --- OpenAI API specific arguments ---
        openai_model_name: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_system_prompt: str = "You are a helpful assistant.",
    ):
        """
        Initializes the PromptedLLM potential.

        Can be initialized either with a local LLM backend (vLLM/HF) or
        with parameters for an OpenAI-compatible API.

        Args:
            llm (AsyncLM, optional): The language model backend (vLLM/HF). Required if not using OpenAI API.
            prompt_ids (list[int], optional): Optional prompt prefix for local LLM backends.
            eos_tokens (list[bytes], optional): EOS tokens for local LLM backends.
            temperature (float, optional): Temperature for sampling/probabilities.
            openai_model_name (str, optional): Model name for OpenAI API. Required if using OpenAI API.
            openai_base_url (str, optional): Base URL for OpenAI-compatible API. Required if using OpenAI API.
            openai_api_key (str, optional): API key for OpenAI API.
            openai_system_prompt (str): System prompt to use with OpenAI API.

        Raises:
            ValueError: If initialization parameters are inconsistent.
        """
        self._is_openai_mode = False
        if openai_model_name and openai_base_url:
            if llm is not None:
                raise ValueError("Cannot provide both 'llm' and OpenAI API parameters.")
            if openai is None:
                 raise ImportError("The 'openai' library is required to use the OpenAI API mode. Please install it (`pip install openai`).") # pragma: no cover
            self._is_openai_mode = True
            self.openai_model_name = openai_model_name
            self.openai_base_url = openai_base_url
            self.openai_api_key = openai_api_key
            self.openai_system_prompt = openai_system_prompt
            self._openai_client = _openai_library.OpenAI(
                base_url=self.openai_base_url,
                api_key=self.openai_api_key,
            )
            # For OpenAI mode, vocab is less relevant in the traditional sense
            # We'll initialize Potential with an empty vocab for now.
            # EOS handling will be done via stop sequences in the API call if needed.
            super().__init__(vocabulary=[])
            self.temperature = temperature
            # Prompt handling needs to be different for OpenAI
            self._openai_prompt_str: Optional[str] = None
            # Note: prompt_ids, eos_tokens, token_maps are not used in OpenAI mode

        elif llm is not None:
            # --- Existing Local LLM Backend Initialization ---
            self.model = llm
            self.prompt_ids = prompt_ids or []

            if not eos_tokens:
                self._eos_tokens = [llm.byte_vocab[self.model.tokenizer.eos_token_id]]
            else:
                self._eos_tokens = eos_tokens

            assert len(set(self._eos_tokens)) == len(self._eos_tokens), (
                "duplicate eos tokens"
            )

            self.token_maps = TokenMappings.create(
                decode=llm.byte_vocab, eos_tokens=self._eos_tokens
            )

            self.temperature = temperature

            V = [x for x in self.token_maps.decode if x not in self._eos_tokens]
            super().__init__(vocabulary=V)
            # --- End Existing Init ---
        else:
             raise ValueError("Must provide either 'llm' (for local backends) or 'openai_model_name' and 'openai_base_url' (for OpenAI API).")


    @classmethod
    def from_name(
        cls,
        name,
        backend=None,
        eos_tokens=None,
        prompt_ids=None,
        temperature=1.0,
        # --- Add OpenAI args ---
        openai_base_url: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_system_prompt: str = "You are a helpful assistant.",
        **kwargs,
    ):
        """Create a `PromptedLLM` from a model name (HF/vLLM) or OpenAI API params.

        Args:
            name (str): Name of the model (used for HF/vLLM backend or as openai_model_name).
            backend (str, optional): `AsyncLM` backend ('vllm', 'hf') or 'openai'.
                Defaults to 'vllm' if CUDA available, else 'hf', if openai_base_url not set.
                If openai_base_url is set, backend defaults to 'openai'.
            openai_base_url (str, optional): Base URL for OpenAI-compatible API. If set, forces 'openai' mode.
            openai_api_key (str, optional): API key for OpenAI API.
            openai_system_prompt (str): System prompt to use with OpenAI API.
            eos_tokens (list[bytes], optional): List of tokens to treat as end-of-sequence tokens.
                Defaults to the EOS token of the language model's tokenizer.
            prompt_ids (list[int], optional): Optional prompt to use as a prompt prefix for all input contexts.
                Must be a list of token IDs. Defaults to None. The prompt ids can be set post-init via `set_prompt_from_str` or `prompt_ids`.
            temperature (float, optional): The temperature to apply to the language model's logits. Defaults to 1.
            **kwargs (dict): Additional arguments passed to AsyncLM constructor

        Returns:
            (PromptedLLM): An instance of PromptedLLM
        """
        if openai_base_url:
            backend = 'openai' # Force openai mode if URL is given

        if backend == 'openai':
            if not openai_base_url:
                raise ValueError("openai_base_url must be provided for 'openai' backend.")
            return cls(
                openai_model_name=name, # Use 'name' as the model identifier
                openai_base_url=openai_base_url,
                openai_api_key=openai_api_key,
                openai_system_prompt=openai_system_prompt,
                temperature=temperature,
            )
        else:
            # --- Existing backend loading ---
            backend = backend or ("vllm" if torch.cuda.is_available() else "hf")
            model = load_model_by_name(name, backend=backend, **kwargs)
            return cls(
                model,
                prompt_ids=prompt_ids,
                eos_tokens=eos_tokens,
                temperature=temperature,
            )
            # --- End existing ---

    @property
    def eos_tokens(self):
        return self._eos_tokens

    @eos_tokens.setter
    def eos_tokens(self, value):
        raise ValueError(
            "Cannot reset eos_tokens after initialization. "
            "Use spawn_new_eos(new_eos_tokens) instead."
        )

    @property
    def prompt(self):
        """
        Get the current prompt as a list of byte sequences corresponding to the prompt token IDs.

        Returns:
            (list[bytes]|None): The current prompt as a list of bytes sequences or None if no prompt_ids are set.
        """
        if not self.prompt_ids:
            return  # pragma: no cover
        return [self.token_maps.decode[x] for x in self.prompt_ids]

    def set_prompt_from_str(self, prompt_str):
        """Set the fixed prompt from a string.

        Modifies `prompt_ids` to be the token IDs of the input prompt according to the language model's tokenizer.

        Args:
            prompt_str (str): The prompt to set.
        """
        # TODO: Handle race condition where prompt_ids reset concurrently.
        if not isinstance(prompt_str, str):
            raise ValueError(
                f"Prompt must a string got {type(prompt_str)}. "
                f"To set the prompt from a list of token IDs (local LLM mode), use prompt_ids."
            )

        if self._is_openai_mode:
             self._openai_prompt_str = prompt_str
        else:
            # Existing logic for local LLMs
            if prompt_str.endswith(" "):
                warnings.warn(
                    "Prompt ends with whitespace, which may affect tokenization. "
                    "Consider removing trailing whitespace.",
                    stacklevel=2,
                )
            self.prompt_ids = self.model.tokenizer.encode(prompt_str)

    def encode_tokens(self, tokens):
        """Encode a list of byte tokens to a list of token IDs in
        the underlying language model's vocabulary.

        Args:
            tokens (list[bytes]): List of byte tokens to encode

        Returns:
            (list[int]): A list of token IDs corresponding to the input tokens.

        Raises:
            ValueError: If any token is not in the vocabulary
        """
        try:
            return [self.token_maps.encode[x] for x in tokens]
        except KeyError as e:
            raise ValueError(f"Token {e.args[0]} not in vocabulary") from e

    def decode_tokens(self, ids):
        """
        Decode a list of token IDs in the language model's vocabulary to a list of byte tokens.

        Args:
            ids (list[int]): A list of token IDs in the language model's vocabulary.

        Returns:
            (list[bytes]): A list of byte tokens corresponding to the input token IDs.
        """
        return [self.token_maps.decode[x] for x in ids]

    def tokenize(self, context_str):
        """Tokenize a string to a list of `bytes` objects, each corresponding to a token in the vocabulary.

        Uses the language model's tokenizer to map `context_str` to a list of token IDs, and then decodes the token IDs to bytes.

        Args:
            context_str (str): A string to encode

        Returns:
            (List[bytes]): A list of byte tokens corresponding to the input string.
        """
        return self.decode_tokens(self.model.tokenizer.encode(context_str))

    async def log_probability(self, context):
        """
        Compute the log probability of `context` given the prompt.

        Args:
            context (list[bytes]): A sequence of bytes tokens.

        Returns:
            (float): The log probability of `context`.
        """
        if self._is_openai_mode:
            raise NotImplementedError("log_probability is not supported in OpenAI API mode.")
        if not context:
            return 0
        context_ids = self.encode_tokens(context)
        return await self._log_probability(context_ids)

    async def _log_probability(self, context_ids):
        if self._is_openai_mode:
             raise NotImplementedError("_log_probability is not supported in OpenAI API mode.")
        prefixes = [self.prompt_ids + context_ids[:i] for i in range(len(context_ids))]
        log_ps = self._maybe_temper(
            await self.model.batch_next_token_logprobs(prefixes)
        )
        target_ids = torch.tensor(context_ids, device=log_ps.device)
        with torch.no_grad():
            token_logprobs = torch.gather(log_ps, 1, target_ids.unsqueeze(1))
            total_logprob = token_logprobs.sum().item()

        return total_logprob

    def _maybe_temper(self, logps):
        if self.temperature == 1:
            return logps
        return torch.log_softmax(logps / self.temperature, dim=-1)

    async def prefix(self, context):
        """
        Compute the log probability of `context` given the prompt.

        Args:
            context (list[bytes]): A sequence of bytes tokens.

        Returns:
            (float): The log probability of `context`.
        """
        if self._is_openai_mode:
            raise NotImplementedError("prefix is not supported in OpenAI API mode.")
        return await self.log_probability(context)

    async def complete(self, context):
        """
        Compute the log probability of `context` and the eos tokens given the prompt.

        If the model has multiple eos tokens, their probabilities will be summed.

        Args:
            context (list[bytes]): A sequence of bytes tokens.

        Returns:
            (float): The log probability of the context.
        """
        if self._is_openai_mode:
            raise NotImplementedError("complete is not supported in OpenAI API mode.")
        context_ids = self.encode_tokens(context)
        logp_context = await self._log_probability(context_ids)
        logp_next = self._maybe_temper(
            await self.model.next_token_logprobs(self.prompt_ids + context_ids)
        )
        logp_eos = torch.logsumexp(logp_next[self.token_maps.eos_idxs], dim=0).item()
        return logp_context + logp_eos

    def _process_logw_next(self, logw_next):
        """Process the log probabilities for the next tokens.

        This function rearranges the log probabilities such that the end-of-sequence (EOS) token's log probability
        is the sum of the log probabilities of `self.eos_tokens`.

        Args:
            logw_next (np.array): The log probabilities for the next tokens.

        Returns:
            (LazyWeights): Processed log probabilities for the next tokens.
        """
        # This is ugly, but it's useful for all potentials to adhere to the convention
        # of keeping the EOS token at the end of the weights array.
        _logw_next = np.full(len(self.vocab) + 1, -np.inf, dtype=logw_next.dtype)
        _logw_next[: len(self.vocab)] = logw_next[
            ~np.isin(np.arange(len(logw_next)), self.token_maps.eos_idxs)
        ]
        _logw_next[-1] = logsumexp(logw_next[self.token_maps.eos_idxs])
        return self.make_lazy_weights(_logw_next)

    async def logw_next(self, context):
        """Get log probabilities for next tokens given the prompt and `context`.

        Args:
            context (List[bytes]): A sequence of bytes tokens.

        Returns:
            (LazyWeights): Log probabilities for next tokens and EOS.
        """
        if self._is_openai_mode:
            raise NotImplementedError("logw_next is not supported in OpenAI API mode.")
        logw_next = self._maybe_temper(
            await self.model.next_token_logprobs(
                self.prompt_ids + self.encode_tokens(context)
            )
        )
        return self._process_logw_next(logw_next.float().cpu().numpy())

    async def batch_logw_next(self, contexts):
        """Get log probabilities for next tokens given the prompt and `context`, for a batch of contexts.

        Args:
            contexts (list[list[bytes]]): A list of sequences of bytes tokens.

        Returns:
            (List[LazyWeights]): Log probabilities for next tokens and EOS for each context.
        """
        if self._is_openai_mode:
            raise NotImplementedError("batch_logw_next is not supported in OpenAI API mode.")
        logw_nexts = self._maybe_temper(
            await self.model.batch_next_token_logprobs(
                [self.prompt_ids + self.encode_tokens(context) for context in contexts]
            )
        )
        return [
            self._process_logw_next(logw_next)
            for logw_next in logw_nexts.float().cpu().numpy()
        ]

    async def generate_completion(
        self,
        user_message: str,
        max_tokens: Optional[int] = None,
        stop: Optional[list[str]] = None,
        thinking: bool = False, # Add thinking parameter
    ) -> str:
        """
        Generates a completion using the configured OpenAI-compatible API.

        Args:
            user_message (str): The user's message/query.
            max_tokens (int, optional): Maximum tokens to generate.
            stop (list[str], optional): Stop sequences for generation.
            thinking (bool): Whether to include detailed thinking in system prompt.

        Returns:
            str: The generated content from the assistant.

        Raises:
            RuntimeError: If not configured for OpenAI API mode.
            ImportError: If openai library is not installed.
            openai.APIError: If the API call fails.
        """
        if not self._is_openai_mode:
            raise RuntimeError("generate_completion can only be used in OpenAI API mode.")
        if _openai_client is None:
             raise ImportError("The 'openai' library is required. Please install it (`pip install openai`).") # pragma: no cover

        messages = []
        system_content = self.openai_system_prompt
        if thinking:
            system_content = f"detailed thinking on, {system_content}" # Prepend thinking instruction
        messages.append({"role": "system", "content": system_content})

        # Add the stored prompt as a previous user message if it exists
        if self._openai_prompt_str:
             # This assumes the prompt acts like an initial user instruction
             # or context. Adjust role if needed (e.g., assistant context).
             messages.append({"role": "user", "content": self._openai_prompt_str})
             # Optionally add a dummy assistant message if the prompt expects
             # the *next* message to be the user query.
             # messages.append({"role": "assistant", "content": "Okay, I understand the context."})


        messages.append({"role": "user", "content": user_message})

        try:
            response = await self._openai_client.chat.completions.create(
                model=self.openai_model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=max_tokens,
                stop=stop,
                # Note: logprobs are not standard in chat completions for *all* tokens
            )
            # Handle potential None content or missing choices
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                 return response.choices[0].message.content.strip()
            else:
                 return "" # Return empty string if no valid response content
        except _openai_library.APIError as e:
            # Handle API errors gracefully
            print(f"OpenAI API error: {e}") # Or use proper logging
            raise # Re-raise the exception


    def __repr__(self):
        return f"PromptedLLM(prompt={self.prompt!r})"

    def spawn(self):
        """
        Spawn a new PromptedLLM instance.
        Copies configuration (local backend or OpenAI API).
        Note: Shares the underlying AsyncLM or OpenAI client instance.
        """
        if self._is_openai_mode:
            new_instance = PromptedLLM(
                openai_model_name=self.openai_model_name,
                openai_base_url=self.openai_base_url,
                openai_api_key=self.openai_api_key,
                openai_system_prompt=self.openai_system_prompt,
                temperature=self.temperature,
            )
            new_instance._openai_prompt_str = self._openai_prompt_str
            # Share the client instance
            new_instance._openai_client = self._openai_client
            return new_instance
        else:
            # Existing spawn logic for local LLMs
            return PromptedLLM(
                self.model,
                prompt_ids=self.prompt_ids.copy(),
                eos_tokens=self._eos_tokens.copy(),
                temperature=self.temperature,
            )

    def spawn_new_eos(self, eos_tokens):
        """
        Create a new PromptedLLM with different EOS tokens (local LLM mode only).
        Raises error in OpenAI mode.
        """
        if self._is_openai_mode:
            raise NotImplementedError("spawn_new_eos is not applicable in OpenAI API mode.")
        return PromptedLLM(
            self.model,
            prompt_ids=self.prompt_ids.copy(),
            eos_tokens=eos_tokens.copy(),
            temperature=self.temperature,
        )

    def to_autobatched(self):
        if self._is_openai_mode:
             # Autobatching is handled differently or not applicable for API calls
             warnings.warn("to_autobatched() called in OpenAI mode; has no effect.")
             return self
        raise ValueError("PromptedLLMs are autobatched by default.") # Keep original error for local mode
