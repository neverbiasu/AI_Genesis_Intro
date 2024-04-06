# import json
# from langchain.llms.base import LLM
# from transformers import AutoTokenizer, AutoModel, AutoConfig
# from typing import List, Optional
# from utils import tool_config_from_file


# class ChatGLM3(LLM):
#     max_token: int = 8192
#     do_sample: bool = False
#     temperature: float = 0.8
#     top_p = 0.8
#     tokenizer: object = None
#     model: object = None
#     history: List = []
#     tool_names: List = []
#     has_search: bool = False

#     def __init__(self):
#         super().__init__()

#     @property
#     def _llm_type(self) -> str:
#         return "ChatGLM3"

#     def load_model(self, model_name_or_path=None):
#         model_config = AutoConfig.from_pretrained(
#             model_name_or_path,
#             trust_remote_code=True
#         )
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             model_name_or_path,
#             trust_remote_code=True
#         )
#         self.model = AutoModel.from_pretrained(
#             model_name_or_path, config=model_config, trust_remote_code=True
#         ).half().cuda()

#     def _tool_history(self, prompt: str):
#         ans = []
#         split_prompt = prompt.split("You have access to the following tools:\n\n")
#         # if "You have access to the following tools:" not in prompt:
#         #     # 没有找到，提供默认行为或跳过处理
#         #     print("Warning: Prompt does not contain expected section for tools. Proceeding without tool configuration.")
#         #     return [], prompt  # 返回空的工具列表和原始提示作为查询

#         tool_prompts = split_prompt[1].split("\n\nUse a json blob")[0].split("\n")
        
#         tool_names = [tool.split(":")[0] for tool in tool_prompts]
#         self.tool_names = tool_names
#         tools_json = []
#         for i, tool in enumerate(tool_names):
#             tool_config = tool_config_from_file(tool)
#             if tool_config:
#                 tools_json.append(tool_config)
#             else:
#                 ValueError(
#                     f"Tool {tool} config not found! Its description is {tool_prompts[i]}"
#                 )

#         ans.append({
#             "role": "system",
#             "content": "Answer the following questions as best as you can. You have access to the following tools:",
#             "tools": tools_json
#         })
#         query = f"""{prompt.split("Human: ")[-1].strip()}"""
#         return ans, query

#     def _extract_observation(self, prompt: str):
#         return_json = prompt.split("Observation: ")[-1].split("\nThought:")[0]
#         self.history.append({
#             "role": "observation",
#             "content": return_json
#         })
#         return

#     def _extract_tool(self):
#         if len(self.history[-1]["metadata"]) > 0:
#             metadata = self.history[-1]["metadata"]
#             content = self.history[-1]["content"]
#             if "tool_call" in content:
#                 for tool in self.tool_names:
#                     if tool in metadata:
#                         input_para = content.split("='")[-1].split("'")[0]
#                         action_json = {
#                             "action": tool,
#                             "action_input": input_para
#                         }
#                         self.has_search = True
#                         return f"""
# Action: 
# ```
# {json.dumps(action_json, ensure_ascii=False)}
# ```"""
#         final_answer_json = {
#             "action": "Final Answer",
#             "action_input": self.history[-1]["content"]
#         }
#         self.has_search = False
#         return f"""
# Action: 
# ```
# {json.dumps(final_answer_json, ensure_ascii=False)}
# ```"""

#     def _call(self, prompt: str, history: List = [], stop: Optional[List[str]] = ["<|user|>"]):
        
#         print("======")
#         print(prompt)
#         print("======")
#         if not self.has_search:
#             self.history, query = self._tool_history(prompt)
#         else:
#             self._extract_observation(prompt)
#             query = ""
#         # print("======")
#         # print(history)
#         # print("======")
#         _, self.history = self.model.chat(
#             self.tokenizer,
#             prompt,  # 直接使用 prompt 变量
#             history=self.history,
#             do_sample=self.do_sample,
#             max_length=self.max_token,
#             temperature=self.temperature,
#         )
#         response = self._extract_tool()
#         history.append((prompt, response))
#         return response

import json
import logging
from typing import Any, List, Optional, Union
from transformers import AutoTokenizer, AutoModel, AutoConfig

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.pydantic_v1 import Field

from langchain_community.llms.utils import enforce_stop_tokens

logger = logging.getLogger(__name__)
HEADERS = {"Content-Type": "application/json"}
DEFAULT_TIMEOUT = 30


def _convert_message_to_dict(message: BaseMessage) -> dict:
    if isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    elif isinstance(message, FunctionMessage):
        message_dict = {"role": "function", "content": message.content}
    else:
        raise ValueError(f"Got unknown type {message}")
    return message_dict


class ChatGLM3(LLM):
    """ChatGLM3 LLM service."""

    model_name: str = Field(default="chatglm3-6b", alias="model")
    endpoint_url: str = "http://127.0.0.1:8000/v1/chat/completions"
    """Endpoint URL to use."""
    model_kwargs: Optional[dict] = None
    """Keyword arguments to pass to the model."""
    max_tokens: int = 20000
    """Max token allowed to pass to the model."""
    temperature: float = 0.1
    """LLM model temperature from 0 to 10."""
    top_p: float = 0.7
    """Top P for nucleus sampling from 0 to 1"""
    prefix_messages: List[BaseMessage] = Field(default_factory=list)
    """Series of messages for Chat input."""
    streaming: bool = False
    """Whether to stream the results or not."""
    http_client: Union[Any, None] = None
    timeout: int = DEFAULT_TIMEOUT

    tokenizer: object = None
    model: object = None

    @property
    def _llm_type(self) -> str:
        return "chat_glm_3"

    @property
    def _invocation_params(self) -> dict:
        """Get the parameters used to invoke the model."""
        params = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stream": self.streaming,
        }
        return {**params, **(self.model_kwargs or {})}

    @property
    def client(self) -> Any:
        import httpx

        return self.http_client or httpx.Client(timeout=self.timeout)

    def load_model(self, model_name_or_path=None):
        model_config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name_or_path, config=model_config, trust_remote_code=True
        ).half().cuda()
        
    def _get_payload(self, prompt: str) -> dict:
        params = self._invocation_params
        messages = self.prefix_messages + [HumanMessage(content=prompt)]
        params.update(
            {
                "messages": [_convert_message_to_dict(m) for m in messages],
            }
        )
        return params

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call out to a ChatGLM3 LLM inference endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

                response = chatglm_llm("Who are you?")
        """
        import httpx

        payload = self._get_payload(prompt)
        logger.debug(f"ChatGLM3 payload: {payload}")

        try:
            response = self.client.post(
                self.endpoint_url, headers=HEADERS, json=payload
            )
        except httpx.NetworkError as e:
            raise ValueError(f"Error raised by inference endpoint: {e}") from e
        except ConnectionRefusedError as e:
            raise ValueError(f"Connection refused by inference endpoint: {e}") from e

        logger.debug(f"ChatGLM3 response: {response}")

        if response.status_code != 200:
            raise ValueError(f"Failed with response: {response}")

        try:
            parsed_response = response.json()

            if isinstance(parsed_response, dict):
                content_keys = "choices"
                if content_keys in parsed_response:
                    choices = parsed_response[content_keys]
                    if len(choices):
                        text = choices[0]["message"]["content"]
                else:
                    raise ValueError(f"No content in response : {parsed_response}")
            else:
                raise ValueError(f"Unexpected response type: {parsed_response}")

        except json.JSONDecodeError as e:
            raise ValueError(
                f"Error raised during decoding response from inference endpoint: {e}."
                f"\nResponse: {response.text}"
            )

        if stop is not None:
            text = enforce_stop_tokens(text, stop)

        return text
