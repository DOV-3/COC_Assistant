from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class InternLM_LLM(LLM):
    # 基于本地 InternLM 自定义 LLM 类
    tokenizer : AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, model_path :str):
        # model_path: InternLM 模型路径
        # 从本地初始化模型
        super().__init__()
        print("正在从本地加载模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(torch.bfloat16).cuda()
        self.model = self.model.eval()
        print("完成本地模型的加载")

    def _call(self, prompt : str, stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any):
        # 重写调用函数
        system_prompt = """You will act as a character in a tabletop role-playing game, as well as an assistant in this game.

When you receive a question starting with <助理>, you will respond as an assistant.
- As an assistant, you can only answer based on the information you know. If you don't know the answer to a question, you should indicate so and not fabricate information to respond.

When you receive a question starting with <NPC>, you will respond as a character in the game.
- As an Character,Never forget you are the Character in the game.
- I will propose actions I plan to take and you will explain what happens when I take those actions.
- Speak in the first person from the perspective of your Character.
- For describing your own body movements, wrap your description in '*'.
- Do not change roles!
- Stop speaking the moment you finish speaking from your perspective.
- You need to respond as the character you are playing, based on the dialogue or environmental descriptions in the question.
- Be creative and imaginative.
        """
        
        messages = [(system_prompt, '')]
        response, history = self.model.chat(self.tokenizer, prompt , history=messages)
        return response
        
    @property
    def _llm_type(self) -> str:
        return "InternLM"