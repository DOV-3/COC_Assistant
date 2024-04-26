from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import os
from LLM import InternLM_LLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import gradio as gr

def load_chain():
    # 加载问答链
    # 定义 Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="/root/FF/model/sentence-transformer")

    # 向量数据库持久化路径
    persist_directory = "/root/FF/code/data_base/vector_db/chroma"

    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
        embedding_function=embeddings
    )

    # 加载自定义 LLM
    llm = InternLM_LLM(model_path = "/root/FF/model/internlm-chat-7b")

    # 定义一个 Prompt Template
    template = """You will act as a character in a tabletop role-playing game, as well as an assistant in this game.

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
    {context}
    问题: {question}
    有用的回答:"""

    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],template=template)

    # 运行 chain
    qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectordb.as_retriever(),return_source_documents=True,chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})
    
    return qa_chain

class Model_center():
    """
    存储检索问答链的对象 
    """
    def __init__(self):
        # 构造函数，加载检索问答链
        self.chain = load_chain()

    def qa_chain_self_answer(self, question: str, chat_history: list = []):
        """
        调用问答链进行回答
        """
        if question == None or len(question) < 1:
            return "", chat_history
        try:
            chat_history.append(
                (question, self.chain({"query": question})["result"]))
            # 将问答结果直接附加到问答历史中，Gradio 会将其展示出来
            return "", chat_history
        except Exception as e:
            return e, chat_history




# 实例化核心功能对象

