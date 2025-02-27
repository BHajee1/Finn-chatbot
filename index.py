from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from PyPDF2 import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import os
from transformers import pipeline
import gradio as gr

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={"device": "cpu"})


projection_data = {
  "monthly_revenue" : [50, 60, 72, 137, 165, 198, 289, 348, 418, 553, 665, 799, 1263.75, 1518.75, 1823.75, 2253.75, 2707.5, 3251.25, 3967.5, 4765, 5721.25, 6932.5, 8323.75, 9992.5, 26529.25, 31847.75, 38227.75, 46024, 55242, 66302.5, 79714.25, 95672.5, 114820.75, 137937.25, 165541.75, 198665.5, 260238, 312306, 374784, 449910, 539913, 647913, 777666, 933222, 1119885, 1344033, 1612863, 1935456, 2632416, 3158926.4, 3790734.8, 4549077.6, 5458921, 6550729, 7861072, 9433317, 11320004.2, 13584203.6, 16301075.6, 19561315.2],
  "monthly_cash_flow" : [-1113950, -2227840, -3341658, -4455389, -5569015, -6682515, -7795863, -8909028, -10021973, -11134760, -12247018, -13359001, -14469907, -15580126, -16689520, -17797924, -18905139, -20010926, -22217016, -23316562, -24413143, -25506165, -26594916, -27642091, -28675730, -29693125, -30691025, -31665531, -32611962, -33524703, -34397014, -35220806, -36682075, -37293930, -37750375, -38075105, -38241777, -38218777, -37968170, -37444434, -36592943, -35348141, -33631368, -31348228, -28385447, -24607095, -19067251, -12196392, -3728315, 6656424, 19341157, 34785855, 53542607, 76273725, 103774118, 136997643, 177088926, 225421520],
  "compound_annual_growth_rate" : 72.34,
  "annual_customer_acquisition_cost" : [4,504, 191, 20, 2, 0],
  "price_strategy" : {
    "type": "recurring",
    "frequency": "quarterly",
    "annual_cost" : [1.00, 1.25, 2.75, 3.00, 3.40]
  },
  ##"business_information" : {
  ##  "business_name": "BoomBrix",
  ##  "business_description": "A start-up company that produces a unique bluetooth speaker that can interact with Lego blocks. The speaker is designed to create a more immersive experience when using Lego blocks for children and Lego-enthusiests.",
  ##}
}


text = f"Projection data = {projection_data}"

documents = [Document(page_content=text, metadata={"source": "revenue_projection_model"})]
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=40)
all_splits = text_splitter.split_documents(documents)

collection = Chroma.from_documents(
  documents=all_splits, embedding=embeddings, persist_directory="doc_vectors")

collection.add_documents(documents)

loader = WebBaseLoader(web_path=("https://mitrarobot.com/"))
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=40)
splits = text_splitter.split_documents(docs)

for s in splits:
    collection.add_documents([s])
    
llm1 = ChatGroq(model_name="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
llm2 = ChatGroq(model_name="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))
llm3 = ChatGroq(model_name="mixtral-8x7b-32768", api_key=os.getenv("GROQ_API_KEY"))


roberta = pipeline("zero-shot-classification",model="facebook/bart-large-mnli", device="cpu")

def llm_router(message):
  candidate_labels = ['billing','revenue','venture', 'other']
  output = roberta(message, candidate_labels)
  max_score_index = output['scores'].index(max(output['scores']))

   # Retrieve the corresponding label
  print(output['labels'][max_score_index])

  if output['labels'][max_score_index] == 'revenue':
    return llm1
  elif output['labels'][max_score_index] == 'venture':
    return llm2
  else:
    return llm3

message = "How can I improve me revenue projection model?"
llm = llm_router(message)
llm.invoke(message)

class Chatbot:
    prompt = """
      Your name is Finn Model. Whereever the reference come, replace with our company, GoProjections.
      GoProjects is a small business that focuses on simplifying the financial modelling side of entreprenuership. You are an assistant for question-answering tasks for GoProjections "model improvement" support.
      You only need to mention this once in the conversation.
      You need to use the following pieces of retrieved context to answer the question.
      Use 3-5 sentences maximum and keep the answer concise. Answer in a professional tone.
      Answer only revenue, business or venture related questions and nothing else. If they ask something like chicken recipes, bring them back to the conversation
      """

    def __init__(self):
      self.messages = [] # setting up a basic memory, Change this to Mongo

      self.messages.append({"role": "system", "content": self.prompt})
      self.messages.append({"role": "user", "content": "This is basic information: User is living in San Francisco, California."})

    def _get_relevant_documents(self, query):
      docs = collection.similarity_search(query, k=4)
      retrieved_string = "\n\n".join(doc.page_content for doc in docs)
      return retrieved_string

    def _get_summary(self, llm):
      conv_history = '\n'.join([f"{msg['role']}: {msg['content']}" for msg in self.messages])
      summary_prompt = """Summarize these conversations keeping only the most relevant details""" + conv_history
      summary = llm.invoke(summary_prompt)
      return summary.content

    def __call__(self, user_message):
      context = self._get_relevant_documents(user_message)
      llm = llm_router(user_message)
      conv_history = self._get_summary(llm)
      query = self.prompt + "| Prior conversations " + conv_history + "| The context for RAG is: " + context + " | The user question is :" + user_message

      revenue_data = " ".join(map(str, projection_data))
      query = self.prompt + "| Prior conversations " + conv_history + "| The context for RAG is: " + context + " | The user question is :" + user_message + " | The revenue data is: " + revenue_data

      ai_message = llm.invoke(query)
      self.messages.append({"role": "user", "content": user_message})
      self.messages.append({"role": "assistant", "content": ai_message.content})
      return ai_message.content + "\n"
  
bot = Chatbot()
print(bot._get_relevant_documents("tell me about revenue projection"))

print(bot("tell me more about GoProjections"))

def chat(message, history):
    return bot(message)

css = """
.gradio-container {background-color: #dfd7c8;}
.gradio-chat-box {background-color: #ffffff;}
"""

demo = gr.ChatInterface(
    fn=chat,
    title="Finn Model",
    description="This is a chatbot built to aid small businesses in improving their revenue projection models.",
    css=css
)
demo.launch(debug=True)