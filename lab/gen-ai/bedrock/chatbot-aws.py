
"""
chatbot examples based on -
https://github.com/aws-samples/amazon-bedrock-workshop/blob/main/04_Chatbot/00_Chatbot_Titan.ipynb
"""

import os
import json
import pandas as pd
from utils import bedrock
from langchain.llms.bedrock import Bedrock
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import CSVLoader, TextLoader
from langchain.indexes.vectorstore import VectorStoreIndexWrapper


def bedrock_client():
    """
    create, validate, return bedrock client
    """
    boto3_bedrock = bedrock.get_bedrock_client(
        region='us-east-1',
        endpoint_url='https://bedrock.us-east-1.amazonaws.com',
        assumed_role='arn:aws:iam::315456707986:role/EC2_Role_Bedrock')
    models = boto3_bedrock.list_foundation_models()
    assert [x['modelId'] for x in models['modelSummaries']]

    return boto3_bedrock

def chatbot_without_context(prompt, temp=0.1, topP=0.9):
    """
    basic chatbot without context
    - temp - closer to zero, higher-probability words
    - topP - word-probablity cut-off
    """
    config = {'temperature': temp, 'topP': topP}
    x = {'inputText': prompt, 'textGenerationConfig': config}
    response = client.invoke_model(body=json.dumps(x), modelId='amazon.titan-tg1-large', accept='application/json', contentType='application/json')
    response = json.loads(response.get('body').read())
    return response.get('results')[0].get('outputText')

# bedrock client
client = bedrock_client()

# # chatbot without context
# response = chatbot_without_context(prompt='what does lytx do?', temp=0.1)

# # chatbot as conversation chain
# model = Bedrock(model_id='amazon.titan-tg1-large', client=client)
# memory = ConversationBufferMemory()
# chain = ConversationChain(llm=model, verbose=False, memory=memory)
# questions = [
#     'hi there',
#     'where is lytx located',
#     'what company are we talking about',
#     'where is general atomics located',
#     'what company are we talking about',
#     'what is the third digit of pi',
#     'who was the first president of USA',
#     'what laws did he pass',
#     'who was the president of USA in 2022',
#     'what laws did he pass',
#     'who was the president of USA in 2018',
#     'what laws did he pass']
# [chain.predict(input=question) for question in questions]
# print(chain.memory.buffer)

# # chatbot as persona and conversation chain
# model = Bedrock(model_id='amazon.titan-tg1-large', client=client)
# # as career coach
# memory = ConversationBufferMemory()
# memory.chat_memory.add_user_message('You will be acting as a career coach. Your goal is to give career advice to users')
# memory.chat_memory.add_ai_message('I am career coach and give career advice')
# chain = ConversationChain(llm=model, verbose=False, memory=memory)
# print('--- as career coach ---')
# print(chain.predict(input='What are the career options in residential construction?'))
# print(chain.predict(input='What are some popular paintings of Piero di Cosimo'))
# # as art critic
# memory = ConversationBufferMemory()
# memory.chat_memory.add_user_message('You will be acting as an art historian. Your goal is to discuss famous paintings')
# memory.chat_memory.add_ai_message('I am an art historian and have deep knowledge of paintings')
# chain = ConversationChain(llm=model, verbose=False, memory=memory)
# print('--- as art historian ---')
# print(chain.predict(input='What are the career options in residential construction?'))
# print(chain.predict(input='What are some popular paintings of Piero di Cosimo'))

# chatbot with explicit context
model = Bedrock(model_id='amazon.titan-tg1-large', client=client)
# docs = CSVLoader(r'/home/ubuntu/bedrock/Amazon_SageMaker_FAQs.csv').load()
docs = CSVLoader(r'/home/ubuntu/bedrock/lytx-coach-notes.csv').load()
# docs = TextLoader(r'/home/ubuntu/bedrock/lytx-coach-notes.txt').load()
# vs = VectorStoreIndexWrapper(vectorstore=FAISS.from_documents(documents=docs, embedding=BedrockEmbeddings(client=client)))
# response = lambda x: print(vs.query(question=x, llm=model))
# response('tell me about r in sagemaker')
# response('how to optimize costs in sagemaker')
# response('what are sagemaker pipelines')
# response('tell me about tyrannosauras rex')
# response('how many cases are of wet condition')
# response('what are key themes coming out of the data')
# response('If we had weather alerting system what could have been avoided based on the provided data?')
