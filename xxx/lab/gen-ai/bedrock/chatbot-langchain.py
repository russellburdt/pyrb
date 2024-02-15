
"""
QA over Documents, based on -
https://python.langchain.com/docs/use_cases/question_answering/
"""

from utils import bedrock
from langchain.llms.bedrock import Bedrock
from langchain.embeddings import BedrockEmbeddings
from langchain.document_loaders import WebBaseLoader, CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.summarize import load_summarize_chain


# bedrock objects
client = bedrock.get_bedrock_client(
    region='us-east-1',
    endpoint_url='https://bedrock.us-east-1.amazonaws.com',
    assumed_role='arn:aws:iam::315456707986:role/EC2_Role_Bedrock')
models = [x['modelId'] for x in client.list_foundation_models()['modelSummaries']]
model = Bedrock(model_id='amazon.titan-tg1-large', client=client)
embedding = BedrockEmbeddings(client=client)

# data pre-processing
loader = CSVLoader(r'/mnt/home/russell.burdt/data/gen-ai/misc/lytx-coach-notes.csv')
data = loader.load()
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = splitter.split_documents(data)
vectorstore = FAISS.from_documents(documents=splits, embedding=embedding)

# summarize docs
summarize = load_summarize_chain(model, chain_type='stuff')
print(summarize.run(splits))

# similar docs
for doc in vectorstore.similarity_search('kirkwood training'):
    print('\n' + doc.page_content)

# QA
qa = RetrievalQA.from_chain_type(model, retriever=vectorstore.as_retriever())
print(qa({'query': 'how many cases are of wet condition'})['result'])
print(qa({'query': 'what are key themes coming out of the data'})['result'])
print(qa({'query': 'If we had weather alerting system what could have been avoided based on the provided data?'})['result'])

# conversation
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
retriever = vectorstore.as_retriever()
chain = ConversationalRetrievalChain.from_llm(model, retriever=retriever, memory=memory)
history = []
# Q1
question = 'what are some causes of bad driving'
answer = chain({'question': question, 'chat_history': history})['answer']
history.append((question, answer))
print(answer)
# Q2
question = 'are the wet conditions everywhere'
answer = chain({'question': question, 'chat_history': history})['answer']
history.append((question, answer))
print(answer)
