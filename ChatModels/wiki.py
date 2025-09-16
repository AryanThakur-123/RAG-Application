import wikipediaapi as wiki
import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
import warnings
from dotenv import load_dotenv
from  langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

warnings.filterwarnings("ignore")
load_dotenv()

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("Hugging Face API token not found.")

api_key = os.getenv("GOOGLE_API_KEY")

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=api_key
)

# llm = HuggingFaceEndpoint(
#     repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct", 
#     huggingfacehub_api_token=hf_token
# )



# model = ChatHuggingFace(llm=llm)

def fetch_wikipedia_summary(topic):
    wiki_wiki = wiki.Wikipedia(user_agent='23f1000966@ds.study.iitm.ac.in', language='en',extract_format=wiki.ExtractFormat.WIKI)
    page = wiki_wiki.page(topic)
    if page.exists():
        return ' '.join(page.text.split('. ')) + '.'
    else:
        return "Topic not found on Wikipedia."
    
if __name__ == "__main__":
    topic = input("Enter the topic: ")
    topic = fetch_wikipedia_summary(topic)
    # print("Fetched Topic Content:", topic)
    f = open("wiki.txt", "w", encoding="utf-8")
    f.write(topic)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(topic)
    # print('splitter=>', text_splitter.split_text(topic))
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vec_store = Chroma.from_texts(chunks, embeddings,collection_name="wiki-collection")
    result = vec_store.similarity_search(topic, k=3)
    # print('result=>', result)
    
    docs_text = "\n\n".join([doc.page_content for doc in result])
    summary_response = model.invoke(docs_text)
    print("Summary:", summary_response.content)

