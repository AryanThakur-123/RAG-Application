import wikipediaapi as wiki
import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("Hugging Face API token not found.")


llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct", 
    huggingfacehub_api_token=hf_token
)

model = ChatHuggingFace(llm=llm)

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
    f = open("wiki.txt", "w", encoding="utf-8")
    f.write(topic)
    summary_response = model.invoke(topic)
    print("Summary:", summary_response.content)

