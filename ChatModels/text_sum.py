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

text_to_summarize = """
Indian Prime Minister Narendra Modi on Friday announced a new initiative to promote the use of electric vehicles (EVs) in the country. The initiative, called 'Electric India', aims to make India a global hub for electric mobility by 2030.
In his address to the nation on the occasion of Independence Day, Modi highlighted the importance of transitioning to electric vehicles to reduce pollution and dependence on fossil fuels. He announced a series of measures to encourage the adoption of EVs, including tax incentives, subsidies for EV manufacturers, and the development of charging infrastructure across the country.
Modi also emphasized the role of innovation and technology in driving the electric vehicle revolution. He called on startups and entrepreneurs to contribute to the development of affordable and efficient electric vehicles that can cater to the needs of the Indian market.
The 'Electric India' initiative is expected to create millions of jobs in the EV sector and boost the country's manufacturing capabilities. The government plans to collaborate with private companies and international partners to accelerate the adoption of electric vehicles and promote sustainable transportation solutions.
The announcement was met with enthusiasm by environmentalists and industry experts, who believe that the initiative could significantly reduce India's carbon footprint and improve air quality in urban areas. However, some experts also cautioned that the success of the initiative would depend on the effective implementation of policies and the availability of affordable electric vehicles for consumers.
The government has set ambitious targets for the adoption of electric vehicles, aiming to have at least 30% of all vehicles on Indian roads be electric by 2030. To achieve this, the government plans to invest in research and development, promote the use of renewable energy for charging infrastructure, and encourage public transportation systems to transition to electric fleets.
"""


prompt = f"Please summarize the following text:\n\n{text_to_summarize}"


summary_response = model.invoke(prompt)


print("Summary:", summary_response.content)