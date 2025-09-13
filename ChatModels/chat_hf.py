from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
llm = HuggingFacePipeline(pipeline=summarizer)

summary = llm("""India, officially the Republic of India, is a country in South Asia. 
It is the seventh-largest country by land area, the most populous country, 
and the most populous democracy in the world. Its capital is New Delhi.
India is bounded by the Indian Ocean on the south, the Arabian Sea on the southwest,
and the Bay of Bengal on the southeast. It shares land borders with Pakistan to the west;
China, Nepal, and Bhutan to the north; and Bangladesh and Myanmar to the east.
In the Indian Ocean, India is in the vicinity of Sri Lanka and the Maldives;
its Andaman and Nicobar Islands share a maritime border with Thailand, Myanmar and Indonesia.
India's diverse culture, languages, and traditions are a result of its long history,
which dates back to the Indus Valley Civilization, one of the world's oldest civilizations.
India has a rich cultural heritage, with influences from various dynasties,
empires, and colonial powers that have ruled the region over millennia.
""")
print('summary',summary)
