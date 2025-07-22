import nest_asyncio
import nltk
import ssl

nest_asyncio.apply()

# Fix SSL certificate issues and download NLTK data
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    print(f"Warning: Could not download NLTK data: {e}")


# Extract the documents from the specified directory
from llama_index.core import SimpleDirectoryReader
docs = SimpleDirectoryReader(input_dir='./data').load_data()

print(f"Loaded {len(docs)} documents from the data directory.")

# import pprint
# Print the loaded documents for debugging
# pprint.pprint(docs)


# Transform

# 1. Hide some keys from llm

# This is useful for debugging and understanding what data structure 
# LlamaIndex creates from your documents, especially to see what
# metadata is available and how the text is stored before you process 
# it further in your RAG pipeline.
# print(docs[0].__dict__)

# quick example of what the LLM and Embeddings see when with a test document
# document = Document(
#     text='This is the super-customized document...',
#     metadata={
#         'filename': 'sample1.txt',
#         'category': 'finance',
#         'author': 'LlamaIndex',
#     },
#     excluded_embed_metadata_keys=['filename'],  # Exclude these keys from LLM processing
#     metadata_separator='\n', 
#     metadata_template='{key}: {value}', 
#     text_template="Metadata:\n{metadata_str}\n-----\nContent:\n{content}",
# )

# print(
#     "The LLM sees this: \n",
#     document.get_content(metadata_mode=MetadataMode.LLM),
# )

# print(
#     "The Embedding model sees this: \n",
#     document.get_content(metadata_mode=MetadataMode.EMBED),
# )



from llama_index.core import Document
from llama_index.core.schema import MetadataMode

# print(docs[0].get_content(metadata_mode=MetadataMode.LLM))
# print(docs[0].get_content(metadata_mode=MetadataMode.EMBED))

for doc in docs:
    # define the metadata and text template for each document
    doc.text_template = "Metadata:\n{metadata_str}\n-----\nContent:\n{content}"

    # exclude page label from llm
    if "page_label" not in doc.excluded_embed_metadata_keys:
        doc.excluded_embed_metadata_keys.append('page_label')


# print(docs[0].get_content(metadata_mode=MetadataMode.LLM))


from llama_index.llms.groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
# print(os.environ['GROQ_API_KEY'])

llm_transformations = Groq(
    model="llama-3.1-8b-instant",
    api_key=os.environ['GROQ_API_KEY'],
)

# Other Transformations splitting text, extracting titles, and questions answered
# These transformations will be applied to the documents before they are processed by the LLM
from llama_index.core.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
)

from llama_index.core.node_parser import SentenceSplitter

text_splitter = SentenceSplitter(
    separator=" ",
    chunk_size=1024,
    chunk_overlap=128,
)

title_extractor = TitleExtractor(
    llm=llm_transformations,
    nodes=5,  # Number of nodes to extract titles from
)

questions_answered_extractor = QuestionsAnsweredExtractor(
    llm=llm_transformations,
    questions=3,  # Number of questions to extract
)


# Combine all transformations into an ingestion pipeline
from llama_index.core.ingestion import IngestionPipeline

ingestion_pipeline = IngestionPipeline(
    transformations=[
        text_splitter,
        title_extractor,
        questions_answered_extractor
    ],
)


nodes = ingestion_pipeline.run(
    documents=docs,
    in_place=True,  # Process documents in place
    show_progress=True,  # Show progress bar
)

print(f"Transformed {len(nodes)} nodes from the documents.")


# pprint.pprint(nodes)
# pprint.pprint(nodes[0].__dict__)

# Print the first node's content and metadata for debugging
print(nodes[0].get_content(metadata_mode=MetadataMode.EMBED))


# Embed the nodes using the LLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

hf_embedding = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

test_embedding = hf_embedding.get_text_embedding('This is a test embedding.')
print(f"Test embedding: {test_embedding}")

# Create Index

from llama_index.core import VectorStoreIndex

index = VectorStoreIndex(
    nodes=nodes,
    embed_model=hf_embedding,
)


# Query

llm_querying = Groq(
    model="llama-3.3-70b-versatile",
    api_key=os.environ['GROQ_API_KEY']
)

query_engine = index.as_query_engine(
    llm=llm_querying,
    similarity_top_k=5,  # Number of similar nodes to consider
    show_progress=True,  # Show progress bar
)

response = query_engine.query(
    "What does this document talk about?",
)

print(f"Query response: {response}")

print(response.__dict__)  # Print the response dictionary for debugging

