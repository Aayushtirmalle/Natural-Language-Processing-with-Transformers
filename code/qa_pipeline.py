from haystack.document_stores import OpenSearchDocumentStore
from haystack import Pipeline
from haystack.pipelines import ExtractiveQAPipeline
from haystack.nodes import TextConverter, PreProcessor, BM25Retriever, FARMReader, PromptTemplate
from haystack.utils import fetch_archive_from_http
import logging
import os
from pprint import pprint
from haystack.utils import print_answers

# Initializing the OpensearchDocumentStore
document_store = OpenSearchDocumentStore()

# Indexing Documents with a Pipeline
doc_dir = "../assets/data/pubmed_medical_intelligence"

# Initialize the pipeline, TextConverter, and PreProcessor
indexing_pipeline = Pipeline()
text_converter = TextConverter()
preprocessor = PreProcessor(
    clean_whitespace=True,
    clean_header_footer=True,
    clean_empty_lines=True,
    split_by="word",
    split_length=200,
    split_overlap=20,
    split_respect_sentence_boundary=True,
)

# Add the nodes into an indexing pipeline.
indexing_pipeline.add_node(component=text_converter, name="TextConverter", inputs=["File"])
indexing_pipeline.add_node(component=preprocessor, name="PreProcessor", inputs=["TextConverter"])
indexing_pipeline.add_node(component=document_store, name="DocumentStore", inputs=["PreProcessor"])

# Run the indexing pipeline to write the text data into the DocumentStore
files_to_index = [doc_dir + "/" + f for f in os.listdir(doc_dir)]
indexing_pipeline.run_batch(file_paths=files_to_index)

# Initializing the Retriever
retriever = BM25Retriever(document_store=document_store)

# Initializing the Reader
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)

# Creating the Retriever-Reader Pipeline
# querying_pipeline = Pipeline()
# querying_pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
# querying_pipeline.add_node(component=reader, name="Reader", inputs=["Retriever"])

querying_pipeline = ExtractiveQAPipeline(reader, retriever)
query = "How does artificial intelligence contribute to reducing drug development time in USA?"
result = querying_pipeline.run(query=query, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}})

# Ask the questions
# prediction = querying_pipeline.run(
#     query="Name the female actresses.", params={"Retriever": {"top_k": 5}, "Reader": {"top_k": 3}}
# )
n = 1
for answer in result['answers']:
    print(f"Top {n}:\n")
    print(f"Answer: {answer.answer}")
    print(f"Score: {answer.score}")
    print(f"Context: {answer.context}")
    print(f"DocumentId: {answer.document_ids}\n")
    n = n + 1