# A Medical Question-Answering System Based On RAG

## INLPT-WS2023 Group 27 
## Team Members:
1. Jiufeng, Li
2. Manoj Tirmalle, Aayush
3. Thiruppathi Kannan, Alagumeena
4. Thiruppathi Kannan, Swathi

## Email Addresses:
* jiufeng.li@stud.uni-heidelberg.de
* aayush.tirmalle@stud.uni-heidelberg.de
* alagumeena.thiruppathi_kannan@stud.uni-heidelberg.de 
* swathi.thiruppathi_kannan@stud.uni-heidelberg.de

## Member Contribution:
1. Jiufeng, Li
- Data Acquisition:
I have developed a data acquisition pipeline that includes querying the PubMed database to gather all relevant medical articles, designing the data model, and establishing the JSON format for storage. Subsequently, the collected data are segmented into smaller chunks using the Langchain text splitter. Consequently, all data are well-structured and stored in JSON format.

- Question-Answering Pipeline:
I have constructed the pipeline for our medical QA system, which involves creating embeddings for all data chunks and uploading them into the Pinecone vector store. This process includes the creation of indices and the formulation of argument prompts for question generation, followed by the answering mechanism.

- Streamlit UI:
I developed a user-friendly webpage using Streamlit to present the final answers and top-k retrieval results from our RAG model. Key sections are highlighted to enhance user experience.

- System Evaluation and Debugging:
My primary role during development was to manage the vector store by establishing connections, initializing, creating, and updating vector indices, all aimed at improving the accuracy of our QA system. For evaluation, I conducted comparisons of our RAG model with leading large language models like GPT-3 and GPT-4, using the embedding metrics.

- Reporting Responsibilities:
I am responsible for writing the Methods/Approach and documenting the Experimental Setup and Results in our report.

2. Manoj Tirmalle, Aayush - 
I've intricately designed scripts to systematically extract information from the PubMed website, going beyond mere retrieval by incorporating a meticulous process of extraction and validation. This comprehensive approach not only ensures the completeness of the collected data but also upholds high standards of data quality, resulting in a robust and reliable dataset. In the evaluation phase, I explored the performance of the model with RAG, GPT-3, and GPT-4. Throughout the pipelining stage, I deepened my understanding of the pipeline's functionality and implementation. Lastly, I conducted system tests to verify the model's compatibility across various computer systems.





3. Thiruppathi Kannan, Alagumeena -
   I did a part of the data acquisition phase of the project, gathering a comprehensive collection of medical articles from PubMed. This involved identifying relevant articles, retrieving the data, and ensuring its quality and completeness for further analysis.  In partnership with Swathi, I engaged in pair programming sessions to develop the user interface (UI) for our project using Streamlit. Together, we designed and implemented interactive features, visualizations, and functionalities to enhance the user experience and facilitate data exploration. I played a significant role in preparing the project report, contributing to various sections such as the introduction and results. 


4. Thiruppathi Kannan, Swathi - I spearheaded the data acquisition phase by sourcing a diverse array of medical articles from PubMed. This involved meticulously selecting pertinent articles, extracting the data, and ensuring its quality and comprehensiveness to facilitate subsequent analysis.  Alongside Alagu, I actively participated in pair programming sessions dedicated to crafting the user interface (UI) using Streamlit. I played a pivotal role in compiling our project report, contributing extensively to various sections including the methodology and discussion. This encompassed synthesizing key findings, structuring content, and upholding clarity, cohesion, and precision to effectively communicate our research outcomes.

## Advisor: 
John Ziegler (ziegler@informatik.uni-heidelberg.de)

## Anti-plagiarism Confirmation:
We affirm that the work for the project was independently completed. 

## **Introduction** (Thiruppathi Kannan, Alagumeena,  Thiruppathi Kannan, Swathi)
Our project aims to develop a sophisticated yet user-friendly platform for efficiently retrieving information from extensive document repositories using advanced NLP techniques.

In today's information age, the volume of textual data available is immense, ranging from research papers and articles to technical documentation and customer support tickets. However, accessing and extracting relevant information from this vast amount of data can be a daunting task, often requiring significant time and effort.

To address this challenge, we have developed a question answering system that leverages state-of-the-art technologies, including langchain framework for text splitting and  NLP processing, Pinecone for scalable embeddings storage, and Streamlit for a seamless frontend interface. By integrating these tools, our system offers users an intuitive way to pose questions and receive accurate answers drawn from extensive document collections.

The Langchian rag model is an integral component of our project, complementing the functionalities of text chunking, embedding, and Streamlit as ui in the realm of advanced Natural Language Processing (NLP). Besides, we utilize ChatGPT to generate the final answer based on the knowledge retrieved from our vector store.

In this report, we will provide an overview of our project's architecture and key components, discuss the methodologies employed in developing our question answering system, and present the results of our evaluation. Additionally, we will explore potential areas for future enhancements and improvements. Through this report, readers can expect to gain a comprehensive understanding of our approach, its effectiveness in addressing text-based information retrieval challenges, and insights into its practical applications.

## **Related Work** (Thiruppathi Kannan, Alagumeena,  Thiruppathi Kannan, Swathi)
In the influential article by [Chen et al. 2017]<sup>1</sup>, the Retriever-Reader Framework is introduced for answering open-domain questions by utilizing a vast collection of documents. The process involves a two-step approach: first, a retriever function `f(D, q) -> (p1, p2, ..., pk)` selects `k` pertinent passages from the document set `D` given a query `q`; then, a reader function `g(q, (p1, p2, ..., pk)) -> a` interprets these passages to formulate an answer `a`, treating it as a reading comprehension challenge. This approach employs a standard tf-idf information-retrieval sparse model and a neural reading comprehension model trained on SQuAD [Rajpurkar et al. 2016]<sup>2</sup> and other distantly-supervised QA datasets.

Inspired by this framework, our methodology enhances the retrieval step by applying the RAG (Retrieval-Augmented Generation) model by [Lewis et al. 2020]<sup>3</sup> to segment the data into smaller chunks, as opposed to the traditional retrieval and reading methods. This technique aims to refine the retrieval phase, enabling more accurate information extraction and potentially boosting the reading comprehension model's capability to understand and respond to inquiries in the medical intelligence domain.


While previous research has laid the groundwork for question answering systems using Elasticsearch, Streamlit, and Haystack, our project contributes novel advancements by leveraging OpenSearch Document Store and tailoring our solution to the specific needs of the healthcare domain. Through this approach, we aim to provide a scalable, efficient, and domain-aware platform for healthcare information retrieval and knowledge extraction.

## **Methods/Approach** (Jiufeng Li)
Our project employs the RAG model to refine document retrieval and text chunking for our QA system. By leveraging fine-grained vector similarity searches through Pinecone, we efficiently retrieve and rank the most relevant information as it shows in Figure 1. This, coupled with Streamlit's interactive UI, enables rapid and accurate answer generation based on state-of-the-art language models, like ChatGPT 4. 

![RAG Pipeline](/assets/images/pipeline.jpg "The Pipeline of Our Medical Question-Answering System")
<p align="center">Figure 1: The Pipeline of Our RAG Medical Question-Answering System</p>

1. *Langchain Framework:*
    We use Langchain, one of the most popular NLP framework, to develop our project. One of the key benefits of using Langchain is its modular design, which allows developers to plug in different components and language models as needed. This flexibility makes it easier to experiment with and deploy various configurations, optimizing for performance and accuracy across a wide range of NLP tasks. Also, it provides a lot of interfaces such as openai embedding models, pinecone, and text splitter. Which is really help for us to develop and integrate other components.

2. *Pinecone Vectore Store:*
  Utilizing Pinecone as our vector store offers several compelling advantages for our knowledge-intensive NLP tasks. Pinecone excels in the efficient creation and management of indexes, even for very large datasets. This efficiency is particularly evident in the vector similarity search, where Pinecone's sophisticated indexing algorithms quickly identify the top k most relevant chunks with remarkable precision. Its ability to fetch these pertinent chunks swiftly greatly enhances the performance of retrieval-augmented models, ensuring that the most contextually appropriate data is used for generating responses. Moreover, Pinecone's architecture is designed to scale seamlessly, accommodating growing data without compromising on speed or accuracy, making it an ideal choice for AI-driven applications where real-time results are paramount.

1. *Streamlit for User Interface (UI):*
  Streamlit serves as the backbone for our system's user interface. Its simplicity and flexibility allow us to create an intuitive and user-friendly platform for posing questions and retrieving information. Streamlit's capabilities in rapidly creating interactive web applications enable a seamless frontend experience for users. Through Streamlit, we provide a visually appealing and accessible interface that enhances the overall user experience.

  4. *Retrieval-Augmented Generation Model:*
   In our project, we integrate the state-of-the-art RAG (Retrieval-Augmented Generation) model within the robust Langchain framework, leveraging Pinecone's vector search capabilities and Streamlit's interactive interface. This synergy facilitates fine-grained retrieval and precise text chunking, essential for a sophisticated Question-Answering (QA) system. The RAG model is pivotal in refining our system's contextual comprehension, enabling it to generate responses that are not only accurate but also contextually enriched. By doing so, it markedly enhances the system's efficiency in sourcing and synthesizing information, thereby producing highly relevant and precise answers drawn from large language models (LLMs). This approach ensures that the answers our QA system generates are informed by a deep and nuanced understanding of the subject matter, setting a new benchmark for accuracy and reliability in user query responses.



## **Experimental Setup and Results** (Jiufeng Li)

### Data
Our project harnesses an automated data aggregation pipeline to meticulously curate a dataset from the PubMed repository, targeting medical articles published between 2013 and 2023 within the ambit of intelligence studies.

The gathered dataset undergoes a rigorous preprocessing regimen, where we employ the Langchain text splitter to judiciously segment abstracts into smaller, well-defined chunks. This granular approach not only enhances the accessibility of the information but also preserves the rich metadata accompanying each article—ranging from the title and authors to DOIs and publication dates—thereby enriching the user experience with insightful context.

Figure 2 encapsulates our robust data model, showcasing the systematic organization and readiness for user interaction.

![Data model](/assets/images/data_model.png "The Pipeline of Our Medical Question-Answering System")
<p align="center">Figure 2: Data model</p>

```python
def custom_text_splitter(text, chunk_size):
    """Split the text into chunks of specified size."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=20
    )
    return text_splitter.create_documents(text)
```

```python
#core code of chunking
abstract = record['AB']
chunks = custom_text_splitter([abstract], chunk_size)
for chunk_id, chunk in enumerate(chunks, start=1):
    print(f"Processing chunk {chunk_id}\n")
    pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{record.get('PMID')}/"
    article = {
        'id': record.get('PMID', ''),
        'doi': record.get('AID', ''),
        'title': record.get('TI', ''),
        'abstract': record.get('AB', ''),
        'chunk-id': chunk_id,
        'chunk': chunk,
        'authors': record.get('AU', []),
        'journal_ref': record.get('JT', ''),
        'published': record.get('DP', ''),
        'source': pubmed_url
    }
    # Depending on the availability and requirement, adjust the fields
    articles.append(article)
```
Subsequently, the partitioned data undergoes JSON serialization, culminating in a structured and query-ready dataset.

Moreover, our system's versatility is showcased by its capacity to adapt to various fields beyond 'intelligence.' By simply specifying an alternative `search_term`, users can leverage our `load_data` function to tailor the dataset to their specific research interests.

### Pipeline 
1. Initializing pinecone vector store  
Our project is underpinned by the Pinecone vector store, which serves as the foundational infrastructure for managing and retrieving vectorized data. The initialization of Pinecone is a critical first step in our workflow, setting the stage for the advanced capabilities that follow.


```python
# Initializing the Pinecone vector store
def load_vectorstore(self):
    from pinecone import Pinecone
    return Pinecone(api_key=self.PINECONE_API_KEY)
```
2. Initializing embedding model for creating embeddings of chunk
```python
# We use ada as our embed model
self.embedding_model = "text-embedding-ada-002" 
self.embed_model = OpenAIEmbeddings(model=self.embedding_model, api_key=self.OPENAI_API_KEY)
```

3. Creating index
```python
# check if index already exists (it shouldn't if this is first time)
if self.index_name not in existing_indexes:
    # if does not exist, create index
    vector_store.create_index(
        self.index_name,
        dimension=1536,  # dimensionality of ada 002
        metric='dotproduct',
        spec=self.spec
    )
```

4. Upsert all the embeddings to pinecone in batch   
Once the index has been successfully created, we iteratively upsert each embedding into the vector store.
```python
for i in tqdm(range(0, len(dataset), batch_size)):
        i_end = min(len(dataset), i + batch_size)
        # get batch of data
        batch = dataset.iloc[i:i_end]

        # generate unique ids for each chunk
        ids = [f"{x['articles']['id']}-{x['articles']['chunk-id']}" for i, x in batch.iterrows()]
        # get text to embed
        texts = [x['articles']['chunk'] for _, x in batch.iterrows()]
        # embed text
        embeds = self.embed_model.embed_documents(texts)
        # get metadata to store in Pinecone
        metadata = [
            {'text': x['articles']['chunk'],
             'source': x['articles']['source'],
             'title': x['articles']['title'],
             'authors': x['articles']['authors'],
             'journal_ref': x['articles']['journal_ref'],
             'published': x['articles']['published']
             } for i, x in batch.iterrows()
        ]
        # add to Pinecone
        index.upsert(vectors=zip(ids, embeds, metadata))
print(index.describe_index_stats())
```
5. Argument Prompt  
Before creating the final query, we use an argument prompt to formulate the final query prompt and pass it to the LLM, such as ChatGPT, to generate the final answer.

```python
def augment_prompt(self, query, k, vectorstore):
      # get top k results from knowledge base
      results = vectorstore.similarity_search(query, k)
      # get the text from the results
      source_knowledge = "\n".join([x.page_content for x in results])
      # feed into an augmented prompt
      augmented_prompt = f"""Using the contexts below, answer the query.
      
      Contexts:
      {source_knowledge}

      Query: {query}"""
      return augmented_prompt, results
```

6. Answering  
After passing the argument prompt to the pretrained LLM, we then receive the final answers to the query.

```python
# Create a llm model
chat = ChatOpenAI(
            openai_api_key=self.OPENAI_API_KEY,
            model=chat_model
        )

# Initialize the chat message queue
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Hi AI, how are you today?"),
    AIMessage(content="I'm great thank you. How can I help you?")
]
augment_prompt, retriever_results = self.augment_prompt(query, k, vectorstore)
prompt = HumanMessage(
    content=augment_prompt
)

# Answerring
res = chat(messages + [prompt])
```

7. Streamlit UI  
To present the results to users, our project utilizes Streamlit, an agile web framework renowned for its simplicity and efficiency. The interface elegantly showcases the *Answer* and the *Top-k Retrieval Results*, with each chunk distinctly highlighted for clarity. Additionally, the application provides interactive elements, allowing users to customize parameters such as 'k' for retrieval depth and to choose from various chat models to tailor the response generation to their preferences.

![Streamlit UI](/assets/images/screenshot.png "UI")
<p align="center">Figure 3: Streamlit UI of Our Medical Question-Answering System</p>

### Evaluation Method (Jiufeng Li, Aayush Manoj Tirmalle)
In evaluating Question Answering (QA) systems, especially those leveraging Large Language Models (LLMs), metrics that accurately reflect the models' understanding and generation capabilities are crucial. Traditional metrics like accuracy, precision, recall, and the F1 score [Chang et al. 2023]<sup>3</sup> provide valuable insights but often fail to capture the semantic nuances of language comprehensively.

The evaluation based on embeddings offers a more granular and semantically rich approach. By converting text into fixed-length vector representations, we can compare the similarity between the embeddings of predicted answers and ground truth answers. This method acknowledges that there are often multiple correct ways to answer a question and that semantic similarity can be a more nuanced indicator of quality than exact lexical matches.

We choose to use embeddings for evaluation because they map text to a high-dimensional space where the distance between vectors corresponds to semantic similarity. This transformation allows for a straightforward comparison of the content, irrespective of the specific choice of words or phrasing.

Cosine similarity is particularly well-suited for this task as it measures the cosine of the angle between two vectors, effectively capturing the orientation (and thus the similarity) regardless of their magnitude. It is a widely-accepted method for comparing document similarity in various natural language processing applications due to its effectiveness in high-dimensional spaces and its robustness to differences in vector length.

In essence, the use of embeddings and cosine similarity in evaluating QA systems reflects a move towards more context-aware, semantically-focused assessment criteria, aligning more closely with the way humans understand and process language. This method provides a more sophisticated means of assessing the performance of LLMs in QA tasks, which is particularly relevant as these models become increasingly central to AI-driven applications.


![Evaluation](/assets/images/evaluation_plot.png )
<p align="center">Figure 4: Evaluation of Our RAG Model</p>
As delineated in Figure 4, the Retrieval-Augmented Generation (RAG) model, enhanced with a pretrained ChatGPT-3.5-turbo model, demonstrates superior performance over standalone GPT-3 and GPT-4 models in the absence of an external knowledge base. This empirical evidence suggests that augmenting pretrained large language models (LLMs) with contextually relevant information segments substantially boosts the model's accuracy. The efficacy of the RAG model is attributable to its strategic integration of retrieval mechanisms, which dynamically enrich the LLM's response generation process with pertinent data, thereby refining its decision-making and predictive capabilities.


### Results
The implementation of the Retrieval-Augmented Generation (RAG) model, incorporating chunks of abstracts, has been shown to enhance the efficiency of our Question Answering (QA) system. By integrating concise and relevant informational snippets directly into the response generation workflow, the RAG model effectively contextualizes the input queries, which enables the QA system to produce more accurate and contextually relevant answers. This methodology capitalizes on the synergistic potential of retrieval-augmented strategies, confirming that the judicious use of targeted data extracts can elevate the performance of large language models in complex QA tasks.


## **Conclusion and Future Work** (Aayush Manoj Tirmalle)
In conclusion, the combination of RAG model and pretrained llm forms a cohesive and powerful solution for efficient question answering in the medical domain. Our project not only leverages state-of-the-art tools but also tailors them to meet the specific needs of medical information retrieval. 

Looking forward, we plan to continue refining our system by staying abreast of advancements in langchain, Pinecone vector store, and Streamlit. Continuous evaluation of the Langchian rag model on evolving our datasets will be a priority. Also, trying to test on different embedding models may helps to improve the accuracy of our system. Additionally, user feedback will guide us in making iterative improvements to the UI, ensuring that the system remains intuitive and user-friendly.

## **References** (Aayush Manoj Tirmalle)
  1. Chen, D., Fisch, A., Weston, J., & Bordes, A. (2017). Reading Wikipedia to Answer Open-Domain Questions. *arXiv preprint arXiv:1704.00051*.
  2. Rajpurkar, P., Zhang, J., Lopyrev, K., & Liang, P. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text. arXiv preprint arXiv:1606.05250.
  3. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. Advances in Neural Information Processing Systems, 33, 9459-9474.
  4. Chang, Y., Wang, X., Wang, J., Wu, Y., Yang, L., Zhu, K., et al. (2023). A Survey on Evaluation of Large Language Models. *arXiv preprint arXiv:2307.03109*.
