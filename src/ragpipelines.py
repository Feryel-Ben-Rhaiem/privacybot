# Import necessary modules and libraries
import os
import openai
from llama_index.core import SimpleDirectoryReader
from llama_index.core import load_index_from_storage
from llama_index.core import Document
from llama_index.core import VectorStoreIndex
from llama_index.core import ServiceContext
from llama_index.llms.openai import OpenAI
import numpy as np
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.indices.postprocessor import SentenceTransformerRerank
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.node_parser import get_leaf_nodes
from llama_index.core import StorageContext
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

from operator import itemgetter

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import FewShotChatMessagePromptTemplate
from langchain_openai import ChatOpenAI

# ------------------------- BASIC RAG PIPELINE -------------------------

class BasicRAGPipeline:
    def __init__(self, api_key, document_path, llm_model, embed_model):
        # Set the API key for OpenAI
        os.environ['OPENAI_API_KEY'] = api_key
        openai.api_key = os.environ["OPENAI_API_KEY"]

        # Load documents and combine them into a single text block
        self.documents = SimpleDirectoryReader(input_files=[document_path]).load_data()
        self.document = Document(text="\n\n".join([doc.text for doc in self.documents]))

        # Initialize LLM and embedding model
        self.llm = OpenAI(model=llm_model, temperature=0.1)
        self.embed_model = embed_model
        self.service_context = self._create_service_context()
        
        # Create an index for retrieval
        self.index = self._create_index()
        self.query_engine = self.index.as_query_engine()

    def _create_service_context(self):
        # Set up the service context with LLM and embedding model
        return ServiceContext.from_defaults(llm=self.llm, embed_model=self.embed_model)

    def _create_index(self):
        # Create a vector store index from the loaded documents
        return VectorStoreIndex.from_documents([self.document], service_context=self.service_context)

    def query(self, query_text):
        # Query the index and return the result
        response = self.query_engine.query(query_text)
        return str(response)

# ------------------------- SENTENCE WINDOW RETRIEVAL PIPELINE -------------------------

class SentenceWindowRAGPipeline:
    def __init__(self, api_key, document_path, llm_model, embed_model, rerank=True):
        os.environ['OPENAI_API_KEY'] = api_key
        openai.api_key = os.environ["OPENAI_API_KEY"]

        # Load documents and initialize LLM and embedding model
        self.documents = SimpleDirectoryReader(input_files=[document_path]).load_data()
        self.llm = OpenAI(model=llm_model, temperature=0.1)
        self.embed_model = embed_model

        # Build a sentence window index
        self.sentence_window_index = self._build_sentence_window_index()

        # Set up post-processing and re-ranking, if enabled
        self.postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
        self.rerank_enabled = rerank
        self.rerank = SentenceTransformerRerank(top_n=2, model="BAAI/bge-reranker-base")

        # Initialize the query engine with or without re-ranking
        self.query_engine = self._initialize_query_engine()

    def _build_sentence_window_index(self):
        # Parse the document into sentence windows and create an index
        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original-text"
        )
        sentence_context = ServiceContext.from_defaults(
            llm=self.llm,
            embed_model=self.embed_model,
            node_parser=node_parser
        )
        sentence_index = VectorStoreIndex.from_documents(
            documents=self.documents, service_context=sentence_context
        )
        return sentence_index

    def _initialize_query_engine(self):
        # Use re-ranking to refine query results if enabled
        if self.rerank_enabled:
            query_engine = self.sentence_window_index.as_query_engine(
                similarity_top_k=6, node_postprocessors=[self.postproc, self.rerank]
            )
        else:
            query_engine = self.sentence_window_index.as_query_engine(
                similarity_top_k=6, node_postprocessors=[self.postproc]
            )
        return query_engine

    def query(self, query_text):
        # Query the index and return the result
        response = self.query_engine.query(query_text)
        return str(response)

# ------------------------- AUTO-MERGING RETRIEVAL PIPELINE -------------------------

class AutoMergingRAGPipeline:
    def __init__(self, api_key, document_path, llm_model, embed_model, save_dir="merging_index", chunk_sizes=None):
        os.environ['OPENAI_API_KEY'] = api_key
        openai.api_key = os.environ["OPENAI_API_KEY"]

        # Load documents and initialize LLM and embedding model
        self.documents = SimpleDirectoryReader(input_files=[document_path]).load_data()
        self.llm = OpenAI(model=llm_model, temperature=0.1)
        self.embed_model = embed_model
        self.chunk_sizes = chunk_sizes or [2048, 512, 128]
        self.save_dir = save_dir

        # Build an auto-merging index
        self.automerging_index = self._build_automerging_index()
        self.query_engine = self._get_automerging_query_engine()

    def _build_automerging_index(self):
        # Parse the document into hierarchical chunks and store them
        node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=self.chunk_sizes)
        nodes = node_parser.get_nodes_from_documents(self.documents)
        leaf_nodes = get_leaf_nodes(nodes)
        merging_context = ServiceContext.from_defaults(
            llm=self.llm,
            embed_model=self.embed_model,
        )
        storage_context = StorageContext.from_defaults()
        storage_context.docstore.add_documents(nodes)

        # Create or load the auto-merging index
        if not os.path.exists(self.save_dir):
            automerging_index = VectorStoreIndex(
                leaf_nodes, storage_context=storage_context, service_context=merging_context
            )
            automerging_index.storage_context.persist(persist_dir=self.save_dir)
        else:
            automerging_index = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=self.save_dir),
                service_context=merging_context,
            )
        return automerging_index

    def _get_automerging_query_engine(self, similarity_top_k=12, rerank_top_n=2):
        # Set up a retriever with auto-merging capabilities and optional re-ranking
        base_retriever = self.automerging_index.as_retriever(similarity_top_k=similarity_top_k)
        retriever = AutoMergingRetriever(
            base_retriever, self.automerging_index.storage_context, verbose=True
        )
        rerank = SentenceTransformerRerank(
            top_n=rerank_top_n, model="BAAI/bge-reranker-base"
        )
        auto_merging_engine = RetrieverQueryEngine.from_args(
            retriever, node_postprocessors=[rerank]
        )
        return auto_merging_engine

    def query(self, query_text):
        # Query the index and return the result
        response = self.query_engine.query(query_text)
        return str(response)
    
# ------------------------- AUTO-MERGING RETRIEVAL WITH PROMPTING PIPELINE -------------------------

class AutoMergingRAGPipelineWithPrompting:
    def __init__(self, api_key, document_path, llm_model, embed_model, save_dir="merging_index", chunk_sizes=None):
        # Set the API key for OpenAI
        os.environ['OPENAI_API_KEY'] = api_key
        openai.api_key = os.environ["OPENAI_API_KEY"]

        # Load documents and initialize LLM and embedding model
        self.documents = SimpleDirectoryReader(input_files=[document_path]).load_data()
        self.llm = OpenAI(model=llm_model, temperature=0.1)
        self.embed_model = embed_model
        self.chunk_sizes = chunk_sizes or [2048, 512, 128]  # Default chunk sizes for hierarchical indexing
        self.save_dir = save_dir

        # Build an auto-merging index from documents
        self.automerging_index = self._build_automerging_index()
        self.query_engine = self._get_automerging_query_engine()

        # Set up LangChain for few-shot prompting to guide response generation
        self.llm_langchain = ChatOpenAI(openai_api_key=api_key, model_name="gpt-4")
        self.few_shot_prompt = self._setup_few_shot_prompt()

    def _build_automerging_index(self):
        # Parse the document into hierarchical chunks and store them in an index
        node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=self.chunk_sizes)
        nodes = node_parser.get_nodes_from_documents(self.documents)
        leaf_nodes = get_leaf_nodes(nodes)  # Get the lowest-level nodes
        merging_context = ServiceContext.from_defaults(llm=self.llm, embed_model=self.embed_model)
        storage_context = StorageContext.from_defaults()
        storage_context.docstore.add_documents(nodes)

        # Create or load the auto-merging index depending on the existence of the save directory
        if not os.path.exists(self.save_dir):
            automerging_index = VectorStoreIndex(leaf_nodes, storage_context=storage_context, service_context=merging_context)
            automerging_index.storage_context.persist(persist_dir=self.save_dir)
        else:
            automerging_index = load_index_from_storage(StorageContext.from_defaults(persist_dir=self.save_dir), service_context=merging_context)
        return automerging_index

    def _get_automerging_query_engine(self, similarity_top_k=12, rerank_top_n=2):
        # Configure a retriever with auto-merging and optional re-ranking
        base_retriever = self.automerging_index.as_retriever(similarity_top_k=similarity_top_k)
        retriever = AutoMergingRetriever(base_retriever, self.automerging_index.storage_context, verbose=True)
        rerank = SentenceTransformerRerank(top_n=rerank_top_n, model="BAAI/bge-reranker-base")
        auto_merging_engine = RetrieverQueryEngine.from_args(retriever, node_postprocessors=[rerank])
        return auto_merging_engine

    def _setup_few_shot_prompt(self):
        # Set up few-shot examples to guide the LLM in generating accurate responses
        few_shot_examples = [
            {"input": "What measures does PMI take when transferring personal data to countries with different data protection standards?", 
             "output": "Philip Morris International (PMI) ensures that when personal data is transferred to countries with different data protection standards, appropriate measures are taken to protect that data in accordance with applicable laws. PMI may transfer personal data based on a European Commission adequacy decision, which certifies that the destination country provides an adequate level of data protection. In cases where such a decision is not available, PMI relies on standard contractual clauses approved by the European Commission or other equivalent mechanisms to safeguard the data. Additionally, data transfers may be necessary to fulfill contractual obligations between the user and PMI or to meet specific requests related to the services provided by PMI."},
            {"input": "How does PMI handle cookies and tracking technologies on its digital touchpoints?", 
             "output": "PMI utilizes cookies and similar tracking technologies across its digital touchpoints to enhance user experience and gather valuable data for business analytics. These technologies allow PMI to collect information about user interactions with its websites and emails, such as the pages visited, the duration of visits, and the content viewed. PMI also employs technologies like web beacons and pixels to monitor user activities across its digital platforms and third-party websites. To respect user preferences, PMI provides options for managing cookie settings, enabling users to accept or refuse certain types of cookies. Detailed information about these mechanisms and how to control them is made available to users on the digital platforms of PMI."},
            {"input": "What specific rights do users in Japan have regarding their personal information?", 
             "output": "Users in Japan are granted specific rights regarding their personal information, and PMI is dedicated to respecting and upholding these rights. Japanese users have the right to be informed about how their personal data will be used and shared, particularly when it involves 'joint use' with other PMI affiliates under Japanese data protection laws. PMI is also responsible for ensuring that any third parties with whom data is shared manage the information responsibly and in compliance with the same data protection standards. Additionally, users in Japan may exercise their rights to access, correct, or delete their personal data, following the procedures established by PMI."}
        ]

        # Create a prompt template with few-shot examples and structured guidance
        few_shot_template = ChatPromptTemplate.from_messages([("human", "{input}"), ("ai", "{output}")])
        few_shot_prompt = FewShotChatMessagePromptTemplate(example_prompt=few_shot_template, examples=few_shot_examples)

        # Construct the final prompt that guides the LLM to generate responses based on the document
        priv_prompt = ChatPromptTemplate.from_messages([
            ("system", 
            "You are a legal expert specializing in PMI's privacy policy and its handling of personal data. Your name is PrivacyBot. Your goal is to provide accurate, concise, and contextually relevant answers based strictly on the information provided in the document. "
            "Avoid using external knowledge. If the document does not contain the necessary information to answer the query, explicitly state that the information is not available. "
            "Always ensure your response is in a professional tone. "
            "Structure your answers clearly and logically, using bullet points or numbered lists if multiple points are addressed. "
            "Avoid speculation or adding any personal opinions. "
            "If the context provided is vague or incomplete, ask clarifying questions instead of assuming details. "
            "Finally, ensure that all responses are compliant with privacy regulations and PMI’s official guidelines."),
            few_shot_prompt,
            ("user", "{question}"),
            ("user", "{context}")
        ])

        return priv_prompt

    def query(self, query_text):
        # Retrieve relevant context using the auto-merging query engine
        context = self.query_engine.query(query_text)
        
        # Format the full prompt with retrieved context and generate a response using the LLM
        full_prompt = self.few_shot_prompt.format_messages(question=query_text, context=context)
        response = self.llm_langchain(full_prompt)

        return str(response.content)