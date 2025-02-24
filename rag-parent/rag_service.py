from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryByteStore
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

class RAGService:
    def __init__(self):
        # Initialize OpenAI components
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            max_tokens=500,
        )

        # Load and process PDF
        self.pdf_path = "DOC-SF238339076816-20230503.pdf"
        self._initialize_document_stores()
        self._setup_retrievers()
        self._setup_chains()

    def _initialize_document_stores(self):
        # Load PDF
        loader = PyPDFLoader(self.pdf_path, extract_images=False)
        self.pages = loader.load_and_split()

        # Initialize text splitters
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True
        )

        # Initialize vector stores
        self.naive_vectordb = Chroma.from_documents(
            self.pages,
            embedding=self.embeddings,
            persist_directory="naiveDB"
        )

        self.child_vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory="chieldVectorDB"
        )

    def _setup_retrievers(self):
        # Setup naive retriever with reranking
        self.naive_retriever = self.naive_vectordb.as_retriever(search_kwargs={"k": 10})
        rerank = CohereRerank(model="rerank-v3.5", top_n=3)
        self.compressor_retriever = ContextualCompressionRetriever(
            base_compressor=rerank,
            base_retriever=self.naive_retriever,
        )

        # Setup parent document retriever
        store = InMemoryByteStore()
        self.parent_retriever = ParentDocumentRetriever(
            vectorstore=self.child_vectorstore,
            docstore=store,
            child_splitter=self.child_splitter,
            parent_splitter=self.parent_splitter,
        )
        self.parent_retriever.add_documents(self.pages)

    def _setup_chains(self):
        # Common prompt template
        template = """
        Você é um especialista em legislação e tecnologia. Responda a pergunta abaixo utilizando o contexto informado.
        Query:
        {question}
        Context:
        {context}
        """
        rag_prompt = ChatPromptTemplate.from_template(template)
        output_parser = StrOutputParser()

        # Setup rerank chain
        rerank_retrieval = RunnableParallel({
            "question": RunnablePassthrough(),
            "context": self.compressor_retriever
        })
        self.rerank_chain = rerank_retrieval | rag_prompt | self.llm | output_parser

        # Setup parent chain
        parent_retrieval = RunnableParallel({
            "question": RunnablePassthrough(),
            "context": self.parent_retriever
        })
        self.parent_chain = parent_retrieval | rag_prompt | self.llm | output_parser

    def process_query(self, question: str, method: str = "rerank") -> str:
        """
        Process a query using the specified RAG method.
        
        Args:
            question (str): The question to be answered
            method (str): The RAG method to use ('rerank' or 'parent')
        
        Returns:
            str: The generated response
        """
        if method == "rerank":
            return self.rerank_chain.invoke(question)
        elif method == "parent":
            return self.parent_chain.invoke(question)
        else:
            raise ValueError("Invalid method. Use 'rerank' or 'parent'.") 