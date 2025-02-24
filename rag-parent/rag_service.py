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
import os
from typing import List

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

    def add_documents(self, file_paths: List[str], allowed_extensions: List[str] = ['.pdf']) -> dict:
        """
        Add new documents to both vector stores.
        
        Args:
            file_paths (List[str]): List of paths to the documents to be added
            allowed_extensions (List[str]): List of allowed file extensions
            
        Returns:
            dict: Status of the operation with details about processed files
        """
        processed_files = []
        failed_files = []
        
        for file_path in file_paths:
            try:
                # Validate file extension
                _, ext = os.path.splitext(file_path)
                if ext.lower() not in allowed_extensions:
                    failed_files.append({
                        "file": file_path,
                        "error": f"Unsupported file type. Allowed types: {', '.join(allowed_extensions)}"
                    })
                    continue

                # Load and process document
                loader = PyPDFLoader(file_path, extract_images=False)
                pages = loader.load_and_split()

                # Add to naive vector store
                self.naive_vectordb.add_documents(pages)
                
                # Add to parent document store
                self.parent_retriever.add_documents(pages)
                
                processed_files.append(file_path)
                
            except Exception as e:
                failed_files.append({
                    "file": file_path,
                    "error": str(e)
                })
        
        return {
            "status": "completed",
            "processed_files": processed_files,
            "failed_files": failed_files,
            "total_processed": len(processed_files),
            "total_failed": len(failed_files)
        } 