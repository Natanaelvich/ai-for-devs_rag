from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
import os
from typing import List, Dict, Any, Tuple

class PDFRAGService:
    def __init__(self):
        # Initialize OpenAI components
        self.embeddings_model = OpenAIEmbeddings()
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            max_tokens=500,
        )
        
        # Load initial PDF if it exists
        self.pdf_path = "DOC-SF238339076816-20230503.pdf"
        self._initialize_vector_store()
        
    def _initialize_vector_store(self):
        """Initialize the vector store with the default PDF if available"""
        try:
            # Check if we have a persisted vector store
            if os.path.exists("text_index"):
                self.vectordb = Chroma(
                    persist_directory="text_index", 
                    embedding_function=self.embeddings_model
                )
            # Otherwise, check if we have the default PDF
            elif os.path.exists(self.pdf_path):
                # Load the PDF
                loader = PyPDFLoader(self.pdf_path, extract_images=False)
                pages = loader.load_and_split()
                
                # Split into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=4000,
                    chunk_overlap=20,
                    length_function=len,
                    add_start_index=True,
                )
                chunks = text_splitter.split_documents(pages)
                
                # Create vector store
                self.vectordb = Chroma.from_documents(
                    chunks, 
                    embedding=self.embeddings_model, 
                    persist_directory="text_index"
                )
                self.vectordb.persist()
            else:
                # Create empty vector store
                self.vectordb = Chroma(
                    embedding_function=self.embeddings_model,
                    persist_directory="text_index"
                )
                
            # Initialize retriever and chain
            self.retriever = self.vectordb.as_retriever(search_kwargs={"k": 3})
            self.chain = load_qa_chain(self.llm, chain_type="stuff")
            
        except Exception as e:
            print(f"Error initializing vector store: {str(e)}")
            # Create empty vector store as fallback
            self.vectordb = Chroma(
                embedding_function=self.embeddings_model,
                persist_directory="text_index"
            )
            self.retriever = self.vectordb.as_retriever(search_kwargs={"k": 3})
            self.chain = load_qa_chain(self.llm, chain_type="stuff")
    
    def ask(self, question: str) -> Tuple[str, List]:
        """
        Process a query about the PDF documents
        
        Args:
            question (str): The question to ask about the documents
            
        Returns:
            Tuple[str, List]: The answer and the context documents used
        """
        context = self.retriever.get_relevant_documents(question)
        answer = self.chain(
            {"input_documents": context, "question": question}, 
            return_only_outputs=True
        )["output_text"]
        
        return answer, context
    
    def add_documents(self, file_paths: List[str], allowed_extensions: set) -> Dict[str, Any]:
        """
        Add PDF documents to the vector store
        
        Args:
            file_paths (List[str]): List of paths to the PDF files
            allowed_extensions (set): Set of allowed file extensions
            
        Returns:
            Dict[str, Any]: Status of the operation
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
                
                # Load the PDF
                loader = PyPDFLoader(file_path, extract_images=False)
                pages = loader.load_and_split()
                
                # Split into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=4000,
                    chunk_overlap=20,
                    length_function=len,
                    add_start_index=True,
                )
                chunks = text_splitter.split_documents(pages)
                
                # Add to vector store
                self.vectordb.add_documents(chunks)
                self.vectordb.persist()
                
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