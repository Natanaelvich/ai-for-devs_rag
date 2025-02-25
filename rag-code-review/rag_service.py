from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
import shutil
from git import Repo
from typing import List, Dict, Any, Optional

class CodeReviewRAGService:
    def __init__(self):
        # Initialize OpenAI components
        self.embeddings = OpenAIEmbeddings(disallowed_special=())
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            max_tokens=500,
        )
        
        # Initialize repository path
        self.repo_path = "./test_repo"
        
        # Initialize vector store if it exists
        self._initialize_vector_store()
        self._setup_chain()
        
    def _initialize_vector_store(self):
        """Initialize the vector store with existing code documents if available"""
        try:
            # Check if we have documents to load
            if os.path.exists(self.repo_path):
                # Load code files
                loader = GenericLoader.from_filesystem(
                    self.repo_path + "/",
                    glob="**/*",
                    suffixes=[".py", ".js", ".ts", ".java", ".c", ".cpp", ".cs", ".go", ".rb", ".php", ".ipynb"],
                    exclude=["**/non-utf-8-encoding.py", "**/node_modules/**", "**/.git/**"],
                    parser=LanguageParser(language=Language.PYTHON, parser_threshold=500)
                )
                
                documents = loader.load()
                
                # Split documents
                python_splitter = RecursiveCharacterTextSplitter.from_language(
                    language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
                )
                
                texts = python_splitter.split_documents(documents)
                
                # Create vector store
                self.db = Chroma.from_documents(texts, self.embeddings)
                self.retriever = self.db.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 8},
                )
            else:
                # Create empty vector store
                self.db = Chroma(embedding_function=self.embeddings)
                self.retriever = self.db.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 8},
                )
        except Exception as e:
            print(f"Error initializing vector store: {str(e)}")
            # Create empty vector store as fallback
            self.db = Chroma(embedding_function=self.embeddings)
            self.retriever = self.db.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 8},
            )
    
    def _setup_chain(self):
        """Set up the retrieval chain for code review"""
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Você é um revisor de código experiente. Forneça informações detalhadas sobre a revisão do código e sugestões de melhorias baseadas no contexto fornecido abaixo: \n\n{context}",
            ),
            ("user", "{input}"),
        ])
        
        document_chain = create_stuff_documents_chain(self.llm, prompt)
        self.retrieval_chain = create_retrieval_chain(self.retriever, document_chain)
    
    def process_query(self, question: str) -> str:
        """
        Process a code review query
        
        Args:
            question (str): The question about the code
            
        Returns:
            str: The generated code review response
        """
        response = self.retrieval_chain.invoke({"input": question})
        return response["answer"]
    
    def add_repository(self, repository_url: str, branch: str = "main") -> Dict[str, Any]:
        """
        Clone a repository and add its code to the vector store
        
        Args:
            repository_url (str): URL of the Git repository
            branch (str): Branch to clone (default: main)
            
        Returns:
            Dict[str, Any]: Status of the operation
        """
        try:
            # Remove existing repo if it exists
            if os.path.exists(self.repo_path):
                shutil.rmtree(self.repo_path)
            
            # Clone the repository
            repo = Repo.clone_from(repository_url, to_path=self.repo_path, branch=branch)
            
            # Load code files
            loader = GenericLoader.from_filesystem(
                self.repo_path + "/",
                glob="**/*",
                suffixes=[".py", ".js", ".ts", ".java", ".c", ".cpp", ".cs", ".go", ".rb", ".php", ".ipynb"],
                exclude=["**/non-utf-8-encoding.py", "**/node_modules/**", "**/.git/**"],
                parser=LanguageParser(language=Language.PYTHON, parser_threshold=500)
            )
            
            documents = loader.load()
            
            # Split documents
            python_splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
            )
            
            texts = python_splitter.split_documents(documents)
            
            # Create new vector store
            self.db = Chroma.from_documents(texts, self.embeddings)
            self.retriever = self.db.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 8},
            )
            
            # Update the chain with the new retriever
            self._setup_chain()
            
            return {
                "status": "success",
                "message": f"Repository cloned and indexed successfully",
                "document_count": len(texts)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to clone repository: {str(e)}"
            }
    
    def add_code_files(self, file_paths: List[str], allowed_extensions: set) -> Dict[str, Any]:
        """
        Add code files to the vector store
        
        Args:
            file_paths (List[str]): List of paths to the code files
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
                
                # Load the file
                loader = GenericLoader.from_filesystem(
                    os.path.dirname(file_path),
                    glob=os.path.basename(file_path),
                    suffixes=[ext],
                    parser=LanguageParser(language=Language.PYTHON, parser_threshold=500)
                )
                
                documents = loader.load()
                
                # Split documents
                python_splitter = RecursiveCharacterTextSplitter.from_language(
                    language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
                )
                
                texts = python_splitter.split_documents(documents)
                
                # Add to vector store
                self.db.add_documents(texts)
                
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