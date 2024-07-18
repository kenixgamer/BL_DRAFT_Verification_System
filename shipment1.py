


# from langchain_community.vectorstores import FAISS



# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_chroma import Chroma
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.llms import Ollama
# from langchain.chains import ConversationChain
# from langchain.chains.conversation.memory import ConversationBufferWindowMemory
# from langchain.prompts import PromptTemplate
# from langchain_community.vectorstores import FAISS
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_community.document_loaders import WebBaseLoader
# # # Initialize the embeddings
# embeddings = OllamaEmbeddings(model="mistral")



# # Load the PDF document
# loader = PyPDFLoader("AVANA DRAFT-Draft.pdf")
# pages = loader.load_and_split()

# # Define the text splitter for manageable chunks
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
# all_splits = text_splitter.split_documents(documents=pages)

# # Initialize the Chroma vector store and create it using the document splits
# db = FAISS.from_documents(all_splits, embeddings)

# # db = Chroma.from_documents(all_splits, embeddings)


# # Optionally, to print some details about the vector store
# print(db)
# # Add any other relevant information you'd like to inspect

# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough

# template = """Answer the question based only on the following context:
# | Feature                | Document: Draft (AVANA DRAFT-Draft.pdf) | Document: Reference (AVANA DRAFT-Reference.pdf) |
# |------------------------|-----------------------------------------|-------------------------------------------------|
# | Bill of Lading No.     | MUN/JEA/24/14740                        | BL  DRAFT 484                                   |
# | Shipper                | Punjab Riceland Agro Foods Pvt Ltd      | Punjab Riceland Agro Foods Pvt Ltd              |
# | Consignee              | AL Janan General Trading Co. LLC        | AL Janan General Trading Co. LLC                |
# | Port of Loading        | Mundra                                  | Mundra                                          |
# | Port of Discharge      | Jebel Ali                               | Jebel Ali                                       |
# | Final Destination      | Jebel Ali                               | Jebel Ali                                       |
# | Vessel/Voyage          | SC Mara/24015W                          | -                                               |
# | Total No. of Packages  | 658 Bags                                | 658 Bags                                        |
# | Gross Weight (Total)   | 25,142 kg                               | 25,142 kg                                       |
# | Net Weight (Total)     | 25,004 kg                               | 25,004 kg                                       |
# | Notify Party 1         | -                                       | AL Janan General Trading Co. LLC                |
# | Notify Party 2         | -                                       | Harmony Ventures General Trading FZCO           |
# | Invoice No.            | PRL/GDM/EXP/484                         | PRL/GDM/EXP/484                                 |
# | Packing Date           | 03/2024                                 | 03/2024                                         |
# | Expiry Date            | 02/2026                                 | 02/2026                                         |
# | Remarks                | -                                       | Carrier not responsible for loss or damage      |

# {context}


# """
# prompt = ChatPromptTemplate.from_template(template)
# llm = Ollama(model="mistral")



# document_chain = create_stuff_documents_chain(llm,prompt)
# retriever = db.as_retriever()
# retrieval_chain = create_retrieval_chain(retriever, document_chain)
# # Assuming 'invoke' needs to pass a dictionary with specific keys
# response = retrieval_chain.invoke({"input": "what is data about"})
# print(response)
# # chain = (
# #     {"context": retriever | format_docs, "question": RunnablePassthrough()}
# #     | prompt
# #     | model
# #     | StrOutputParser()
# # )

# # chain.invoke("What did the president say about technology?")

























































from langchain_community.vectorstores import FAISS


from langchain_community.document_loaders import PyPDFLoader

from langchain_community.embeddings import OllamaEmbeddings
# from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
# # Initialize the embeddings
embeddings = OllamaEmbeddings(model="mistral")
pdf_filepaths = [
    "AVANA DRAFT-Draft.pdf",
    "AVANA DRAFT-Reference.pdf"
    # Add more file paths as needed
]

# Load and split the PDFs into pages
# pages = []
# for filepath in pdf_filepaths:
#     loader = PyPDFLoader(filepath)
#     pages.extend(loader.load_and_split())

# # Define the text splitter for manageable chunks
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# # You must ensure the documents passed to split_documents are properly formatted
# all_splits = []
# for page in pages:
#     splits = text_splitter.split_text(page.page_content)
#     all_splits.extend(splits)

# # Initialize the Chroma vector store with the document splits
# db = FAISS.from_documents(all_splits, embeddings)





# The rest of your processing and setup can follow

loaders = (PyPDFLoader(filepath) for filepath in pdf_filepaths)

# Load and split the PDFs into pages
pages = (loader.load_and_split() for loader in loaders)
# pages = (loader.load_and_split() for loader in loaders)

# Flatten the list of lists into a single list of pages
# loader = [page for sublist in pages for page in sublist]
# print(context)
# pages = loader.load_and_split()


# Define the text splitter for manageable chunks

# # You must ensure the documents passed to split_documents are properly formatted
# all_splits = text_splitter.split_documents(documents)

# Initialize the Chroma vector store and create it using the document splits
db = FAISS.from_documents(pages, embeddings)
print(db)


from langchain_core.prompts import ChatPromptTemplate
template = """
Based on the provided context, do side-by-side comparison in tabular formate like below formate :

| Feature                | Document: Draft (AVANA DRAFT-Draft.pdf) | Document: Reference (AVANA DRAFT-Reference.pdf) |
|------------------------|-----------------------------------------|-------------------------------------------------|
| Bill of Lading No.     |                                         |                                                 |
| Shipper                |                                         |                                                 |
| Consignee              |                                         |                                                 |
| Port of Loading        |                                         |                                                 |
| Port of Discharge      |                                         |                                                 |
| Final Destination      |                                         |                                                 |
| Vessel/Voyage          |                                         |                                                 |
| Total No. of Packages  |                                         |                                                 |
| Gross Weight (Total)   |                                         |                                                 |
| Net Weight (Total)     |                                         |                                                 |
| Notify Party 1         |                                         |                                                 |
| Notify Party 2         |                                         |                                                 |
| Invoice No.            |                                         |                                                 |
| Packing Date           |                                         |                                                 |
| Expiry Date            |                                         |                                                 |
| Remarks  

"""
prompt = ChatPromptTemplate.from_template(template)
llm = Ollama(model="mistral")



document_chain = create_stuff_documents_chain(llm,prompt)
retriever = db.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)
# Assuming 'invoke' needs to pass a dictionary with specific keys
# response = retrieval_chain.invoke({"context": "extract the necessary values for comparison from the documents"})

response = retrieval_chain.invoke({"input": "compare both document side by side"})
print(response)








































# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_chroma import Chroma
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.llms import Ollama
# from langchain.chains import  create_retrieval_chain
# from langchain.prompts import PromptTemplate
# from langchain_core.prompts import ChatPromptTemplate

# # Initialize the embeddings
# embeddings = OllamaEmbeddings(model="mistral")

# pdf_filepaths = ["AVANA DRAFT-Draft.pdf", "AVANA DRAFT-Reference.pdf"]
# loaders = [PyPDFLoader(filepath) for filepath in pdf_filepaths]

# # Load and split the PDFs into pages
# pages = [loader.load_and_split() for loader in loaders]

# # Flatten the list of lists into a single list of pages
# flat_pages = [page for sublist in pages for page in sublist]

# # Define the text splitter for manageable chunks
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
# all_splits = text_splitter.split_documents(documents=flat_pages)

# # Initialize the Chroma vector store and create it using the document splits
# db = FAISS.from_documents(all_splits, embeddings)

# prompt_template =""" Answer the question based only on the following table formate:
# | Feature                | Document: Draft (AVANA DRAFT-Draft.pdf) | Document: Reference (AVANA DRAFT-Reference.pdf) |
# |------------------------|-----------------------------------------|-------------------------------------------------|
# | Bill of Lading No.     | MUN/JEA/24/14740                        | BL  DRAFT 484                                   |
# | Shipper                | Punjab Riceland Agro Foods Pvt Ltd      | Punjab Riceland Agro Foods Pvt Ltd              |
# | Consignee              | AL Janan General Trading Co. LLC        | AL Janan General Trading Co. LLC                |
# | Port of Loading        | Mundra                                  | Mundra                                          |
# | Port of Discharge      | Jebel Ali                               | Jebel Ali                                       |
# | Final Destination      | Jebel Ali                               | Jebel Ali                                       |
# | Vessel/Voyage          | SC Mara/24015W                          | -                                               |
# | Total No. of Packages  | 658 Bags                                | 658 Bags                                        |
# | Gross Weight (Total)   | 25,142 kg                               | 25,142 kg                                       |
# | Net Weight (Total)     | 25,004 kg                               | 25,004 kg                                       |
# | Notify Party 1         | -                                       | AL Janan General Trading Co. LLC                |
# | Notify Party 2         | -                                       | Harmony Ventures General Trading FZCO           |
# | Invoice No.            | PRL/GDM/EXP/484                         | PRL/GDM/EXP/484                                 |
# | Packing Date           | 03/2024                                 | 03/2024                                         |
# | Expiry Date            | 02/2026                                 | 02/2026                                         |
# | Remarks                | -                                       | Carrier not responsible for loss or damage      |

# """

# prompt = ChatPromptTemplate.from_template(prompt_template)
# llm = Ollama(model="mistral")

# # # Assuming the custom function to create a chain
# # document_chain = ConversationChain(llm, prompt)
# retriever = db.as_retriever()
# retrieval_chain = create_retrieval_chain(retriever, document_chain)

# # Assuming 'invoke' needs to pass a dictionary with specific keys
# response = retrieval_chain.invoke({"input": "compare both document side by side in table formate"})
# print(response)























































# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_chroma import Chroma
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.llms import Ollama
# from langchain.chains import  create_retrieval_chain
# from langchain.prompts import PromptTemplate
# from langchain_core.prompts import ChatPromptTemplate



# from langchain_community.vectorstores import FAISS


# from langchain_community.document_loaders import PyPDFLoader

# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_chroma import Chroma
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.llms import Ollama
# from langchain.chains import ConversationChain
# from langchain.chains.conversation.memory import ConversationBufferWindowMemory
# from langchain.prompts import PromptTemplate
# from langchain_community.vectorstores import FAISS
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_community.document_loaders import WebBaseLoader


# from langchain_chroma import FAISS, PyPDFLoader
# from langchain.schema import Document
# from langchain_core.embeddings import OllamaEmbeddings
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough

# # Initialize Ollama embeddings
# embeddings = OllamaEmbeddings(model="mistral")

# # List of PDF file paths
# pdf_filepaths = [
#     "AVANA DRAFT-Draft.pdf",
#     "AVANA DRAFT-Reference.pdf"
#     # Add more file paths as needed
# ]

# # Load PDFs and extract content
# documents = []
# for filepath in pdf_filepaths:
#     loader = PyPDFLoader(filepath)
#     pages = loader.load_and_split()
#     content = " ".join(page.text for page in pages)  # Combine text from all pages
#     documents.append(Document(page_content=content))  # Create Document objects with content

# # Initialize the Chroma vector store with document contents
# db = FAISS.from_documents(documents, embeddings)

# # Define the prompt template for the comparison
# template = """
# Based on the provided context, do a side-by-side comparison in the following table format:

# | Feature                | Document: Draft (AVANA DRAFT-Draft.pdf) | Document: Reference (AVANA DRAFT-Reference.pdf) |
# |------------------------|-----------------------------------------|-------------------------------------------------|
# | Bill of Lading No.     | {billoflading_draft}                    | {billoflading_reference}                        |
# | Shipper                | {shipper_draft}                         | {shipper_reference}                             |
# | Consignee              | {consignee_draft}                       | {consignee_reference}                           |
# | Port of Loading        | {portloading_draft}                     | {portloading_reference}                         |
# | Port of Discharge      | {portdischarge_draft}                   | {portdischarge_reference}                       |
# | Final Destination      | {finaldestination_draft}                | {finaldestination_reference}                    |
# | Vessel/Voyage          | {vesselvoyage_draft}                    | {vesselvoyage_reference}                        |
# | Total No. of Packages  | {totalpackages_draft}                   | {totalpackages_reference}                       |
# | Gross Weight (Total)   | {grossweight_draft}                     | {grossweight_reference}                         |
# | Net Weight (Total)     | {netweight_draft}                       | {netweight_reference}                           |
# | Notify Party 1         | {notifyparty1_draft}                    | {notifyparty1_reference}                        |
# | Notify Party 2         | {notifyparty2_draft}                    | {notifyparty2_reference}                        |
# | Invoice No.            | {invoiceno_draft}                       | {invoiceno_reference}                           |
# | Packing Date           | {packingdate_draft}                     | {packingdate_reference}                         |
# | Expiry Date            | {expirydate_draft}                      | {expirydate_reference}                          |
# | Remarks                | {remarks_draft}                         | {remarks_reference}                             |
# """
# prompt = ChatPromptTemplate.from_template(template)

# # Setup the document chain and retrieval chain
# llm = Ollama(model="mistral")
# document_chain = RunnablePassthrough(llm, prompt, StrOutputParser())
# retriever = db.as_retriever()
# retrieval_chain = create_retrieval_chain(retriever, document_chain)

# # Invoke the chain to perform the comparison
# response = retrieval_chain.invoke({"input": "compare both documents side by side"})
# print(response)
