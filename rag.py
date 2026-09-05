import os
from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import requests
from bs4 import BeautifulSoup

load_dotenv()

# Constants
CHUNK_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
COLLECTION_NAME = "news_market_research"

llm = None
vector_store = None
qa_chain = None

def scrape_url(url):
    """Custom scraper with headers to bypass blocking"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove scripts and styles
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        if len(text.strip()) < 100:
            print(f"Warning: very little content scraped from {url}")
            return None
        return Document(page_content=text, metadata={"source": url})

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

def initialize_components():
    global llm, vector_store, qa_chain

    if llm is None:
        llm = ChatGroq(model=os.getenv("GROQ_MODEL"), temperature=0.9, max_tokens=500)

    if vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True}
        )

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=str(VECTORSTORE_DIR)
        )

    if qa_chain is None:
        qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            retriever=vector_store.as_retriever(),
            chain_type="stuff"
        )


def process_urls(urls):
    """
    This function scraps data from a url and stores it in a vector db
    :param urls: input urls
    :return:
    """
    yield "Initializing Components"
    initialize_components()

    yield "Resetting vector store...✅"
    vector_store.reset_collection()

    yield "Loading data...✅"
    data = []
    failed_urls = []
    for url in urls:
        doc = scrape_url(url)
        if doc:
            print(f"SOURCE: {doc.metadata.get('source')}")
            print(f"LENGTH: {len(doc.page_content)} chars")
            print(f"PREVIEW: {doc.page_content[:300]}")
            print("---")
            data.append(doc)
        else:
            failed_urls.append(url)

    if failed_urls:
        yield f"⚠️ Failed to load: {', '.join(failed_urls)}"

    if not data:
        yield "No data loaded! URLs might be blocked ❌"
        return

    yield "Splitting text into chunks...✅"
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE
    )
    docs = text_splitter.split_documents(data)

    yield "Add chunks to vector database...✅"
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs, ids=uuids)

    yield "Done adding docs to vector database...✅"

def generate_answer(query):
    if not vector_store:
        raise RuntimeError("Vector database is not initialized ")

    result = qa_chain.invoke({"question": query}, return_only_outputs=True)
    sources = result.get("sources", "")

    return result['answer'], sources


if __name__ == "__main__":
    urls = [
        "https://www.cnbc.com/2026/08/27/august-apartment-rents-turn-positive-for-the-first-time-in-four-years.html",
        "https://www.cnbc.com/2026/09/02/demand-for-riskier-mortgages-rises-along-with-interest-rates.html"
    ]

    for status in process_urls(urls):
        print(status)
    answer, sources = generate_answer("How much rent grew in August?")
    print(f"Answer: {answer}")
    print(f"Sources: {sources}")

