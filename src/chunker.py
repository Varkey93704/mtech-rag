import tiktoken

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 1. Load the document
file_path = "data/documents/chunking_test.txt"

with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

documents = [
    Document(
        page_content=text,
        metadata={"source": file_path},
    )
]


# 2. Create the text splitter
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=512,
    chunk_overlap=50,
)


# 3. Split the document
chunks = text_splitter.split_documents(documents)


# 4. Create tokenizer for inspecting token counts
encoding = tiktoken.get_encoding("cl100k_base")


# 5. Inspect the chunks
print(f"Number of chunks: {len(chunks)}")

for i, chunk in enumerate(chunks):
    token_count = len(encoding.encode(chunk.page_content))

    print(f"\n--- CHUNK {i + 1} ---")
    print(f"Token count: {token_count}")
    print(chunk.page_content)

    print("\nMetadata:")
    print(chunk.metadata)