from langchain_core.documents import Document

file_path = "data/documents/regression_testing.txt"

with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

documents = [
    Document(
        page_content=text,
        metadata={"source": file_path},
    )
]

print(f"Number of documents: {len(documents)}")

for document in documents:
    print("\n--- CONTENT ---")
    print(document.page_content)

    print("\n--- METADATA ---")
    print(document.metadata)