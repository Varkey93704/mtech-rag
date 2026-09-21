from langchain_community.document_loaders import TextLoader

loader = TextLoader("data/documents/regression_testing.txt")

documents = loader.load()

print(f"Number of documents: {len(documents)}")

for document in documents:
    print("\n--- CONTENT ---")
    print(document.page_content)

    print("\n--- METADATA ---")
    print(document.metadata)