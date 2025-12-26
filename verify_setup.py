"""Verify that all packages are installed and working correctly."""

print("=" * 60)
print("VERIFICATION TEST - Advanced Multi-Source RAG")
print("=" * 60)
print()

packages_to_test = [
    ("torch", "PyTorch"),
    ("faiss", "FAISS"),
    ("spacy", "spaCy"),
    ("transformers", "Transformers"),
    ("sentence_transformers", "Sentence-Transformers"),
    ("llama_index.core", "LlamaIndex"),
    ("langchain", "LangChain"),
    ("networkx", "NetworkX"),
    ("streamlit", "Streamlit"),
    ("pandas", "Pandas"),
    ("pypdf", "PyPDF"),
    ("bs4", "BeautifulSoup4"),
]

failed = []
passed = []

for module_name, display_name in packages_to_test:
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", "OK")
        print(f"[OK] {display_name:25s} {version}")
        passed.append(display_name)
    except Exception as e:
        print(f"[FAIL] {display_name:25s} ERROR: {str(e)[:50]}")
        failed.append(display_name)

print()
print("=" * 60)
print(f"SUMMARY: {len(passed)}/{len(packages_to_test)} packages working")
print("=" * 60)

if failed:
    print("\n[X] FAILED PACKAGES:")
    for pkg in failed:
        print(f"  - {pkg}")
    exit(1)
else:
    print("\n[SUCCESS] ALL PACKAGES WORKING CORRECTLY!")
    print("\nYour environment is ready for development!")
    exit(0)
