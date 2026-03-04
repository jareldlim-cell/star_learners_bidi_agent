"""
manage_corpus.py
Interactive CLI tool for managing Vertex AI RAG corpus
"""

import vertexai
from vertexai.preview import rag
import requests
import tempfile
import os
import re
from urllib.parse import urlparse

# ==========================================
# CONFIGURATION
# ==========================================

PROJECT_ID = "tridorian-sg-vertex-ai"
LOCATION = "asia-southeast1"

CORPUS_PATH = (
    "projects/644602727268/locations/asia-southeast1/"
    "ragCorpora/2305843009213693952"
)

# ==========================================
# INITIALIZE VERTEX AI
# ==========================================

vertexai.init(project=PROJECT_ID, location=LOCATION)

# ==========================================
# FUNCTIONS
# ==========================================

def create_corpus():
    name = input("Enter corpus display name: ")
    corpus = rag.create_corpus(display_name=name)
    print("\n✅ Corpus created:")
    print(corpus.name)


def import_file():
    print("\nChoose import type:")
    print("1. GCS File (gs://...)")
    print("2. Web URL (https://...)")

    choice = input("Select option (1-2): ")

    if choice == "1":
        path = input("Enter GCS file path (e.g. gs://bucket/file.pdf): ")
        try:
            rag.import_files(
                corpus_name=CORPUS_PATH,
                paths=[path],
            )
            print("✅ GCS Import initiated successfully.")
        except Exception as e:
            print(f"❌ GCS Import failed: {e}")

    elif choice == "2":
        url = input("Enter public web URL (e.g. https://starlearners.com.sg/): ")
        try:
            # 1. Fetch the content
            headers = {"User-Agent": "Mozilla/5.0 (Vertex-AI-RAG-Tool)"}
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()

            # 2. Generate a clean filename from the URL
            parsed = urlparse(url)
            clean_name = re.sub(r"[^a-zA-Z0-9]", "_", parsed.netloc + parsed.path).strip("_")
            filename = f"{clean_name if clean_name else 'web_page'}.html"

            # 3. Create a temporary file and upload it
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = os.path.join(temp_dir, filename)
                with open(temp_file_path, "w", encoding="utf-8") as f:
                    f.write(response.text)

                # Use upload_file for direct ingestion from a local source
                rag.upload_file(
                    corpus_name=CORPUS_PATH,
                    path=temp_file_path,
                    display_name=url,
                )
            print(f"✅ Web page '{url}' scraped and uploaded successfully.")
        except Exception as e:
            print(f"❌ Web import failed: {e}")
    else:
        print("Invalid option.")

def list_files():
    files = rag.list_files(corpus_name=CORPUS_PATH)
    print("\n📂 Files in Corpus:\n")

    file_list = list(files)
    if not file_list:
        print("⚠️ No files found in corpus.")
        return

    for f in file_list:
        print("File ID:", f.name)
        print("Source :", f.display_name)
        print("----")


def delete_file():
    list_files()
    file_id = input("\nEnter FULL File ID to delete: ")

    confirm = input("Are you sure? (yes/no): ")
    if confirm.lower() == "yes":
        rag.delete_file(name=file_id)
        print("🗑 File deleted successfully.")
    else:
        print("Cancelled.")


def delete_corpus():
    confirm = input("⚠️ This will permanently delete the corpus. Type DELETE to confirm: ")
    if confirm == "DELETE":
        rag.delete_corpus(name=CORPUS_PATH)
        print("🔥 Corpus deleted successfully.")
    else:
        print("Cancelled.")


# ==========================================
# MENU
# ==========================================

def menu():
    while True:
        print("\n====== Vertex RAG Corpus Manager ======")
        print("1. Create Corpus")
        print("2. Import File (GCS)")
        print("3. List Files")
        print("4. Delete File")
        print("5. Delete Entire Corpus")
        print("6. Exit")
        print("========================================")

        choice = input("Select option (1-6): ")

        if choice == "1":
            create_corpus()
        elif choice == "2":
            import_file()
        elif choice == "3":
            list_files()
        elif choice == "4":
            delete_file()
        elif choice == "5":
            delete_corpus()
        elif choice == "6":
            print("Goodbye 👋")
            break
        else:
            print("Invalid option. Try again.")


# ==========================================
# RUN
# ==========================================

if __name__ == "__main__":
    menu()