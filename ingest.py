"""
ingest.py
Script để nạp tài liệu vào Weaviate knowledge base.
Chạy một lần trước khi dùng chatbot.

Ví dụ:
  python ingest.py --files docs/faq.txt docs/manual.txt
  python ingest.py --dir ./docs
"""
import argparse
import os
from pathlib import Path
from langchain_core.documents import Document

from rag.weaviate_store import get_weaviate_client, get_vector_store, ingest_text_files


def ingest_from_dir(vector_store, directory: str):
    txt_files = list(Path(directory).rglob("*.txt"))
    if not txt_files:
        print(f"[!] Không tìm thấy file .txt trong {directory}")
        return 0
    paths = [str(f) for f in txt_files]
    print(f"[*] Tìm thấy {len(paths)} file: {paths}")
    count = ingest_text_files(vector_store, paths)
    return count


def main():
    parser = argparse.ArgumentParser(description="Ingest tài liệu vào Weaviate")
    parser.add_argument("--files", nargs="+", help="Danh sách file .txt")
    parser.add_argument("--dir",   help="Thư mục chứa file .txt")
    parser.add_argument(
        "--text",
        help="Nhập text trực tiếp (để test nhanh)",
    )
    args = parser.parse_args()

    print("[*] Kết nối Weaviate...")
    client = get_weaviate_client()
    vs = get_vector_store(client)
    print("[✓] Kết nối OK")

    total = 0

    if args.text:
        doc = Document(page_content=args.text, metadata={"source": "manual_input"})
        from rag.weaviate_store import ingest_documents
        ingest_documents(vs, [doc])
        total += 1
        print(f"[✓] Đã ingest 1 document từ --text")

    if args.files:
        count = ingest_text_files(vs, args.files)
        total += count
        print(f"[✓] Đã ingest {count} chunks từ {len(args.files)} file")

    if args.dir:
        count = ingest_from_dir(vs, args.dir)
        total += count
        print(f"[✓] Đã ingest {count} chunks từ thư mục {args.dir}")

    if total == 0:
        print("Không có dữ liệu nào được ingest. Dùng --files, --dir, hoặc --text")

    client.close()
    print("Xong!")


if __name__ == "__main__":
    main()
