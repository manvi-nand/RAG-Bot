import argparse
from pathlib import Path

from app.db import init_db
from app.embeddings import ingest_folder


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs from data/ into the vector DB.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest even if a PDF was already ingested (deletes existing chunks first).",
    )
    args = parser.parse_args()
    init_db()
    data_dir = Path("data")
    processed, skipped = ingest_folder(data_dir, force=args.force)
    print(f"Processed {len(processed)} PDF(s): {processed}")
    print(f"Skipped {len(skipped)} PDF(s): {skipped}")


if __name__ == "__main__":
    main()
