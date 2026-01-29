from pathlib import Path

from app.db import init_db
from app.embeddings import ingest_folder


def main():
    init_db()
    data_dir = Path("data")
    processed, skipped = ingest_folder(data_dir)
    print(f"Processed {len(processed)} PDF(s): {processed}")
    print(f"Skipped {len(skipped)} PDF(s): {skipped}")


if __name__ == "__main__":
    main()
