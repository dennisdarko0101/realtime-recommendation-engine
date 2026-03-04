#!/usr/bin/env python3
"""Generate synthetic demo dataset and save to disk."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.generator import generate_dataset


def main() -> None:
    data_dir = Path("data/store")
    print(f"Generating dataset → {data_dir}")

    store = generate_dataset(n_users=1000, n_items=500, n_interactions=50_000)
    store.save(str(data_dir))

    print(f"  Users:        {store.n_users}")
    print(f"  Items:        {store.n_items}")
    print(f"  Interactions: {store.n_interactions}")
    print("Done!")


if __name__ == "__main__":
    main()
