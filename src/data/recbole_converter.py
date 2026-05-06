"""Convert processed DataFrames to RecBole atomic file format."""

from pathlib import Path
import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)


def create_inter_file(df: pd.DataFrame, output_dir: str) -> Path:
    """Create .inter file for RecBole.

    Format: tab-separated with typed header.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    filepath = out_path / "amazon-electronics.inter"

    # RecBole atomic file: header with type annotations
    header = "user_id:token\titem_id:token\trating:float\ttimestamp:float"
    rows = []
    for _, row in df.iterrows():
        rows.append(
            f"{row['user_id']}\t{row['item_id']}\t{row['rating']}\t{row['timestamp']}"
        )

    with open(filepath, "w") as f:
        f.write(header + "\n")
        f.write("\n".join(rows) + "\n")

    logger.info(f"Created RecBole .inter file: {filepath} ({len(rows)} interactions)")
    return filepath


def create_item_file(metadata_df: pd.DataFrame, output_dir: str) -> Path:
    """Create .item file for RecBole."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    filepath = out_path / "amazon-electronics.item"

    header = "item_id:token\tcategory:token_seq\ttitle:token_seq"
    rows = []
    for _, row in metadata_df.iterrows():
        cat = str(row.get("category", "")).replace("\t", " ")
        title = str(row.get("title", "")).replace("\t", " ")
        rows.append(f"{row['item_id']}\t{cat}\t{title}")

    with open(filepath, "w") as f:
        f.write(header + "\n")
        f.write("\n".join(rows) + "\n")

    logger.info(f"Created RecBole .item file: {filepath} ({len(rows)} items)")
    return filepath


def convert_to_recbole(
    interactions_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    output_dir: str = "data/recbole/amazon-electronics",
) -> tuple[Path, Path]:
    """Convert both DataFrames to RecBole format."""
    inter_path = create_inter_file(interactions_df, output_dir)
    item_path = create_item_file(metadata_df, output_dir)
    return inter_path, item_path
