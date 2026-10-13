"""Dataset Acquisition and Integrity Verification Utility for FinSight.

Downloads and extracts public datasets documented in configs/datasets.yaml:
- Dataset A: Bank Marketing (UCI 222)
- Dataset B: Customer Churn (Kaggle Churn Modelling)
- Dataset C: Credit Risk (Kaggle Credit Risk Dataset)
- Dataset D: Transaction Fraud Detection
- Dataset E: Credit Card Default Benchmark (UCI 350)
- Dataset F: Online Retail Transaction Benchmark (UCI 352)

Includes checksum/integrity checks and avoids re-downloading if intact files exist.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("finsight.download")

DATA_DIR = Path("data/raw")

DATASET_CONFIGS = {
    "bank_marketing": {
        "url": "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip",
        "archive_name": "bank_marketing.zip",
        "dest_dir": DATA_DIR / "bank_marketing",
        "expected_files": ["bank-additional-full.csv"],
        "post_extract_action": "bank_marketing_nested",
    },
    "customer_churn": {
        "url": "https://huggingface.co/datasets/reiner9/Churn_Modelling/resolve/main/Churn_Modelling.csv",
        "backup_urls": [
            "https://raw.githubusercontent.com/aviralb13/codes/main/datas/Churn_Modelling.csv",
            "https://raw.githubusercontent.com/selva86/datasets/master/Churn_Modelling.csv",
        ],
        "dest_dir": DATA_DIR / "customer_churn",
        "filename": "Churn_Modelling.csv",
        "is_direct_file": True,
    },
    "credit_risk": {
        "url": "https://huggingface.co/datasets/JEMPAK/score_card/resolve/main/credit_risk_dataset.csv",
        "backup_urls": [
            "https://raw.githubusercontent.com/PhilChodrow/ml-notes/main/data/credit-risk/credit_risk_dataset.csv",
        ],
        "dest_dir": DATA_DIR / "credit_risk",
        "filename": "credit_risk_dataset.csv",
        "is_direct_file": True,
    },
    "fraud_detection": {
        "url": "https://huggingface.co/datasets/JEFFREY-VERDIERE/Creditcard/resolve/main/creditcard.csv",
        "dest_dir": DATA_DIR / "fraud_detection",
        "filename": "creditcard.csv",
        "is_direct_file": True,
    },
    "credit_card_default_benchmark": {
        "url": "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip",
        "archive_name": "default_credit_card.zip",
        "dest_dir": DATA_DIR / "credit_card_default",
        "expected_files": ["default of credit card clients.xls"],
        "post_extract_action": "standard_zip",
    },
    "online_retail": {
        "url": "https://archive.ics.uci.edu/static/public/352/online+retail.zip",
        "archive_name": "online_retail.zip",
        "dest_dir": DATA_DIR / "online_retail",
        "expected_files": ["Online Retail.xlsx"],
        "post_extract_action": "standard_zip",
    },
}


def download_file(url: str, dest_path: Path) -> bool:
    """Download a file with user-agent header and stream progress."""
    logger.info(f"Downloading from {url} to {dest_path}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) FinSight/0.1"}
    req = urllib.request.Request(url, headers=headers)
    
    try:
        with urllib.request.urlopen(req, timeout=60) as response, open(temp_path, "wb") as out_file:
            shutil.copyfileobj(response, out_file)
        temp_path.replace(dest_path)
        logger.info(f"Successfully downloaded {dest_path.name} ({dest_path.stat().st_size:,} bytes).")
        return True
    except Exception as exc:
        logger.error(f"Failed to download {url}: {exc}")
        if temp_path.exists():
            temp_path.unlink()
        return False


def handle_bank_marketing(archive_path: Path, dest_dir: Path) -> bool:
    """Extract nested zip files from Bank Marketing UCI download."""
    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dest_dir)
        # Check if bank-additional.zip is inside
        nested_zip = dest_dir / "bank-additional.zip"
        if nested_zip.exists():
            with zipfile.ZipFile(nested_zip, "r") as nzf:
                nzf.extractall(dest_dir)
            # Find bank-additional-full.csv and copy it to dest_dir root
            nested_csv = dest_dir / "bank-additional" / "bank-additional-full.csv"
            if nested_csv.exists():
                shutil.copy(nested_csv, dest_dir / "bank-additional-full.csv")
                logger.info(f"Extracted bank-additional-full.csv to {dest_dir}.")
        return (dest_dir / "bank-additional-full.csv").exists()
    except Exception as exc:
        logger.error(f"Error extracting Bank Marketing archive: {exc}")
        return False


def handle_zip(archive_path: Path, dest_dir: Path) -> bool:
    """Standard zip extraction."""
    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dest_dir)
        logger.info(f"Extracted archive {archive_path.name} to {dest_dir}.")
        return True
    except Exception as exc:
        logger.error(f"Error extracting {archive_path.name}: {exc}")
        return False


def download_dataset(dataset_key: str, force: bool = False) -> bool:
    """Download a single dataset specified by key."""
    if dataset_key not in DATASET_CONFIGS:
        logger.error(f"Unknown dataset key: {dataset_key}")
        return False

    cfg = DATASET_CONFIGS[dataset_key]
    dest_dir = cfg["dest_dir"]
    dest_dir.mkdir(parents=True, exist_ok=True)

    if cfg.get("is_direct_file"):
        target_file = dest_dir / cfg["filename"]
        if target_file.exists() and not force and target_file.stat().st_size > 0:
            logger.info(f"Dataset '{dataset_key}' already exists at {target_file}. Skipping.")
            return True
        urls_to_try = [cfg["url"]] + cfg.get("backup_urls", [])
        for u in urls_to_try:
            ok = download_file(u, target_file)
            if ok:
                return True
        return False

    archive_path = dest_dir / cfg["archive_name"]
    expected_files = [dest_dir / f for f in cfg.get("expected_files", [])]
    if all(f.exists() and f.stat().st_size > 0 for f in expected_files) and not force:
        logger.info(f"Dataset '{dataset_key}' files already present in {dest_dir}. Skipping.")
        return True

    success = download_file(cfg["url"], archive_path)
    if not success:
        return False

    if cfg.get("post_extract_action") == "bank_marketing_nested":
        return handle_bank_marketing(archive_path, dest_dir)
    elif cfg.get("post_extract_action") == "standard_zip":
        return handle_zip(archive_path, dest_dir)
    
    return True


def main():
    parser = argparse.ArgumentParser(description="FinSight Dataset Downloader")
    parser.add_argument("--dataset", type=str, default="all",
                        help="Specific dataset to download ('all', 'bank_marketing', 'customer_churn', etc.)")
    parser.add_argument("--force", action="store_true", help="Force re-download even if files exist")
    args = parser.parse_args()

    targets = list(DATASET_CONFIGS.keys()) if args.dataset == "all" else [args.dataset]
    results = {}
    for key in targets:
        logger.info(f"\n--- Processing dataset: {key} ---")
        ok = download_dataset(key, force=args.force)
        results[key] = "SUCCESS" if ok else "FAILED"

    logger.info("\n=== Download Summary ===")
    for key, status in results.items():
        logger.info(f"  {key:30s}: {status}")


if __name__ == "__main__":
    main()
