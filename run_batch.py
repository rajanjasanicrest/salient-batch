"""Editable runner: change CONFIG_FILE, then run this file."""

import logging
from pathlib import Path

from image_batch import run_config


# Point this at the JSON payload you want to run.
CONFIG_FILE = Path(__file__).with_name("config.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
    succeeded, failed = run_config(CONFIG_FILE)
    print(f"Finished: {succeeded} succeeded, {failed} failed")
