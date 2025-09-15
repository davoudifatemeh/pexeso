import argparse

from utils.config import Config
from offline import run_offline
from online import run_online


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--query_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    cfg = Config()
    run_offline(args.dataset_dir, args.output_dir, cfg)
    run_online(args.query_dir, args.output_dir, cfg)
