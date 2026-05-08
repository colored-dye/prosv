import argparse
import glob
from loguru import logger
import os
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True, type=str, help="Directory to save config.json")
    parser.add_argument("--file_names", required=True, type=str, nargs='+')
    args = parser.parse_args()

    for file_name in args.file_names:
        files = glob.glob(f"{args.output_dir}/{file_name}*")
        if len(files) == 1:
            logger.error("No files to merge.")
            exit(1)

        dfs = []
        for f in files:
            df = pd.read_parquet(f)
            dfs.append(df)
        df = pd.concat(dfs, axis=0, ignore_index=True)

        save_file = Path(args.output_dir, file_name)
        files.remove(str(save_file))

        df.to_parquet(save_file, index=False)
        logger.warning(f"Saved to {save_file}")

        for file in files:
            os.remove(file)
            logger.warning(f"Removed {file}")


if __name__ == "__main__":
    main()
