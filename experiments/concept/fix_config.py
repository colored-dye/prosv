import argparse
import os
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True, type=str, help="Directory to save config.json")
    args = parser.parse_args()

    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    new_config = {}
    new_config["embed_dim"] = config["embed_dim"]
    new_config["low_rank_dim"] = config["low_rank_dim"]
    new_config["target_module"] = config["target_modules"][0]
    new_config["layer"] = config["layers"][0]
    new_config["intervention_type"] = config["class"]
    new_config["concept"] = config["concept"]

    with open(config_path, "w") as f:
        json.dump(new_config, f, indent=2)

if __name__ == "__main__":
    main()
