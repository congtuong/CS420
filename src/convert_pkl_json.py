import pickle
import json
import argparse
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--pkl", type=str)
parser.add_argument("--output", type=str)
args = parser.parse_args()


output_dir = Path(args.output)
output_dir.mkdir(parents=True, exist_ok=True)


data = pickle.load(open(args.pkl, "rb"))
for i in range(len(data)):
    result = {}

    img_path = Path(data[i]["img_path"])
    result['pred_instances'] = data[i]["pred_instances"]
    result['scores'] = data[i]["scores"]
    result['labels'] = data[i]["labels"]

    json.dump(result, open(output_dir / (img_path.stem + ".json"), "w"))