import argparse
import json


parser = argparse.ArgumentParser(description="Concatenate predictions")
parser.add_argument("--f1", type=str, help="input json file")
parser.add_argument("--f2", type=str, help="input json file")
parser.add_argument("--output", type=str, help="output json file")
args = parser.parse_args()


d1 = json.load(open(args.f1, "r"))
d2 = json.load(open(args.f2, "r"))

d1.extend(d2)

with open(args.output, "w+") as f:
    json.dump(d1, f, indent=2)
