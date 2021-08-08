import json
import os
import re
from argparse import ArgumentParser
from collections import defaultdict

task_results_regex = re.compile("task_results_(\w+)_(\d+_\d+_\d+).json")


def main(args):
    results = defaultdict(lambda: {
        "model": [],
        "model_checkpoint": [],
        "date": [],
        "gc": [],
        "sr": [],
        "plw_sr": [],
        "plw_gc": []
    })

    for root, dirs, files in os.walk(args.folder):
        for file in files:
            if args.results_prefix in file and file.endswith(".json"):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath) as in_file:
                        curr_results = json.load(in_file)
                except:
                    continue

                if "results" not in curr_results or "all" not in curr_results["results"]:
                    continue

                model_name = os.path.basename(root)
                match = re.match(task_results_regex, file)

                if match is not None:
                    split = match.group(1)
                    date = match.group(2)

                    results[split]["model"].append(model_name)
                    results[split]["model_checkpoint"].append(curr_results.get("model_checkpoint"))
                    results[split]["date"].append(date)
                    results[split]["gc"].append(
                        curr_results["results"]["all"]["goal_condition_success"]["goal_condition_success_rate"])
                    results[split]["sr"].append(curr_results["results"]["all"]["success"]["success_rate"])
                    results[split]["plw_sr"].append(curr_results["results"]["all"]["path_length_weighted_success_rate"])
                    results[split]["plw_gc"].append(
                        curr_results["results"]["all"]["path_length_weighted_goal_condition_success_rate"])

    for split, split_results in results.items():
        print(f"Split: {split}")
        print("Date\tModel\tSR\tGC\tPLW_SR\tPLW_GC")

        for date, model, model_checkpoint, gc, sr, plw_sr, plw_gc in zip(split_results["date"],
                                                                         split_results["model"],
                                                                         split_results["model_checkpoint"],
                                                                         split_results["gc"], split_results["sr"],
                                                                         split_results["plw_sr"],
                                                                         split_results["plw_gc"]):
            print(f"{date}\t{model}\t{model_checkpoint}\t{sr}\t{gc}\t{plw_sr}\t{plw_gc}")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--results_prefix", type=str, default="task_results")

    args = parser.parse_args()
    main(args)
