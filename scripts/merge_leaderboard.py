import json
import os
from argparse import ArgumentParser
from datetime import datetime


def main(args):
    with open(args.tests_seen) as in_file:
        tests_seen = json.load(in_file)

    with open(args.tests_unseen) as in_file:
        tests_unseen = json.load(in_file)

    results = {"tests_seen": tests_seen["tests_seen"], "tests_unseen": tests_unseen["tests_unseen"]}

    save_path = os.path.dirname(args.tests_seen)

    output_path = os.path.join(save_path,
                               f'tests_actseqs_dump_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}.json')

    with open(output_path, mode="w") as out_file:
        json.dump(results, out_file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-ts", dest="tests_seen", type=str)
    parser.add_argument("-tus", dest="tests_unseen", type=str)
    args = parser.parse_args()

    main(args)
