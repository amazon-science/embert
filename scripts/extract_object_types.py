import json
from argparse import ArgumentParser

from ai2thor.controller import Controller
from tqdm import tqdm


def main(args):
    # These are object types that are available only after the execution of specific actions
    # More details are available here: https://ai2thor.allenai.org/ithor/documentation/objects/object-types
    object_types = {"AppleSliced", "BreadSliced", "LettuceSliced", "EggCracked", "PotatoSliced", "TomatoSliced"}
    controller = Controller()
    controller.start()

    for scene in tqdm(controller.scene_names(), desc="Looking for object types in all scenes..."):
        controller.reset(scene)
        e = controller.step(action=dict(action="Pass"))
        for obj in e.metadata['objects']:
            object_types.add(obj['objectType'])

    controller.stop()

    print(f"Completed object types extraction. Found {len(object_types)} object types!")
    for i, x in enumerate(object_types):
        if i == 20:
            break
        print(f"{i + 1}) {x}")

    print(f"Saving vocabulary to file '{args.vocab_file}'")

    with open(args.vocab_file, mode="w") as out_file:
        json.dump({
            o_type: idx
            for idx, o_type in enumerate(object_types)
        }, out_file)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--vocab_file', type=str, default='configs/ai2thor_vocab.json')

    args = parser.parse_args()
    main(args)
