import argparse

from configs import PRESET_REGISTRY, get_config
from train import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Challenge version Transformer training")
    parser.add_argument(
        "--preset",
        type=str,
        default="baseline",
        choices=sorted(PRESET_REGISTRY.keys()),
        help="Experiment preset to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config(args.preset)
    artifacts = train_model(config)

    print("\nTranslation Samples")
    for sample in artifacts["samples"]:
        print(f"Source   : {sample['source']}")
        print(f"Reference: {sample['reference']}")
        print(f"Generated: {sample['generated']}")
        print("-" * 50)


if __name__ == "__main__":
    main()
