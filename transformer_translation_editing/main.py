from train import train_model


def main() -> None:
    artifacts = train_model()

    print("\nTranslation Samples")
    for sample in artifacts["samples"]:
        print(f"Source   : {sample['source']}")
        print(f"Reference: {sample['reference']}")
        print(f"Generated: {sample['generated']}")
        print("-" * 50)


if __name__ == "__main__":
    main()

