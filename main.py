import argparse
from models.builder import get_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="vit")
    args = parser.parse_args()

    model = get_model(args.model, num_classes=2)
    print(model)


if __name__ == "__main__":
    main()
