import os
import yaml
from ultralytics import YOLO


def get_base_dir():
    # หา root project อัตโนมัติ
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config():
    BASE_DIR = get_base_dir()
    config_path = os.path.join(BASE_DIR, "configs", "train_config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_paths(config):
    BASE_DIR = get_base_dir()

    dataset_root = os.path.join(BASE_DIR, config["dataset_root"])
    dataset_folder = config["dataset_folder"]

    test_images_path = os.path.join(
        dataset_root,
        dataset_folder,
        "test",
        "images"
    )

    exp_name = f"{config['name']}_{dataset_folder}"

    model_path = os.path.join(
        BASE_DIR,
        config["project"],
        exp_name,
        "weights",
        "best.pt"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    if not os.path.exists(test_images_path):
        raise FileNotFoundError(f"Test images not found at: {test_images_path}")

    return model_path, test_images_path, exp_name


def main():
    config = load_config()
    model_path, test_images_path, exp_name = build_paths(config)

    print(f"Using model: {model_path}")
    print(f"Predicting on: {test_images_path}\n")

    model = YOLO(model_path)

    model.predict(
        source=test_images_path,
        conf=0.25,
        imgsz=config["imgsz"],
        device=config["device"],
        save=True,
        project=os.path.join(get_base_dir(), config["project"]),
        name=f"predict_{config['dataset_folder']}"
    )

    print("\nPrediction complete.")


if __name__ == "__main__":
    main()
