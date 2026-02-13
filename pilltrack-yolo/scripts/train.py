import os
import yaml
import torch
from ultralytics import YOLO


def load_config():
    # หา root project แบบ dynamic
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(BASE_DIR, "configs", "train_config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_data_path(config):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    dataset_root = os.path.join(BASE_DIR, config["dataset_root"])
    dataset_folder = config["dataset_folder"]

    data_path = os.path.join(dataset_root, dataset_folder, "data.yaml")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"data.yaml not found at: {data_path}")

    return data_path


def main():
    config = load_config()
    data_path = build_data_path(config)

    print("========== SYSTEM INFO ==========")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=================================\n")

    print(f"Using dataset: {data_path}\n")

    model = YOLO(config["model"])

    model.train(
        data=data_path,
        epochs=config["epochs"],
        imgsz=config["imgsz"],
        batch=config["batch"],
        patience=config["patience"],
        lr0=config["lr0"],
        degrees=config["degrees"],
        translate=config["translate"],
        scale=config["scale"],
        fliplr=config["fliplr"],
        hsv_h=config["hsv_h"],
        hsv_s=config["hsv_s"],
        hsv_v=config["hsv_v"],
        device=config["device"],
        workers=config["workers"],
        project=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            config["project"]
        ),
        name=f"{config['name']}_{config['dataset_folder']}"
    )


if __name__ == "__main__":
    main()
