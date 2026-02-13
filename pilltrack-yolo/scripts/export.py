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


def build_model_path(config):
    BASE_DIR = get_base_dir()
    dataset_folder = config["dataset_folder"]

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

    return model_path, exp_name


def main():
    config = load_config()
    model_path, exp_name = build_model_path(config)

    print(f"Using model: {model_path}\n")

    model = YOLO(model_path)

    # ===== Export ONNX =====
    model.export(
        format="onnx",
        imgsz=config["imgsz"],
        device=config["device"]
    )

    # ===== TensorRT (ถ้าใช้ NVIDIA GPU) =====
    # model.export(format="engine", device=config["device"])

    print(f"\nExport complete for experiment: {exp_name}")


if __name__ == "__main__":
    main()
