import argparse
import json
import sys
from pathlib import Path

from db.db_manager import DBManager
from logic.settings_manager import SettingsManager
from logic.xfeat_core import XFeatCore
from logic.location_consumer import LocationConsumer


def process_image(consumer: LocationConsumer, image_path: Path) -> None:
    """Processes a single image, writing JSON to stdout and errors to stderr."""
    if not image_path.is_file():
        sys.stderr.write(
            json.dumps(
                {
                    "file": str(image_path),
                    "status": "error",
                    "message": "File not found",
                }
            )
            + "\n"
        )
        sys.stderr.flush()
        return

    try:
        result = consumer.localize(str(image_path))

        output_data = {
            "file": image_path.name,
            "status": "success" if result else "failed",
            "data": result,
        }

        # Pure JSON to standard output for pipeline consumption
        sys.stdout.write(json.dumps(output_data) + "\n")
        sys.stdout.flush()

    except Exception as e:
        error_data = {"file": image_path.name, "status": "error", "message": str(e)}
        # Errors to standard error to prevent corrupting the data pipeline
        sys.stderr.write(json.dumps(error_data) + "\n")
        sys.stderr.flush()


def main() -> None:

    parser = argparse.ArgumentParser(description="Headless XFeat Location Consumer Pipeline")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-i", "--image", type=str, help="Path to a single image file")
    group.add_argument("-d", "--directory", type=str, help="Path to a directory of images")

    parser.add_argument("--db", type=str, required=True, help="Path to the SQLite database")
    parser.add_argument("--model", type=str, required=True, help="Path to the XFeat ONNX model")
    parser.add_argument(
        "--config",
        type=str,
        default="system_preferences.json",
        help="Path to JSON configuration file",
    )

    args = parser.parse_args()
    try:
        db_manager = DBManager(args.db)
        settings_manager = SettingsManager(args.config)

        cv_params = settings_manager.get_cv_params()
        gem_p = int(cv_params.get("gemPoolingPower", 3))

        xfeat_model = XFeatCore(model_path=args.model, gem_p=gem_p)

        consumer = LocationConsumer(
            db_manager=db_manager,
            xfeat_model=xfeat_model,
            settings_manager=settings_manager,
        )
    except Exception as e:
        sys.stderr.write(f"Pipeline initialization failed: {e}\n")
        sys.exit(1)

    # Execution Phase
    if args.image:
        process_image(consumer, Path(args.image))

    elif args.directory:
        dir_path = Path(args.directory)
        if not dir_path.is_dir():
            sys.stderr.write(f"Directory not found or invalid: {dir_path}\n")
            sys.exit(1)

        # Recursive search using rglob to find all files in all subfolders
        files_to_process = [f for f in dir_path.rglob("*") if f.is_file()]

        # Output to stderr so it shows in the console, not the jsonl file
        sys.stderr.write(
            f"[INFO] Found {len(files_to_process)} files to process in {dir_path}\n"
        )
        sys.stderr.flush()

        for file_path in files_to_process:
            process_image(consumer, file_path)


if __name__ == "__main__":
    main()

