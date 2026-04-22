import json
import os

FILES = [
    "hoodies_sweatshirts.json"
]

IMAGES_FOLDER = "images"


def clean_json_file(input_file):
    output_file = input_file.replace(".json", "_clean.json")

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    clean_data = []
    removed_count = 0

    for item in data:
        image_name = item.get("image_name", "").strip()

        if not image_name:
            removed_count += 1
            continue

        image_path = os.path.join(IMAGES_FOLDER, image_name)

        if os.path.exists(image_path):
            clean_data.append(item)
        else:
            removed_count += 1

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(clean_data, f, ensure_ascii=False, indent=2)

    print(f"\n{input_file}")
    print(f"Original: {len(data)}")
    print(f"Removed: {removed_count}")
    print(f"Remaining: {len(clean_data)}")
    print(f"Saved as: {output_file}")


def main():
    for file in FILES:
        if os.path.exists(file):
            clean_json_file(file)
        else:
            print(f"\n{file} NOT FOUND")


if __name__ == "__main__":
    main()