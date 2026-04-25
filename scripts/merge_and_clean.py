import json
import os
import pandas as pd

# كل الملفات اللي عايز تدمجها
FILES = [
    "hoodies_sweatshirts_clean.json",
    "tops_clean.json",
    "shorts_clean.json",
    "shoes_clean.json",
    "hats_headwear_clean.json",
    "pants_tights_clean.json"
]
all_data = []

# تحميل البيانات
for file in FILES:
    if os.path.exists(file):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"{file}: {len(data)} items")
            all_data.extend(data)
    else:
        print(f"{file} NOT FOUND")

print("\nTotal before cleaning:", len(all_data))

# إزالة الدوبلكيت باستخدام product_url
seen = set()
clean_data = []

for item in all_data:
    url = item.get("product_url")
    if url and url not in seen:
        seen.add(url)
        clean_data.append(item)

print("Total after removing duplicates:", len(clean_data))

# حفظ JSON النهائي
with open("final_dataset.json", "w", encoding="utf-8") as f:
    json.dump(clean_data, f, ensure_ascii=False, indent=2)

# تحويل لـ CSV
df = pd.DataFrame(clean_data)
df.to_csv("final_dataset.csv", index=False, encoding="utf-8-sig")

# تقرير
print("\n===== LABEL COUNTS =====")
print(df["label"].value_counts())

print("\nSaved:")
print("- final_dataset.json")
print("- final_dataset.csv")