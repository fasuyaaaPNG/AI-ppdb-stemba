import sys
import os
import json
from huggingface_hub import login as hf_login
from datasets import load_dataset, Dataset
import pandas as pd

# Fungsi untuk memuat token dari file HUGGINGFACE_TOKEN
def load_token_from_file():
    with open("HUGGINGFACE_TOKEN.txt", "r") as file:
        for line in file:
            if line.startswith("TOKEN="):
                return line.strip().split("=")[1]
    raise ValueError("Token not found in HUGGINGFACE_TOKEN file.")

# Fungsi untuk login ke Hugging Face
def user_login():
    token = load_token_from_file()
    hf_login(token=token)

# Fungsi untuk menampilkan dataset dengan indeks
def dataset_view():
    dataset = load_dataset("Lvyn/bot-ppdb", split="train")
    data_df = dataset.to_pandas()

    grouped = data_df.groupby(data_df.index // 2)
    for index, group in grouped:
        user = group.iloc[0]["content"]
        assistant = group.iloc[1]["content"]
        print(f"Index {index + 1}:")
        print(f"  User: {user}")
        print(f"  Assistant: {assistant}")
        print("-" * 30)

# Fungsi untuk memproses input indeks
def process_indices(indices_input, max_index):
    indices_to_remove = set()
    for item in indices_input:
        if "-" in item:
            start, end = map(int, item.split("-"))
            if start < 1 or end > max_index or start > end:
                raise ValueError(f"Invalid range: {item}")
            indices_to_remove.update(range(start - 1, end))
        else:
            index = int(item) - 1
            if index < 0 or index >= max_index:
                raise ValueError(f"Invalid index: {index + 1}")
            indices_to_remove.add(index)
    return sorted(indices_to_remove)

# Fungsi untuk menghapus beberapa data berdasarkan indeks
def dataset_remove():
    dataset = load_dataset("Lvyn/bot-ppdb", split="train")
    data_df = dataset.to_pandas()

    dataset_view()
    try:
        indices_input = input("Enter the indices to remove (e.g., 1 3 5-7): ").strip().split()
        max_index = len(data_df) // 2
        rows_to_remove = []

        indices_to_remove = process_indices(indices_input, max_index)
        for index in indices_to_remove:
            rows_to_remove.extend([index * 2, index * 2 + 1])

        updated_df = data_df.drop(rows_to_remove).reset_index(drop=True)

        updated_dataset = Dataset.from_pandas(updated_df)
        updated_dataset.push_to_hub("Lvyn/bot-ppdb", private=True)

        print("Data successfully removed and updated.")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Fungsi untuk menambahkan data baru dari input manual
def dataset_modify():
    dataset = load_dataset("Lvyn/bot-ppdb", split="train")
    data_df = dataset.to_pandas()

    user_value = input("Enter User value: ").strip()
    assistant_value = input("Enter Assistant value: ").strip()

    new_data = pd.DataFrame({
        "role": ["user", "assistant"],
        "content": [user_value, assistant_value]
    })

    updated_df = pd.concat([data_df, new_data], ignore_index=True)
    updated_dataset = Dataset.from_pandas(updated_df)

    updated_dataset.push_to_hub("Lvyn/bot-ppdb", private=True)
    print("Data successfully added and saved.")

# Fungsi untuk menambahkan data dari file JSON lokal
def dataset_add_from_file():
    try:
        file_path = input("Enter the path to the JSON file: ").strip()
        if not os.path.isfile(file_path):
            print("File not found!")
            return

        with open(file_path, "r") as file:
            new_data = json.load(file)

        if not isinstance(new_data, list):
            print("Invalid file format! The file must contain a list of dictionaries.")
            return

        # Convert JSON to DataFrame
        new_data_df = pd.DataFrame(new_data)

        # Validate required columns
        if not set(["role", "content"]).issubset(new_data_df.columns):
            print("Invalid JSON structure! The file must contain 'role' and 'content' fields.")
            return

        # Load existing dataset
        dataset = load_dataset("Lvyn/bot-ppdb", split="train")
        data_df = dataset.to_pandas()

        # Concatenate and update the dataset
        updated_df = pd.concat([data_df, new_data_df], ignore_index=True)
        updated_dataset = Dataset.from_pandas(updated_df)

        updated_dataset.push_to_hub("Lvyn/bot-ppdb", private=True)
        print("Data successfully added from file and saved.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Menu utama
def menu():
    while True:
        print("\n--- MENU ---")
        print("1. View Data")
        print("2. Remove Data")
        print("3. Add Data (Manual)")
        print("4. Add Data from Local File")
        print("5. Exit")
        input_user = input("Enter your option: ")

        match input_user:
            case "1":
                dataset_view()
            case "2":
                dataset_remove()
            case "3":
                dataset_modify()
            case "4":
                dataset_add_from_file()
            case "5":
                print("Exiting...")
                break
            case _:
                print("Invalid option! Please try again.")

# Fungsi utama
def main():
    try:
        user_login()
        menu()
    except Exception as e:
        print(f"An error occurred: {e}")
    except KeyboardInterrupt:
        print("\nExit.......")
        exit()

# Jalankan skrip
if __name__ == "__main__":
    main()
