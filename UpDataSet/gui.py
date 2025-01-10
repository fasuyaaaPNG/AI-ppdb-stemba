import sys
import os
from huggingface_hub import login as hf_login
from datasets import load_dataset, Dataset
import pandas as pd
import streamlit as st

# Fungsi untuk memuat token dari file HUGGINGFACE_TOKEN
def load_token_from_file():
    with open("HUGGINGFACE_TOKEN", "r") as file:
        for line in file:
            if line.startswith("TOKEN="):
                return line.strip().split("=")[1]
    raise ValueError("Token not found in HUGGINGFACE_TOKEN file.")

# Fungsi untuk login ke Hugging Face
def user_login():
    token = load_token_from_file()
    hf_login(token=token)

# Fungsi untuk menampilkan dataset
def dataset_view():
    st.title("View Dataset")
    dataset = load_dataset("Lvyn/bot-ppdb", split="train")
    data_df = dataset.to_pandas()

    grouped = data_df.groupby(data_df.index // 2)
    for index, group in grouped:
        user = group.iloc[0]["content"]
        assistant = group.iloc[1]["content"]
        st.write(f"**Index {index + 1}:**")
        st.write(f"- User: {user}")
        st.write(f"- Assistant: {assistant}")
        st.markdown("---")

# Fungsi untuk memproses input indeks
def process_indices(indices_input, max_index):
    indices_to_remove = set()
    for item in indices_input:
        if "-" in item:
            start, end = map(int, item.split("-"))
            if start < 1 or end > max_index or start > end:
                st.error(f"Invalid range: {item}")
                return []
            indices_to_remove.update(range(start - 1, end))
        else:
            index = int(item) - 1
            if index < 0 or index >= max_index:
                st.error(f"Invalid index: {index + 1}")
                return []
            indices_to_remove.add(index)
    return sorted(indices_to_remove)

# Fungsi untuk menghapus data
def dataset_remove():
    st.title("Remove Data")
    dataset = load_dataset("Lvyn/bot-ppdb", split="train")
    data_df = dataset.to_pandas()

    st.write("### Current Dataset:")
    dataset_view()

    indices_input = st.text_input("Enter indices to remove (e.g., 1 3 5-7)")
    if st.button("Remove Data"):
        try:
            max_index = len(data_df) // 2
            rows_to_remove = []

            indices_to_remove = process_indices(indices_input.split(), max_index)
            for index in indices_to_remove:
                rows_to_remove.extend([index * 2, index * 2 + 1])

            updated_df = data_df.drop(rows_to_remove).reset_index(drop=True)

            updated_dataset = Dataset.from_pandas(updated_df)
            updated_dataset.push_to_hub("Lvyn/bot-ppdb", private=True)

            st.success("Data successfully removed and updated.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Fungsi untuk menambahkan data baru
def dataset_add():
    st.title("Add Data")
    dataset = load_dataset("Lvyn/bot-ppdb", split="train")
    data_df = dataset.to_pandas()

    st.write("### Add New Entries")
    user_inputs = st.text_area("Enter User values (one per line)")
    assistant_inputs = st.text_area("Enter Assistant values (one per line)")

    if st.button("Add Data"):
        user_values = user_inputs.split("\n")
        assistant_values = assistant_inputs.split("\n")

        if len(user_values) != len(assistant_values):
            st.error("The number of User and Assistant entries must match.")
        else:
            new_data = pd.DataFrame({
                "role": ["user", "assistant"] * len(user_values),
                "content": [item for pair in zip(user_values, assistant_values) for item in pair]
            })

            updated_df = pd.concat([data_df, new_data], ignore_index=True)
            updated_dataset = Dataset.from_pandas(updated_df)

            updated_dataset.push_to_hub("Lvyn/bot-ppdb", private=True)
            st.success("Data successfully added and saved.")

# Menu utama
def main():
    st.sidebar.title("Menu")
    option = st.sidebar.selectbox("Select an option:", ["View Data", "Remove Data", "Add Data"])

    user_login()

    if option == "View Data":
        dataset_view()
    elif option == "Remove Data":
        dataset_remove()
    elif option == "Add Data":
        dataset_add()

if __name__ == "__main__":
    main()