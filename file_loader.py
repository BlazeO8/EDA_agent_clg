
import os
import pandas as pd


def read_uploaded_file(uploaded_file):
    """Reads an uploaded file into a pandas DataFrame based on its extension.

    Supports: CSV, Excel (.xlsx, .xls), JSON, Parquet, and Pickle.
    """
    # Get the file name/path depending on how the file is passed
    if hasattr(uploaded_file, "name"):
        file_name = uploaded_file.name
    else:
        file_name = str(uploaded_file)

    # Extract the file extension
    _, ext = os.path.splitext(file_name)
    ext = ext.lower()

    # Read based on extension
    try:
        if ext == ".csv":
            # You can add delimiter=';' if needed based on your data
            df = pd.read_csv(uploaded_file)

        elif ext in [".xls", ".xlsx"]:
            df = pd.read_excel(uploaded_file)

        elif ext == ".json":
            df = pd.read_json(uploaded_file)

        elif ext == ".parquet":
            df = pd.read_parquet(uploaded_file)

        elif ext in [".pkl", ".pickle"]:
            df = pd.read_pickle(uploaded_file)

        elif ext == ".tsv":
            df = pd.read_csv(uploaded_file, sep="\t")

        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        return df

    except Exception as e:
        raise RuntimeError(f"Error reading file {file_name}: {e}")
