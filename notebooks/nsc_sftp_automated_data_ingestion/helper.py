import os
import pandas as pd
import re
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession
from azure.storage.blob import BlobServiceClient
import traceback
import paramiko

from datetime import datetime

class CustomLogger:
    def __init__(self, log_file: str = "sftp.log"):
        self.log_file = log_file

    def _log(self, level: str, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "a") as f:
            f.write(f"{timestamp} - {level} - {message}\n")

    def info(self, message: str) -> None:
        self._log("INFO", message)

    def warning(self, message: str) -> None:
        self._log("WARNING", message)

    def error(self, message: str) -> None:
        self._log("ERROR", message)

    def debug(self, message: str) -> None:
        self._log("DEBUG", message)

    def exception(self, message: str) -> None:
        """Logs an error message with traceback info."""
        tb = traceback.format_exc()
        self._log("ERROR", f"{message}\n{tb}")

def process_and_save_file(volume_dir, file_name, df):
    local_file_path = os.path.join(volume_dir, file_name)  # Define the local file path

    print(f"Saving to Volumes {local_file_path}")
    df.columns = [re.sub(r"[^a-zA-Z0-9_]", "_", col) for col in df.columns]
    df.to_csv(local_file_path, index=False)
    print(f"Saved {file_name} to {local_file_path}")

    return local_file_path

def move_file_to_blob(dbfs_file_path, blob_container_name, blob_file_name, connection_string):
    # Create a blob service client
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    
    # Get the container client
    container_client = blob_service_client.get_container_client(blob_container_name)
    
    # Create the container if it doesn't exist
    #container_client.create_container()

    # Create a blob client for our target blob
    blob_client = container_client.get_blob_client(blob_file_name)
    
    # Read the file from DBFS (note the '/dbfs' prefix)
    with open(dbfs_file_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)

    print(f"File moved to Blob Storage: {blob_file_name}")

def initialize_data(path):
    spark = SparkSession.builder.appName("Data Initialization App").getOrCreate()

    def is_table_format(p):
        return '.' in p and not p.endswith(('.csv', '.xlsx'))

    # Function to convert a Spark DataFrame to a CSV file
    def convert_table_to_csv(table_path):
        # Extract just the final part of the table name
        final_table_name = table_path.split('.')[-1] + ".csv"
        output_path = f"/tmp/{final_table_name}"
        df = spark.read.table(table_path).toPandas()
        df.to_csv(output_path, index=False)
        display(f"Table {table_path} has been converted to {output_path}")
        return output_path

    # Function to load a CSV or XLSX file into a Pandas DataFrame
    def load_file(file_path):
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            return pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Only .csv and .xlsx are supported.")

    if is_table_format(path):
        # If it's a table, convert it to a CSV file
        file_path = convert_table_to_csv(path)
        return pd.read_csv(file_path), file_path
    else:
        # If it's a file, load it directly
        return load_file(path), path
    
def validate_filepath(filepath: str, keyword: str) -> bool:
    """
    Validates that the given filepath:
      1. Contains the specified keyword.
      2. Matches one of the two valid patterns:
         - Dot-delimited path starting with "sst_dev"
         - Unix-style path starting with "/Volumes/sst_dev" and ending with a filename.ext

    Args:
        filepath (str): The filepath to validate.
        keyword (str): The substring that must be present in the filepath.

    Returns:
        bool: True if both conditions are met, otherwise False.
    """
    # Check for the presence of the keyword in the filepath.
    if keyword not in filepath:
        return False

    # Compile a regular expression that matches either pattern.
    pattern = re.compile(
        r'^(?:'
        r'staging_sst_01(?:\.[A-Za-z0-9_]+)+'  # Pattern 1: dot-separated path starting with sst_dev.
        r'|'
        r'/Volumes/staging_sst_01(?:/[A-Za-z0-9_]+)*/[A-Za-z0-9_]+\.[A-Za-z0-9]+'  # Pattern 2: Unix-like path.
        r')$'
    )
    
    # Check if the filepath matches the pattern.
    return bool(pattern.match(filepath))

def remove_from_sftp(host, user, password=None, remote_folder=None, file_name=None):
    """
    Connects to the SFTP server and removes a specific file.
    """
    # Setup SSH client
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname=host, username=user, password=password)

    sftp = ssh.open_sftp()
    try:
        remote_path = os.path.join(remote_folder, file_name)
        # Check existence (optional)
        try:
            sftp.stat(remote_path)
        except FileNotFoundError:
            print(f"File does not exist: {remote_path}")
            return
        # Remove file
        sftp.remove(remote_path)
        print(f"Removed file: {remote_path}")

        # List remaining files (for confirmation)
        entries = sftp.listdir(remote_folder)
        file_info = {
            fname: {
                "last_modified": datetime.fromtimestamp(
                    sftp.stat(os.path.join(remote_folder, fname)).st_mtime
                ).strftime("%Y-%m-%d %H:%M:%S"),
                "size_bytes": sftp.stat(os.path.join(remote_folder, fname)).st_size
            }
            for fname in entries
        }
        print("Remaining files in directory:", file_info)

    finally:
        sftp.close()
        ssh.close()