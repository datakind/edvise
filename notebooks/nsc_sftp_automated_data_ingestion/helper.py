import hashlib
import os
import re
import shlex
import stat
import traceback

from datetime import datetime, timezone

import pandas as pd


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


def connect_sftp(host: str, username: str, password: str, port: int = 22):
    """
    Return (transport, sftp_client). Caller must close both.
    """
    import paramiko

    transport = paramiko.Transport((host, port))
    transport.connect(username=username, password=password)
    sftp = paramiko.SFTPClient.from_transport(transport)
    print(f"Connected successfully to {host}")
    return transport, sftp


def list_receive_files(sftp, remote_dir: str, source_system: str):
    """
    List non-directory files in remote_dir with metadata.
    Returns list[dict] with keys: source_system, sftp_path, file_name, file_size, file_modified_time
    """
    results = []
    for attr in sftp.listdir_attr(remote_dir):
        if stat.S_ISDIR(attr.st_mode):
            continue

        file_name = attr.filename
        file_size = int(attr.st_size) if attr.st_size is not None else None
        mtime = (
            datetime.fromtimestamp(int(attr.st_mtime), tz=timezone.utc)
            if attr.st_mtime
            else None
        )

        results.append(
            {
                "source_system": source_system,
                "sftp_path": remote_dir,
                "file_name": file_name,
                "file_size": file_size,
                "file_modified_time": mtime,
            }
        )
    return results


def _hash_file(path, algo="sha256", chunk_size=8 * 1024 * 1024):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _remote_hash(ssh, remote_path, algo="sha256"):
    cmd = None
    if algo.lower() == "sha256":
        cmd = f"sha256sum -- {shlex.quote(remote_path)}"
    elif algo.lower() == "md5":
        cmd = f"md5sum -- {shlex.quote(remote_path)}"
    else:
        return None

    try:
        _, stdout, stderr = ssh.exec_command(cmd, timeout=300)
        out = stdout.read().decode("utf-8", "replace").strip()
        err = stderr.read().decode("utf-8", "replace").strip()
        if err:
            return None
        # Format: "<hash>  <filename>"
        return out.split()[0]
    except Exception:
        return None


def download_sftp_atomic(
    sftp,
    remote_path,
    local_path,
    *,
    chunk: int = 150,
    verify="size",  # "size" | "sha256" | "md5" | None
    ssh_for_remote_hash=None,  # paramiko.SSHClient if you want remote hash verify
    progress=True,
):
    """
    Atomic + resumable SFTP download that never trims data in situ.
    Writes to local_path + '.part' and moves into place after verification.
    """
    remote_size = sftp.stat(remote_path).st_size
    tmp_path = f"{local_path}.part"
    chunk_size = chunk * 1024 * 1024
    offset = 0
    if os.path.exists(tmp_path):
        part_size = os.path.getsize(tmp_path)
        # If local .part is larger than remote, start fresh.
        if part_size <= remote_size:
            offset = part_size
        else:
            os.remove(tmp_path)

    # Open remote and local
    with sftp.file(remote_path, "rb") as rf:
        try:
            try:
                rf.set_pipelined(True)
            except Exception:
                pass

            if offset:
                rf.seek(offset)

            # Append if resuming, write if fresh
            with open(tmp_path, "ab" if offset else "wb") as lf:
                transferred = offset

                while transferred < remote_size:
                    to_read = min(chunk_size, remote_size - transferred)
                    data = rf.read(to_read)
                    if not data:
                        # don't accept short-read silently
                        raise IOError(
                            f"Short read at {transferred:,} of {remote_size:,} bytes"
                        )
                    lf.write(data)
                    transferred += len(data)
                    if progress and remote_size:
                        print(f"{transferred / remote_size:.2%} transferred...")
                lf.flush()
                os.fsync(lf.fileno())

        finally:
            # SFTPFile closed by context manager
            pass

    # Mandatory size verification
    local_size = os.path.getsize(tmp_path)
    if local_size != remote_size:
        raise IOError(
            f"Post-download size mismatch (local {local_size:,}, remote {remote_size:,})"
        )

    if verify in {"sha256", "md5"}:
        algo = verify
        local_hash = _hash_file(tmp_path, algo=algo)
        remote_hash = None
        if ssh_for_remote_hash is not None:
            remote_hash = _remote_hash(ssh_for_remote_hash, remote_path, algo=algo)

        if remote_hash and (remote_hash != local_hash):
            # Clean up .part so next run starts fresh
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            raise IOError(
                f"{algo.upper()} mismatch: local={local_hash} remote={remote_hash}"
            )

    # Move atomically into place
    os.replace(tmp_path, local_path)
    if progress:
        print("Download complete (atomic & verified).")


def ensure_plan_table(spark, plan_table: str):
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {plan_table} (
          file_fingerprint STRING,
          file_name STRING,
          local_path STRING,
          institution_id STRING,
          inst_col STRING,
          file_size BIGINT,
          file_modified_time TIMESTAMP,
          planned_at TIMESTAMP
        )
        USING DELTA
        """
    )


def normalize_col(name: str) -> str:
    """
    Same column normalization as the current script.
    """
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("_")
    return name


def detect_institution_column(cols, inst_col_pattern):
    """
    Detect institution id column using the same regex logic as the current script.
    Returns the matched column name or None.
    """
    return next((c for c in cols if inst_col_pattern.search(c)), None)


def extract_institution_ids(local_path: str, *, renames, inst_col_pattern):
    """
    Read staged file with the same parsing approach (pandas read_csv),
    normalize/rename columns, detect institution column, return (inst_col, unique_ids).
    """
    df = pd.read_csv(local_path, on_bad_lines="warn")
    df = df.rename(columns={c: normalize_col(c) for c in df.columns})
    df = df.rename(columns=renames)

    inst_col = detect_institution_column(df.columns, inst_col_pattern)
    if inst_col is None:
        return None, []

    # Make IDs robust: drop nulls, strip whitespace, keep as string
    series = df[inst_col].dropna()

    # Some files store as numeric; normalize to integer-like strings when possible
    ids = set()
    for v in series.tolist():
        # Handle pandas/numpy numeric types
        try:
            if isinstance(v, (int,)):
                ids.add(str(v))
                continue
            if isinstance(v, float):
                # If 323100.0 -> "323100"
                if v.is_integer():
                    ids.add(str(int(v)))
                else:
                    ids.add(str(v).strip())
                continue
        except Exception:
            pass

        s = str(v).strip()
        if s == "" or s.lower() == "nan":
            continue
        # If it's "323100.0" as string, coerce safely
        if re.fullmatch(r"\d+\.0+", s):
            s = s.split(".")[0]
        ids.add(s)

    return inst_col, sorted(ids)


def output_file_name_from_sftp(file_name: str) -> str:
    return f"{os.path.basename(file_name).split('.')[0]}.csv"


def databricksify_inst_name(inst_name: str) -> str:
    """
    Follow DK standardized rules for naming conventions used in Databricks.
    """
    name = inst_name.lower()
    dk_replacements = {
        "community technical college": "ctc",
        "community college": "cc",
        "of science and technology": "st",
        "university": "uni",
        "college": "col",
    }

    for old, new in dk_replacements.items():
        name = name.replace(old, new)

    special_char_replacements = {" & ": " ", "&": " ", "-": " "}
    for old, new in special_char_replacements.items():
        name = name.replace(old, new)

    final_name = name.replace(" ", "_")

    pattern = "^[a-z0-9_]*$"
    if not re.match(pattern, final_name):
        raise ValueError("Unexpected character found in Databricks compatible name.")
    return final_name


_schema_cache: dict[str, set[str]] = {}
_bronze_volume_cache: dict[str, str] = {}  # key: f"{catalog}.{schema}" -> volume_name


def list_schemas_in_catalog(spark, catalog: str) -> set[str]:
    if catalog in _schema_cache:
        return _schema_cache[catalog]

    rows = spark.sql(f"SHOW SCHEMAS IN {catalog}").collect()

    schema_names: set[str] = set()
    for row in rows:
        d = row.asDict()
        for k in ["databaseName", "database_name", "schemaName", "schema_name", "name"]:
            v = d.get(k)
            if v:
                schema_names.add(v)
                break
        else:
            schema_names.add(list(d.values())[0])

    _schema_cache[catalog] = schema_names
    return schema_names


def find_bronze_schema(spark, catalog: str, inst_prefix: str) -> str:
    target = f"{inst_prefix}_bronze"
    schemas = list_schemas_in_catalog(spark, catalog)
    if target not in schemas:
        raise ValueError(f"Bronze schema not found: {catalog}.{target}")
    return target


def find_bronze_volume_name(spark, catalog: str, schema: str) -> str:
    key = f"{catalog}.{schema}"
    if key in _bronze_volume_cache:
        return _bronze_volume_cache[key]

    vols = spark.sql(f"SHOW VOLUMES IN {catalog}.{schema}").collect()
    if not vols:
        raise ValueError(f"No volumes found in {catalog}.{schema}")

    # Usually "volume_name", but be defensive
    def _get_vol_name(row):
        d = row.asDict()
        for k in ["volume_name", "volumeName", "name"]:
            if k in d:
                return d[k]
        return list(d.values())[0]

    vol_names = [_get_vol_name(v) for v in vols]
    bronze_like = [v for v in vol_names if "bronze" in str(v).lower()]
    if bronze_like:
        _bronze_volume_cache[key] = bronze_like[0]
        return bronze_like[0]

    raise ValueError(
        f"No volume containing 'bronze' found in {catalog}.{schema}. Volumes={vol_names}"
    )


def update_manifest(
    spark,
    manifest_table: str,
    file_fingerprint: str,
    *,
    status: str,
    error_message: str | None,
):
    """
    Update ingestion_manifest for this file_fingerprint.
    Assumes upstream inserted status=NEW already.
    """
    from pyspark.sql import types as T

    now_ts = datetime.now(timezone.utc)

    # ingested_at only set when we finish BRONZE_WRITTEN
    row = {
        "file_fingerprint": file_fingerprint,
        "status": status,
        "error_message": error_message,
        "ingested_at": now_ts if status == "BRONZE_WRITTEN" else None,
        "processed_at": now_ts,
    }

    schema = T.StructType(
        [
            T.StructField("file_fingerprint", T.StringType(), False),
            T.StructField("status", T.StringType(), False),
            T.StructField("error_message", T.StringType(), True),
            T.StructField("ingested_at", T.TimestampType(), True),
            T.StructField("processed_at", T.TimestampType(), False),
        ]
    )
    df = spark.createDataFrame([row], schema=schema)
    df.createOrReplaceTempView("manifest_updates")

    spark.sql(
        f"""
        MERGE INTO {manifest_table} AS t
        USING manifest_updates AS s
        ON t.file_fingerprint = s.file_fingerprint
        WHEN MATCHED THEN UPDATE SET
          t.status = s.status,
          t.error_message = s.error_message,
          t.ingested_at = COALESCE(s.ingested_at, t.ingested_at),
          t.processed_at = s.processed_at
        """
    )


def process_and_save_file(volume_dir, file_name, df):
    local_file_path = os.path.join(volume_dir, file_name)  # Define the local file path

    print(f"Saving to Volumes {local_file_path}")
    df.columns = [re.sub(r"[^a-zA-Z0-9_]", "_", col) for col in df.columns]
    df.to_csv(local_file_path, index=False)
    print(f"Saved {file_name} to {local_file_path}")

    return local_file_path


def move_file_to_blob(
    dbfs_file_path, blob_container_name, blob_file_name, connection_string
):
    from azure.storage.blob import BlobServiceClient

    # Create a blob service client
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    # Get the container client
    container_client = blob_service_client.get_container_client(blob_container_name)

    # Create the container if it doesn't exist
    # container_client.create_container()

    # Create a blob client for our target blob
    blob_client = container_client.get_blob_client(blob_file_name)

    # Read the file from DBFS (note the '/dbfs' prefix)
    with open(dbfs_file_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)

    print(f"File moved to Blob Storage: {blob_file_name}")


def initialize_data(path):
    from databricks.connect import DatabricksSession

    spark = DatabricksSession.builder.getOrCreate()

    def is_table_format(p):
        return "." in p and not p.endswith((".csv", ".xlsx"))

    # Function to convert a Spark DataFrame to a CSV file
    def convert_table_to_csv(table_path):
        # Extract just the final part of the table name
        final_table_name = table_path.split(".")[-1] + ".csv"
        output_path = f"/tmp/{final_table_name}"
        df = spark.read.table(table_path).toPandas()
        df.to_csv(output_path, index=False)
        print(f"Table {table_path} has been converted to {output_path}")
        return output_path

    # Function to load a CSV or XLSX file into a Pandas DataFrame
    def load_file(file_path):
        if file_path.endswith(".csv"):
            return pd.read_csv(file_path)
        elif file_path.endswith(".xlsx"):
            return pd.read_excel(file_path)
        else:
            raise ValueError(
                "Unsupported file format. Only .csv and .xlsx are supported."
            )

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
        r"^(?:"
        r"staging_sst_01(?:\.[A-Za-z0-9_]+)+"  # Pattern 1: dot-separated path starting with sst_dev.
        r"|"
        r"/Volumes/staging_sst_01(?:/[A-Za-z0-9_]+)*/[A-Za-z0-9_]+\.[A-Za-z0-9]+"  # Pattern 2: Unix-like path.
        r")$"
    )

    # Check if the filepath matches the pattern.
    return bool(pattern.match(filepath))


def remove_from_sftp(host, user, password=None, remote_folder=None, file_name=None):
    """
    Connects to the SFTP server and removes a specific file.
    """
    import paramiko

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
                "size_bytes": sftp.stat(os.path.join(remote_folder, fname)).st_size,
            }
            for fname in entries
        }
        print("Remaining files in directory:", file_info)

    finally:
        sftp.close()
        ssh.close()
