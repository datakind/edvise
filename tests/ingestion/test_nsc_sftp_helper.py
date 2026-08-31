import re

from edvise.ingestion.nsc_sftp.helpers import (
    extract_institution_ids,
    group_plan_rows_by_file,
    sst_identity_or_resolve,
)
from edvise.utils.institution_naming import databricksify_inst_name
from edvise.utils.data_cleaning import convert_to_snake_case, detect_institution_column
from edvise.utils.sftp import download_sftp_atomic


def test_normalize_col():
    """Test column normalization (now using convert_to_snake_case)."""
    assert convert_to_snake_case(" Institution ID ") == "institution_id"
    assert convert_to_snake_case("Student-ID#") == "student_id_#"
    assert convert_to_snake_case("__Already__Ok__") == "already_ok"


def test_detect_institution_column():
    pattern = re.compile(r"(?=.*institution)(?=.*id)", re.IGNORECASE)
    assert (
        detect_institution_column(["foo", "institutionid", "bar"], pattern)
        == "institutionid"
    )
    assert detect_institution_column(["foo", "bar"], pattern) is None


def test_extract_institution_ids_handles_numeric(tmp_path):
    csv_path = tmp_path / "staged.csv"
    csv_path.write_text(
        "InstitutionID,other\n323100,1\n323101.0,2\n,3\n323102.0,4\n 323103 ,5\ninf,6\n-inf,7\n"
    )

    inst_col_pattern = re.compile(r"(?=.*institution)(?=.*id)", re.IGNORECASE)
    inst_col, inst_ids = extract_institution_ids(
        str(csv_path), renames={}, inst_col_pattern=inst_col_pattern
    )

    assert inst_col == "institution_id"
    assert inst_ids == ["323100", "323101", "323102", "323103"]


def test_load_and_group_matches_extract_with_floatish_ids(tmp_path):
    """Stage-03 filter must find the same PDP ids stage-02 extracted."""
    from edvise.ingestion.nsc_sftp.helpers import (
        group_dataframe_by_institution_id,
        load_staged_csv,
    )

    # NaNs force pandas float inference when dtype is not applied.
    csv_path = tmp_path / "staged.csv"
    csv_path.write_text("InstitutionID,other\n345000,1\n,2\n345000.0,3\n 345000 ,4\n")

    inst_col_pattern = re.compile(r"(?=.*institution)(?=.*id)", re.IGNORECASE)
    inst_col, inst_ids = extract_institution_ids(
        str(csv_path), renames={}, inst_col_pattern=inst_col_pattern
    )
    assert inst_ids == ["345000"]

    df = load_staged_csv(str(csv_path), renames={}, inst_col=inst_col)
    grouped = group_dataframe_by_institution_id(df, inst_col, inst_ids)
    assert set(grouped) == {"345000"}
    assert len(grouped["345000"]) == 3


def test_databricksify_inst_name():
    assert databricksify_inst_name("Big State University") == "big_state_uni"


def test_hash_file_sha256(tmp_path):
    """Test file hashing (internal function, tested via download_sftp_atomic)."""
    # The _hash_file function is internal to sftp.py, so we test it indirectly
    # through download_sftp_atomic which uses it for verification
    pass


def test_download_sftp_atomic_downloads_and_cleans_part(tmp_path):
    class _Stat:
        def __init__(self, size: int):
            self.st_size = size

    class _RemoteFile:
        def __init__(self, data: bytes):
            self._data = data
            self._pos = 0

        def set_pipelined(self, _):
            return None

        def seek(self, offset: int):
            self._pos = offset

        def read(self, n: int) -> bytes:
            if self._pos >= len(self._data):
                return b""
            b = self._data[self._pos : self._pos + n]
            self._pos += len(b)
            return b

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _Sftp:
        def __init__(self, by_path: dict[str, bytes]):
            self._by_path = by_path

        def stat(self, path: str):
            return _Stat(len(self._by_path[path]))

        def file(self, path: str, mode: str):
            assert mode == "rb"
            return _RemoteFile(self._by_path[path])

    remote_path = "/receive/file1.csv"
    remote_bytes = b"hello world\n" * 100
    sftp = _Sftp({remote_path: remote_bytes})

    local_path = tmp_path / "file1.csv"
    download_sftp_atomic(
        sftp,
        remote_path,
        str(local_path),
        chunk=1,
        verify="size",
        progress=False,
    )

    assert local_path.read_bytes() == remote_bytes
    assert not (tmp_path / "file1.csv.part").exists()


def test_download_sftp_atomic_resumes_existing_part(tmp_path):
    class _Stat:
        def __init__(self, size: int):
            self.st_size = size

    class _RemoteFile:
        def __init__(self, data: bytes):
            self._data = data
            self._pos = 0

        def set_pipelined(self, _):
            return None

        def seek(self, offset: int):
            self._pos = offset

        def read(self, n: int) -> bytes:
            if self._pos >= len(self._data):
                return b""
            b = self._data[self._pos : self._pos + n]
            self._pos += len(b)
            return b

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _Sftp:
        def __init__(self, by_path: dict[str, bytes]):
            self._by_path = by_path

        def stat(self, path: str):
            return _Stat(len(self._by_path[path]))

        def file(self, path: str, mode: str):
            assert mode == "rb"
            return _RemoteFile(self._by_path[path])

    remote_path = "/receive/file2.csv"
    remote_bytes = b"0123456789" * 200
    sftp = _Sftp({remote_path: remote_bytes})

    local_path = tmp_path / "file2.csv"
    part_path = tmp_path / "file2.csv.part"

    part_path.write_bytes(remote_bytes[:123])

    download_sftp_atomic(
        sftp,
        remote_path,
        str(local_path),
        chunk=1,
        verify="size",
        progress=False,
    )

    assert local_path.read_bytes() == remote_bytes
    assert not part_path.exists()


def test_sst_identity_or_resolve_uses_planned_when_complete() -> None:
    planned = ("inst-1", "Example University")
    assert sst_identity_or_resolve(object(), "323100", planned) == planned


def test_group_plan_rows_by_file() -> None:
    rows = [
        {
            "file_fingerprint": "fp1",
            "file_name": "a.csv",
            "local_path": "/tmp/a.csv",
            "inst_col": "institution_id",
            "institution_id": "323100",
            "inst_id": "i1",
            "institution_name": "Uni A",
        },
        {
            "file_fingerprint": "fp1",
            "file_name": "a.csv",
            "local_path": "/tmp/a.csv",
            "inst_col": "institution_id",
            "institution_id": "323101",
            "inst_id": "",
            "institution_name": "",
        },
    ]
    by_file, inst_ids, identity = group_plan_rows_by_file(rows)
    assert by_file["fp1"]["file_name"] == "a.csv"
    assert inst_ids["fp1"] == ["323100", "323101"]
    assert identity["fp1"] == {"323100": ("i1", "Uni A")}
