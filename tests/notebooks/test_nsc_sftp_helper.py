import importlib.util
import re
from pathlib import Path


def _load_helper_module():
    repo_root = Path(__file__).resolve().parents[2]
    helper_path = (
        repo_root
        / "notebooks"
        / "nsc_sftp_automated_data_ingestion"
        / "helper.py"
    )
    spec = importlib.util.spec_from_file_location("nsc_sftp_helper", helper_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_normalize_col():
    helper = _load_helper_module()
    assert helper.normalize_col(" Institution ID ") == "institution_id"
    assert helper.normalize_col("Student-ID#") == "student_id"
    assert helper.normalize_col("__Already__Ok__") == "already_ok"


def test_detect_institution_column():
    helper = _load_helper_module()
    pattern = re.compile(r"(?=.*institution)(?=.*id)", re.IGNORECASE)
    assert (
        helper.detect_institution_column(["foo", "institutionid", "bar"], pattern)
        == "institutionid"
    )
    assert helper.detect_institution_column(["foo", "bar"], pattern) is None


def test_extract_institution_ids_handles_numeric(tmp_path):
    helper = _load_helper_module()
    csv_path = tmp_path / "staged.csv"
    csv_path.write_text(
        "InstitutionID,other\n"
        "323100,1\n"
        "323101.0,2\n"
        ",3\n"
        "323102.0,4\n"
        " 323103 ,5\n"
    )

    inst_col_pattern = re.compile(r"(?=.*institution)(?=.*id)", re.IGNORECASE)
    inst_col, inst_ids = helper.extract_institution_ids(
        str(csv_path), renames={}, inst_col_pattern=inst_col_pattern
    )

    assert inst_col == "institutionid"
    assert inst_ids == ["323100", "323101", "323102", "323103"]


def test_output_file_name_from_sftp():
    helper = _load_helper_module()
    assert helper.output_file_name_from_sftp("some_file.txt") == "some_file.csv"
    assert helper.output_file_name_from_sftp("/a/b/c/my.data.csv") == "my.csv"


def test_databricksify_inst_name():
    helper = _load_helper_module()
    assert helper.databricksify_inst_name("Big State University") == "big_state_uni"


def test_hash_file_sha256(tmp_path):
    helper = _load_helper_module()
    fp = tmp_path / "x.bin"
    fp.write_bytes(b"abc")
    assert (
        helper._hash_file(str(fp))
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_download_sftp_atomic_downloads_and_cleans_part(tmp_path):
    helper = _load_helper_module()

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
    helper.download_sftp_atomic(
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
    helper = _load_helper_module()

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

    helper.download_sftp_atomic(
        sftp,
        remote_path,
        str(local_path),
        chunk=1,
        verify="size",
        progress=False,
    )

    assert local_path.read_bytes() == remote_bytes
    assert not part_path.exists()

