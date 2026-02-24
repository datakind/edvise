"""
SFTP utilities for file transfer operations.

Provides functions for connecting to SFTP servers, listing files, and downloading
files with atomic operations and verification.
"""

import hashlib
import logging
import os
import shlex
import stat
from datetime import datetime, timezone
from typing import Optional

LOGGER = logging.getLogger(__name__)


def connect_sftp(host: str, username: str, password: str, port: int = 22):
    """
    Connect to an SFTP server.

    Args:
        host: SFTP server hostname
        username: SFTP username
        password: SFTP password
        port: SFTP port (default: 22)

    Returns:
        Tuple of (transport, sftp_client). Caller must close both.

    Example:
        >>> transport, sftp = connect_sftp("example.com", "user", "pass")
        >>> try:
        ...     files = list_receive_files(sftp, "/remote/path", "NSC")
        ... finally:
        ...     sftp.close()
        ...     transport.close()
    """
    import paramiko

    transport = paramiko.Transport((host, port))
    transport.connect(username=username, password=password)
    sftp = paramiko.SFTPClient.from_transport(transport)
    LOGGER.info(f"Connected successfully to {host}:{port}")
    return transport, sftp


def list_receive_files(
    sftp, remote_dir: str, source_system: str
) -> list[dict[str, any]]:
    """
    List non-directory files in remote directory with metadata.

    Args:
        sftp: Paramiko SFTPClient instance
        remote_dir: Remote directory path to list
        source_system: Source system identifier (e.g., "NSC")

    Returns:
        List of dictionaries with keys: source_system, sftp_path, file_name,
        file_size, file_modified_time

    Example:
        >>> files = list_receive_files(sftp, "/receive", "NSC")
        >>> for f in files:
        ...     print(f["file_name"], f["file_size"])
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


def _hash_file(path: str, algo: str = "sha256", chunk_size: int = 8 * 1024 * 1024) -> str:
    """
    Compute hash of a file.

    Args:
        path: File path
        algo: Hash algorithm ("sha256" or "md5")
        chunk_size: Chunk size for reading file

    Returns:
        Hexadecimal hash string
    """
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _remote_hash(ssh, remote_path: str, algo: str = "sha256") -> Optional[str]:
    """
    Compute hash of a remote file using SSH command.

    Args:
        ssh: Paramiko SSHClient instance
        remote_path: Remote file path
        algo: Hash algorithm ("sha256" or "md5")

    Returns:
        Hexadecimal hash string, or None if computation fails
    """
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
    remote_path: str,
    local_path: str,
    *,
    chunk: int = 150,
    verify: str = "size",  # "size" | "sha256" | "md5" | None
    ssh_for_remote_hash=None,  # paramiko.SSHClient if you want remote hash verify
    progress: bool = True,
) -> None:
    """
    Atomic and resumable SFTP download with verification.

    Writes to local_path + '.part' and moves into place after verification.
    Supports resuming interrupted downloads.

    Args:
        sftp: Paramiko SFTPClient instance
        remote_path: Remote file path
        local_path: Local destination path
        chunk: Chunk size in MB (default: 150)
        verify: Verification method: "size", "sha256", "md5", or None
        ssh_for_remote_hash: SSHClient for remote hash verification (optional)
        progress: Whether to print progress (default: True)

    Raises:
        IOError: If download fails, size mismatch, or hash mismatch

    Example:
        >>> download_sftp_atomic(sftp, "/remote/file.csv", "/local/file.csv")
        >>> # With hash verification:
        >>> download_sftp_atomic(
        ...     sftp, "/remote/file.csv", "/local/file.csv",
        ...     verify="sha256", ssh_for_remote_hash=ssh
        ... )
    """
    remote_size = sftp.stat(remote_path).st_size
    tmp_path = f"{local_path}.part"
    chunk_size = chunk * 1024 * 1024
    offset = 0

    # Check for existing partial download
    if os.path.exists(tmp_path):
        part_size = os.path.getsize(tmp_path)
        # If local .part is larger than remote, start fresh
        if part_size <= remote_size:
            offset = part_size
            if progress:
                LOGGER.info(f"Resuming download from {offset:,} bytes")
        else:
            os.remove(tmp_path)
            if progress:
                LOGGER.warning("Partial file larger than remote, starting fresh")

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
                        pct = transferred / remote_size
                        if pct % 0.1 < 0.01 or transferred == remote_size:  # Print every 10%
                            LOGGER.info(f"{pct:.1%} transferred ({transferred:,}/{remote_size:,} bytes)")
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

    # Optional hash verification
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
        LOGGER.info(f"Download complete (atomic & verified): {local_path}")
