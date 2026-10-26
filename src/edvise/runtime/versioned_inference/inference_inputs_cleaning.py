import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

PROFILE = "2052166062819251"
CATALOG = "staging_sst_01"
PREFIX = f"dbfs:/Volumes/{CATALOG}/"
DELETE = "--delete" in sys.argv


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["databricks", *args, "--profile", PROFILE],
        text=True,
        capture_output=True,
    )


def assert_delete_path(path: str) -> str:
    if not path.startswith(PREFIX):
        raise SystemExit(f"REFUSE not under {PREFIX}: {path}")
    parts = path[len(PREFIX) :].strip("/").split("/")
    if len(parts) != 3 or parts[1] != "bronze_volume" or parts[2] != "inference_inputs":
        raise SystemExit(f"REFUSE unexpected shape: {path}")
    schema = parts[0]
    if not schema or schema in {".", ".."} or any(c in schema for c in "/\\"):
        raise SystemExit(f"REFUSE bad schema: {path}")
    if "bronze" not in schema or "bronze" not in path or "inference_inputs" not in path:
        raise SystemExit(f"REFUSE missing bronze or inference_inputs: {path}")
    if not path.rstrip("/").endswith("/inference_inputs"):
        raise SystemExit(f"REFUSE path does not end at inference_inputs: {path}")
    for banned in ("training_inputs", "silver_volume", "gcs_uploads", "dev_sst_02"):
        if banned in path:
            raise SystemExit(f"REFUSE banned token {banned}: {path}")
    return path


catalogs = json.loads(run("catalogs", "list", "-o", "json").stdout)
names = {row.get("name") for row in catalogs}
if CATALOG not in names:
    raise SystemExit(
        f"REFUSE: profile {PROFILE} has no catalog {CATALOG}. Catalogs: {sorted(names)}"
    )

raw = run("schemas", "list", CATALOG, "--limit", "500", "-o", "json")
if raw.returncode != 0:
    raise SystemExit(raw.stderr or raw.stdout)
schemas = sorted(
    row["name"] for row in json.loads(raw.stdout) if "bronze" in row.get("name", "")
)
candidates = [
    assert_delete_path(f"{PREFIX}{schema}/bronze_volume/inference_inputs")
    for schema in schemas
]


def probe(path: str) -> tuple[str, int | None]:
    assert_delete_path(path)
    proc = run("fs", "ls", path)
    if proc.returncode != 0:
        return path, None
    count = len([line for line in proc.stdout.splitlines() if line.strip()])
    return path, count


found: list[tuple[str, int]] = []
with ThreadPoolExecutor(max_workers=8) as pool:
    for path, count in pool.map(probe, candidates):
        if count is not None:
            found.append((path, count))

found.sort()
print(f"profile {PROFILE}")
print(f"bronze schemas checked: {len(candidates)}")
print(f"inference_inputs present: {len(found)}")
for path, count in found:
    assert_delete_path(path)
    print(f"{count:4d}  {path}")

if not DELETE:
    print("list only. Re-run with --delete to remove these directories.")
    raise SystemExit(0)

if not found:
    print("nothing to delete")
    raise SystemExit(0)

typed = input(f"Type {CATALOG} to delete these {len(found)} directories: ")
if typed.strip() != CATALOG:
    raise SystemExit("REFUSE: confirmation did not match. Nothing deleted.")

for path, _ in found:
    assert_delete_path(path)
    proc = run("fs", "rm", "-r", path)
    status = "ok" if proc.returncode == 0 else "FAIL"
    print(status, path)
    if proc.returncode != 0:
        print(proc.stderr or proc.stdout)
        raise SystemExit(1)

print("done")