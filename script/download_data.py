from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


DOI = "10.6084/m9.figshare.30043246"
FIGSHARE_ARTICLE_ID = "30043246"
FIGSHARE_VERSION = "1"
FILE_ID = "57670756"
ARCHIVE_NAME = "data.rar"
DOWNLOAD_URL = f"https://ndownloader.figshare.com/files/{FILE_ID}"
EXPECTED_SIZE = 1_767_736_956
EXPECTED_MD5 = "3550c6bc7d2f7e49232f45f3d41f3082"
DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def format_bytes(size: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def resolve_output_path(path: Path) -> Path:
    if path.exists() and path.is_dir():
        return path / ARCHIVE_NAME
    return path


def part_path_for(path: Path) -> Path:
    return path.with_name(f"{path.name}.part")


def calculate_md5(path: Path, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, f"{path} does not exist"

    size = path.stat().st_size
    if size != EXPECTED_SIZE:
        return False, (
            f"size mismatch: expected {EXPECTED_SIZE} bytes "
            f"({format_bytes(EXPECTED_SIZE)}), got {size} bytes ({format_bytes(size)})"
        )

    checksum = calculate_md5(path)
    if checksum.lower() != EXPECTED_MD5.lower():
        return False, f"MD5 mismatch: expected {EXPECTED_MD5}, got {checksum}"

    return True, "size and MD5 match"


def response_total_size(response, downloaded: int) -> int | None:
    headers = getattr(response, "headers", {})
    content_range = headers.get("Content-Range")
    if content_range and "/" in content_range:
        total = content_range.rsplit("/", 1)[-1]
        if total.isdigit():
            return int(total)

    content_length = headers.get("Content-Length")
    if content_length and content_length.isdigit():
        return downloaded + int(content_length)

    return EXPECTED_SIZE


def print_progress(downloaded: int, total: int | None) -> None:
    if total:
        percent = min(downloaded / total * 100, 100)
        message = (
            f"\rDownloaded {format_bytes(downloaded)} / "
            f"{format_bytes(total)} ({percent:5.1f}%)"
        )
    else:
        message = f"\rDownloaded {format_bytes(downloaded)}"
    print(message, end="", flush=True)


def open_download_response(url: str, resume_from: int, timeout: int):
    headers = {}
    if resume_from > 0:
        headers["Range"] = f"bytes={resume_from}-"
    request = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(request, timeout=timeout)


def should_retry(error: BaseException) -> bool:
    if isinstance(error, urllib.error.HTTPError):
        return error.code in {408, 429, 500, 502, 503, 504}
    return isinstance(error, (urllib.error.URLError, TimeoutError, ConnectionError))


def download_archive(
    url: str,
    destination: Path,
    chunk_size: int,
    timeout: int,
    retries: int,
    resume: bool,
) -> None:
    temporary = part_path_for(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, retries + 2):
        resume_from = temporary.stat().st_size if resume and temporary.exists() else 0
        downloaded = resume_from

        try:
            with open_download_response(url, resume_from, timeout) as response:
                status = getattr(response, "status", response.getcode())
                mode = "ab" if resume_from > 0 and status == 206 else "wb"

                if resume_from > 0 and status != 206:
                    print("Server did not honor resume request; restarting download.")
                    downloaded = 0

                total = response_total_size(response, downloaded)
                last_update = 0.0
                with temporary.open(mode) as handle:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        handle.write(chunk)
                        downloaded += len(chunk)
                        now = time.monotonic()
                        if now - last_update >= 1:
                            print_progress(downloaded, total)
                            last_update = now
                print_progress(downloaded, total)
                print()
            return
        except urllib.error.HTTPError as error:
            if error.code == 416 and temporary.exists():
                print("Existing partial file cannot be resumed; restarting download.")
                temporary.unlink()
                continue
            if attempt > retries + 1 or not should_retry(error):
                raise
            print(f"Download failed ({error}); retrying {attempt}/{retries}...")
            time.sleep(min(2 ** attempt, 30))
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            if attempt > retries + 1 or not should_retry(error):
                raise
            print(f"Download failed ({error}); retrying {attempt}/{retries}...")
            time.sleep(min(2 ** attempt, 30))


def check_available_space(destination: Path) -> None:
    temporary = part_path_for(destination)
    existing_size = temporary.stat().st_size if temporary.exists() else 0
    remaining_size = max(EXPECTED_SIZE - existing_size, 0)
    free_size = shutil.disk_usage(destination.parent).free
    if free_size < remaining_size:
        raise RuntimeError(
            f"not enough free space in {destination.parent}: "
            f"need at least {format_bytes(remaining_size)}, available {format_bytes(free_size)}"
        )


def find_extractor() -> tuple[str, list[str]] | None:
    unrar = shutil.which("unrar")
    if unrar:
        return unrar, [unrar, "x", "-o+", "{archive}", "{target}"]

    seven_zip_modern = shutil.which("7zz")
    if seven_zip_modern:
        return seven_zip_modern, [seven_zip_modern, "x", "{archive}", "-o{target}", "-y"]

    seven_zip = shutil.which("7z")
    if seven_zip:
        return seven_zip, [seven_zip, "x", "{archive}", "-o{target}", "-y"]

    bsdtar = shutil.which("bsdtar")
    if bsdtar:
        return bsdtar, [bsdtar, "-xf", "{archive}", "-C", "{target}"]

    return None


def extract_archive(archive: Path, target: Path) -> None:
    extractor = find_extractor()
    if extractor is None:
        raise RuntimeError("RAR extraction requires one of: 7z, unrar, bsdtar")

    tool, template = extractor
    target.mkdir(parents=True, exist_ok=True)
    command = [
        item.format(archive=str(archive), target=str(target))
        for item in template
    ]
    print(f"Extracting with {Path(tool).name} into {target}")
    subprocess.run(command, check=True)


def remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def print_dataset_info(output: Path, extract_dir: Path) -> None:
    print(f"DOI: {DOI}.v{FIGSHARE_VERSION}")
    print(f"Figshare article: {FIGSHARE_ARTICLE_ID}, file: {FILE_ID}")
    print(f"Download URL: {DOWNLOAD_URL}")
    print(f"Archive: {output}")
    print(f"Expected size: {EXPECTED_SIZE} bytes ({format_bytes(EXPECTED_SIZE)})")
    print(f"Expected MD5: {EXPECTED_MD5}")
    print(f"Extraction target: {extract_dir}")
    print(f"Expected extracted data directory: {extract_dir / 'data'}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_output = repo_root() / ARCHIVE_NAME
    parser = argparse.ArgumentParser(
        description="Download the isoST Figshare dataset archive with resume and MD5 verification."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="archive path, or an existing directory to place data.rar in",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="extract the verified RAR archive after download",
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=None,
        help="directory used with --extract; default is the archive parent directory",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an invalid existing archive and restart partial downloads",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="discard any .part file and start downloading from byte 0",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="only verify the existing archive; do not download",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print planned actions without downloading or extracting",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="network timeout in seconds",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="number of retries for transient network errors",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="download chunk size in bytes",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    output = resolve_output_path(args.output.expanduser()).resolve()
    extract_dir = output.parent if args.extract_dir is None else args.extract_dir.expanduser().resolve()
    temporary = part_path_for(output)
    resume = not args.no_resume

    print_dataset_info(output, extract_dir)

    if args.dry_run:
        if args.force:
            print("Would remove existing archive/partial files before downloading.")
        elif resume and temporary.exists():
            print(f"Would resume from {temporary} ({format_bytes(temporary.stat().st_size)}).")
        else:
            print("Would download the archive and verify size/MD5.")
        if args.extract:
            print("Would extract the verified archive after download.")
        return 0

    if args.force:
        remove_if_exists(output)
        remove_if_exists(temporary)
    elif args.no_resume:
        remove_if_exists(temporary)

    if args.verify_only:
        ok, message = verify_archive(output)
        print(f"Verification: {message}")
        return 0 if ok else 1

    if output.exists():
        ok, message = verify_archive(output)
        if ok:
            print(f"Existing archive verified: {message}")
            if args.extract:
                extract_archive(output, extract_dir)
            return 0
        raise RuntimeError(f"existing archive is invalid ({message}); rerun with --force")

    check_available_space(output)
    download_archive(
        DOWNLOAD_URL,
        output,
        args.chunk_size,
        args.timeout,
        args.retries,
        resume,
    )

    ok, message = verify_archive(temporary)
    if not ok:
        raise RuntimeError(f"downloaded file failed verification ({message}); keeping {temporary}")

    os.replace(temporary, output)
    print(f"Archive verified and saved to {output}")

    if args.extract:
        extract_archive(output, extract_dir)

    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        print("Interrupted; partial download is kept for resume.", file=sys.stderr)
        return 130
    except (RuntimeError, OSError, subprocess.CalledProcessError, urllib.error.URLError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
