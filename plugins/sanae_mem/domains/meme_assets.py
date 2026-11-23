from __future__ import annotations

import base64
import hashlib
import http.client
import ipaddress
import socket
from urllib.parse import urljoin, urlsplit
import io
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..infra.contracts import ViveMindError

try:
    from PIL import Image, ImageOps
except Exception:
    Image = None
    ImageOps = None

logger = logging.getLogger(__name__)

CHAT_STICKER_MAX_EDGE = 300
CHAT_STICKER_COPY_VERSION = "v1"
IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"})
FORMAT_TO_MIME = {
    "PNG": ("image/png", ".png"),
    "JPEG": ("image/jpeg", ".jpg"),
    "GIF": ("image/gif", ".gif"),
    "WEBP": ("image/webp", ".webp"),
    "BMP": ("image/bmp", ".bmp"),
}
IMAGE_HINT_RE = re.compile(r"\[Image attached at:\s*([^\]]+)\]")
_VISION_IMAGE_URL_RE = re.compile(
    r"^\[[^\r\n]*\bvision_analyze\b[^\r\n]*\bimage_url:\s*([^\r\n]+?)\s*(?:~\s*)?\]\s*$",
    re.IGNORECASE | re.MULTILINE,
)


class MemeChatCopyError(ViveMindError, RuntimeError):
    """A chat-sticker send copy cannot be prepared without violating fidelity."""


def asset_root(data_dir: str | Path) -> Path:
    root = Path(data_dir) / "memes"
    root.mkdir(parents=True, exist_ok=True)
    return root


def asset_path(data_dir: str | Path, rel: str) -> Path:
    root = asset_root(data_dir).resolve()
    target = (root / str(rel or "")).resolve()
    if target != root and root not in target.parents:
        raise ValueError("meme asset path escapes the meme root")
    return target


MAX_IMAGE_BYTES = 20 * 1024 * 1024


def _fetch_http_image(url: str) -> bytes:
    """Fetch a bounded HTTP image, pinning each connection to validated public IPs."""
    for redirects in range(4):
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("image URL must be an unauthenticated HTTP(S) URL")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        addresses = socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
        if not addresses:
            raise ValueError("image host has no addresses")
        for _, _, _, _, address in addresses:
            ip = ipaddress.ip_address(address[0])
            ip = getattr(ip, "ipv4_mapped", None) or ip
            if (not ip.is_global or ip.is_loopback or ip.is_private or ip.is_link_local
                    or ip.is_multicast or ip.is_unspecified or ip.is_reserved):
                raise ValueError("image URL resolves to a non-public address")

        def connect_validated(_address, timeout=15.0, source_address=None):
            error = None
            for family, socktype, proto, _, address in addresses:
                sock = socket.socket(family, socktype, proto)
                try:
                    sock.settimeout(timeout)
                    if source_address:
                        sock.bind(source_address)
                    sock.connect(address)
                    return sock
                except OSError as exc:
                    error = exc
                    sock.close()
            raise error or OSError("image connection failed")

        cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        conn = cls(parsed.hostname, port, timeout=15.0)
        conn._create_connection = connect_validated
        try:
            path = parsed.path or "/"
            if parsed.query:
                path += "?" + parsed.query
            conn.request("GET", path, headers={"Accept": "image/*", "User-Agent": "ViveMind"})
            response = conn.getresponse()
            if response.status in {301, 302, 303, 307, 308}:
                location = response.getheader("Location")
                if redirects == 3 or not location:
                    raise ValueError("image redirect limit or missing target")
                url = urljoin(url, location)
                continue
            if response.status != 200:
                raise ValueError(f"image HTTP status {response.status}")
            length = response.getheader("Content-Length")
            if length is not None and int(length) > MAX_IMAGE_BYTES:
                raise ValueError("image exceeds maximum size")
            data = response.read(MAX_IMAGE_BYTES + 1)
            if len(data) > MAX_IMAGE_BYTES:
                raise ValueError("image exceeds maximum size")
            return data
        finally:
            conn.close()
    raise ValueError("image redirect limit")


def read_source(src: str) -> bytes:
    text = str(src or "").strip()
    if not text:
        raise ValueError("image source is empty")
    if text.startswith("data:"):
        return decode_data_url(text)
    if text.startswith(("http://", "https://")):
        return _fetch_http_image(text)
    if text.lower().startswith("file://"):
        text = text[7:]
    path = Path(os.path.expanduser(text))
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"image not found: {path}")
    return path.read_bytes()


def decode_data_url(value: str) -> bytes:
    header, _, payload = str(value).partition(",")
    if ";base64" not in header:
        raise ValueError("image data URL must be base64-encoded")
    try:
        return base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise ValueError(f"invalid image data URL: {exc}") from exc


def validate_image(data: bytes) -> Dict[str, Any]:
    if not data:
        raise ValueError("image is empty")
    if Image is None:
        raise RuntimeError("Pillow is required for meme image storage")
    try:
        with Image.open(io.BytesIO(data)) as image:
            fmt = str(image.format or "").upper()
            if fmt not in FORMAT_TO_MIME:
                raise ValueError(f"unsupported image format: {fmt or 'unknown'}")
            image.load()
            width, height = image.size
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"cannot decode image: {exc}") from exc
    mime, ext = FORMAT_TO_MIME[fmt]
    return {"mime": mime, "ext": ext, "width": int(width), "height": int(height)}


def store_file(root: Path, data: bytes, sha256: str, ext: str) -> Tuple[str, bool]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rel = f"{sha256[:2]}/{sha256}{ext}"
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return rel, False
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, target)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
    return rel, True


def remove_file(data_dir: str | Path, rel: str) -> None:
    try:
        path = asset_path(data_dir, rel)
    except ValueError:
        return
    try:
        path.unlink(missing_ok=True)
        if path.parent != asset_root(data_dir).resolve():
            path.parent.rmdir()
    except OSError:
        logger.debug("sanae_mem: meme asset cleanup skipped for %s", rel)


def file_sha256(path: str | Path) -> str:
    with Path(path).open("rb") as image:
        return hashlib.file_digest(image, "sha256").hexdigest()


def copy_asset(
    data_dir: str | Path, archive_root: str | Path, record_id: int, asset: Any,
) -> Dict[str, Any]:
    source = asset_path(data_dir, str(asset["path"]))
    if not source.is_file():
        raise FileNotFoundError(f"meme asset missing from disk: {source}")
    target_dir = Path(archive_root) / "memes" / str(int(record_id))
    target_dir.mkdir(parents=True, exist_ok=True)
    suffix = source.suffix.lower() or ".bin"
    target = target_dir / f"{int(asset['id'])}-{str(asset['sha256'])[:12]}{suffix}"
    shutil.copy2(source, target)
    return {
        "id": int(asset["id"]),
        "path": str(asset["path"] or ""),
        "sha256": str(asset["sha256"] or ""),
        "mime": str(asset["mime"] or ""),
        "bytes": int(asset["bytes"] or 0),
        "width": int(asset["width"] or 0),
        "height": int(asset["height"] or 0),
        "archived_path": str(target),
    }


def chat_root(data_dir: str | Path) -> Path:
    return Path(data_dir) / "memes" / "chat"


def chat_size(width: int, height: int) -> tuple[int, int]:
    longest = max(int(width), int(height))
    if longest <= CHAT_STICKER_MAX_EDGE:
        return int(width), int(height)
    scale = CHAT_STICKER_MAX_EDGE / longest
    return max(1, round(int(width) * scale)), max(1, round(int(height) * scale))


def chat_target(data_dir: str | Path, row: Any) -> Path:
    suffix = Path(str(row["path"] or "")).suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        suffix = ".jpg"
    if suffix not in IMAGE_EXTS:
        raise MemeChatCopyError(f"unsupported chat copy image type: {suffix or 'unknown'}")
    name = f"{row['sha256']}_{CHAT_STICKER_COPY_VERSION}_{CHAT_STICKER_MAX_EDGE}{suffix}"
    return chat_root(data_dir) / name


def remove_chat_copy(data_dir: str | Path, row: Any) -> None:
    try:
        target = chat_target(data_dir, row)
    except (MemeChatCopyError, ValueError, OSError):
        return
    try:
        target.unlink(missing_ok=True)
        if target.parent != chat_root(data_dir).resolve():
            target.parent.rmdir()
    except OSError:
        logger.debug("sanae_mem: chat copy cleanup skipped for %s", target)


def save_static_copy(source: Path, target: Path, fmt: str) -> None:
    if Image is None or ImageOps is None:
        raise MemeChatCopyError("Pillow is unavailable")
    try:
        with Image.open(source) as image:
            image = ImageOps.exif_transpose(image)
            width, height = image.size
            target_size = chat_size(width, height)
            image = image.resize(target_size, Image.Resampling.LANCZOS)
            if fmt == "JPEG":
                image.convert("RGB").save(target, format="JPEG", quality=90, optimize=True)
            elif fmt == "PNG":
                image.save(target, format="PNG")
            elif fmt == "WEBP":
                image.save(target, format="WEBP", quality=90)
            elif fmt == "BMP":
                image.save(target, format="BMP")
            else:
                raise MemeChatCopyError(f"unsupported chat copy format: {fmt}")
    except MemeChatCopyError:
        raise
    except Exception as exc:
        raise MemeChatCopyError(f"static chat copy failed: {exc}") from exc


def save_gif_copy(source: Path, target: Path) -> None:
    if Image is None:
        raise MemeChatCopyError("Pillow is unavailable")
    try:
        with Image.open(source) as image:
            frame_count = int(getattr(image, "n_frames", 1) or 1)
            loop = image.info.get("loop")
            target_size = chat_size(*image.size)
            durations: List[int] = []
            frames: List[Any] = []
            for index in range(frame_count):
                image.seek(index)
                durations.append(int(image.info.get("duration", 0) or 0))
                frame = image.convert("RGBA")
                frame = frame.resize(target_size, Image.Resampling.LANCZOS)
                alpha = frame.getchannel("A").point(lambda value: 255 if value >= 128 else 0)
                frame.putalpha(alpha)
                frames.append(frame)
            if not frames:
                raise MemeChatCopyError("GIF has no decodable frames")
            save_kwargs: Dict[str, Any] = {
                "save_all": True,
                "append_images": frames[1:],
                "duration": durations,
                "disposal": 2,
                "optimize": False,
            }
            if loop is not None:
                save_kwargs["loop"] = loop
            frames[0].save(target, format="GIF", **save_kwargs)

        with Image.open(target) as check:
            saved_frame_count = int(getattr(check, "n_frames", 1) or 1)
            if saved_frame_count < 1 or saved_frame_count != len(frames):
                raise MemeChatCopyError("GIF chat copy lost frames")
            if max(check.size) > CHAT_STICKER_MAX_EDGE:
                raise MemeChatCopyError("GIF chat copy exceeds sticker size limit")
    except MemeChatCopyError:
        raise
    except Exception as exc:
        raise MemeChatCopyError(f"GIF chat copy failed: {exc}") from exc


def chat_copy(data_dir: str | Path, row: Any) -> tuple[Path, int, int]:
    source = asset_path(data_dir, str(row["path"]))
    if not source.is_file():
        raise MemeChatCopyError(f"chat copy source missing: {source}")
    try:
        with Image.open(source) as image:
            fmt = str(image.format or "").upper()
            is_gif = fmt == "GIF" or source.suffix.lower() == ".gif"
            if is_gif:
                width, height = image.size
            elif ImageOps is not None:
                width, height = ImageOps.exif_transpose(image).size
            else:
                width, height = image.size
    except Exception as exc:
        raise MemeChatCopyError(f"chat copy source unreadable: {exc}") from exc

    if max(width, height) <= CHAT_STICKER_MAX_EDGE:
        return source, int(width), int(height)

    target = chat_target(data_dir, row)
    expected_width, expected_height = chat_size(width, height)
    if target.is_file():
        try:
            with Image.open(target) as image:
                if image.size == (expected_width, expected_height):
                    return target, expected_width, expected_height
        except Exception:
            pass
        target.unlink(missing_ok=True)

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        if is_gif:
            save_gif_copy(source, tmp)
        else:
            save_static_copy(source, tmp, fmt or source.suffix.lstrip(".").upper())
        with Image.open(tmp) as check:
            if check.size != (expected_width, expected_height):
                raise MemeChatCopyError(
                    f"chat copy size mismatch: expected {(expected_width, expected_height)}, "
                    f"got {check.size}"
                )
        os.replace(tmp, target)
    except MemeChatCopyError:
        raise
    except Exception as exc:
        raise MemeChatCopyError(f"chat copy write failed: {exc}") from exc
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
    return target, expected_width, expected_height


def image_bytes(value: Any, depth: int = 0) -> List[bytes]:
    if depth > 6 or value is None:
        return []
    if isinstance(value, bytes):
        return [value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("data:image/") or text.startswith("data:application/octet-stream"):
            try:
                return [decode_data_url(text)]
            except Exception:
                return []
        if text.startswith(("http://", "https://")):
            return []

        def read_path(candidate: str) -> Optional[bytes]:
            candidate = candidate.strip().strip("'\"")
            if candidate.lower().startswith("file://"):
                candidate = candidate[7:]
            path = Path(os.path.expanduser(candidate))
            if path.suffix.lower() not in IMAGE_EXTS:
                return None
            try:
                return path.read_bytes() if path.is_file() else None
            except OSError:
                return None

        hinted = [match.group(1) for match in IMAGE_HINT_RE.finditer(text)]
        noted = []
        for match in _VISION_IMAGE_URL_RE.finditer(text):
            candidate = match.group(1).strip()
            candidate = re.sub(r"\s*~?\]\s*$", "", candidate).strip().strip("'\"")
            if candidate:
                noted.append(candidate)
        candidates = [*hinted, *noted]
        if candidates:
            images = [data for data in (read_path(item) for item in candidates) if data]
            if images:
                return images
        direct = read_path(text)
        return [direct] if direct else []
    if isinstance(value, list):
        images: List[bytes] = []
        for item in value:
            images.extend(image_bytes(item, depth + 1))
        return images
    if isinstance(value, dict):
        kind = str(value.get("type") or "")
        if kind in {"image", "image_url", "input_image", "image_path"}:
            for key in ("image_url", "url", "path", "image_path"):
                if key in value:
                    found = image_bytes(value.get(key), depth + 1)
                    if found:
                        return found
            source = value.get("source")
            if isinstance(source, dict):
                if str(source.get("type") or "") == "base64":
                    try:
                        raw = base64.b64decode(str(source.get("data") or ""), validate=True)
                        return [raw] if raw else []
                    except Exception:
                        return []
            data = value.get("data")
            if isinstance(data, str) and data:
                try:
                    raw = base64.b64decode(data, validate=True)
                    return [raw] if raw else []
                except Exception:
                    return []
            return []
        images: List[bytes] = []
        for key in ("image_url", "url", "path", "image_path", "source", "content"):
            if key in value:
                images.extend(image_bytes(value.get(key), depth + 1))
        return images
    return []
