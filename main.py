"""Context-aware image captioning for group chats.

Every image seen in an allowlisted group is captioned in the background
(through the 8084 cache, so repeated memes are free) and kept in a small
per-session ring buffer. When the model is about to answer, the most recent
captions are injected as context, so "@bot 这图啥意思" works even though the
image came a few messages earlier.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import sqlite3
import time
from collections import defaultdict, deque
from pathlib import Path

import httpx

from astrbot.api import logger, star
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image
from astrbot.api.provider import ProviderRequest

ROOT = Path(r"D:\QQNTBOT")
BOT_DB = ROOT / "apps" / "AstrBot" / "data" / "data_v4.db"
CAPTION_URL = "http://127.0.0.1:8084/v1/chat/completions"
CAPTION_KEY = "sk-ocr-local"
CAPTION_MODEL = "PaddleOCR-VL-1.6"

MAX_PER_SESSION = 6        # 每个会话保留的图片释义条数
INJECT_WINDOW_SEC = 900    # 15 分钟内的图才算上下文
INJECT_MAX = 3             # 最多注入几条
INJECT_MAX_CHARS = 300     # 注入总长度上限
MAX_CAPTION_CHARS = 90     # 单条释义截断
COLLECT_DIR = (
    ROOT
    / "apps"
    / "AstrBot"
    / "data"
    / "plugin_data"
    / "meme_manager"
    / "packs"
    / "unsorted"
    / "memes"
    / "未分类"
)   # 新收的图进"未分类"这个收件箱包，furry 包只读不动
COLLECT_ENABLED = True
COLLECT_MAX_BYTES = 8 * 1024 * 1024


def _allowed_groups() -> set[str]:
    """Read the group allowlist from the AstrBot config."""
    cfg_path = ROOT / "apps" / "AstrBot" / "data" / "cmd_config.json"
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
        entries = cfg["platform_settings"]["id_whitelist"]
    except Exception:  # noqa: BLE001
        return set()
    return {str(e) for e in entries if str(e).isdigit()}


def load_meme_index() -> dict[str, tuple[str, str, str]]:
    """Map image hash to (pack, category, description) for every installed pack.

    Returns:
        Mapping of sha256 prefix to pack name, category name and its description.
    """
    index: dict[str, tuple[str, str, str]] = {}
    packs_root = COLLECT_DIR.parent.parent.parent
    if not packs_root.exists():
        return index
    for pack_dir in packs_root.iterdir():
        manifest = pack_dir / "manifest.json"
        if not manifest.is_file():
            continue
        try:
            meta = json.loads(manifest.read_text(encoding="utf-8-sig"))
        except Exception:  # noqa: BLE001
            continue
        categories = meta.get("categories") or {}
        memes_dir = pack_dir / "memes"
        if not memes_dir.is_dir():
            continue
        for category_dir in memes_dir.iterdir():
            if not category_dir.is_dir():
                continue
            desc = str((categories.get(category_dir.name) or {}).get("description") or category_dir.name)
            for image in category_dir.iterdir():
                if not image.is_file():
                    continue
                try:
                    digest = hashlib.sha256(image.read_bytes()).hexdigest()[:16]
                except OSError:
                    continue
                index[digest] = (pack_dir.name, category_dir.name, desc)
    return index


class Main(star.Star):
    """Caption group images in the background and inject them as context."""

    def __init__(self, context: star.Context) -> None:
        self.context = context
        self._recent: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=MAX_PER_SESSION)
        )
        self._inflight: set[str] = set()
        self.meme_index = load_meme_index()
        logger.info(f"[vision] ready, meme index: {len(self.meme_index)} images")

    def _image_ref(self, comp: Image) -> str | None:
        """Return a usable reference (data URL or http URL or local path)."""
        for attr in ("url", "file", "path"):
            value = getattr(comp, attr, None)
            if value:
                return str(value)
        return None

    async def _caption(self, ref: str) -> str | None:
        """Ask the cache proxy for a caption of one image."""
        if ref.startswith(("http://", "https://")):
            image_url = ref
        else:
            path = Path(ref)
            if not path.exists():
                return None
            data = base64.b64encode(path.read_bytes()).decode()
            image_url = f"data:image/jpeg;base64,{data}"
        body = {
            "model": CAPTION_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "用一句中文说明这图：先照抄图上所有文字，再一句话描述画面。",
                        },
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            "max_tokens": 120,
            "temperature": 0,
        }
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    CAPTION_URL,
                    headers={"Authorization": f"Bearer {CAPTION_KEY}"},
                    json=body,
                )
                resp.raise_for_status()
                text = resp.json()["choices"][0]["message"]["content"].strip()
            return text[:MAX_CAPTION_CHARS] or None
        except Exception as exc:  # noqa: BLE001 - captions are best effort
            logger.warning(f"[vision] caption failed: {exc}")
            return None

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def watch_images(self, event: AstrMessageEvent) -> None:
        """Caption every image posted in an allowlisted group."""
        group = str(event.get_group_id() or "")
        if group not in _allowed_groups():
            return
        images = [m for m in event.get_messages() if isinstance(m, Image)]
        if not images:
            return
        umo = event.unified_msg_origin
        group_key = umo.rsplit("_", 1)[-1]
        for comp in images[:2]:
            ref = self._image_ref(comp)
            if not ref or ref in self._inflight:
                continue
            self._inflight.add(ref)
            asyncio.create_task(self._store_caption(group_key, ref))

    async def _store_caption(self, group_key: str, ref: str) -> None:
        """Caption one image, keep a local copy, and buffer it for context."""
        try:
            data = await self._read_bytes(ref)
            digest = hashlib.sha256(data).hexdigest()[:16] if data else ""
            # 群里发的正好是表情库里收录过的图：直接给出已标记的含义，不跑 OCR
            known = self.meme_index.get(digest)
            if known:
                pack, category, desc = known
                note = f"[表情库·{category}] {desc}"
                self._recent[group_key].append((time.time(), note))
                logger.info(f"[vision] known meme ({pack}/{category})")
                return
            if COLLECT_ENABLED and data:
                await self._collect_bytes(data)
            text = await self._caption(ref)
            if text:
                self._recent[group_key].append((time.time(), text))
                logger.info(f"[vision] captioned {group_key}: {text[:40]}")
        finally:
            self._inflight.discard(ref)

    async def _read_bytes(self, ref: str) -> bytes | None:
        """Load the raw image bytes from a URL or a local path."""
        try:
            if ref.startswith(("http://", "https://")):
                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.get(ref)
                    resp.raise_for_status()
                    return resp.content
            path = Path(ref)
            return path.read_bytes() if path.exists() else None
        except Exception:  # noqa: BLE001 - reading is best effort
            return None

    async def _collect_bytes(self, data: bytes) -> None:
        """Save new stickers into the unsorted inbox, skipping known ones."""
        try:
            if not data or len(data) > COLLECT_MAX_BYTES:
                return
            digest = hashlib.sha256(data).hexdigest()[:16]
            packs_root = COLLECT_DIR.parent.parent.parent
            if any(p.name.startswith(digest) for p in packs_root.rglob("*") if p.is_file()):
                return
            COLLECT_DIR.mkdir(parents=True, exist_ok=True)
            suffix = ".gif" if data[:3] == b"GIF" else ".png" if data[:4] == b"\x89PNG" else ".jpg"
            target = COLLECT_DIR / f"{digest}{suffix}"
            if not target.exists():
                target.write_bytes(data)
                logger.info(f"[vision] collected -> 未分类/{target.name}")
        except Exception as exc:  # noqa: BLE001 - collecting is best effort
            logger.debug(f"[vision] collect skipped: {exc}")

    @filter.on_llm_request()
    async def inject_context(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        """Add recent image captions to the prompt."""
        group = str(event.get_group_id() or "")
        if not group:
            return
        buf = self._recent.get(group)
        if not buf:
            return
        now = time.time()
        picked = []
        budget = INJECT_MAX_CHARS
        for ts, text in reversed(buf):
            if now - ts > INJECT_WINDOW_SEC:
                break
            if len(picked) >= INJECT_MAX or budget <= 0:
                break
            picked.append(f"- {text}")
            budget -= len(text) + 2
        if not picked:
            return
        req.system_prompt = (req.system_prompt or "") + (
            "\n\n[群里最近发过的图（按时间从早到晚），需要时可以参考，不要主动复述]\n"
            + "\n".join(reversed(picked))
        )
