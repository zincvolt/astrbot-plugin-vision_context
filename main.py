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

# 指令自声明（插件开发规范要求，见 docs/PLUGIN-DEV.md）：本插件无斜线指令
DECLARED_COMMANDS: dict[str, tuple[str, str]] = {}
DECLARED_BEHAVIOR = "无斜线指令：为 /读图 提供图片上下文缓存与同图去重（群聊+私聊，私聊与 /私聊 白名单同口径）"

ROOT = Path(r"D:\QQNTBOT")
BOT_DB = ROOT / "apps" / "AstrBot" / "data" / "data_v4.db"

# 识图后端按顺序试：先画面描述（VL），再只认文字的 OCR。
# 8084 那个缓存代理已经不在 start-all 里了，别再把请求发过去空等超时。
CAPTION_ENDPOINTS = [
    (
        "http://127.0.0.1:8085/v1/chat/completions",
        "sk-vl-local",
        "Qwen3VL-4B-Instruct",
        "用中文简短描述这张图：有文字就照抄要点，再说画面是什么，不超过 80 字。",
    ),
    (
        "http://127.0.0.1:8083/v1/chat/completions",
        "sk-ocr-local",
        "PaddleOCR-VL-1.6",
        "用一句中文说明这图：先照抄图上所有文字，再一句话描述画面。",
    ),
]

# 与 peer / meme_manager 共用同一张表，谁先识图谁写，别处直接复用
VISION_CACHE_DB = ROOT / "data" / "vision_cache.db"
VISION_STICKER_BYTES = 1536 * 1024   # 表情包体积上限，和 meme_manager 一致
VISION_KEEP_DAYS = 30                # 大图描述留多久
VISION_KEEP_HITS = 2                 # 大图低于这个调用次数，过期就清
VISION_GC_INTERVAL = 3600            # 最快多久清理一次

MAX_PER_SESSION = 6        # 每个会话保留的图片释义条数
INJECT_WINDOW_SEC = 900    # 15 分钟内的图才算上下文
INJECT_MAX = 3             # 最多注入几条
INJECT_MAX_CHARS = 300     # 注入总长度上限
# 长度按图分档：表情包一句话，截图/长图可能整段文案，60 字会把内容全裁掉，
# 裁掉的描述还会被缓存复用，等于永久丢掉图上的文字
STICKER_PROFILE_BYTES = 300 * 1024
STICKER_CAPTION_CHARS = 90
CONTENT_CAPTION_CHARS = 320
MAX_CAPTION_CHARS = STICKER_CAPTION_CHARS   # 兼容旧引用
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
COLLECT_MAX_BYTES = VISION_STICKER_BYTES   # 长图/截图/原图照片不进表情库

# 「还没归类的表情，暂存区」这类占位文案对模型是零信息噪声，注入出去
# 反而误导（模型以为知道图内容了）。低质描述一律不注入、不当描述用，
# 该图按「没描述」处理（peer 那边会把原图发给模型兜底）。
PLACEHOLDER_WORDS = ("暂存", "未归类", "未分类")

# 判官触发的主动接话，只在这条话真的在谈图时才注入历史图描述。
# 词表跟 peer 的读图闸保持一致（peer 改了这边跟着变），peer 没加载时用兜底词表。
try:
    from astrbot_plugin_peer.main import VISION_HINTS as IMAGE_REF_HINTS
except Exception:  # noqa: BLE001 - peer 未加载时的本地兜底
    IMAGE_REF_HINTS = (
        "看图", "这图", "那张图", "刚才那张", "上一张", "图片", "图里", "图上",
        "这是啥", "这啥", "是啥", "啥玩意", "什么东西", "啥东西", "啥意思",
        "什么意思", "长啥样", "啥图", "什么图", "识别", "翻译", "描述一下",
    )


def caption_shaky(text: str) -> bool:
    """Heuristic: is this caption too low-quality to trust as the image?

    免费规则筛查，专抓 OCR 时代的纯文字抄写（「ARK RHODES ISLAND…」这种
    只有 logo 文字、没有画面描述）和表情库占位文案。宁可错杀：被杀掉的
    描述不发不发不注入，原图走真看图，只贵一次。

    Args:
        text: Cached caption or meme-pack description.

    Returns:
        True when the caption should not be trusted.
    """
    if not text or not text.strip():
        return True
    text = text.strip()
    if len(text) < 10:
        return True
    if any(word in text for word in PLACEHOLDER_WORDS):
        return True
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return cjk / len(text) < 0.3


def _cache_conn() -> sqlite3.Connection:
    """Open the shared caption cache.

    The schema is created here as well as in ``peer`` so either plugin can
    start first.

    Returns:
        A short-lived sqlite3 connection.
    """
    VISION_CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(VISION_CACHE_DB, timeout=3)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute(
        "CREATE TABLE IF NOT EXISTS captions("
        "sha TEXT PRIMARY KEY, caption TEXT NOT NULL, bytes INTEGER NOT NULL,"
        " hits INTEGER NOT NULL DEFAULT 0, created REAL NOT NULL,"
        " last_used REAL NOT NULL)"
    )
    return con


def cache_get(sha: str) -> str:
    """Return a stored caption and count the hit.

    Args:
        sha: SHA-256 of the raw image bytes.

    Returns:
        The caption, or an empty string when absent.
    """
    try:
        con = _cache_conn()
        try:
            row = con.execute(
                "SELECT caption FROM captions WHERE sha = ?", (sha,)
            ).fetchone()
            if not row:
                return ""
            con.execute(
                "UPDATE captions SET hits = hits + 1, last_used = ? WHERE sha = ?",
                (time.time(), sha),
            )
            con.commit()
            return row[0]
        finally:
            con.close()
    except Exception as exc:  # noqa: BLE001 - 缓存挂了不能挡识图
        logger.debug(f"[vision] cache read failed: {exc}")
        return ""


def cache_put(sha: str, caption: str, size: int) -> None:
    """Store a caption for later reuse.

    Args:
        sha: SHA-256 of the raw image bytes.
        caption: Text produced by the captioning backend.
        size: Size of the raw image in bytes.
    """
    now = time.time()
    try:
        con = _cache_conn()
        try:
            con.execute(
                "INSERT INTO captions(sha, caption, bytes, hits, created, last_used)"
                " VALUES(?,?,?,1,?,?)"
                " ON CONFLICT(sha) DO UPDATE SET caption = excluded.caption,"
                " bytes = excluded.bytes",
                (sha, caption, size, now, now),
            )
            con.commit()
        finally:
            con.close()
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"[vision] cache write failed: {exc}")


def cache_gc() -> int:
    """Drop stale captions for large images.

    Sticker-sized entries stay forever: they are the ones people ask about
    again, and holding them costs almost nothing.

    Returns:
        Number of rows removed.
    """
    cutoff = time.time() - VISION_KEEP_DAYS * 86400
    try:
        con = _cache_conn()
        try:
            cur = con.execute(
                "DELETE FROM captions WHERE last_used < ? AND hits < ? AND bytes > ?",
                (cutoff, VISION_KEEP_HITS, VISION_STICKER_BYTES),
            )
            con.commit()
            return cur.rowcount or 0
        finally:
            con.close()
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"[vision] cache gc failed: {exc}")
        return 0


# 本地服务 / 云端兜底各自的熔断时间戳，和上次清理时间
_LOCAL_DOWN_UNTIL = 0.0
_CLOUD_DOWN_UNTIL = 0.0
_GC_AT = 0.0
# 群图扎堆时别把十几个 flash 请求同时打出去
_CLOUD_SEM = asyncio.Semaphore(1)
# 第三方端点解析结果缓存：(配置文件 mtime, (base, key, model) | None)
_CLOUD_CACHE: tuple[float, tuple[str, str, str] | None] = (0.0, None)


def _allowed_groups() -> set[str]:
    """Read the group allowlist from the AstrBot config."""
    cfg_path = ROOT / "apps" / "AstrBot" / "data" / "cmd_config.json"
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
        entries = cfg["platform_settings"]["id_whitelist"]
    except Exception:  # noqa: BLE001
        return set()
    return {str(e) for e in entries if str(e).isdigit()}


def _private_watch(qq: str) -> bool:
    """Whether background captioning should run for this private-chat user.

    与 /私聊 白名单同一口径：总闸关（enable_id_white_list=False）= 谁都放；
    总闸开 = 只放 id_whitelist 里 `qq:FriendMessage:<qq>` 那些人。
    """
    cfg_path = ROOT / "apps" / "AstrBot" / "data" / "cmd_config.json"
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
        ps = cfg["platform_settings"]
        if not ps.get("enable_id_white_list", False):
            return True
        entries = ps.get("id_whitelist", [])
    except Exception:  # noqa: BLE001
        return False
    privs = {
        str(e).rsplit(":", 1)[-1] for e in entries if "FriendMessage" in str(e)
    }
    return qq in privs


CMD_CONFIG = ROOT / "apps" / "AstrBot" / "data" / "cmd_config.json"


def _cloud_endpoint() -> tuple[str, str, str] | None:
    """Find a third-party provider that can look at images.

    Prefers a ``flash`` model — one call per new sticker, then the text is
    cached forever. Local bases are skipped because they were already tried.

    Returns:
        ``(api_base, api_key, model)``, or None when nothing usable is configured.
    """
    global _CLOUD_CACHE

    try:
        mtime = CMD_CONFIG.stat().st_mtime
    except OSError:
        return None
    if _CLOUD_CACHE[0] == mtime:
        return _CLOUD_CACHE[1]

    try:
        cfg = json.loads(CMD_CONFIG.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None

    sources = {s.get("id"): s for s in cfg.get("provider_sources", [])}
    best: tuple[int, tuple[str, str, str]] | None = None
    for provider in cfg.get("provider", []):
        if "image" not in (provider.get("modalities") or []):
            continue
        source = sources.get(provider.get("provider_source_id")) or {}
        base = str(source.get("api_base") or "").rstrip("/")
        keys = source.get("key") or []
        key = source.get("api_key") or (keys[0] if keys else "")
        model = str(provider.get("model") or "")
        if not base or not key or not model:
            continue
        if "127.0.0.1" in base or "localhost" in base:
            continue      # 本地端点上面已经试过了
        # 便宜的 flash 优先；没有就用任意一个支持看图的
        score = 2 if "flash" in model.lower() else 1
        if best is None or score > best[0]:
            best = (score, (base, key, model))

    result = best[1] if best else None
    _CLOUD_CACHE = (mtime, result)
    if result:
        logger.info(f"[vision] 云端兜底: {result[2]} @ {result[0]}")
    else:
        logger.info("[vision] 没有配置可用的云端识图 provider")
    return result


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


def _caption_plan(size: int, raw: bytes = b"") -> tuple[str, int, int]:
    """Pick the caption instruction and length limit for one image.

    表情包一句话就够；QQ空间截图、聊天记录这类一张图就是一段文案，
    60 字会把内容裁掉，之后每次复用都拿到残缺描述。

    Args:
        size: Raw image size in bytes.
        raw: Raw bytes, used to measure the aspect ratio when available.

    Returns:
        ``(instruction, max_tokens, char_cap)``.
    """
    tall = False
    if raw:
        try:
            import io

            from PIL import Image as _Img

            with _Img.open(io.BytesIO(raw)) as im:
                w, h = im.size
            tall = h / max(w, 1) >= 1.35 or w / max(h, 1) >= 2.5
        except Exception:  # noqa: BLE001 - 量不出尺寸就按体积判断
            tall = False

    if size > STICKER_PROFILE_BYTES or tall:
        return (
            "这是截图或长图。请把图上所有文字按从上到下的顺序尽量完整地抄下来"
            "（保留换行），再用一句话说明画面。文字多就都抄，不要总结，"
            f"最多 {CONTENT_CAPTION_CHARS} 字。",
            600,
            CONTENT_CAPTION_CHARS,
        )
    return (
        "用中文一句话说清这张表情包：有文字就照抄，再说画面和情绪，"
        f"不超过 {STICKER_CAPTION_CHARS} 字。",
        200,
        STICKER_CAPTION_CHARS,
    )


class Main(star.Star):
    """Caption group images in the background and inject them as context."""

    def __init__(self, context: star.Context) -> None:
        self.context = context
        self._recent: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=MAX_PER_SESSION)
        )
        self._inflight: set[str] = set()
        self._collected: set[str] = set()   # 本进程已经收过的图，省掉重复 stat
        self.meme_index = load_meme_index()
        logger.info(f"[vision] ready, meme index: {len(self.meme_index)} images")

    def _image_ref(self, comp: Image) -> str | None:
        """Return a usable reference (data URL or http URL or local path)."""
        for attr in ("url", "file", "path"):
            value = getattr(comp, attr, None)
            if value:
                return str(value)
        return None

    async def _caption(self, ref: str, size: int = 0, raw: bytes = b"") -> str | None:
        """Caption one image, trying local backends before spending tokens.

        Args:
            ref: data URL, http URL or local path of the image.
            size: Raw image size in bytes, used to decide whether a cloud
                fallback is worth it.
            raw: Raw bytes when the caller already has them.

        Returns:
            Caption text, or None when every backend is unavailable.
        """
        global _LOCAL_DOWN_UNTIL

        if ref.startswith("data:"):
            image_url = ref
        elif ref.startswith(("http://", "https://")):
            image_url = ref
        else:
            path = Path(ref)
            if not path.exists():
                return None
            raw = path.read_bytes()
            data = base64.b64encode(raw).decode()
            image_url = f"data:image/jpeg;base64,{data}"

        if not size:
            size = len(raw)
        instruction, max_tokens, cap = _caption_plan(size, raw)

        if time.time() >= _LOCAL_DOWN_UNTIL:
            for url, key, model, _base_hint in CAPTION_ENDPOINTS:
                body = {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": instruction},
                                {"type": "image_url", "image_url": {"url": image_url}},
                            ],
                        }
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0,
                }
                try:
                    async with httpx.AsyncClient(timeout=90) as client:
                        resp = await client.post(
                            url, headers={"Authorization": f"Bearer {key}"}, json=body
                        )
                        resp.raise_for_status()
                        text = resp.json()["choices"][0]["message"]["content"].strip()
                    if text:
                        return text[:cap]
                except Exception as exc:  # noqa: BLE001 - 换下一个后端
                    logger.debug(f"[vision] {model} unavailable: {exc}")
                    continue
            _LOCAL_DOWN_UNTIL = time.time() + 120
            logger.info("[vision] 本地识图服务不可用，120 秒内直接走 flash")

        # 本地全不通（第三方模式会停掉 8083/8085）：用 flash 补一次描述。
        # 只做表情包体积的图——大图多半是长截图，30 天后就要清，不值得花 token。
        if 0 < size <= VISION_STICKER_BYTES:
            text = await self._caption_cloud(image_url, instruction)
            if text:
                return text[:cap]
        return None

    async def _caption_cloud(
        self, image_url: str, instruction: str | None = None, max_tokens: int = 200
    ) -> str | None:
        """Caption one image with the third-party vision provider.

        This is the fallback for third-party mode, where the local VL servers
        are stopped on purpose. One flash call per new sticker, then the text
        is cached and never paid for again.

        Args:
            image_url: data URL of the image.
            instruction: Prompt for this image, picked by :func:`_caption_plan`.
            max_tokens: Output budget; screenshots need more than stickers.

        Returns:
            Caption text, or None when the call failed.
        """
        if time.time() < _CLOUD_DOWN_UNTIL:
            return None
        endpoint = _cloud_endpoint()
        if not endpoint:
            return None
        base, key, model = endpoint
        if not instruction:
            instruction = (
                "用中文一句话说清这张表情包：有文字就照抄，再说画面和情绪，"
                f"不超过 {STICKER_CAPTION_CHARS} 字。"
            )
        body = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0,
            # 不关掉思考，max_tokens 会被 reasoning_content 全吃光，
            # content 返回空字符串（踩过两次的坑）
            "thinking": {"type": "disabled"},
        }
        # 串行：群图扎堆时别同时打十几个请求出去
        async with _CLOUD_SEM:
            try:
                async with httpx.AsyncClient(timeout=60) as client:
                    resp = await client.post(
                        f"{base}/chat/completions",
                        headers={"Authorization": f"Bearer {key}"},
                        json=body,
                    )
                    resp.raise_for_status()
                    message = resp.json()["choices"][0]["message"]
                    text = (message.get("content") or "").strip()
                    if not text and message.get("reasoning_content"):
                        logger.warning(
                            "[vision] flash 只回了思考内容，content 为空（检查 thinking 参数）"
                        )
                if text:
                    logger.info(f"[vision] flash 补描述 {text[:30]}")
                    return text
            except Exception as exc:  # noqa: BLE001 - 补描述是尽力而为
                logger.warning(f"[vision] flash 描述失败: {exc}")
        return None

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def watch_images(self, event: AstrMessageEvent) -> None:
        """Caption images posted in allowlisted groups and private chats.

        私聊图进缓存是省 token 的关键：第三方模式下 peer 见到有缓存的图
        就用文字描述替换原图（一张原图 1000~2000 token）。私聊 session
        key 用 ``u<qq>``，群聊用群号，互不串。
        """
        group = str(event.get_group_id() or "")
        if group:
            if group not in _allowed_groups():
                return
            session = group
        else:
            sender = str(event.get_sender_id() or "")
            if not sender or not _private_watch(sender):
                return
            session = f"u{sender}"
        # 消息本体里的图 + 引用消息里的图都算。「引用一张图问 bot」很常见，
        # 引用里的图不收进 ring buffer 的话，模型永远只能说看不到
        images = [m for m in event.get_messages() if isinstance(m, Image)]
        for comp in event.get_messages():
            if type(comp).__name__ == "Reply":
                chain = getattr(comp, "chain", None) or []
                images += [m for m in chain if isinstance(m, Image)]
        if not images:
            return
        seen_refs: set[str] = set()
        for comp in images[:3]:
            ref = self._image_ref(comp)
            if not ref or ref in self._inflight or ref in seen_refs:
                continue
            seen_refs.add(ref)
            self._inflight.add(ref)
            asyncio.create_task(self._store_caption(session, ref))

    async def _store_caption(self, session: str, ref: str) -> None:
        """Caption one image once, then reuse the stored text forever.

        Args:
            session: Group number, or ``u<qq>`` for a private chat.
            ref: Where to read the image from.
        """
        global _GC_AT

        try:
            data = await self._read_bytes(ref)
            if not data:
                return
            sha = hashlib.sha256(data).hexdigest()

            # 同一张图（表情包会反复刷）只识一次，之后 0 秒 0 token 返回
            cached = cache_get(sha)
            if cached:
                self._recent[session].append((time.time(), cached))
                logger.info(f"[vision] cache hit {sha[:8]}")
                return

            # 群里发的正好是表情库里收录过的图：直接给出已标记的含义，不跑 OCR。
            # 「未分类」收件箱的描述是占位文案（「还没归类的表情，暂存区」），
            # 当描述用只会误导模型——跳过，照常走识图流程让 flash 真看一眼，
            # 描述进缓存后下次 cache hit 拿到的就是真内容。
            known = self.meme_index.get(sha[:16])
            if known and known[1] != "未分类":
                pack, category, desc = known
                note = f"[表情库·{category}] {desc}"
                self._recent[session].append((time.time(), note))
                logger.info(f"[vision] known meme ({pack}/{category})")
                return

            if COLLECT_ENABLED:
                await self._collect_bytes(data)
            text = await self._caption(ref, len(data), data)
            if text:
                cache_put(sha, text, len(data))
                self._recent[session].append((time.time(), text))
                logger.info(f"[vision] captioned {sha[:8]}: {text[:40]}")

            now = time.time()
            if now - _GC_AT > VISION_GC_INTERVAL:
                _GC_AT = now
                removed = await asyncio.to_thread(cache_gc)
                if removed:
                    logger.info(f"[vision] cache gc removed {removed} stale caption(s)")
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
        """Save new stickers into the unsorted inbox, skipping known ones.

        Args:
            data: Raw image bytes.
        """
        try:
            if not data:
                return
            if len(data) > COLLECT_MAX_BYTES:
                # 表情包几十到几百 KB；长图/截图/原图照片动辄几 MB，
                # 放进来会把表情库灌满
                logger.info(
                    f"[vision] 不收：{len(data) / 1048576:.1f} MiB 超过表情包体积上限"
                )
                return
            digest = hashlib.sha256(data).hexdigest()[:16]
            if digest in self._collected:
                return
            if next(COLLECT_DIR.glob(f"{digest}*"), None) is not None:
                self._collected.add(digest)
                return
            COLLECT_DIR.mkdir(parents=True, exist_ok=True)
            suffix = (
                ".gif" if data[:3] == b"GIF"
                else ".png" if data[:4] == b"\x89PNG"
                else ".jpg"
            )
            target = COLLECT_DIR / f"{digest}{suffix}"
            target.write_bytes(data)
            self._collected.add(digest)
            logger.info(f"[vision] collected -> 未分类/{target.name}")
        except Exception as exc:  # noqa: BLE001 - collecting is best effort
            logger.debug(f"[vision] collect skipped: {exc}")

    @filter.on_llm_request()
    async def inject_context(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        """Add recent image captions to the prompt (group and private)."""
        group = str(event.get_group_id() or "")
        if group:
            session = group
            scope = "群里"
        else:
            sender = str(event.get_sender_id() or "")
            if not sender:
                return
            session = f"u{sender}"
            scope = "私聊里"
        buf = self._recent.get(session)
        if not buf:
            return
        if group and event.get_extra("vision_judge_wake"):
            # 判官触发的主动接话：判官不看图、也没人叫它，注入的东西它
            # 需求说了算——只有当前这条话真的在谈图（带图或指图问）时，
            # 历史图描述才帮得上忙；否则就是往无关闲聊里塞 15 分钟前的
            # 旧图描述（2026-11-21）。直通唤醒/艾特唤醒不打这个标记，
            # 注入行为不变。
            has_image = bool(req.image_urls) or any(
                isinstance(m, Image) for m in event.get_messages()
            )
            text = req.prompt or ""
            if not has_image and not any(h in text for h in IMAGE_REF_HINTS):
                logger.debug("[vision] judge wake, message not image-related, skip inject")
                return
        now = time.time()
        # peer 已经把当前这条消息的图释义拼进 prompt 了，别再说一遍
        already = req.prompt or ""
        picked = []
        budget = INJECT_MAX_CHARS
        for ts, text in reversed(buf):
            if now - ts > INJECT_WINDOW_SEC:
                break
            if len(picked) >= INJECT_MAX or budget <= 0:
                break
            if text and text in already:
                continue
            if caption_shaky(text):
                continue  # 占位/纯 OCR 抄写这类低质描述，注入出去只会误导
            picked.append(f"- {text}")
            budget -= len(text) + 2
        if not picked:
            return
        joined = "\n".join(reversed(picked))
        req.system_prompt = (req.system_prompt or "") + (
            f"\n\n[{scope}最近发过的图（按时间从早到晚），对方问「这图/刚才那张图」"
            "或引用消息带图时，优先对照下面的描述回答，不要说看不到；"
            "描述可能不完整，本轮消息里如果附了原图，以原图为准]\n"
            + joined
        )
        logger.info(
            f"[vision] injected {len(picked)} caption(s): "
            + " | ".join(p[:40] for p in picked)  # picked 是 "- 文本" 字符串列表
        )
