"""
sunrgbd.py — базові утиліти для роботи з набором даних SUN RGB-D.

Покриває:
  * пошук сцен у довільній структурі каталогів SUN RGB-D;
  * коректне декодування 16-бітних карт глибини (бітовий зсув!);
  * зчитування внутрішніх параметрів камери (intrinsics);
  * парсинг 2D-полігональних анотацій з annotation2Dfinal/index.json;
  * нормалізацію назв класів;
  * зворотну проєкцію пікселів у 3D (використовується у розділі 3 роботи).

Автор: Поліна Сушко
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Константи
# ---------------------------------------------------------------------------

#: Максимальна достовірна дальність сенсорів SUN RGB-D (метри).
#: Значення понад цю межу вважаються недостовірними і обнуляються.
MAX_DEPTH_M = 8.0

#: Категорії, які НЕ є екземплярами об'єктів ("stuff", а не "things").
#: Їх виключаємо з instance-сегментації: вони не мають чітких меж екземплярів.
STUFF_CLASSES = {
    "wall", "floor", "ceiling", "room", "background", "unknown",
    "wall decoration", "floor mat", "ceiling light",
}

#: Приведення синонімів та типових варіантів написання до єдиної назви.
#: Ключ — нормалізована назва з датасету, значення — канонічна назва.
SYNONYMS = {
    "chairs": "chair", "office chair": "chair", "armchair": "chair",
    "desk chair": "chair", "swivel chair": "chair",
    "tables": "table", "desk": "table", "coffee table": "table",
    "dining table": "table", "side table": "table", "end table": "table",
    "night stand": "nightstand", "bedside table": "nightstand",
    "monitor": "monitor", "computer monitor": "monitor", "screen": "monitor",
    "tv": "television", "tv screen": "television",
    "garbage bin": "garbage bin", "trash can": "garbage bin",
    "trash bin": "garbage bin", "garbage can": "garbage bin",
    "waste basket": "garbage bin", "bin": "garbage bin",
    "book shelf": "bookshelf", "bookshelves": "bookshelf",
    "shelves": "shelf", "shelfs": "shelf",
    "cabinets": "cabinet", "kitchen cabinet": "cabinet",
    "couch": "sofa", "sofa chair": "sofa",
    "pillows": "pillow", "cushion": "pillow",
    "lamps": "lamp", "table lamp": "lamp", "desk lamp": "lamp",
    "picture frame": "picture", "pictures": "picture",
    "books": "book", "papers": "paper", "bottles": "bottle",
    "boxes": "box", "cups": "cup", "mug": "cup",
    "computer": "computer", "pc": "computer", "cpu": "computer",
    "keyboard ": "keyboard", "white board": "whiteboard",
    "door ": "door", "doors": "door", "window ": "window", "windows": "window",
}

_TRAILING_NUM = re.compile(r"[\s_\-#]*\d+\s*$")
_MULTISPACE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Структури даних
# ---------------------------------------------------------------------------

@dataclass
class Scene:
    """Одна сцена (кадр) набору SUN RGB-D."""

    scene_dir: Path
    rgb_path: Path
    depth_path: Path
    intrinsics_path: Path | None
    annotation_path: Path | None
    sensor: str                 # kv1 / kv2 / xtion / realsense
    collection: str             # NYUdata / b3dodata / sun3ddata / ...
    rel_key: str                # унікальний ключ сцени відносно кореня
    group_key: str = ""         # ключ групування для split (див. assign_groups)
    scene_type: str = ""        # тип приміщення зі scene.txt

    @property
    def name(self) -> str:
        return self.scene_dir.name


@dataclass
class ObjectAnnotation:
    """Один анотований об'єкт (полігон) у сцені."""

    raw_name: str
    name: str                                  # нормалізована назва
    polygons: list[list[float]] = field(default_factory=list)  # [x1,y1,x2,y2,...]


# ---------------------------------------------------------------------------
# Пошук сцен
# ---------------------------------------------------------------------------

def _first_file(directory: Path, patterns: Sequence[str]) -> Path | None:
    """Повертає перший файл у каталозі, що відповідає одному з шаблонів."""
    if not directory.is_dir():
        return None
    for pattern in patterns:
        files = sorted(directory.glob(pattern))
        if files:
            return files[0]
    return None


def _infer_sensor_and_collection(rel_parts: Sequence[str]) -> tuple[str, str]:
    """Визначає сенсор і підколекцію за відносним шляхом сцени."""
    sensors = {"kv1", "kv2", "xtion", "realsense"}
    sensor, collection = "unknown", "unknown"
    for i, part in enumerate(rel_parts):
        if part in sensors:
            sensor = part
            if i + 1 < len(rel_parts):
                collection = rel_parts[i + 1]
            break
    if sensor == "unknown" and rel_parts:
        collection = rel_parts[0]
    return sensor, collection


def find_scenes(root: Path, require_annotation: bool = True) -> list[Scene]:
    """Рекурсивно знаходить усі сцени SUN RGB-D під каталогом `root`.

    Сценою вважається каталог, який містить підкаталоги `image/` та `depth/`.
    Структура SUN RGB-D неоднорідна (різні сенсори — різна вкладеність),
    тому пошук ведеться за наявністю цих підкаталогів, а не за фіксованим шляхом.
    """
    root = Path(root).expanduser().resolve()
    scenes: list[Scene] = []

    for image_dir in root.rglob("image"):
        if not image_dir.is_dir():
            continue
        scene_dir = image_dir.parent

        rgb = _first_file(image_dir, ["*.jpg", "*.png", "*.jpeg"])
        if rgb is None:
            continue

        # depth_bfx — покращена (inpainted) карта глибини, використовуємо її
        # за наявності; інакше — сира depth.
        depth = _first_file(scene_dir / "depth_bfx", ["*.png"])
        depth_source = "depth_bfx"
        if depth is None:
            depth = _first_file(scene_dir / "depth", ["*.png"])
            depth_source = "depth"
        if depth is None:
            continue

        intr = scene_dir / "intrinsics.txt"
        intr = intr if intr.is_file() else None

        ann = scene_dir / "annotation2Dfinal" / "index.json"
        if not ann.is_file():
            alt = scene_dir / "annotation2D" / "index.json"
            ann = alt if alt.is_file() else None
        if require_annotation and ann is None:
            continue

        rel = scene_dir.relative_to(root)
        sensor, collection = _infer_sensor_and_collection(rel.parts)

        scene_type = ""
        scene_txt = scene_dir / "scene.txt"
        if scene_txt.is_file():
            try:
                scene_type = scene_txt.read_text(errors="ignore").strip()
            except OSError:
                pass

        scenes.append(
            Scene(
                scene_dir=scene_dir,
                rgb_path=rgb,
                depth_path=depth,
                intrinsics_path=intr,
                annotation_path=ann,
                sensor=sensor,
                collection=collection,
                rel_key=rel.as_posix(),
                scene_type=scene_type,
            )
        )
        # позначка, з якого джерела взято глибину, знадобиться в EDA
        scenes[-1].depth_source = depth_source  # type: ignore[attr-defined]

    scenes.sort(key=lambda s: s.rel_key)
    return scenes


def assign_groups(scenes: Iterable[Scene], flat_threshold: int = 30) -> None:
    """Проставляє `group_key` для коректного (без витоку) поділу на вибірки.

    Проблема: у частині колекцій (sun3ddata) багато кадрів знято з однієї
    кімнати підряд — якщо такі кадри потраплять і в train, і в test, оцінка
    якості буде завищеною. Тому кадри однієї послідовності групуємо разом.

    Евристика: якщо батьківський каталог містить багато сцен (>= flat_threshold),
    вважаємо його пласкою колекцією незалежних кадрів (напр. NYUdata) і кожна
    сцена утворює власну групу. Інакше — це послідовність, і групою є батьківський
    каталог.
    """
    scenes = list(scenes)
    parent_counts: dict[str, int] = {}
    for s in scenes:
        parent = s.scene_dir.parent.as_posix()
        parent_counts[parent] = parent_counts.get(parent, 0) + 1

    for s in scenes:
        parent = s.scene_dir.parent.as_posix()
        if parent_counts[parent] >= flat_threshold:
            s.group_key = s.rel_key           # незалежний кадр
        else:
            s.group_key = parent              # послідовність кадрів


# ---------------------------------------------------------------------------
# Глибина
# ---------------------------------------------------------------------------

def decode_depth(raw: np.ndarray, max_depth_m: float = MAX_DEPTH_M) -> np.ndarray:
    """Декодує 16-бітну карту глибини SUN RGB-D у метри.

    УВАГА. Глибина в SUN RGB-D зберігається зі зсувом бітів: три молодші біти
    перенесено у старші. Без зворотного зсуву карта виглядає як шум.
    Еквівалент оригінального MATLAB-коду:
        bitor(bitshift(d, -3), bitshift(d, 13))

    Повертає float32-масив у метрах; недостовірні значення (0 або > max) = 0.
    """
    d = np.asarray(raw).astype(np.uint16)
    high = (d >> 3).astype(np.uint16)
    low = ((d.astype(np.uint32) << 13) & 0xFFFF).astype(np.uint16)
    decoded = np.bitwise_or(high, low).astype(np.float32) / 1000.0
    decoded[~np.isfinite(decoded)] = 0.0
    decoded[decoded > max_depth_m] = 0.0
    decoded[decoded < 0.0] = 0.0
    return decoded


def read_depth(path: Path, max_depth_m: float = MAX_DEPTH_M) -> np.ndarray:
    """Зчитує та декодує карту глибини у метри."""
    from PIL import Image

    with Image.open(path) as img:
        raw = np.array(img)
    if raw.ndim == 3:                      # трапляються 3-канальні png
        raw = raw[..., 0]
    return decode_depth(raw, max_depth_m=max_depth_m)


def read_rgb(path: Path) -> np.ndarray:
    """Зчитує RGB-зображення як uint8-масив (H, W, 3)."""
    from PIL import Image

    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


# ---------------------------------------------------------------------------
# Параметри камери
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def as_matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


def read_intrinsics(path: Path | None) -> Intrinsics | None:
    """Зчитує матрицю внутрішніх параметрів камери з intrinsics.txt.

    Файл містить 9 чисел (матриця 3x3). УВАГА: параметри РІЗНІ для різних
    сенсорів, тому їх не можна задавати константою на весь набір даних.
    """
    if path is None or not Path(path).is_file():
        return None
    try:
        values = [float(v) for v in Path(path).read_text().split()]
    except (OSError, ValueError):
        return None
    if len(values) < 9:
        return None
    k = np.array(values[:9], dtype=np.float64).reshape(3, 3)
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    if not all(np.isfinite([fx, fy, cx, cy])) or fx <= 0 or fy <= 0:
        return None
    return Intrinsics(fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy))


def backproject(
    depth_m: np.ndarray,
    intr: Intrinsics,
    mask: np.ndarray | None = None,
    rgb: np.ndarray | None = None,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Зворотна проєкція: (u, v, глибина) -> 3D-координати (X, Y, Z).

    Формули класичної pinhole-моделі камери:
        Z = depth(v, u)
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

    Повертає (points Nx3, colors Nx3 або None). Враховуються лише пікселі
    з достовірною глибиною (> 0) і, за наявності, всередині маски.
    """
    h, w = depth_m.shape[:2]
    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    z = depth_m[::stride, ::stride]

    valid = z > 0
    if mask is not None:
        valid &= mask[::stride, ::stride].astype(bool)
    if not valid.any():
        return np.empty((0, 3), np.float32), (None if rgb is None else np.empty((0, 3), np.float32))

    z = z[valid].astype(np.float32)
    u = us[valid].astype(np.float32)
    v = vs[valid].astype(np.float32)

    x = (u - intr.cx) * z / intr.fx
    y = (v - intr.cy) * z / intr.fy
    points = np.stack([x, y, z], axis=1)

    colors = None
    if rgb is not None:
        colors = (rgb[::stride, ::stride][valid].astype(np.float32) / 255.0)
    return points, colors


# ---------------------------------------------------------------------------
# Анотації
# ---------------------------------------------------------------------------

def normalize_name(raw: str) -> str:
    """Приводить назву класу до канонічного вигляду.

    У SUN RGB-D назви вводили вручну, тому трапляються 'chair', 'Chair ',
    'chair2', 'chairs'. Без нормалізації один клас розпадається на десятки.
    """
    name = str(raw).strip().lower()
    name = name.replace("_", " ")
    name = _TRAILING_NUM.sub("", name)          # 'chair 2' -> 'chair'
    name = _MULTISPACE.sub(" ", name).strip()
    name = SYNONYMS.get(name, name)
    return name


def _coerce_coords(value) -> list[float]:
    """Координати в index.json бувають числами, рядками або вкладеними списками."""
    out: list[float] = []
    if value is None:
        return out
    if not isinstance(value, (list, tuple)):
        value = [value]
    for item in value:
        if isinstance(item, (list, tuple)):
            out.extend(_coerce_coords(item))
            continue
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            continue
    return out


def load_annotations(path: Path | None) -> list[ObjectAnnotation]:
    """Парсить annotation2Dfinal/index.json у список об'єктів з полігонами.

    Формат (LabelMe-подібний):
        {"objects": [{"name": "chair"}, null, ...],
         "frames":  [{"polygon": [{"object": 0, "x": [...], "y": [...]}, ...]}]}

    Файли подекуди пошкоджені або містять порожні/видалені об'єкти —
    усі такі випадки пропускаються без винятків.
    """
    if path is None or not Path(path).is_file():
        return []
    try:
        data = json.loads(Path(path).read_text(errors="ignore"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, dict):
        return []

    objects = data.get("objects") or []
    if not isinstance(objects, list):
        return []

    names: list[str | None] = []
    for obj in objects:
        if isinstance(obj, dict):
            names.append(obj.get("name"))
        else:
            names.append(None)

    collected: dict[int, ObjectAnnotation] = {}

    def _add(obj_idx: int, xs: list[float], ys: list[float]) -> None:
        if obj_idx < 0 or obj_idx >= len(names):
            return
        raw = names[obj_idx]
        if not raw:
            return
        n = min(len(xs), len(ys))
        if n < 3:                                # полігон має бути хоча б трикутником
            return
        flat: list[float] = []
        for i in range(n):
            flat.extend((xs[i], ys[i]))
        entry = collected.get(obj_idx)
        if entry is None:
            entry = ObjectAnnotation(raw_name=str(raw), name=normalize_name(raw))
            collected[obj_idx] = entry
        entry.polygons.append(flat)

    frames = data.get("frames") or []
    if isinstance(frames, list):
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            for poly in frame.get("polygon") or []:
                if not isinstance(poly, dict):
                    continue
                obj_idx = poly.get("object")
                if not isinstance(obj_idx, int):
                    continue
                _add(obj_idx, _coerce_coords(poly.get("x")), _coerce_coords(poly.get("y")))

    # запасний варіант: полігони, вкладені безпосередньо в objects
    if not collected:
        for idx, obj in enumerate(objects):
            if not isinstance(obj, dict):
                continue
            poly = obj.get("polygon")
            if isinstance(poly, dict):
                _add(idx, _coerce_coords(poly.get("x")), _coerce_coords(poly.get("y")))
            elif isinstance(poly, list):
                for p in poly:
                    if isinstance(p, dict):
                        _add(idx, _coerce_coords(p.get("x")), _coerce_coords(p.get("y")))

    return [collected[k] for k in sorted(collected)]


# ---------------------------------------------------------------------------
# Полігони -> маски / геометрія
# ---------------------------------------------------------------------------

def clip_polygon(poly: Sequence[float], width: int, height: int) -> list[float]:
    """Обрізає координати полігона до меж зображення."""
    pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
    return pts.reshape(-1).tolist()


def polygons_to_mask(polygons: Sequence[Sequence[float]], width: int, height: int) -> np.ndarray:
    """Растеризує набір полігонів в одну бінарну маску (uint8)."""
    import cv2

    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons:
        pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
        if pts.shape[0] < 3:
            continue
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    return mask


def _area_via_pycocotools(
    polygons: Sequence[Sequence[float]], width: int, height: int
) -> float | None:
    """Обчислює площу так само, як це робить COCOeval (за наявності pycocotools).

    Це важливо для узгодженості: метрика AP розділяє об'єкти на small/medium/large
    саме за полем `area`, тому воно має рахуватися тим самим кодом, що й
    під час оцінювання.
    """
    try:
        from pycocotools import mask as mask_utils
    except ImportError:
        return None
    try:
        rles = mask_utils.frPyObjects([list(p) for p in polygons], height, width)
        rle = mask_utils.merge(rles)
        return float(mask_utils.area(rle))
    except Exception:  # noqa: BLE001
        return None


def polygons_bbox_area(
    polygons: Sequence[Sequence[float]], width: int, height: int
) -> tuple[list[float], float]:
    """Обчислює COCO-bbox [x, y, w, h] та площу маски в пікселях.

    Площа рахується растеризацією (а не формулою Гаусса), щоб коректно
    обробити самоперетини та кілька полігонів одного об'єкта. За наявності
    pycocotools використовується його реалізація — для повної узгодженості
    з подальшим оцінюванням за метрикою AP.
    """
    import cv2

    all_pts = []
    for poly in polygons:
        pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
        if pts.shape[0] >= 3:
            all_pts.append(pts)
    if not all_pts:
        return [0.0, 0.0, 0.0, 0.0], 0.0

    stacked = np.vstack(all_pts)
    x0 = float(np.clip(stacked[:, 0].min(), 0, width - 1))
    y0 = float(np.clip(stacked[:, 1].min(), 0, height - 1))
    x1 = float(np.clip(stacked[:, 0].max(), 0, width - 1))
    y1 = float(np.clip(stacked[:, 1].max(), 0, height - 1))

    bw, bh = max(x1 - x0, 0.0), max(y1 - y0, 0.0)
    if bw < 1 or bh < 1:
        return [x0, y0, bw, bh], 0.0

    area = _area_via_pycocotools(polygons, width, height)
    if area is None:
        # запасний варіант: растеризація в межах обмежувальної рамки
        canvas_w, canvas_h = int(np.ceil(bw)) + 1, int(np.ceil(bh)) + 1
        canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        for pts in all_pts:
            shifted = pts - np.array([x0, y0])
            cv2.fillPoly(canvas, [np.round(shifted).astype(np.int32)], 1)
        area = float(canvas.sum())
    return [x0, y0, bw, bh], area


# ---------------------------------------------------------------------------
# Допоміжне
# ---------------------------------------------------------------------------

def iter_batched(items: Sequence, size: int) -> Iterator[Sequence]:
    for i in range(0, len(items), size):
        yield items[i : i + size]
