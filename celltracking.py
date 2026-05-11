import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import ndimage
from scipy.spatial import KDTree
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from skimage.filters import threshold_isodata
import warnings
warnings.filterwarnings("ignore")

def load_image_sequence(folder: str) -> list[np.ndarray]:
    paths = sorted(Path(folder).glob("*"))
    paths = [p for p in paths if p.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}]
    if not paths:
        raise FileNotFoundError(f"No supported images found in: {folder}")
    frames = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in paths]
    print(f"Loaded {len(frames)} frames from '{folder}'")
    return frames

def sobel_edge_detection(image: np.ndarray, blur_sigma: int = 1) -> np.ndarray:
    if blur_sigma > 0:
        ksize = 2 * (3 * blur_sigma) + 1
        image = cv2.GaussianBlur(image, (ksize, ksize), blur_sigma)
    image = image.astype(np.float64)
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    Gx = ndimage.convolve(image, kernel_x)
    Gy = ndimage.convolve(image, kernel_y)
    magnitude = np.hypot(Gx, Gy)
    return (magnitude / magnitude.max() * 255).astype(np.uint8) if magnitude.max() > 0 else magnitude.astype(np.uint8)

def isodata_threshold(edge_image: np.ndarray) -> np.ndarray:
    thresh = threshold_isodata(edge_image)
    return edge_image >= thresh

def zhang_suen_skeletonize(binary: np.ndarray) -> np.ndarray:
    return skeletonize(binary)

def reconstruct_cell_areas(skeleton: np.ndarray,
                            closing_radius: int = 4,
                            min_cell_area: int = 5,
                            max_cell_area: int = 200) -> tuple[np.ndarray, list]:

    from scipy.ndimage import distance_transform_edt
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max

    skel_uint8 = skeleton.astype(np.uint8) * 255

   
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * closing_radius + 1, 2 * closing_radius + 1)
    )
    closed = cv2.morphologyEx(skel_uint8, cv2.MORPH_CLOSE, kernel)


    filled = ndimage.binary_fill_holes(closed.astype(bool))

    distance   = distance_transform_edt(filled)
    coords     = peak_local_max(distance, min_distance=3,
                                labels=filled.astype(np.uint8))
    mask_peaks = np.zeros(distance.shape, dtype=bool)
    if coords.size > 0:
        mask_peaks[tuple(coords.T)] = True
    markers      = label(mask_peaks)
    labeled_mask = watershed(-distance, markers, mask=filled) if markers.max() > 0 \
                   else label(filled)


    props = regionprops(labeled_mask)
    for prop in props:
        if prop.area < min_cell_area or prop.area > max_cell_area:
            labeled_mask[labeled_mask == prop.label] = 0

    labeled_mask = label(labeled_mask > 0)
    return labeled_mask, regionprops(labeled_mask)


class CellTracker:
    def __init__(self):
        self.next_label = 1
        self.cell_positions: dict = {}
        self.birth_events: list = []
        self.death_events: list = []
        self.frame0_origin: dict = {}
        self._current_labels: dict = {}

    def initialise_frame(self, props: list) -> dict:
        label_map = {}
        for prop in props:
            lbl = self._new_label()
            centroid = prop.centroid
            label_map[lbl] = centroid
            self.cell_positions[lbl] = [(0, centroid)]
            self.frame0_origin[lbl] = lbl
        self._current_labels = label_map
        return label_map

    def update_frame(self, frame_idx: int, props: list,
                     nn_lookahead: int = 5, death_lookback: int = 2) -> dict:
        if not self._current_labels:
            return self.initialise_frame(props)

        prev_labels    = self._current_labels
        prev_centroids = np.array(list(prev_labels.values()))
        prev_lbls      = list(prev_labels.keys())
        curr_centroids = np.array([p.centroid for p in props]) if props else np.empty((0, 2))
        new_label_map  = {}

        if len(curr_centroids) == 0:
            self._current_labels = new_label_map
            return new_label_map

        if len(prev_centroids) == 0:
            for centroid in curr_centroids:
                lbl = self._new_label()
                new_label_map[lbl] = tuple(centroid)
                self.cell_positions[lbl] = [(frame_idx, tuple(centroid))]
                self.frame0_origin[lbl] = None
            self._current_labels = new_label_map
            return new_label_map

        curr_tree = KDTree(curr_centroids)
        used_curr = set()

        for lbl, pos_p in zip(prev_lbls, prev_centroids):
            _, idx_c = curr_tree.query(pos_p, k=1)
            idx_c = int(idx_c)
            if idx_c in used_curr:
                continue
            used_curr.add(idx_c)
            centroid_c = tuple(curr_centroids[idx_c])
            new_label_map[lbl] = centroid_c
            if lbl not in self.cell_positions:
                self.cell_positions[lbl] = []
            self.cell_positions[lbl].append((frame_idx, centroid_c))

        prev_tree = KDTree(prev_centroids)
        for idx_c, centroid_c in enumerate(curr_centroids):
            if idx_c not in used_curr:
                _, idx_p_near = prev_tree.query(centroid_c, k=1)
                parent_lbl    = prev_lbls[int(idx_p_near)]
                parent_history = self.cell_positions.get(parent_lbl, [])
                recent_frames  = [f for f, _ in parent_history
                                  if frame_idx - nn_lookahead <= f <= frame_idx]
                if len(recent_frames) >= 1:
                    d1, d2 = self._new_label(), self._new_label()
                    origin = self.frame0_origin.get(parent_lbl)
                    self.frame0_origin[d1] = origin
                    self.frame0_origin[d2] = origin
                    existing = new_label_map.pop(parent_lbl, None)
                    if existing:
                        new_label_map[d1] = existing
                        self.cell_positions[d1] = [(frame_idx, existing)]
                    new_label_map[d2] = tuple(centroid_c)
                    self.cell_positions[d2] = [(frame_idx, tuple(centroid_c))]
                    self.birth_events.append({
                        "frame": frame_idx, "parent_label": parent_lbl,
                        "frame0_origin": origin, "daughter_labels": [d1, d2],
                    })
                else:
                    lbl = self._new_label()
                    new_label_map[lbl] = tuple(centroid_c)
                    self.cell_positions[lbl] = [(frame_idx, tuple(centroid_c))]
                    self.frame0_origin[lbl] = None

        for lbl in set(prev_lbls) - set(new_label_map.keys()):
            history  = self.cell_positions.get(lbl, [])
            lookback = [f for f, _ in history if frame_idx - death_lookback <= f < frame_idx]
            if len(lookback) >= death_lookback:
                self.death_events.append({
                    "frame": frame_idx, "label": lbl,
                    "frame0_origin": self.frame0_origin.get(lbl),
                    "last_position": history[-1][1] if history else None,
                })

        self._current_labels = new_label_map
        return new_label_map

    def _new_label(self) -> int:
        lbl = self.next_label
        self.next_label += 1
        return lbl


def quantify_phenotypes(tracker: CellTracker, frame0_labels: set,
                        total_frames: int, stasis_threshold: float = 0.9) -> dict:
    dividing_origins = {ev["frame0_origin"] for ev in tracker.birth_events
                        if ev["frame0_origin"] is not None}
    dying_origins    = {ev["frame0_origin"] for ev in tracker.death_events
                        if ev["frame0_origin"] is not None}
    dying_only = dying_origins - dividing_origins

    min_frames_required = int(np.ceil(stasis_threshold * total_frames))

    def frames_tracked(origin_lbl: int) -> int:
        all_labels = {lbl for lbl, orig in tracker.frame0_origin.items()
                      if orig == origin_lbl}
        frames_seen = set()
        for lbl in all_labels:
            for frame_idx, _ in tracker.cell_positions.get(lbl, []):
                frames_seen.add(frame_idx)
        return len(frames_seen)

    cell_fates = {}
    for lbl in frame0_labels:
        if lbl in dividing_origins:
            cell_fates[lbl] = "dividing"
        elif lbl in dying_only:
            cell_fates[lbl] = "dying"
        elif frames_tracked(lbl) >= min_frames_required:
            cell_fates[lbl] = "stasis"
        else:
            cell_fates[lbl] = "unclassified"

    n_stasis       = sum(1 for f in cell_fates.values() if f == "stasis")
    n_unclassified = sum(1 for f in cell_fates.values() if f == "unclassified")

    return {
        "total_frame0":    len(frame0_labels),
        "n_dividing":      len(dividing_origins),
        "n_dying":         len(dying_only),
        "n_stasis":        n_stasis,
        "n_unclassified":  n_unclassified,
        "cell_fates":      cell_fates,
        "birth_events":    tracker.birth_events,
        "death_events":    tracker.death_events,
    }


def run_pipeline(frames: list[np.ndarray],
                 blur_sigma: int = 1,
                 closing_radius: int = 4,
                 min_cell_area: int = 5,
                 max_cell_area: int = 200,
                 nn_lookahead: int = 5,
                 death_lookback: int = 2) -> dict:
    edge_maps, skeletons, labeled_masks, all_props = [], [], [], []

    for i, frame in enumerate(frames):
        print(f"[1-4/6] Processing frame {i+1}/{len(frames)} ...")
        edges        = sobel_edge_detection(frame, blur_sigma=blur_sigma)
        binary_edges = isodata_threshold(edges)
        skeleton     = zhang_suen_skeletonize(binary_edges)
        labeled_mask, props = reconstruct_cell_areas(
            skeleton, closing_radius, min_cell_area, max_cell_area)
        edge_maps.append(edges)
        skeletons.append(skeleton)
        labeled_masks.append(labeled_mask)
        all_props.append(props)

    print("[5/6] Tracking cells ...")
    tracker       = CellTracker()
    label_history = []
    for i, props in enumerate(all_props):
        lmap = tracker.initialise_frame(props) if not tracker._current_labels \
               else tracker.update_frame(i, props, nn_lookahead, death_lookback)
        label_history.append(lmap)

    frame0_labels = {lbl for lbl, origin in tracker.frame0_origin.items() if origin == lbl}

    print("[6/6] Quantifying phenotypes ...")
    phenotypes = quantify_phenotypes(tracker, frame0_labels, total_frames=len(frames))

    print(f"\n=== Results ===")
    print(f"  Total (frame 0)  : {phenotypes['total_frame0']}")
    print(f"  Dividing         : {phenotypes['n_dividing']}")
    print(f"  Dying            : {phenotypes['n_dying']}")
    print(f"  Stasis           : {phenotypes['n_stasis']}")
    print(f"  Unclassified     : {phenotypes['n_unclassified']}")

    return {
        "edge_maps":     edge_maps,
        "skeletons":     skeletons,
        "labeled_masks": labeled_masks,
        "label_history": label_history,
        "phenotypes":    phenotypes,
        "tracker":       tracker,
    }


PALETTE = {
    "dividing":     "#39d353",
    "dying":        "#f55d3e",
    "stasis":       "#4ea8de",
    "unclassified": "#888888",
    "bg":           "#0d1117",
    "fg":           "#e6edf3",
    "grid":         "#21262d",
}


def _gray_to_rgb(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    lo, hi = img.min(), img.max()
    if hi > lo:
        img = (img - lo) / (hi - lo)
    return (np.stack([img] * 3, axis=-1) * 255).astype(np.uint8)


def _label_to_rgb(labeled: np.ndarray) -> np.ndarray:
    from skimage.color import label2rgb
    return (label2rgb(labeled, bg_label=0) * 255).astype(np.uint8)


def _build_frame0_origin(result: dict) -> dict:
    label_history = result["label_history"]
    phenotypes    = result["phenotypes"]
    frame0_labels = set(label_history[0].keys()) if label_history else set()
    frame0_origin = {lbl: lbl for lbl in frame0_labels}
    for ev in phenotypes["birth_events"]:
        for d in ev["daughter_labels"]:
            frame0_origin[d] = ev["frame0_origin"]
    for ev in phenotypes["death_events"]:
        frame0_origin[ev["label"]] = ev["frame0_origin"]
    return frame0_origin


def _draw_cell_overlay(frame: np.ndarray, label_map: dict,
                       cell_fates: dict, frame0_origin: dict) -> np.ndarray:
    rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    fate_colours = {
        "dividing":     (57,  211, 83),
        "dying":        (62,  93,  245),
        "stasis":       (222, 168, 78),
        "unclassified": (136, 136, 136),
        "unknown":      (180, 180, 180),
    }
    for lbl, centroid in label_map.items():
        origin = frame0_origin.get(lbl)
        fate   = cell_fates.get(origin, "unknown") if origin is not None else "unknown"
        colour = fate_colours[fate]
        cy, cx = int(centroid[0]), int(centroid[1])
        cv2.circle(rgb, (cx, cy), 4, colour, -1)
        cv2.putText(rgb, str(origin if origin else lbl),
                    (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, colour, 1, cv2.LINE_AA)
    return rgb


def visualize_pipeline(frames: list[np.ndarray], result: dict,
                       output_dir: str = "viz_output", dpi: int = 150) -> None:
    """Save a 2x3 diagnostic grid for every frame."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cell_fates    = result["phenotypes"]["cell_fates"]
    frame0_origin = _build_frame0_origin(result)
    titles = ["Raw Frame", "Sobel Edges", "Isodata Threshold",
              "Skeleton", "Cell Areas", "Tracked (colour = fate)"]

    for i in range(len(frames)):
        fig = plt.figure(figsize=(18, 7), facecolor=PALETTE["bg"])
        fig.suptitle(f"Frame {i+1} / {len(frames)}",
                     color=PALETTE["fg"], fontsize=14,
                     fontfamily="monospace", y=1.01)
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.08)

        binary = result["edge_maps"][i] >= threshold_isodata(result["edge_maps"][i])
        skel   = result["skeletons"][i]

        panels = [
            _gray_to_rgb(frames[i]),
            _gray_to_rgb(result["edge_maps"][i]),
            _gray_to_rgb(binary.astype(np.uint8) * 255),
            _gray_to_rgb((skel.astype(np.uint8) * 255) if skel.dtype == bool else skel),
            _label_to_rgb(result["labeled_masks"][i]),
            _draw_cell_overlay(
                frames[i],
                result["label_history"][i] if i < len(result["label_history"]) else {},
                cell_fates, frame0_origin),
        ]

        for j, (panel, title) in enumerate(zip(panels, titles)):
            ax = fig.add_subplot(gs[j // 3, j % 3])
            ax.imshow(panel)
            ax.set_title(title, color=PALETTE["fg"], fontsize=9,
                         fontfamily="monospace", pad=4)
            ax.axis("off")

        fig.get_axes()[-1].legend(handles=[
            mpatches.Patch(color=PALETTE["dividing"],     label="Dividing"),
            mpatches.Patch(color=PALETTE["dying"],        label="Dying"),
            mpatches.Patch(color=PALETTE["stasis"],       label="Stasis"),
            mpatches.Patch(color=PALETTE["unclassified"], label="Unclassified"),
        ], loc="lower right", fontsize=7, framealpha=0.5,
           facecolor=PALETTE["bg"], labelcolor=PALETTE["fg"])

        save_path = out / f"frame_{i+1:04d}.png"
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor=PALETTE["bg"])
        plt.close(fig)
        print(f"  Saved {save_path}")

    print(f"\nDiagnostic frames written to '{out}/'")


def visualize_summary(frames: list[np.ndarray], result: dict,
                      output_path: str = "viz_output/summary.png", dpi: int = 150) -> None:
    """Save a summary figure: annotated frame 0 + phenotype bar chart."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    phenotypes    = result["phenotypes"]
    cell_fates    = phenotypes["cell_fates"]
    frame0_origin = _build_frame0_origin(result)
    label_history = result["label_history"]

    overlay = _draw_cell_overlay(
        frames[0], label_history[0] if label_history else {},
        cell_fates, frame0_origin)

    fig = plt.figure(figsize=(14, 6), facecolor=PALETTE["bg"])
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.12, width_ratios=[1.4, 1])

    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(overlay)
    ax_img.set_title("Frame 0 — Cell Fates", color=PALETTE["fg"],
                     fontsize=12, fontfamily="monospace", pad=8)
    ax_img.axis("off")
    ax_img.legend(handles=[
        mpatches.Patch(color=PALETTE["dividing"],     label="Dividing"),
        mpatches.Patch(color=PALETTE["dying"],        label="Dying"),
        mpatches.Patch(color=PALETTE["stasis"],       label="Stasis"),
        mpatches.Patch(color=PALETTE["unclassified"], label="Unclassified"),
    ], loc="lower right", fontsize=9, framealpha=0.6,
       facecolor=PALETTE["bg"], labelcolor=PALETTE["fg"])

    ax_bar = fig.add_subplot(gs[1])
    ax_bar.set_facecolor(PALETTE["bg"])
    counts  = [phenotypes["n_dividing"], phenotypes["n_dying"],
               phenotypes["n_stasis"],   phenotypes["n_unclassified"]]
    colours = [PALETTE["dividing"], PALETTE["dying"],
               PALETTE["stasis"],   PALETTE["unclassified"]]
    bars = ax_bar.bar(["Dividing", "Dying", "Stasis", "Unclassified"],
                      counts, color=colours, width=0.55, zorder=3)
    for bar, count in zip(bars, counts):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(counts, default=1) * 0.02,
                    str(count), ha="center", va="bottom",
                    color=PALETTE["fg"], fontsize=11,
                    fontfamily="monospace", fontweight="bold")

    ax_bar.set_title(f"Phenotype Counts  (n={phenotypes['total_frame0']} cells)",
                     color=PALETTE["fg"], fontsize=11, fontfamily="monospace", pad=8)
    ax_bar.set_ylabel("Number of cells", color=PALETTE["fg"],
                      fontfamily="monospace", fontsize=9)
    ax_bar.tick_params(colors=PALETTE["fg"], labelsize=10)
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.spines[["left", "bottom"]].set_color(PALETTE["grid"])
    ax_bar.yaxis.grid(True, color=PALETTE["grid"], linewidth=0.6, zorder=0)
    ax_bar.set_axisbelow(True)
    ax_bar.set_ylim(0, max(counts, default=1) * 1.2)
    for lbl in ax_bar.get_xticklabels():
        lbl.set_fontfamily("monospace")
        lbl.set_color(PALETTE["fg"])

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"Summary figure saved to '{output_path}'")


if __name__ == "__main__":
    folder = "/data/14dC01"

    frames = load_image_sequence(folder)
    result = run_pipeline(frames)

    print("\nGenerating per-frame diagnostics ...")
    visualize_pipeline(frames, result, output_dir="viz_output")

    print("\nGenerating summary figure ...")
    visualize_summary(frames, result, output_path="viz_output/summary.png")
