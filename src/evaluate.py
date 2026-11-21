"""
Evaluation Module for Ocean Plastic CNN Classification
======================================================
Dự án: Phân loại rác thải nhựa môi trường biển (TrashCan 1.0 — Instance)
Mã công việc: S1-EVAL (Sprint 1)
Tác giả: Nhóm Đánh giá & Tối ưu hóa (Evaluation & Optimization)

Module này cung cấp toàn bộ công cụ cần thiết cho quy trình đánh giá mô hình CNN:
- Tính toán các chỉ số: Accuracy, Precision, Recall, F1-score (Per-class, Macro, Weighted).
- Tạo báo cáo phân loại (Classification Report) dạng chuỗi bảng và dictionary.
- Tính toán và trực quan hóa Confusion Matrix (raw counts và normalized).
- Trực quan hóa đường cong học tập (Learning Curves: Loss và Accuracy).
- Trích xuất và phân tích lỗi phân loại (Error Analysis: Top Confused Pairs, Misclassified Samples).
- Thiết kế bảng so sánh thực nghiệm (Experiment Comparison Table).
- Xuất kết quả đánh giá ra các định dạng chuẩn (JSON, CSV, Markdown).
- Chạy đánh giá một checkpoint trên một tập:
    * Keras (E01–E02): evaluate_run(run_dir, processed_dir, split)
    * PyTorch MobileNetV2 (E03–E04): evaluate_torch_checkpoint(run_dir, ckpt, processed_dir, split)
  Tập "test" chỉ được chạy MỘT lần cho mô hình cuối.

Dòng lệnh (PyTorch):
    python -m src.evaluate --run runs/E04_<thời gian> --ckpt best.pt --processed-dir <...>/v2_crop224 --split val
    python -m src.evaluate            # không tham số: chạy smoke test của module
"""

import argparse
import os
import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Bảo đảm mã hóa UTF-8 an toàn trên Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)

try:
    from src.preprocessing import class_names, load_class_mapping, plastic_indices
except ImportError:  # pragma: no cover
    from preprocessing import class_names, load_class_mapping, plastic_indices


def compute_classification_metrics(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    labels: Optional[List] = None,
    target_names: Optional[List[str]] = None,
    zero_division: Union[int, float] = 0.0,
) -> Dict[str, Any]:
    """
    Tính toán bộ chỉ số đánh giá hoàn chỉnh cho bài toán phân loại nhiều lớp:
    Accuracy, Macro/Weighted Precision, Recall, F1-score, và Per-class metrics.

    Parameters
    ----------
    y_true : Union[List, np.ndarray]
        Nhãn thực tế (1D array hoặc list).
    y_pred : Union[List, np.ndarray]
        Nhãn dự đoán từ mô hình (1D array hoặc list).
    labels : Optional[List], default=None
        Danh sách nhãn chỉ định để tính toán và sắp xếp thứ tự.
    target_names : Optional[List[str]], default=None
        Tên hiển thị tương ứng với các nhãn.
    zero_division : Union[int, float], default=0.0
        Giá trị gán khi xảy ra chia cho 0 (khi một lớp không có dự đoán nào).

    Returns
    -------
    Dict[str, Any]
        Dictionary chứa các chỉ số tổng hợp và chi tiết từng lớp:
        - 'accuracy': float
        - 'macro_precision': float
        - 'macro_recall': float
        - 'macro_f1': float (Primary Metric)
        - 'weighted_precision': float
        - 'weighted_recall': float
        - 'weighted_f1': float
        - 'per_class': Dict[str, Dict[str, float]]
        - 'total_samples': int
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError(
            f"Kích thước y_true ({len(y_true_arr)}) và y_pred ({len(y_pred_arr)}) không khớp."
        )

    # Xác định danh sách nhãn duy nhất nếu không được truyền vào
    if labels is None:
        unique_labels = np.unique(np.concatenate([y_true_arr, y_pred_arr]))
        try:
            labels = sorted(unique_labels.tolist())
        except TypeError:
            labels = list(unique_labels)
    else:
        labels = list(labels)

    num_classes = len(labels)
    if target_names is None:
        target_names = [str(lbl) for lbl in labels]
    elif len(target_names) != num_classes:
        raise ValueError(
            f"Số lượng target_names ({len(target_names)}) phải bằng số lượng labels ({num_classes})."
        )

    # 1. Tính toán Overall Accuracy
    acc = float(accuracy_score(y_true_arr, y_pred_arr))

    # 2. Macro Averages (Chỉ số không trọng số - Primary Metric)
    macro_prec = float(
        precision_score(
            y_true_arr,
            y_pred_arr,
            labels=labels,
            average="macro",
            zero_division=zero_division,
        )
    )
    macro_rec = float(
        recall_score(
            y_true_arr,
            y_pred_arr,
            labels=labels,
            average="macro",
            zero_division=zero_division,
        )
    )
    macro_f1 = float(
        f1_score(
            y_true_arr,
            y_pred_arr,
            labels=labels,
            average="macro",
            zero_division=zero_division,
        )
    )

    # 3. Weighted Averages (Có trọng số theo support)
    weighted_prec = float(
        precision_score(
            y_true_arr,
            y_pred_arr,
            labels=labels,
            average="weighted",
            zero_division=zero_division,
        )
    )
    weighted_rec = float(
        recall_score(
            y_true_arr,
            y_pred_arr,
            labels=labels,
            average="weighted",
            zero_division=zero_division,
        )
    )
    weighted_f1 = float(
        f1_score(
            y_true_arr,
            y_pred_arr,
            labels=labels,
            average="weighted",
            zero_division=zero_division,
        )
    )

    # 4. Per-Class Metrics
    per_class_prec = precision_score(
        y_true_arr,
        y_pred_arr,
        labels=labels,
        average=None,
        zero_division=zero_division,
    )
    per_class_rec = recall_score(
        y_true_arr,
        y_pred_arr,
        labels=labels,
        average=None,
        zero_division=zero_division,
    )
    per_class_f1 = f1_score(
        y_true_arr,
        y_pred_arr,
        labels=labels,
        average=None,
        zero_division=zero_division,
    )

    per_class_dict = {}
    for idx, name in enumerate(target_names):
        lbl = labels[idx]
        support = int(np.sum(y_true_arr == lbl))
        per_class_dict[name] = {
            "label": lbl,
            "precision": float(per_class_prec[idx]),
            "recall": float(per_class_rec[idx]),
            "f1_score": float(per_class_f1[idx]),
            "support": support,
        }

    return {
        "accuracy": acc,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_prec,
        "weighted_recall": weighted_rec,
        "weighted_f1": weighted_f1,
        "per_class": per_class_dict,
        "total_samples": len(y_true_arr),
        "target_names": target_names,
    }


def generate_classification_report(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    labels: Optional[List] = None,
    target_names: Optional[List[str]] = None,
    digits: int = 4,
    zero_division: Union[int, float] = 0.0,
    output_dict: bool = False,
) -> Union[str, Dict]:
    """
    Sinh báo cáo phân loại chi tiết (Classification Report) theo từng lớp.

    Parameters
    ----------
    y_true : Union[List, np.ndarray]
        Nhãn thực tế.
    y_pred : Union[List, np.ndarray]
        Nhãn dự đoán.
    labels : Optional[List], default=None
        Danh sách nhãn chỉ định.
    target_names : Optional[List[str]], default=None
        Danh sách tên lớp hiển thị.
    digits : int, default=4
        Số chữ số thập phân hiển thị trong bảng.
    zero_division : Union[int, float], default=0.0
        Xử lý khi chia cho 0.
    output_dict : bool, default=False
        Nếu True, trả về dict thay vì string.

    Returns
    -------
    Union[str, Dict]
        Chuỗi bảng định dạng hoặc dictionary kết quả.
    """
    return classification_report(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        target_names=target_names,
        digits=digits,
        zero_division=zero_division,
        output_dict=output_dict,
    )


def compute_confusion_matrix(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    labels: Optional[List] = None,
    normalize: Optional[str] = None,
) -> np.ndarray:
    """
    Tính toán Confusion Matrix giữa nhãn thực tế và nhãn dự đoán.

    Parameters
    ----------
    y_true : Union[List, np.ndarray]
        Nhãn thực tế.
    y_pred : Union[List, np.ndarray]
        Nhãn dự đoán.
    labels : Optional[List], default=None
        Danh sách nhãn chỉ định và thứ tự của ma trận.
    normalize : Optional[str], default=None
        Phương thức chuẩn hóa ma trận:
        - None: giữ nguyên số lượng đếm mẫu (raw counts).
        - 'true': chuẩn hóa theo nhãn thực tế (mỗi hàng tổng = 1.0, tương ứng Recall).
        - 'pred': chuẩn hóa theo nhãn dự đoán (mỗi cột tổng = 1.0, tương ứng Precision).
        - 'all': chuẩn hóa theo tổng số mẫu toàn bộ ma trận.

    Returns
    -------
    np.ndarray
        Ma trận kích thước (C, C).
    """
    return confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        normalize=normalize,
    )


def plot_confusion_matrix(
    cm: np.ndarray,
    target_names: Optional[List[str]] = None,
    normalize: bool = False,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
    show: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Vẽ và trực quan hóa Confusion Matrix dạng Heatmap chuyên nghiệp.
    Hỗ trợ hiển thị cả số lượng mẫu thực tế và tỷ lệ phần trăm trong từng ô.

    Parameters
    ----------
    cm : np.ndarray
        Ma trận nhầm lẫn kích thước (C, C).
    target_names : Optional[List[str]], default=None
        Danh sách tên lớp cho các trục.
    normalize : bool, default=False
        Nếu True, chuẩn hóa ma trận theo hàng (True class).
    title : str, default="Confusion Matrix"
        Tiêu đề biểu đồ.
    cmap : str, default="Blues"
        Bảng màu Matplotlib.
    figsize : Tuple[int, int], default=(8, 6)
        Kích thước khung hình vẽ.
    save_path : Optional[str], default=None
        Đường dẫn file để lưu biểu đồ (ví dụ: 'results/confusion_matrix.png').
    show : bool, default=False
        Có gọi plt.show() hay không.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Figure và Axes của biểu đồ.
    """
    num_classes = cm.shape[0]
    if target_names is None:
        target_names = [f"Class {i}" for i in range(num_classes)]

    # Chuẩn hóa nếu được yêu cầu
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        # Tránh chia cho 0 với hàng rỗng
        cm_display = np.divide(
            cm.astype(float),
            row_sums,
            out=np.zeros_like(cm, dtype=float),
            where=row_sums != 0,
        )
        title_suffix = " (Normalized by True Class)"
    else:
        cm_display = cm
        title_suffix = " (Raw Counts)"

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_display, interpolation="nearest", cmap=cmap)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

    ax.set_title(f"{title}{title_suffix}", fontsize=14, fontweight="bold", pad=15)
    tick_marks = np.arange(num_classes)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(target_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(target_names, fontsize=10)

    # Hiển thị số liệu trong từng ô
    thresh = (cm_display.max() + cm_display.min()) / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            val = cm_display[i, j]
            if normalize:
                text = f"{val:.1%}\n({int(cm[i, j])})"
            else:
                text = f"{int(val)}"

            color = "white" if val > thresh else "black"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color=color,
                fontsize=9,
                fontweight="medium",
            )

    ax.set_ylabel("True Class (Nhãn thực tế)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Class (Nhãn dự đoán)", fontsize=11, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[evaluate] Đã lưu Confusion Matrix tại: {save_path}")

    if show:
        plt.show()

    return fig, ax


def plot_learning_curves(
    history: Dict[str, List[float]],
    metrics: Tuple[str, str] = ("loss", "accuracy"),
    save_path: Optional[str] = None,
    title_prefix: str = "CNN Baseline",
    figsize: Tuple[int, int] = (14, 5),
    show: bool = False,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """
    Vẽ đường cong học tập (Learning Curves) biểu diễn Loss và Accuracy
    của cả tập Training và Validation qua từng Epoch. Tự động đánh dấu
    Epoch tốt nhất (Best Epoch: min val_loss và max val_accuracy).

    Parameters
    ----------
    history : Dict[str, List[float]]
        Dictionary lịch sử huấn luyện từ Keras/PyTorch:
        Phải chứa ít nhất các khóa 'loss' và 'val_loss' (cho loss),
        cùng 'accuracy'/'acc' và 'val_accuracy'/'val_acc' (cho accuracy).
    metrics : Tuple[str, str], default=('loss', 'accuracy')
        Cặp metric cần biểu diễn.
    save_path : Optional[str], default=None
        Đường dẫn file để lưu biểu đồ.
    title_prefix : str, default="CNN Baseline"
        Tiền tố tiêu đề mô hình.
    figsize : Tuple[int, int], default=(14, 5)
        Kích thước figure.
    show : bool, default=False
        Có hiển thị plt.show() hay không.

    Returns
    -------
    Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]
        Figure và 2 Axes biểu đồ Loss và Accuracy.
    """
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=figsize)

    # 1. Chuẩn hóa tên khóa của Loss
    train_loss = history.get("loss")
    val_loss = history.get("val_loss")

    if train_loss is None:
        raise KeyError("History dictionary không chứa khóa 'loss'.")

    epochs = range(1, len(train_loss) + 1)

    # --- Vẽ biểu đồ Loss ---
    ax_loss.plot(
        epochs,
        train_loss,
        "o-",
        color="#1f77b4",
        linewidth=2,
        label="Training Loss",
        markersize=4,
    )
    if val_loss is not None:
        ax_loss.plot(
            epochs,
            val_loss,
            "s--",
            color="#d62728",
            linewidth=2,
            label="Validation Loss",
            markersize=4,
        )
        best_loss_idx = int(np.argmin(val_loss))
        best_loss_epoch = epochs[best_loss_idx]
        best_loss_val = val_loss[best_loss_idx]
        ax_loss.scatter(
            [best_loss_epoch],
            [best_loss_val],
            color="#2ca02c",
            s=120,
            zorder=5,
            edgecolors="black",
            label=f"Best Val Loss ({best_loss_val:.4f} @ Ep {best_loss_epoch})",
        )

    ax_loss.set_title(f"{title_prefix} — Training & Validation Loss", fontsize=12, fontweight="bold")
    ax_loss.set_xlabel("Epoch", fontsize=11)
    ax_loss.set_ylabel("Loss (Cross-Entropy)", fontsize=11)
    ax_loss.grid(True, linestyle="--", alpha=0.6)
    ax_loss.legend(loc="upper right", frameon=True, fontsize=10)

    # 2. Chuẩn hóa tên khóa của Accuracy
    train_acc = history.get("accuracy") or history.get("acc")
    val_acc = history.get("val_accuracy") or history.get("val_acc")

    if train_acc is not None:
        ax_acc.plot(
            epochs,
            train_acc,
            "o-",
            color="#1f77b4",
            linewidth=2,
            label="Training Accuracy",
            markersize=4,
        )
        if val_acc is not None:
            ax_acc.plot(
                epochs,
                val_acc,
                "s--",
                color="#2ca02c",
                linewidth=2,
                label="Validation Accuracy",
                markersize=4,
            )
            best_acc_idx = int(np.argmax(val_acc))
            best_acc_epoch = epochs[best_acc_idx]
            best_acc_val = val_acc[best_acc_idx]
            ax_acc.scatter(
                [best_acc_epoch],
                [best_acc_val],
                color="#ff7f0e",
                s=120,
                zorder=5,
                edgecolors="black",
                label=f"Best Val Acc ({best_acc_val:.2%} @ Ep {best_acc_epoch})",
            )

        ax_acc.set_title(f"{title_prefix} — Training & Validation Accuracy", fontsize=12, fontweight="bold")
        ax_acc.set_xlabel("Epoch", fontsize=11)
        ax_acc.set_ylabel("Accuracy", fontsize=11)
        ax_acc.grid(True, linestyle="--", alpha=0.6)
        ax_acc.legend(loc="lower right", frameon=True, fontsize=10)
    else:
        ax_acc.text(
            0.5,
            0.5,
            "No Accuracy Metric Found in History",
            ha="center",
            va="center",
            transform=ax_acc.transAxes,
        )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[evaluate] Đã lưu biểu đồ Learning Curves tại: {save_path}")

    if show:
        plt.show()

    return fig, (ax_loss, ax_acc)


def identify_misclassified_samples(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    probabilities: Optional[np.ndarray] = None,
    target_names: Optional[List[str]] = None,
    sample_metadata: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Trích xuất toàn bộ các mẫu phân loại sai và tính toán các chỉ số lỗi:
    - Độ tin cậy của nhãn dự đoán sai (Confidence).
    - Độ tin cậy của nhãn thực tế (True Confidence).
    - Độ chênh lệch giữa Top-1 và Top-2 xác suất (Margin).
    - Phân loại mức độ nghiêm trọng: 'high_confidence_error', 'moderate_error', 'low_confidence_error'.

    Parameters
    ----------
    y_true : Union[List, np.ndarray]
        Nhãn thực tế.
    y_pred : Union[List, np.ndarray]
        Nhãn dự đoán.
    probabilities : Optional[np.ndarray], default=None
        Ma trận xác suất softmax kích thước (N, C).
    target_names : Optional[List[str]], default=None
        Danh sách tên lớp.
    sample_metadata : Optional[List[Dict[str, Any]]], default=None
        Metadata kèm theo từng mẫu (ví dụ: tên file ảnh, video_id).

    Returns
    -------
    List[Dict[str, Any]]
        Danh sách các dictionary chứa thông tin chi tiết từng mẫu lỗi.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    misclassified_indices = np.where(y_true_arr != y_pred_arr)[0]

    errors: List[Dict[str, Any]] = []
    for idx in misclassified_indices:
        t_label = y_true_arr[idx]
        p_label = y_pred_arr[idx]

        t_name = target_names[t_label] if (target_names and isinstance(t_label, (int, np.integer))) else str(t_label)
        p_name = target_names[p_label] if (target_names and isinstance(p_label, (int, np.integer))) else str(p_label)

        record: Dict[str, Any] = {
            "sample_index": int(idx),
            "true_label": int(t_label) if isinstance(t_label, (int, np.integer)) else t_label,
            "pred_label": int(p_label) if isinstance(p_label, (int, np.integer)) else p_label,
            "true_class_name": t_name,
            "pred_class_name": p_name,
        }

        if probabilities is not None:
            probs = probabilities[idx]
            conf = float(np.max(probs))
            sorted_probs = np.sort(probs)[::-1]
            margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else conf

            # True class confidence
            if isinstance(t_label, (int, np.integer)) and t_label < len(probs):
                true_conf = float(probs[t_label])
            else:
                true_conf = 0.0

            if conf >= 0.80:
                severity = "high_confidence_error"
            elif conf < 0.50:
                severity = "low_confidence_error"
            else:
                severity = "moderate_error"

            record.update({
                "confidence": conf,
                "true_confidence": true_conf,
                "margin": margin,
                "severity": severity,
            })

        if sample_metadata and idx < len(sample_metadata):
            record["metadata"] = sample_metadata[idx]

        errors.append(record)

    return errors


def find_top_confused_pairs(
    cm: np.ndarray,
    target_names: Optional[List[str]] = None,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Trích xuất danh sách Top K cặp lớp bị mô hình nhầm lẫn nhiều nhất từ Confusion Matrix.

    Parameters
    ----------
    cm : np.ndarray
        Ma trận nhầm lẫn kích thước (C, C).
    target_names : Optional[List[str]], default=None
        Danh sách tên lớp.
    top_k : int, default=5
        Số lượng cặp nhầm lẫn hàng đầu cần trích xuất.

    Returns
    -------
    List[Dict[str, Any]]
        Danh sách các cặp nhầm lẫn sắp xếp giảm dần theo số lượng mẫu bị nhầm:
        - 'true_idx', 'pred_idx'
        - 'true_name', 'pred_name'
        - 'count': int
        - 'percentage_of_true': float (tỷ lệ trên tổng số mẫu thực tế của lớp đó)
    """
    num_classes = cm.shape[0]
    if target_names is None:
        target_names = [f"Class {i}" for i in range(num_classes)]

    confused_pairs = []
    for i in range(num_classes):
        true_total = int(np.sum(cm[i, :]))
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                count = int(cm[i, j])
                pct = float(count / true_total) if true_total > 0 else 0.0
                confused_pairs.append({
                    "true_idx": i,
                    "pred_idx": j,
                    "true_name": target_names[i],
                    "pred_name": target_names[j],
                    "count": count,
                    "percentage_of_true": pct,
                })

    confused_pairs.sort(key=lambda x: x["count"], reverse=True)
    return confused_pairs[:top_k]


def plot_misclassified_samples(
    images: Union[List, np.ndarray],
    misclassified_info: List[Dict[str, Any]],
    target_names: Optional[List[str]] = None,
    max_samples: int = 9,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 10),
    show: bool = False,
) -> Optional[Tuple[plt.Figure, np.ndarray]]:
    """
    Hiển thị lưới hình ảnh của các mẫu bị phân loại sai kèm nhãn thực tế,
    nhãn dự đoán và độ tin cậy để phục vụ phân tích nguyên nhân môi trường biển.

    Parameters
    ----------
    images : Union[List, np.ndarray]
        Danh sách hoặc mảng chứa hình ảnh tương ứng (H, W, C) hoặc (H, W).
    misclassified_info : List[Dict[str, Any]]
        Kết quả từ hàm `identify_misclassified_samples`.
    target_names : Optional[List[str]], default=None
        Danh sách tên lớp.
    max_samples : int, default=9
        Số lượng mẫu tối đa cần hiển thị trên lưới (thường là 4 hoặc 9).
    save_path : Optional[str], default=None
        Đường dẫn file lưu ảnh.
    figsize : Tuple[int, int], default=(10, 10)
        Kích thước hình ảnh.
    show : bool, default=False
        Có gọi plt.show() hay không.

    Returns
    -------
    Optional[Tuple[plt.Figure, np.ndarray]]
        Figure và Axes nếu có mẫu sai, ngược lại None.
    """
    if not misclassified_info:
        print("[evaluate] Không có mẫu nào bị phân loại sai.")
        return None

    samples_to_show = misclassified_info[:max_samples]
    n = len(samples_to_show)
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = np.atleast_1d(axes).flatten()

    for idx, (ax, info) in enumerate(zip(axes_flat, samples_to_show)):
        sample_idx = info["sample_index"]
        img = images[sample_idx]

        # Kiểm tra chuẩn hóa ảnh nếu nằm ngoài khoảng [0, 1] hoặc [0, 255]
        if isinstance(img, np.ndarray):
            if img.max() > 1.0 and img.dtype != np.uint8:
                img_disp = (img / 255.0).clip(0, 1)
            else:
                img_disp = img
        else:
            img_disp = img

        ax.imshow(img_disp)
        ax.axis("off")

        true_name = info.get("true_class_name", f"Class {info['true_label']}")
        pred_name = info.get("pred_class_name", f"Class {info['pred_label']}")
        conf_str = f" ({info['confidence']:.1%})" if "confidence" in info else ""

        title = f"True: {true_name}\nPred: {pred_name}{conf_str}"
        ax.set_title(title, fontsize=10, color="#d62728", fontweight="bold", pad=8)

    # Tắt các ô phụ trống
    for ax in axes_flat[n:]:
        ax.axis("off")

    plt.suptitle(
        "Underwater Misclassified Samples Analysis (Phân Tích Mẫu Phân Loại Sai)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[evaluate] Đã lưu lưới ảnh phân tích lỗi tại: {save_path}")

    if show:
        plt.show()

    return fig, axes


def format_comparison_table(
    experiments: List[Dict[str, Any]],
    output_format: str = "markdown",
) -> Union[str, pd.DataFrame]:
    """
    Tạo bảng so sánh các thí nghiệm mô hình theo chuẩn khoa học.

    Parameters
    ----------
    experiments : List[Dict[str, Any]]
        Danh sách kết quả của các lần chạy thực nghiệm. Mỗi dictionary chứa:
        - 'exp_id': str
        - 'model_name': str
        - 'augmentation': str
        - 'val_loss': float
        - 'test_accuracy': float
        - 'test_macro_f1': float (Primary Metric)
        - 'test_macro_recall': float
        - 'test_macro_precision': float
        - 'parameters': str (optional)
        - 'notes': str (optional)
    output_format : str, default="markdown"
        Định dạng trả về: 'markdown', 'dataframe', hoặc 'latex'.

    Returns
    -------
    Union[str, pd.DataFrame]
        Bảng so sánh định dạng tương ứng.
    """
    df = pd.DataFrame(experiments)

    # Đổi tên cột cho trực quan
    col_mapping = {
        "exp_id": "Exp ID",
        "model_name": "Mô Hình",
        "augmentation": "Kỹ Thuật Augmentation",
        "val_loss": "Val Loss",
        "test_loss": "Test Loss",
        "test_accuracy": "Accuracy",
        "test_macro_precision": "Macro Prec",
        "test_macro_recall": "Macro Rec",
        "test_macro_f1": "Macro F1 (Chính)",
        "parameters": "Tham Số",
        "notes": "Ghi Chú",
    }
    df = df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns})

    # Định dạng các cột số thập phân
    float_cols = ["Val Loss", "Test Loss", "Accuracy", "Macro Prec", "Macro Rec", "Macro F1 (Chính)"]
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (float, np.floating)) else x)

    if output_format.lower() == "markdown":
        try:
            return df.to_markdown(index=False)
        except (ImportError, ModuleNotFoundError):
            # Fallback thuần Python không phụ thuộc thư viện tabulate ngoài
            headers = [str(c) for c in df.columns]
            rows = [[str(val) for val in row] for row in df.values]
            col_widths = [len(h) for h in headers]
            for row in rows:
                for idx, val in enumerate(row):
                    col_widths[idx] = max(col_widths[idx], len(val))
            header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
            separator_line = "|-" + "-|-".join("-" * col_widths[i] for i in range(len(headers))) + "-|"
            data_lines = [
                "| " + " | ".join(val.ljust(col_widths[i]) for i, val in enumerate(row)) + " |"
                for row in rows
            ]
            return "\n".join([header_line, separator_line] + data_lines)
    elif output_format.lower() == "latex":
        return df.to_latex(index=False)
    else:
        return df


def save_evaluation_results(
    results: Dict[str, Any],
    save_path: str,
    format: str = "json",
) -> None:
    """
    Lưu trữ kết quả đánh giá ra file để lưu vết thực nghiệm (reproducibility).

    Parameters
    ----------
    results : Dict[str, Any]
        Dictionary kết quả từ `compute_classification_metrics`.
    save_path : str
        Đường dẫn file đích (ví dụ: 'results/evaluation_results.json').
    format : str, default="json"
        Định dạng lưu trữ ('json' hoặc 'markdown').
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    if format.lower() == "json":
        # Hàm hỗ trợ serialize các đối tượng numpy
        def default_serializer(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False, default=default_serializer)
        print(f"[evaluate] Đã lưu kết quả đánh giá dạng JSON tại: {save_path}")

    elif format.lower() == "markdown":
        lines = [
            "# Kết Quả Đánh Giá Mô Hình CNN (Evaluation Summary)",
            "",
            "## 1. Các Chỉ Số Tổng Hợp",
            f"- **Accuracy:** {results.get('accuracy', 0.0):.4f}",
            f"- **Macro F1-score (Chỉ số chính):** {results.get('macro_f1', 0.0):.4f}",
            f"- **Macro Precision:** {results.get('macro_precision', 0.0):.4f}",
            f"- **Macro Recall:** {results.get('macro_recall', 0.0):.4f}",
            f"- **Weighted F1-score:** {results.get('weighted_f1', 0.0):.4f}",
            f"- **Tổng số mẫu kiểm thử:** {results.get('total_samples', 0)}",
            "",
            "## 2. Chi Tiết Từng Lớp (Per-Class Performance)",
            "| Lớp Rác Thải | Precision | Recall | F1-Score | Số Mẫu (Support) |",
            "|---|---|---|---|---|",
        ]
        per_class = results.get("per_class", {})
        for name, m in per_class.items():
            lines.append(
                f"| `{name}` | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1_score']:.4f} | {m['support']} |"
            )

        with open(save_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[evaluate] Đã lưu báo cáo đánh giá dạng Markdown tại: {save_path}")
    else:
        raise ValueError(f"Định dạng {format} không được hỗ trợ. Chọn 'json' hoặc 'markdown'.")


def smoke_test() -> bool:
    """
    Hàm kiểm thử tự động (Smoke Test) xác minh tính toàn vẹn của tất cả các hàm
    trong module với dữ liệu giả lập có kết quả kỳ vọng đã biết.
    """
    print("=" * 60)
    print("BẮT ĐẦU SMOKE TEST MODULE SRC/EVALUATE.PY")
    print("=" * 60)

    # 1. Dữ liệu giả định với 4 lớp rác thải đại dương
    classes = ["trash_bag", "trash_bottle", "trash_can", "trash_net"]
    y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 3, 3])
    y_pred = np.array([0, 0, 1, 1, 1, 2, 2, 2, 3, 0])
    # Đúng 7/10 mẫu -> Accuracy kỳ vọng = 0.70

    # 2. Kiểm thử compute_classification_metrics
    metrics = compute_classification_metrics(y_true, y_pred, target_names=classes)
    assert abs(metrics["accuracy"] - 0.70) < 1e-5, f"Lỗi Accuracy: {metrics['accuracy']}"
    assert "macro_f1" in metrics
    assert "per_class" in metrics
    print(f"[✓] Metrics computation passed. Accuracy: {metrics['accuracy']:.2f}, Macro F1: {metrics['macro_f1']:.4f}")

    # 3. Kiểm thử classification_report
    report_str = generate_classification_report(y_true, y_pred, target_names=classes)
    assert "trash_bag" in report_str
    print("[✓] Classification report generated successfully.")

    # 4. Kiểm thử confusion_matrix
    cm = compute_confusion_matrix(y_true, y_pred)
    assert cm.shape == (4, 4)
    assert np.trace(cm) == 7  # 7 mẫu đúng
    print("[✓] Confusion matrix computed successfully.")

    # 5. Kiểm thử tìm top confused pairs
    confused = find_top_confused_pairs(cm, target_names=classes, top_k=2)
    assert len(confused) > 0
    print(f"[✓] Top confused pairs extracted: {confused[0]['true_name']} -> {confused[0]['pred_name']} ({confused[0]['count']} samples)")

    # 6. Kiểm thử trích xuất mẫu lỗi
    probs = np.zeros((10, 4))
    for i in range(10):
        probs[i, y_pred[i]] = 0.85
        probs[i, (y_pred[i] + 1) % 4] = 0.15

    errors = identify_misclassified_samples(y_true, y_pred, probabilities=probs, target_names=classes)
    assert len(errors) == 3  # 3 mẫu sai
    assert errors[0]["severity"] == "high_confidence_error"
    print(f"[✓] Misclassified samples identified successfully: {len(errors)} errors found.")

    # 7. Kiểm thử vẽ learning curves
    mock_history = {
        "loss": [0.9, 0.6, 0.4, 0.25],
        "val_loss": [0.95, 0.65, 0.45, 0.35],
        "accuracy": [0.55, 0.72, 0.85, 0.92],
        "val_accuracy": [0.50, 0.68, 0.80, 0.88],
    }
    test_lc_path = "results/test_smoke_learning_curves.png"
    fig_lc, _ = plot_learning_curves(mock_history, save_path=test_lc_path)
    plt.close(fig_lc)
    assert os.path.exists(test_lc_path)
    os.remove(test_lc_path)
    print("[✓] Learning curves plotted and verified.")

    # 8. Kiểm thử vẽ confusion matrix
    test_cm_path = "results/test_smoke_cm.png"
    fig_cm, _ = plot_confusion_matrix(cm, target_names=classes, save_path=test_cm_path)
    plt.close(fig_cm)
    assert os.path.exists(test_cm_path)
    os.remove(test_cm_path)
    print("[✓] Confusion matrix heatmap plotted and verified.")

    # 9. Kiểm thử lưu kết quả JSON & Markdown
    test_json_path = "results/test_smoke_metrics.json"
    test_md_path = "results/test_smoke_metrics.md"
    save_evaluation_results(metrics, test_json_path, format="json")
    save_evaluation_results(metrics, test_md_path, format="markdown")
    assert os.path.exists(test_json_path)
    assert os.path.exists(test_md_path)
    os.remove(test_json_path)
    os.remove(test_md_path)
    print("[✓] Results export (JSON & Markdown) verified.")

    print("=" * 60)
    print("TOÀN BỘ SMOKE TEST HOÀN TẤT THÀNH CÔNG! MODULE SẴN SÀNG SỬ DỤNG.")
    print("=" * 60)
    return True


# =============================================================================
# CHẠY ĐÁNH GIÁ MỘT CHECKPOINT (Keras E01–E02)
# Kết quả theo quy ước tên file:
#     results/metrics/<EXP_ID>_<split>.json, <EXP_ID>_<split>_predictions.csv
#     results/figures/<EXP_ID>_<split>_confusion_matrix.png, <EXP_ID>_learning_curves.png,
#                     <EXP_ID>_<split>_misclassified.png
# Báo cáo hai cấp: tổng thể 22 lớp và nhóm lớp liên quan nhựa.
# =============================================================================
def group_metrics(y_true, y_pred, indices: list[int]) -> dict[str, Any]:
    """Macro P/R/F1 chỉ trên các lớp trong `indices` (mô hình vẫn dự đoán 22 lớp).

    Lớp không có mẫu thật trong tập đánh giá bị loại khỏi trung bình và được liệt kê.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    present = [i for i in indices if (y_true == i).any()]
    if not present:
        return {"classes": indices, "classes_evaluated": [], "macro_precision": None, "macro_recall": None, "macro_f1": None}
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=present, zero_division=0)
    return {
        "classes": indices,
        "classes_evaluated": present,
        "classes_without_samples": [i for i in indices if i not in present],
        "macro_precision": float(p.mean()), "macro_recall": float(r.mean()), "macro_f1": float(f.mean()),
        "support": int(s.sum()),
    }


def summarize(y_true, y_pred, probs, mapping: list[dict[str, Any]], exp_id: str, split: str) -> dict[str, Any]:
    names = class_names(mapping)
    labels = list(range(len(names)))
    overall = compute_classification_metrics(y_true, y_pred, labels=labels, target_names=names)
    present = sorted(set(np.asarray(y_true).tolist()))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=present, zero_division=0)
    return {
        "exp_id": exp_id,
        "split": split,
        "num_samples": int(len(y_true)),
        "num_classes": len(names),
        "overall": overall,
        "macro_f1_present_classes": float(f.mean()),   # chỉ tính lớp có mẫu trong tập này
        "plastic_related": group_metrics(y_true, y_pred, plastic_indices(mapping)),
        "note": "Kết quả trên TrashCan-Instance; lớp có support nhỏ cần diễn giải thận trọng.",
    }


def predict_split(model, processed_dir: str | Path, split: str, mapping: list[dict[str, Any]],
                  input_size: int = 128, batch_size: int = 128) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Dự đoán toàn bộ ảnh của một tập theo thứ tự cố định. Trả (y_true, probs, file_paths)."""
    import tensorflow as tf
    names = class_names(mapping)
    ds = tf.keras.utils.image_dataset_from_directory(
        str(Path(processed_dir) / split), labels="inferred", label_mode="int", class_names=names, color_mode="rgb",
        image_size=(input_size, input_size), batch_size=batch_size, shuffle=False)
    paths = list(ds.file_paths)
    y_true = np.concatenate([y.numpy() for _, y in ds]) if paths else np.array([], dtype=int)
    probs = model.predict(ds, verbose=0) if paths else np.zeros((0, len(names)))
    return y_true, probs, paths


def evaluate_run(run_dir: str | Path, processed_dir: str | Path, split: str = "val",
                 results_dir: str | Path = "results", checkpoint: str = "best.keras") -> dict[str, Any]:
    """Đánh giá checkpoint trong run_dir trên `split`; ghi file kết quả; trả dict chỉ số."""
    import tensorflow as tf
    if split == "test":
        print("[CHÚ Ý] Đang đánh giá tập TEST — chỉ chạy MỘT lần cho mô hình cuối (ADR mô hình cuối).")
    run_dir, results_dir = Path(run_dir), Path(results_dir)
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    exp_id = cfg["exp_id"]
    mapping = load_class_mapping(run_dir / "class_mapping.json")
    names = class_names(mapping)
    model = tf.keras.models.load_model(str(run_dir / checkpoint))

    y_true, probs, paths = predict_split(model, processed_dir, split, mapping, cfg["input_shape"][0])
    y_pred = probs.argmax(axis=1)
    result = summarize(y_true, y_pred, probs, mapping, exp_id, split)
    result.update({"run_dir": str(run_dir), "checkpoint": checkpoint, "data_version": cfg.get("data_version")})

    (results_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (results_dir / "figures").mkdir(parents=True, exist_ok=True)
    with open(results_dir / "metrics" / f"{exp_id}_{split}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=float)
    pd.DataFrame({"file_path": [Path(p).name for p in paths], "y_true": y_true, "y_pred": y_pred,
                  "true_name": [names[i] for i in y_true], "pred_name": [names[i] for i in y_pred],
                  "confidence": probs.max(axis=1) if len(probs) else []}
                 ).to_csv(results_dir / "metrics" / f"{exp_id}_{split}_predictions.csv", index=False)

    cm = compute_confusion_matrix(y_true, y_pred, labels=list(range(len(names))), normalize="true")
    plot_confusion_matrix(np.nan_to_num(cm), target_names=names, normalize=True, figsize=(14, 12),
                             title=f"{exp_id} — {split} (normalized)",
                             save_path=str(results_dir / "figures" / f"{exp_id}_{split}_confusion_matrix.png"))
    hist_path = run_dir / "history.csv"
    if hist_path.exists():
        plot_learning_curves(pd.read_csv(hist_path).to_dict(orient="list"), title_prefix=exp_id,
                                save_path=str(results_dir / "figures" / f"{exp_id}_learning_curves.png"))
    wrong = identify_misclassified_samples(y_true, y_pred, probabilities=probs, target_names=names)
    if wrong:
        from PIL import Image
        wrong = sorted(wrong, key=lambda w: -w.get("confidence", 0))[:16]
        imgs = [np.asarray(Image.open(paths[w["sample_index"]]).convert("RGB")) for w in wrong]
        plot_misclassified_samples(imgs, [{**w, "sample_index": i} for i, w in enumerate(wrong)], target_names=names,
                                      max_samples=16, figsize=(12, 12),
                                      save_path=str(results_dir / "figures" / f"{exp_id}_{split}_misclassified.png"))
    return result


def comparison_table(metric_files: list[str | Path]) -> pd.DataFrame:
    """Bảng so sánh các thí nghiệm từ file results/metrics/<EXP_ID>_<split>.json."""
    rows = []
    for p in metric_files:
        r = json.loads(Path(p).read_text(encoding="utf-8"))
        o, g = r["overall"], r["plastic_related"]
        rows.append({"exp_id": r["exp_id"], "split": r["split"], "n": r["num_samples"],
                     "accuracy": o["accuracy"], "macro_f1": o["macro_f1"], "weighted_f1": o.get("weighted_f1"),
                     "macro_f1_present": r.get("macro_f1_present_classes", o["macro_f1"]), "plastic_macro_f1": g["macro_f1"],
                     "data_version": r.get("data_version")})
    return pd.DataFrame(rows).round(4)


# =============================================================================
# CHẠY ĐÁNH GIÁ MỘT CHECKPOINT (PyTorch MobileNetV2, E03–E04)
# =============================================================================
def evaluate_torch_checkpoint(run_dir: str | Path, checkpoint: str, processed_dir: str | Path, split: str = "val",
                              out_dir: str | Path | None = None, device: str | None = None) -> dict[str, Any]:
    """Đánh giá một checkpoint PyTorch (state_dict do src/train.py lưu) trên `split`.

    Đọc <run_dir>/config.json; ghi cạnh checkpoint (hoặc vào out_dir):
        <EXP_ID>_<split>.json                  Macro F1, Macro F1 nhóm nhựa, chỉ số từng lớp, kích thước, thời gian
        <EXP_ID>_<split>_confusion_counts.csv  ma trận nhầm lẫn (số mẫu)
        <EXP_ID>_<split>_predictions.csv       đường dẫn ảnh, nhãn thật, nhãn dự đoán
    split="test" chỉ dùng MỘT lần cho mô hình cuối.
    """
    import torch
    from torch.utils.data import DataLoader
    try:
        from src.train import (build_model, build_transform, compute_metrics, find_class_mapping, measure_efficiency,
                               ordered_image_folder, predict)
    except ImportError:  # pragma: no cover
        from train import (build_model, build_transform, compute_metrics, find_class_mapping, measure_efficiency,
                           ordered_image_folder, predict)

    if split not in ("train", "val", "test"):
        raise ValueError(f"split phải là train/val/test, nhận: {split}")
    if split == "test":
        print("[CHÚ Ý] Đang đánh giá tập TEST — chỉ chạy MỘT lần cho mô hình cuối.")
    run = Path(run_dir)
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    mapping = load_class_mapping(find_class_mapping(processed_dir, cfg["data"].get("class_mapping_path", "class_mapping.json")))
    names, plastic = class_names(mapping), plastic_indices(mapping)
    if names != list(cfg["data"].get("class_names", names)):
        raise ValueError("Thứ tự lớp khác với lúc huấn luyện")

    ds = ordered_image_folder(Path(processed_dir) / split, names, build_transform(cfg))
    loader = DataLoader(ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=False, num_workers=0)
    model = build_model(cfg, pretrained=False).to(device)
    model.load_state_dict(torch.load(run / checkpoint, map_location=device))
    loss, y, p = predict(model, loader, device)
    metrics = compute_metrics(y, p, names, plastic)
    metrics.update({"exp_id": cfg["exp_id"], "checkpoint": checkpoint, "split": split, "num_samples": int(len(y)),
                    "num_classes": len(names), "loss_unweighted": loss, "data_version": cfg["data"].get("data_version"),
                    "run_dir": str(run), "efficiency": measure_efficiency(model, cfg, device)})

    out = Path(out_dir) if out_dir else run
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{cfg['exp_id']}_{split}"
    (out / f"{stem}.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    cm = confusion_matrix(y, p, labels=list(range(len(names))))
    pd.DataFrame(cm, index=names, columns=names).to_csv(out / f"{stem}_confusion_counts.csv")
    pd.DataFrame({"path": [Path(s[0]).relative_to(processed_dir).as_posix() for s in ds.samples],
                  "y_true": y, "y_pred": p}).to_csv(out / f"{stem}_predictions.csv", index=False)
    o = metrics["overall"]
    print(f"{cfg['exp_id']} / {checkpoint} / {split}: macro_f1 {o['macro_f1']:.4f}  "
          f"plastic6 {metrics['plastic_related']['macro_f1']:.4f}  acc {o['accuracy']:.4f}")
    return metrics


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Đánh giá checkpoint PyTorch (E03–E04); không tham số = smoke test")
    ap.add_argument("--run", help="thư mục run, ví dụ runs/E04_20261121-1000")
    ap.add_argument("--ckpt", default="best.pt", help="checkpoint trong thư mục run")
    ap.add_argument("--processed-dir")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    if not a.run:
        smoke_test()
        return
    if not a.processed_dir:
        ap.error("--processed-dir là bắt buộc khi có --run")
    evaluate_torch_checkpoint(a.run, a.ckpt, a.processed_dir, a.split, a.out_dir, a.device)


if __name__ == "__main__":
    main()
