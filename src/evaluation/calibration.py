import json
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize
from pathlib import Path

def inverse_sigmoid(p, eps=1e-7):
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))

def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))

def compute_log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-7) -> float:
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper) if i > 0 else (y_prob >= bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean((y_prob[in_bin] >= 0.5) == y_true[in_bin])
            avg_confidence_in_bin = np.mean(np.maximum(y_prob[in_bin], 1.0 - y_prob[in_bin]))
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return float(ece)

class TemperatureScaler:
    """
    Learns scalar Temperature T > 0 on validation set to calibrate probabilities:
    P_calibrated = Sigmoid(Logit / T)
    """
    def __init__(self, temperature: float = 1.0):
        self.temperature = float(temperature)

    def fit(self, val_logits: np.ndarray, val_labels: np.ndarray):
        val_logits = np.array(val_logits, dtype=np.float64)
        val_labels = np.array(val_labels, dtype=np.float64)

        def nll_func(t):
            t_val = max(t[0], 1e-3)
            scaled_logits = val_logits / t_val
            # Log-sum-exp numerically stable sigmoid loss
            loss = np.mean(np.log1p(np.exp(-scaled_logits * (2 * val_labels - 1))))
            return loss

        res = minimize(nll_func, x0=[1.0], method='Nelder-Mead', bounds=[(0.01, 10.0)])
        self.temperature = float(max(res.x[0], 0.01))
        print(f"[CALIBRATION] Optimized Temperature T = {self.temperature:.4f}")
        return self.temperature

    def calibrate_logits(self, logits: np.ndarray) -> np.ndarray:
        return np.array(logits, dtype=np.float64) / self.temperature

    def calibrate_probs(self, probs: np.ndarray) -> np.ndarray:
        logits = inverse_sigmoid(np.array(probs, dtype=np.float64))
        cal_logits = self.calibrate_logits(logits)
        return 1.0 / (1.0 + np.exp(-cal_logits))

    def save(self, filepath: str = "checkpoints/calibration_params.json"):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump({"temperature": self.temperature}, f, indent=2)
        print(f"[CALIBRATION] Parameters saved to {filepath}")

    def load(self, filepath: str = "checkpoints/calibration_params.json"):
        fpath = Path(filepath)
        if fpath.exists():
            with open(fpath, "r") as f:
                data = json.load(f)
                self.temperature = float(data.get("temperature", 1.0))
            print(f"[CALIBRATION] Loaded Temperature T = {self.temperature:.4f} from {filepath}")
        else:
            print(f"[CALIBRATION] Checkpoint {filepath} not found. Using default T = 1.0")

def evaluate_calibration(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "brier_score": compute_brier_score(y_true, y_prob),
        "log_loss": compute_log_loss(y_true, y_prob),
        "ece": compute_ece(y_true, y_prob)
    }
