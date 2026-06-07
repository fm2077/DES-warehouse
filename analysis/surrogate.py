'''
Builds a Gaussian Process (GP) surrogate model for the warehouse throughput.
GP is adequate surrogate given the low dimensionality of parameter space and its UQ benefits. NN is another alternative.
Parameters are distilled down based on sensitivity analysis results.
'''

import sys
sys.path.insert(0, ".")
import numpy as np
import matplotlib.pyplot as plt
import pickle                                                                               # to save the surrogate model
import json
from scipy.stats import qmc                                                                 # qmc for quasi-monte carlo methods such as LHS
from sklearn.gaussian_process import GaussianProcessRegressor                               # the GP fit function
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel               # choice of kernel in GP. WhiteKernel for de-noising
from sklearn.model_selection import LeaveOneOut                                             # choice of splitter, LOO-CV for surrogate validation
from sklearn.base import clone                                                              # create new unfitted estimator / surrogate
from tqdm import tqdm

from sim.task import Task
from main import run_sim


with open("config.json") as f:
    REF_CONFIG = json.load(f)


#─────────PARAMETER SPACE──────────

PARAMS = {
    "names": ["task_arrival_rate", "num_robots",
              "grid_dim_1", "grid_dim_2",
              "robot_speed", "deadlock_timeout", "num_stations"],                           # influential parameters from sensitivity analysis
    "bounds": [[0.5,  5.0],                                                                 # task_arrival_rate
               [3,    25],                                                                  # num_robots
               [10,   30],                                                                  # grid_dim_1
               [10,   30],                                                                  # grid_dim_2
               [0.5,  2.0],                                                                 # robot_speed
               [1.0,  10.0],                                                                # deadlock_timeout
               [5,    20]],                                                                 # num_stations , all these params are design choices and constrained by physics or cost
    "int": [False, True, True, True, False, False, True]                                    # boolean flags to round to whole numbers after sampling for integer parameters
}

#─────────SAMPLING──────────

def sample_LHS(n: int) -> np.ndarray:
# generates n samples using Latin Hypercube Sampling strategy over PARAMS space.
# returns an (n, d) array where d is parameter space dimensionality, scaled to physical range.
# n does not have to be power of 2.

    d = len(PARAMS["names"])
    sampler = qmc.LatinHypercube(d=d, seed=REF_CONFIG["seed"])                              # LHS covers each parameter interval exactly once. If params are nonlinear or interactive, has to be later augmented with adaptive sampling
    unit_samples = sampler.random(n=n)                                                      # sampled from uniform dist.

    lower_bounds = [b[0] for b in PARAMS["bounds"]]
    upper_bounds = [b[1] for b in PARAMS["bounds"]]
    samples = qmc.scale(unit_samples, lower_bounds, upper_bounds)                           # scale back to physical range

    for i, is_int in enumerate(PARAMS["int"]):
        if is_int:
            samples[:, i] = np.round(samples[:, i])                                         # round integer parameters
    return samples


#─────────SIMULATION DATA COLLECTION──────────

def train_sim(X: np.ndarray) -> np.ndarray:
# collects output y for surrogate regression, using the simulation engine and sampled params X

    y = np.zeros(len(X))
    for i, sample in enumerate(tqdm(X, desc="Collecting training data", unit="sim")):
        Task._task_id = 0
        config = REF_CONFIG.copy()
        for j, name in enumerate(PARAMS["names"]):
            config[name] = int(sample[j]) if PARAMS["int"][j] else float(sample[j])       # build a simulation config for each sample
        try:
            metrics, _ = run_sim(config)
            y[i] = metrics.throughput()
        except Exception as e:
            print(f"run {i} failed: {e}, setting y to 0.0")
            y[i] = 0.0
    return y


#─────────SURROGATE FIT──────────

def fit_GP(X: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
    '''
    fits a GP surrogate to normalized samples and throughput outputs and returns the surrogate object.
    the kernel is standard squared exponential (RBF) scaled by ConstantKernel.
    GP output is standardized to zero mean, unit variance for kernel inversion stability.
    to avoid local optima, the kernel reinitialized and re-optimized multiple times.
    '''    

    X_norm = normalize(X)
    # define the constant scaling and correlation length scale bounds for optimizer to search in. 
    # whitekernel models the noise / wiggle as sample points grow closer to each other. adds term sigma_n^2*I to the kernel matrix.
    # alternatively can be ensemble averaged to kill the noise.
    kernel = ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) \
            * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 10.0)) \
            + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1.0))                                   
    GP = GaussianProcessRegressor(
        kernel = kernel,
        normalize_y = True,
        n_restarts_optimizer = 10,                                                          # number of reinitializations to find MLE params
        alpha = 1e-6                                                                        # small number added to diag terms of kernel matrix for stability and conditioning
    )
    GP.fit(X_norm, y)
    print(f" fitted kernel: {GP.kernel_}")
    return GP


#─────────SURROGATE VALIDATION──────────

def validate_GP(GP: GaussianProcessRegressor, X: np.ndarray, y: np.ndarray) -> dict:
    # uses Q^2 score (LOO-CV) for fit quality evaluation, along with NRMSE and uncertainty calibration.
    # complexity of LOO is O(n^3) for each sample, total samples O(n^4). for large n use closed-from solution to save cost. (see fast_validate_GP)

    X_norm = normalize(X)
    n = len(y)                                                                              # num samples
    LOO = LeaveOneOut()
    y_pred_LOO = np.zeros(n)                                                                # needed for LOO, NRMSE, UC
    sigma_LOO = np.zeros(n)                                                                 # needed for UC

    for train_idx, test_idx in tqdm(LOO.split(X_norm),
                                    total = n,
                                    desc = "LOO-CV validation",
                                    unit = "fold"):                                         # with LOO, fold=1
        GP_LOO = clone(GP)
        GP_LOO.fit(X_norm[train_idx], y[train_idx])
        y_pred_LOO[test_idx], sigma_LOO[test_idx] = GP_LOO.predict(X_norm[test_idx], return_std=True)

    ss_res = np.sum((y - y_pred_LOO)**2)                                                    # sum of squared residuals
    ss_tot = np.sum((y - np.mean(y))**2)                                                    # total sum of squares
    Q2 = 1 - ss_res/ss_tot                                                                  # Q^2 score with LOO-CV since R^2 overfits and it is not reliable metric

    RMSE = np.sqrt(np.mean((y-y_pred_LOO)**2))
    NRMSE = RMSE / (np.max(y) - np.min(y))                                                  # normalized root mean square error, globally interpretable, should be less than 10%

    errors = np.abs(y-y_pred_LOO)
    cal_68 = np.mean(errors <= sigma_LOO)                                                   # check if 68% of errors fall within 1 x sigma
    cal_95 = np.mean(errors <= 2*sigma_LOO)                                                 # check if 95% of errors fall within 2 x sigma

    results = {
        "Q2": Q2,
        "RMSE": RMSE,
        "NRMSE": NRMSE,
        "uncertainty_calibration_68": cal_68,
        "uncertainty_calibration_95": cal_95,
        "y_pred_LOO": y_pred_LOO,                                                           # pass for visualization
        "sigma_LOO": sigma_LOO
    }
    print(f"SURROGATE VALIDATION")
    print(f"{'Q²':<25} {Q2:.4f}   (target > 0.90)")
    print(f"{'RMSE':<25} {RMSE:.4f}")
    print(f"{'NRMSE':<25} {NRMSE:.4f}  (target < 0.10)")
    print(f"{'68% calibration':<25} {cal_68:.4f}  (target ≈ 0.68)")
    print(f"{'95% calibration':<25} {cal_95:.4f}  (target ≈ 0.95)")

    return results


def fast_validate_GP(GP: GaussianProcessRegressor, X: np.ndarray, y: np.ndarray) -> dict:
    # Analytical LOO-CV using GP kernel matrix inverse. Complexity O(n^3) instead of O(n^4)

    X_norm = normalize(X)
    GP_full = clone(GP)
    GP_full.fit(X_norm, y)
    K = GP_full.kernel_(X_norm)                                                             # get the full GP kernel matrix and inverse only once
    K[np.diag_indices_from(K)] += GP_full.alpha                                             # add nugget to diagonal terms manually for inversion stability
    K_inv = np.linalg.inv(K)                                                                # calculate K^(-1) matrix
    y_mean = y.mean(); y_std = y.std(); y_norm = (y-y_mean)/y_std                           # GP works with standardized output, standardize manually
    K_inv_y = K_inv @ y_norm                                                                # calculate K^(-1)y vector
    K_inv_diag = np.diag(K_inv)                                                             # calculate K^(-1) diagonal terms
    y_pred_norm = y_norm - K_inv_y/K_inv_diag                                               # normalized version of y_pred_LOO, analytical formula
    sigma_norm = np.sqrt(1.0/K_inv_diag)                                                    # normalized version of sigma_LOO, analytical formula
    y_pred_LOO = y_pred_norm*y_std + y_mean                                                 # physical scale y_pred_LOO
    sigma_LOO = sigma_norm*y_std                                                            # physical scale y_pred_LOO

    ss_res = np.sum((y - y_pred_LOO)**2)                                                    # sum of squared residuals
    ss_tot = np.sum((y - np.mean(y))**2)                                                    # total sum of squares
    Q2 = 1 - ss_res/ss_tot                                                                  # Q^2 score with LOO-CV since R^2 overfits and it is not reliable metric

    RMSE = np.sqrt(np.mean((y-y_pred_LOO)**2))
    NRMSE = RMSE / (np.max(y) - np.min(y))                                                  # normalized root mean square error, globally interpretable, should be less than 10%

    errors = np.abs(y-y_pred_LOO)
    cal_68 = np.mean(errors <= sigma_LOO)                                                   # check if 68% of errors fall within 1 x sigma
    cal_95 = np.mean(errors <= 2*sigma_LOO)                                                 # check if 95% of errors fall within 2 x sigma

    results = {
        "Q2": Q2,
        "RMSE": RMSE,
        "NRMSE": NRMSE,
        "uncertainty_calibration_68": cal_68,
        "uncertainty_calibration_95": cal_95,
        "y_pred_LOO": y_pred_LOO,                                                           # pass for visualization
        "sigma_LOO": sigma_LOO
    }
    print(f"SURROGATE VALIDATION")
    print(f"{'Q²':<25} {Q2:.4f}   (target > 0.90)")
    print(f"{'RMSE':<25} {RMSE:.4f}")
    print(f"{'NRMSE':<25} {NRMSE:.4f}  (target < 0.10)")
    print(f"{'68% calibration':<25} {cal_68:.4f}  (target ≈ 0.68)")
    print(f"{'95% calibration':<25} {cal_95:.4f}  (target ≈ 0.95)")

    return results


#─────────3D SURFACE PLOT──────────

def plot_surface(gp: GaussianProcessRegressor, X: np.ndarray, y: np.ndarray, val: dict, resolution: int = 50,
                 x_param: str = "task_arrival_rate", y_param: str = "num_robots", color_param: str = "robot_speed",
                 color_param_vals: list = None) -> None:
    # x num_robots or user specifies,
    # y arrival_rate or user specifies,
    # z GP mean throughput prediction,
    # a third variable tht user specifies for 3 contour levels displayed as overlaid surfaces, 
    # scatter sample points.

    xi = PARAMS["names"].index(x_param)                                                     # get three primary visualized parameters
    yi = PARAMS["names"].index(y_param)
    ci = PARAMS["names"].index(color_param)
    nominal = {name: round((b[0]+b[1])/2) if is_int else (b[0]+b[1])/2
               for name, b, is_int in zip(PARAMS["names"], PARAMS["bounds"], PARAMS["int"])} # for non-visualized parameters, fix a nominal value at mean of the range
    if color_param_vals is None:
        blo, bhi = PARAMS["bounds"][ci]
        color_param_vals = [blo, (blo+bhi)/2, bhi]                                          # set 3 levels (min,med,max) for colored variable

    x_grid = np.linspace(PARAMS["bounds"][xi][0], PARAMS["bounds"][xi][1], resolution)
    y_grid = np.linspace(PARAMS["bounds"][yi][0], PARAMS["bounds"][yi][1], resolution)
    XG, YG = np.meshgrid(x_grid, y_grid)

    fig = plt.figure(figsize=(16, 6))

    # ──────LEFT: THROUGHPUT SURFACE───────

    ax1 = fig.add_subplot(121, projection="3d")                                             # 1 row, 2 cols, subplot 1
    surface_colors = ["steelblue", "coral", "green"]                                        # 3 surfaces have 3 distinct colors
    for color_val, c in zip(color_param_vals, surface_colors):                              # loop over the 3 surfaces
        grid_flat = np.tile(                                                                # build full d-dim grid, all params at nominal
            [nominal[n] for n in PARAMS["names"]],
            (resolution*resolution, 1)
        )
        grid_flat[:, xi] = XG.ravel()                                                       # override x parameter
        grid_flat[:, yi] = YG.ravel()                                                       # override y parameter
        grid_flat[:, ci] = color_val  
        mu, sigma = gp.predict(normalize(grid_flat), return_std=True)                       # mean and var over full d-dim grid
        MU = mu.reshape(resolution, resolution)                                             # map for contour
        SIGMA = sigma.reshape(resolution, resolution)
        sigma_norm = (SIGMA-SIGMA.min())/(SIGMA.max()-SIGMA.min()+1e-10)                    # provide normalized uncertainty for color intensity 
        ax1.plot_surface(XG, YG, MU,                                                        # throughput is mean GP output az axis z
                         facecolors=plt.cm.plasma(sigma_norm),                              # uncertainty changes color intensity
                         alpha=0.5, linewidth=0,                                            # some transparency for overlaying surfaces
                         label=f"{color_param}={color_val:.1f}")


    ax1.scatter(X[:, xi], X[:, yi], y,                                                     # train points / samples overlaid as scattered points
                color="white", edgecolors="black", s=30, zorder=5)
    ax1.set_xlabel(x_param); ax1.set_ylabel(y_param)
    ax1.set_zlabel("Throughput (tasks/min)")
    ax1.set_title(f"GP Surrogate — Mean Throughput\n({color_param} low/mid/high)")
    ax1.legend(loc="upper left", fontsize=8)

    # ──────RIGHT: LOO PREDICTION VS ACTUAL───────
   
    y_pred_LOO = val["y_pred_LOO"]
    ax2 = fig.add_subplot(122)                                                              # 1 row, 2 cols, subplot 2 
    ax2.scatter(y, y_pred_LOO, color="steelblue", edgecolors="black", s=40, alpha=0.8)
    lims = [min(y.min(), y_pred_LOO.min()) * 0.95,
            max(y.max(), y_pred_LOO.max()) * 1.05]
    ax2.plot(lims, lims, "r--", linewidth=1.5, label="perfect prediction")
    ax2.set_xlabel("Actual Throughput (simulation)"); ax2.set_ylabel("LOO Predicted Throughput (GP)")
    ax2.set_title("LOO Cross Validation\nPredicted vs Actual")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("outputs/surrogate_surface.png", dpi=300)
    plt.show()


#─────────SAVE / LOAD MODEL──────────

def save_model(gp: GaussianProcessRegressor, X: np.ndarray, y: np.ndarray) -> None:
    # pickles the surrogate model for reuse.

    model_data = {"gp": gp, "X": X, "y":y, "params": PARAMS}
    with open("outputs/surrogate_model.pkl", "wb") as f:
        pickle.dump(model_data, f)
    print("Saved surrogate model at: outputs/surrogate_model.pkl")


def load_model() -> dict:
    # loads a pickled surrogate model.
    with open("outputs/surrogate_model.pkl", "rb") as f:
        model_data = pickle.load(f)
    print("Loaded surrogate model from: outputs/surrogate_model.pkl")
    return model_data


#─────────HELPERS──────────

def normalize(X: np.ndarray) -> np.ndarray:
    l = np.array([b[0] for b in PARAMS["bounds"]])
    u = np.array([b[1] for b in PARAMS["bounds"]])
    return (X - l)/(u - l)                                                                  # normalize to [0, 1] range for GP fitting


def denormalize(X_norm: np.ndarray) -> np.ndarray:
    l = np.array([b[0] for b in PARAMS["bounds"]])
    u = np.array([b[1] for b in PARAMS["bounds"]])
    return X_norm*(u - l) + l                                                               # map back to physical range


#─────────MAIN──────────

if __name__ == "__main__":
    X = sample_LHS(n = 145)                                                                 # rule of thumb is n = (10-20)d
    y = train_sim(X)
    GP = fit_GP(X, y)
    #val = validate_GP(GP, X, y)                                                            # do not use for large n, inefficient
    val = fast_validate_GP(GP, X, y)
    plot_surface(GP, X, y, val)
    save_model(GP, X, y)
