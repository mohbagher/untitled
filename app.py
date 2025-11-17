"""
Interactive RIS learning dashboard built with Plotly Dash.
The app exposes multiple channel/path loss models and intuitive explanations
for parameter changes.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
from dash import Dash, Input, Output, State, dcc, html

# --------------------------------------------------------------------------------------
# Physical constants and helper conversions
# --------------------------------------------------------------------------------------
C = 3e8  # Speed of light in m/s


def db_to_linear(value_db: float) -> float:
    """Convert dB quantity (power) to linear scale."""
    return 10 ** (value_db / 10)


def linear_to_db(value_linear: float) -> float:
    """Convert linear quantity (power) to dB scale."""
    return 10 * math.log10(value_linear)


# --------------------------------------------------------------------------------------
# Path loss and link budget models
# --------------------------------------------------------------------------------------

def wavelength(fc_ghz: float) -> float:
    """Return wavelength in meters for a given carrier frequency in GHz."""
    fc_hz = fc_ghz * 1e9
    return C / fc_hz


def free_space_path_loss(distance_m: np.ndarray, wavelength_m: float) -> np.ndarray:
    """Friis free-space path loss in dB for distance array.

    PL_FS(d) = 20 log10(4 pi d / lambda)
    """
    distance_m = np.maximum(distance_m, 1e-9)  # avoid log of zero
    return 20 * np.log10(4 * math.pi * distance_m / wavelength_m)


def friis_received_power(
    pt_dbm: float, gt_dbi: float, gr_dbi: float, distance_m: np.ndarray, wavelength_m: float
) -> np.ndarray:
    """Received power in dBm using Friis free-space model."""
    pl_db = free_space_path_loss(distance_m, wavelength_m)
    return pt_dbm + gt_dbi + gr_dbi - pl_db


def tang_ris_path_loss(
    d1_m: float,
    d2_m: np.ndarray,
    wavelength_m: float,
    gt_dbi: float,
    gr_dbi: float,
    m_elements: int,
    n_elements: int,
    dx_lambda: float,
    dy_lambda: float,
    ft: float,
    fr: float,
    reflection_mag: float,
) -> np.ndarray:
    """Tang-style refined far-field RIS path loss (simplified).

    PL_RIS(d1, d2) = 16 pi^2 (d1 d2)^2 / (Gt Gr (M N dx dy)^2 Ft Fr A^2)
    - Gt, Gr are linear gains inside the formula (converted from dBi here).
    - dx, dy are expressed as multiples of wavelength.
    - Returned value is in dB.
    """
    # Convert parameters to linear domain
    gt_linear = db_to_linear(gt_dbi)
    gr_linear = db_to_linear(gr_dbi)

    effective_area = m_elements * n_elements * dx_lambda * dy_lambda * (wavelength_m**2)
    reflection_mag = max(reflection_mag, 1e-9)
    ft = max(ft, 1e-9)
    fr = max(fr, 1e-9)

    numerator = (16 * math.pi**2) * (d1_m**2) * (d2_m**2)
    denominator = gt_linear * gr_linear * (effective_area**2) * ft * fr * (reflection_mag**2)

    pl_linear = numerator / denominator
    return linear_to_db(pl_linear)


def tang_ris_received_power(
    pt_dbm: float,
    d1_m: float,
    d2_m: np.ndarray,
    wavelength_m: float,
    gt_dbi: float,
    gr_dbi: float,
    m_elements: int,
    n_elements: int,
    dx_lambda: float,
    dy_lambda: float,
    ft: float,
    fr: float,
    reflection_mag: float,
) -> np.ndarray:
    """Received power via RIS using Tang refined model in dBm."""
    pl_db = tang_ris_path_loss(
        d1_m,
        d2_m,
        wavelength_m,
        gt_dbi,
        gr_dbi,
        m_elements,
        n_elements,
        dx_lambda,
        dy_lambda,
        ft,
        fr,
        reflection_mag,
    )
    return pt_dbm - pl_db


def floating_intercept_received_power(
    pt_dbm: float,
    gt_dbi: float,
    gr_dbi: float,
    alpha: float,
    beta: float,
    distance_m: np.ndarray,
) -> np.ndarray:
    """Received power (dBm) using Floating Intercept path loss model.

    PL_FI(d) = alpha + 10 beta log10(d)
    """
    distance_m = np.maximum(distance_m, 1e-9)
    pl_db = alpha + 10 * beta * np.log10(distance_m)
    return pt_dbm + gt_dbi + gr_dbi - pl_db


def close_in_received_power(
    pt_dbm: float,
    gt_dbi: float,
    gr_dbi: float,
    pl_d0_db: float,
    path_loss_exponent: float,
    distance_m: np.ndarray,
    d0_m: float = 1.0,
) -> np.ndarray:
    """Received power (dBm) using Close-In reference model.

    PL_CI(d) = PL(d0) + 10 n log10(d / d0)
    """
    distance_m = np.maximum(distance_m, 1e-9)
    pl_db = pl_d0_db + 10 * path_loss_exponent * np.log10(distance_m / d0_m)
    return pt_dbm + gt_dbi + gr_dbi - pl_db


def snr_from_power(pr_dbm: np.ndarray, noise_dbm: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return SNR in linear and dB domains given received power and noise power (dBm)."""
    pr_linear_mw = db_to_linear(pr_dbm)
    noise_linear_mw = db_to_linear(noise_dbm)
    gamma_linear = pr_linear_mw / noise_linear_mw
    gamma_db = linear_to_db(np.maximum(gamma_linear, 1e-12))
    return gamma_linear, gamma_db


def achievable_rate(gamma_linear: np.ndarray) -> np.ndarray:
    """Spectral efficiency (bps/Hz) using Shannon formula."""
    return np.log2(1 + gamma_linear)


# --------------------------------------------------------------------------------------
# Dataclass to track user parameters for explanations
# --------------------------------------------------------------------------------------
@dataclass
class ParameterSnapshot:
    params: Dict[str, float]

    @staticmethod
    def from_inputs(inputs: Dict[str, float]) -> "ParameterSnapshot":
        return ParameterSnapshot(params=inputs)

    def diff(self, other: "ParameterSnapshot") -> List[str]:
        changes = []
        if other is None:
            return ["Initial configuration loaded."]
        for key, new_val in self.params.items():
            old_val = other.params.get(key)
            if old_val is None:
                continue
            if isinstance(new_val, float):
                # consider change significant if > small threshold
                changed = abs(new_val - old_val) > 1e-6
            else:
                changed = new_val != old_val
            if changed:
                changes.append(f"{key} changed from {old_val} to {new_val}")
        return changes


# --------------------------------------------------------------------------------------
# Dash application setup
# --------------------------------------------------------------------------------------
external_stylesheets = [dbc.themes.BOOTSTRAP]
app: Dash = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "RIS Learning Lab"


def build_controls() -> html.Div:
    """Construct the control panel layout."""
    return dbc.Card(
        [
            dbc.CardHeader(html.H4("Controls", className="mb-0")),
            dbc.CardBody(
                [
                    html.H5("General settings"),
                    dbc.Row(
                        [
                            dbc.Col(dbc.Input(id="fc", type="number", value=3.5, min=0.1, step=0.1, addon_before="fc (GHz)"), width=12),
                            dbc.Col(dbc.Input(id="pt", type="number", value=30, step=1, addon_before="Pt (dBm)"), width=12),
                            dbc.Col(dbc.Input(id="gt", type="number", value=10, step=0.5, addon_before="Gt (dBi)"), width=6),
                            dbc.Col(dbc.Input(id="gr", type="number", value=10, step=0.5, addon_before="Gr (dBi)"), width=6),
                            dbc.Col(dbc.Input(id="noise", type="number", value=-90, step=1, addon_before="Noise (dBm)"), width=6),
                            dbc.Col(dbc.Input(id="bandwidth", type="number", value=20, step=1, addon_before="BW (MHz)"), width=6),
                        ],
                        className="gy-2",
                    ),
                    html.Hr(),
                    html.H5("Geometry"),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("BS→RIS distance d1 (m)"),
                                    dcc.Slider(id="d1", min=1, max=200, step=1, value=30,
                                               marks={0: "0", 50: "50", 100: "100", 150: "150", 200: "200"}),
                                ],
                                width=12,
                            ),
                            dbc.Col(dbc.Input(id="d2_min", type="number", value=5, step=1, addon_before="d2 min (m)"), width=6),
                            dbc.Col(dbc.Input(id="d2_max", type="number", value=150, step=1, addon_before="d2 max (m)"), width=6),
                            dbc.Col(dbc.Input(id="d2_points", type="number", value=80, step=1, min=10, addon_before="# points"), width=6),
                            dbc.Col(
                                dbc.RadioItems(
                                    id="direct_mode",
                                    options=[
                                        {"label": "Direct distance = d1 + d2", "value": "sum"},
                                        {"label": "Independent direct distance", "value": "independent"},
                                    ],
                                    value="sum",
                                ),
                                width=12,
                            ),
                            dbc.Col(dbc.Input(id="direct_dist", type="number", value=50, step=1, addon_before="Direct d (m)"), width=6),
                        ],
                        className="gy-2",
                    ),
                    html.Hr(),
                    html.H5("RIS parameters"),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("M elements (x-axis)"),
                                    dcc.Slider(id="m_elements", min=1, max=128, step=1, value=32,
                                               marks={1: "1", 32: "32", 64: "64", 128: "128"}),
                                ],
                                width=12,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("N elements (y-axis)"),
                                    dcc.Slider(id="n_elements", min=1, max=128, step=1, value=32,
                                               marks={1: "1", 32: "32", 64: "64", 128: "128"}),
                                ],
                                width=12,
                            ),
                            dbc.Col(dbc.Input(id="dx", type="number", value=0.5, step=0.1, addon_before="dx (λ)"), width=6),
                            dbc.Col(dbc.Input(id="dy", type="number", value=0.5, step=0.1, addon_before="dy (λ)"), width=6),
                            dbc.Col(dbc.Input(id="reflection", type="number", value=1.0, step=0.05, min=0, max=1, addon_before="A"), width=6),
                            dbc.Col(dbc.Input(id="ft", type="number", value=1.0, step=0.1, addon_before="Ft"), width=6),
                            dbc.Col(dbc.Input(id="fr", type="number", value=1.0, step=0.1, addon_before="Fr"), width=6),
                        ],
                        className="gy-2",
                    ),
                    html.Hr(),
                    html.H5("Model selection"),
                    dbc.Checklist(
                        id="models",
                        options=[
                            {"label": "Direct free space (Friis)", "value": "friis"},
                            {"label": "RIS refined far field (Tang)", "value": "ris"},
                            {"label": "Floating Intercept (FI)", "value": "fi"},
                            {"label": "Close-In (CI)", "value": "ci"},
                        ],
                        value=["friis", "ris"],
                    ),
                    html.Hr(),
                    html.H5("Plot options"),
                    dbc.Checklist(
                        id="plots",
                        options=[
                            {"label": "Received power vs distance", "value": "pr_vs_d"},
                            {"label": "Path loss vs distance", "value": "pl_vs_d"},
                            {"label": "SNR vs distance", "value": "snr_vs_d"},
                            {"label": "Achievable rate vs distance", "value": "rate_vs_d"},
                            {"label": "Received power vs RIS size", "value": "pr_vs_elements"},
                            {"label": "Rate vs transmit power", "value": "rate_vs_pt"},
                        ],
                        value=["pr_vs_d", "snr_vs_d", "rate_vs_d"],
                    ),
                    html.Hr(),
                    html.H5("FI / CI parameters"),
                    dbc.Row(
                        [
                            dbc.Col(dbc.Input(id="alpha", type="number", value=30, step=0.5, addon_before="α (FI)"), width=6),
                            dbc.Col(dbc.Input(id="beta", type="number", value=3.0, step=0.1, addon_before="β (FI)"), width=6),
                            dbc.Col(dbc.Input(id="pl_d0", type="number", value=40, step=0.5, addon_before="PL(d0) (CI)"), width=6),
                            dbc.Col(dbc.Input(id="n_exp", type="number", value=2.2, step=0.1, addon_before="n (CI)"), width=6),
                            dbc.Col(dbc.Input(id="d0", type="number", value=1.0, step=0.1, addon_before="d0 (m)"), width=6),
                        ],
                        className="gy-2",
                    ),
                ]
            ),
        ],
        className="shadow-sm",
    )


def empty_graph(message: str) -> dcc.Graph:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=300)
    return dcc.Graph(figure=fig)


app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(build_controls(), width=4),
                dbc.Col(
                    [
                        html.H3("Reconfigurable Intelligent Surface Learning Lab"),
                        html.P("Explore how geometry, frequency, and RIS design shape path loss, SNR, and rate."),
                        dcc.Loading(id="plots-area", type="default", children=html.Div(id="graphs")),
                        html.Hr(),
                        html.H4("Explanation and tips"),
                        html.Div(id="explanation", className="lead"),
                    ],
                    width=8,
                ),
            ],
            className="my-3",
        ),
        dcc.Store(id="param-store"),
    ],
    fluid=True,
)


# --------------------------------------------------------------------------------------
# Helper functions to build figures
# --------------------------------------------------------------------------------------

def generate_distance_curves(
    distances: np.ndarray,
    models_selected: List[str],
    pt_dbm: float,
    gt_dbi: float,
    gr_dbi: float,
    lam: float,
    d1_m: float,
    m_elements: int,
    n_elements: int,
    dx_lambda: float,
    dy_lambda: float,
    ft: float,
    fr: float,
    reflection_mag: float,
    alpha: float,
    beta: float,
    pl_d0: float,
    n_exp: float,
    d0_m: float,
    direct_distances: np.ndarray,
):
    """Compute received powers for each model across distances."""
    results = {}

    if "friis" in models_selected:
        results["friis"] = friis_received_power(pt_dbm, gt_dbi, gr_dbi, direct_distances, lam)

    if "ris" in models_selected:
        results["ris"] = tang_ris_received_power(
            pt_dbm,
            d1_m,
            distances,
            lam,
            gt_dbi,
            gr_dbi,
            m_elements,
            n_elements,
            dx_lambda,
            dy_lambda,
            ft,
            fr,
            reflection_mag,
        )

    if "fi" in models_selected:
        results["fi"] = floating_intercept_received_power(
            pt_dbm, gt_dbi, gr_dbi, alpha, beta, direct_distances
        )

    if "ci" in models_selected:
        results["ci"] = close_in_received_power(
            pt_dbm, gt_dbi, gr_dbi, pl_d0, n_exp, direct_distances, d0_m
        )

    return results


def build_pr_figure(distances: np.ndarray, results: Dict[str, np.ndarray]) -> go.Figure:
    fig = go.Figure()
    for name, values in results.items():
        fig.add_trace(go.Scatter(x=distances, y=values, mode="lines", name=model_name(name)))
    fig.update_layout(
        title="Received power vs distance",
        xaxis_title="Distance d2 (m)",
        yaxis_title="Pr (dBm)",
        template="plotly_white",
        height=350,
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    return fig


def build_pl_figure(distances: np.ndarray, results: Dict[str, np.ndarray], pt_dbm: float, gt_dbi: float, gr_dbi: float) -> go.Figure:
    fig = go.Figure()
    for name, pr in results.items():
        pl = pt_dbm + gt_dbi + gr_dbi - pr
        fig.add_trace(go.Scatter(x=distances, y=pl, mode="lines", name=model_name(name)))
    fig.update_layout(
        title="Path loss vs distance",
        xaxis_title="Distance d2 (m)",
        yaxis_title="PL (dB)",
        template="plotly_white",
        height=350,
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    return fig


def build_snr_figure(distances: np.ndarray, results: Dict[str, np.ndarray], noise_dbm: float) -> go.Figure:
    fig = go.Figure()
    for name, pr in results.items():
        gamma_lin, gamma_db = snr_from_power(pr, noise_dbm)
        fig.add_trace(go.Scatter(x=distances, y=gamma_db, mode="lines", name=model_name(name)))
    fig.update_layout(
        title="SNR vs distance",
        xaxis_title="Distance d2 (m)",
        yaxis_title="SNR (dB)",
        template="plotly_white",
        height=350,
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    return fig


def build_rate_figure(distances: np.ndarray, results: Dict[str, np.ndarray], noise_dbm: float, bandwidth_mhz: float) -> go.Figure:
    fig = go.Figure()
    for name, pr in results.items():
        gamma_lin, _ = snr_from_power(pr, noise_dbm)
        spectral_eff = achievable_rate(gamma_lin)
        rate_bps = spectral_eff * bandwidth_mhz * 1e6
        fig.add_trace(
            go.Scatter(
                x=distances,
                y=rate_bps / 1e6,
                mode="lines",
                name=model_name(name),
                hovertemplate="d2=%{x:.1f} m<br>Rate=%{y:.2f} Mbps",
            )
        )
    fig.update_layout(
        title="Achievable rate vs distance",
        xaxis_title="Distance d2 (m)",
        yaxis_title="Throughput (Mbps)",
        template="plotly_white",
        height=350,
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    return fig


def build_ris_size_figure(
    m_elements: int,
    n_elements: int,
    base_params: Dict[str, float],
    lam: float,
    distance_m: float,
    models_selected: List[str],
) -> go.Figure:
    sizes = np.unique(np.clip(np.round(np.logspace(0, np.log10(max(m_elements * n_elements, 1)), num=8)), 1, None)).astype(int)
    sizes = np.clip(sizes, 1, 16384)
    fig = go.Figure()
    for name in models_selected:
        if name != "ris":
            continue
        pr_values = []
        for total_elements in sizes:
            m_side = int(max(1, round(math.sqrt(total_elements))))
            n_side = int(max(1, total_elements // m_side))
            pr = tang_ris_received_power(
                base_params["pt"],
                base_params["d1"],
                np.array([distance_m]),
                lam,
                base_params["gt"],
                base_params["gr"],
                m_side,
                n_side,
                base_params["dx"],
                base_params["dy"],
                base_params["ft"],
                base_params["fr"],
                base_params["reflection"],
            )[0]
            pr_values.append(pr)
        fig.add_trace(
            go.Scatter(
                x=sizes,
                y=pr_values,
                mode="lines+markers",
                name="RIS path (Tang)",
            )
        )
    fig.update_layout(
        title=f"Received power vs RIS elements (d2 midpoint {distance_m:.1f} m)",
        xaxis_title="Total elements (M×N)",
        yaxis_title="Pr (dBm)",
        template="plotly_white",
        height=350,
    )
    fig.update_xaxes(type="log", showgrid=True)
    fig.update_yaxes(showgrid=True)
    return fig


def build_rate_vs_pt_figure(
    pt_dbm_base: float,
    models_selected: List[str],
    distances: np.ndarray,
    direct_distances: np.ndarray,
    lam: float,
    ris_params: Dict[str, float],
    noise_dbm: float,
    bandwidth_mhz: float,
    alpha: float,
    beta: float,
    pl_d0: float,
    n_exp: float,
    d0_m: float,
) -> go.Figure:
    pt_values = np.linspace(pt_dbm_base - 10, pt_dbm_base + 20, 8)
    fig = go.Figure()
    d2_mid = distances[len(distances) // 2]
    direct_mid = direct_distances[len(direct_distances) // 2]

    for name in models_selected:
        pr_list = []
        for pt_val in pt_values:
            if name == "friis":
                pr = friis_received_power(pt_val, ris_params["gt"], ris_params["gr"], np.array([direct_mid]), lam)[0]
            elif name == "ris":
                pr = tang_ris_received_power(
                    pt_val,
                    ris_params["d1"],
                    np.array([d2_mid]),
                    lam,
                    ris_params["gt"],
                    ris_params["gr"],
                    int(ris_params["m"]),
                    int(ris_params["n"]),
                    ris_params["dx"],
                    ris_params["dy"],
                    ris_params["ft"],
                    ris_params["fr"],
                    ris_params["reflection"],
                )[0]
            elif name == "fi":
                pr = floating_intercept_received_power(pt_val, ris_params["gt"], ris_params["gr"], alpha, beta, np.array([direct_mid]))[0]
            elif name == "ci":
                pr = close_in_received_power(pt_val, ris_params["gt"], ris_params["gr"], pl_d0, n_exp, np.array([direct_mid]), d0_m)[0]
            else:
                pr = None
            if pr is not None:
                gamma_lin, _ = snr_from_power(np.array([pr]), noise_dbm)
                se = achievable_rate(gamma_lin)[0]
                pr_list.append(se * bandwidth_mhz * 1e6 / 1e6)  # Mbps
        if pr_list:
            fig.add_trace(go.Scatter(x=pt_values, y=pr_list, mode="lines+markers", name=model_name(name)))

    fig.update_layout(
        title="Achievable rate vs transmit power",
        xaxis_title="Pt (dBm)",
        yaxis_title="Throughput (Mbps)",
        template="plotly_white",
        height=350,
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    return fig


def model_name(key: str) -> str:
    return {
        "friis": "Direct LOS (Friis)",
        "ris": "RIS path (Tang)",
        "fi": "FI model",
        "ci": "CI model",
    }.get(key, key)


# --------------------------------------------------------------------------------------
# Explanation builder
# --------------------------------------------------------------------------------------

def build_explanation(changes: List[str], results: Dict[str, np.ndarray], distances: np.ndarray, pt_dbm: float, noise_dbm: float, m: int, n: int, reflection: float) -> str:
    if not changes:
        changes = ["No parameter changes detected (using current settings)."]

    bullets = [f"- {c}" for c in changes]

    # Observations based on results
    observations = []
    if "ris" in results and "friis" in results:
        ris_gain = np.median(results["ris"] - results["friis"])
        observations.append(
            f"RIS median received power sits about {ris_gain:.1f} dB relative to the direct path over the scanned range."
        )
    if "ris" in results:
        observations.append(
            f"RIS aperture (M×N) = {m*n} elements. Higher element counts roughly square the gain; halving reflection magnitude A reduces power by about 6 dB."
        )
    if distances.size > 0:
        obs_range = (distances[0], distances[-1])
        observations.append(
            f"Plotted distances span {obs_range[0]:.1f}–{obs_range[1]:.1f} m. Watch how SNR drops with distance because free-space loss scales with 20·log10(d)."
        )
    observations.append(
        f"Noise floor is {noise_dbm} dBm; raising transmit power or RIS aperture moves curves upward and increases achievable rate."
    )

    hints = [
        "Try halving A from 1 to 0.5 and note the ~6 dB drop on the RIS curve.",
        "Sweep M and N upward; in far field the power improvement is roughly 20·log10(MN).",
        "Toggle FI/CI models to see how empirical exponents modify decay compared to Friis.",
    ]

    return "\n".join([
        "Parameter changes:",
        *bullets,
        "\nWhat to expect:",
        *[f"- {o}" for o in observations],
        "\nExperiments to try:",
        *[f"- {h}" for h in hints],
    ])


# --------------------------------------------------------------------------------------
# Dash callbacks
# --------------------------------------------------------------------------------------


@app.callback(
    Output("graphs", "children"),
    Output("explanation", "children"),
    Output("param-store", "data"),
    [
        Input("fc", "value"),
        Input("pt", "value"),
        Input("gt", "value"),
        Input("gr", "value"),
        Input("noise", "value"),
        Input("bandwidth", "value"),
        Input("d1", "value"),
        Input("d2_min", "value"),
        Input("d2_max", "value"),
        Input("d2_points", "value"),
        Input("direct_mode", "value"),
        Input("direct_dist", "value"),
        Input("m_elements", "value"),
        Input("n_elements", "value"),
        Input("dx", "value"),
        Input("dy", "value"),
        Input("reflection", "value"),
        Input("ft", "value"),
        Input("fr", "value"),
        Input("models", "value"),
        Input("plots", "value"),
        Input("alpha", "value"),
        Input("beta", "value"),
        Input("pl_d0", "value"),
        Input("n_exp", "value"),
        Input("d0", "value"),
        State("param-store", "data"),
    ],
)
def update_dashboard(
    fc,
    pt,
    gt,
    gr,
    noise,
    bandwidth,
    d1,
    d2_min,
    d2_max,
    d2_points,
    direct_mode,
    direct_dist,
    m_elements,
    n_elements,
    dx,
    dy,
    reflection,
    ft,
    fr,
    models_selected,
    plots_selected,
    alpha,
    beta,
    pl_d0,
    n_exp,
    d0,
    stored_params,
):
    # Prepare parameter snapshot
    current_params = ParameterSnapshot.from_inputs(
        {
            "fc": fc,
            "pt": pt,
            "gt": gt,
            "gr": gr,
            "noise": noise,
            "bandwidth": bandwidth,
            "d1": d1,
            "d2_min": d2_min,
            "d2_max": d2_max,
            "d2_points": d2_points,
            "direct_mode": direct_mode,
            "direct_dist": direct_dist,
            "m": m_elements,
            "n": n_elements,
            "dx": dx,
            "dy": dy,
            "reflection": reflection,
            "ft": ft,
            "fr": fr,
            "alpha": alpha,
            "beta": beta,
            "pl_d0": pl_d0,
            "n_exp": n_exp,
            "d0": d0,
        }
    )
    previous_snapshot = ParameterSnapshot(params=stored_params) if stored_params else None
    changes = current_params.diff(previous_snapshot)

    # Distance vectors
    d2_points = int(max(5, d2_points or 50))
    d2_array = np.linspace(float(d2_min), float(d2_max), d2_points)
    if direct_mode == "sum":
        direct_distance = d2_array + float(d1)
    else:
        direct_distance = np.full_like(d2_array, float(direct_dist))

    lam = wavelength(float(fc))

    # Compute results for each model
    results = generate_distance_curves(
        d2_array,
        models_selected,
        float(pt),
        float(gt),
        float(gr),
        lam,
        float(d1),
        int(m_elements),
        int(n_elements),
        float(dx),
        float(dy),
        float(ft),
        float(fr),
        float(reflection),
        float(alpha),
        float(beta),
        float(pl_d0),
        float(n_exp),
        float(d0),
        direct_distance,
    )

    figures = []
    if not plots_selected:
        figures.append(empty_graph("Select plot options to visualize results."))
    else:
        if "pr_vs_d" in plots_selected:
            figures.append(dcc.Graph(figure=build_pr_figure(d2_array, results)))
        if "pl_vs_d" in plots_selected:
            figures.append(dcc.Graph(figure=build_pl_figure(d2_array, results, float(pt), float(gt), float(gr))))
        if "snr_vs_d" in plots_selected:
            figures.append(dcc.Graph(figure=build_snr_figure(d2_array, results, float(noise))))
        if "rate_vs_d" in plots_selected:
            figures.append(dcc.Graph(figure=build_rate_figure(d2_array, results, float(noise), float(bandwidth))))
        if "pr_vs_elements" in plots_selected:
            figures.append(
                dcc.Graph(
                    figure=build_ris_size_figure(
                        int(m_elements),
                        int(n_elements),
                        {
                            "pt": float(pt),
                            "d1": float(d1),
                            "gt": float(gt),
                            "gr": float(gr),
                            "dx": float(dx),
                            "dy": float(dy),
                            "ft": float(ft),
                            "fr": float(fr),
                            "reflection": float(reflection),
                        },
                        lam,
                        float((d2_min + d2_max) / 2),
                        models_selected,
                    )
                )
            )
        if "rate_vs_pt" in plots_selected:
            figures.append(
                dcc.Graph(
                    figure=build_rate_vs_pt_figure(
                        float(pt),
                        models_selected,
                        d2_array,
                        direct_distance,
                        lam,
                        {
                            "gt": float(gt),
                            "gr": float(gr),
                            "d1": float(d1),
                            "m": float(m_elements),
                            "n": float(n_elements),
                            "dx": float(dx),
                            "dy": float(dy),
                            "ft": float(ft),
                            "fr": float(fr),
                            "reflection": float(reflection),
                        },
                        float(noise),
                        float(bandwidth),
                        float(alpha),
                        float(beta),
                        float(pl_d0),
                        float(n_exp),
                        float(d0),
                    )
                )
            )

    explanation_text = build_explanation(
        changes, results, d2_array, float(pt), float(noise), int(m_elements), int(n_elements), float(reflection)
    )

    return figures, explanation_text, current_params.params


if __name__ == "__main__":
    app.run_server(debug=False)
