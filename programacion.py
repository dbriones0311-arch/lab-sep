from __future__ import annotations

import cmath
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Sequence, Tuple, Type

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# =============================================================================
# 1. UTILIDADES Y VALIDACIÓN
# =============================================================================


class ParameterError(ValueError):
    """Parámetro físico fuera de su rango válido."""


def require_range(name: str, value: float, lo: float, hi: float) -> float:
    """Valida que ``lo <= value <= hi``.

    Args:
        name: Nombre legible del parámetro.
        value: Valor a validar.
        lo: Límite inferior.
        hi: Límite superior.

    Returns:
        El mismo valor si es válido.

    Raises:
        ParameterError: Si el valor está fuera de rango o no es finito.
    """
    if not (math.isfinite(value) and lo <= value <= hi):
        raise ParameterError(f"{name} = {value} fuera de rango [{lo}, {hi}]")
    return value


def magnetizing_current_pu(vf_pu: float) -> float:
    """Corriente de magnetización didáctica de un transformador.

    Modelo exponencial: 1 % a V/f = 1 pu y crecimiento rápido al saturar.

    Args:
        vf_pu: Relación V/f en por unidad.

    Returns:
        Corriente de excitación en pu de la corriente nominal.
    """
    return 0.01 * math.exp(22.0 * (vf_pu - 1.0))


# =============================================================================
# 2. CAPA DE FÍSICA (independiente de la interfaz)
# =============================================================================


@dataclass(frozen=True)
class PowerBalanceModel:
    """Equilibrio estático P-f y Q-V del sistema (Módulo 1).

    Attributes:
        f0: Frecuencia nominal [Hz].
        s_base_mw: Potencia base del sistema [MW].
        droop: Estatismo equivalente R [pu].
        damping: Autorregulación de carga D [pu/pu].
        s_sc_mva: Potencia de cortocircuito en el nodo [MVA].
    """

    f0: float = 60.0
    s_base_mw: float = 1000.0
    droop: float = 0.05
    damping: float = 1.0
    s_sc_mva: float = 3000.0

    def __post_init__(self) -> None:
        require_range("Estatismo R", self.droop, 0.01, 0.12)
        require_range("Amortiguamiento D", self.damping, 0.0, 5.0)
        require_range("Potencia de cortocircuito", self.s_sc_mva, 100.0, 1e5)

    def steady_frequency(self, p_gen_mw: float, p_load_mw: float) -> float:
        """Frecuencia de régimen permanente tras la regulación primaria [Hz]."""
        beta = self.damping + 1.0 / self.droop
        return self.f0 * (1.0 + (p_gen_mw - p_load_mw) / self.s_base_mw / beta)

    def steady_voltage(self, q_gen_mvar: float, q_load_mvar: float) -> float:
        """Tensión nodal aproximada [pu]: ΔV ≈ ΔQ / S_cc."""
        return 1.0 + (q_gen_mvar - q_load_mvar) / self.s_sc_mva


class MotorStartingModel:
    """Arranque de un motor de inducción desde una red con impedancia Thevenin.

    El motor se modela como admitancia constante dependiente del deslizamiento
    s(t) = exp(-3 t / t_acc). La red es una fuente ideal con Z_cc (X/R dado).
    """

    METHODS: ClassVar[Tuple[str, ...]] = (
        "Directo", "Estrella-Triángulo", "Arrancador suave", "VFD")
    TORQUE_FACTOR: ClassVar[Dict[str, float]] = {
        "Directo": 1.0, "Estrella-Triángulo": 1 / 3, "Arrancador suave": 0.25,
        "VFD": 1.5}
    LOCKED_TORQUE_PU: ClassVar[float] = 1.8

    def __init__(self, p_kw: float, v_ll_v: float = 440.0, efficiency: float = 0.93,
                 pf_rated: float = 0.85, ilr_multiple: float = 6.5,
                 pf_locked: float = 0.3, s_sc_kva: float = 5000.0,
                 x_over_r: float = 5.0, t_acc_s: float = 8.0,
                 soft_limit: float = 3.0) -> None:
        """Crea el modelo validando todos los parámetros."""
        self.p_kw = require_range("Potencia", p_kw, 0.5, 5000.0)
        self.v_ll_v = require_range("Tensión", v_ll_v, 100.0, 15000.0)
        self.efficiency = require_range("Rendimiento", efficiency, 0.5, 1.0)
        self.pf_rated = require_range("FP nominal", pf_rated, 0.5, 1.0)
        self.ilr_multiple = require_range("I_arranque/I_n", ilr_multiple, 3.0, 9.0)
        self.pf_locked = require_range("FP rotor bloqueado", pf_locked, 0.1, 0.6)
        self.s_sc_kva = require_range("S_cc", s_sc_kva, 100.0, 1e7)
        self.x_over_r = require_range("X/R", x_over_r, 0.5, 30.0)
        self.t_acc_s = require_range("Tiempo de aceleración", t_acc_s, 0.5, 60.0)
        self.soft_limit = require_range("Límite arrancador suave", soft_limit, 1.5, 5.0)

    @property
    def i_rated_a(self) -> float:
        """Corriente nominal de línea [A]."""
        return self.p_kw * 1000 / (math.sqrt(3) * self.v_ll_v * self.efficiency * self.pf_rated)

    @property
    def s_rated_kva(self) -> float:
        """Potencia aparente nominal [kVA]."""
        return self.p_kw / (self.efficiency * self.pf_rated)

    def _shape(self, method: str, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Múltiplo de corriente (en I_n) y factor de potencia vs. tiempo."""
        if method not in self.METHODS:
            raise ParameterError(f"Método desconocido: {method}")
        s = np.exp(-3.0 * t / self.t_acc_s)
        direct = 1.0 + (self.ilr_multiple - 1.0) * s
        pf = self.pf_locked + (self.pf_rated - self.pf_locked) * (1.0 - s)
        if method == "Directo":
            mult = direct
        elif method == "Estrella-Triángulo":
            mult = np.where(t < 0.8 * self.t_acc_s, direct / 3.0, direct)
        elif method == "Arrancador suave":
            mult = np.minimum(direct, self.soft_limit)
        else:  # VFD
            mult, pf = np.full_like(t, 1.1), np.full_like(t, 0.95)
        return mult, pf

    def simulate(self, method: str, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simula tensión en el PCC y corriente de línea.

        Args:
            method: Uno de ``METHODS``.
            t: Vector de tiempo [s].

        Returns:
            Tupla (tensión [pu], corriente [A]).
        """
        mult, pf = self._shape(method, t)
        i_pu = mult * self.s_rated_kva / self.s_sc_kva          # base S_cc, Z_cc = 1 pu
        z_src = cmath.rect(1.0, math.atan(self.x_over_r))
        v = np.abs(1.0 / (1.0 + z_src * i_pu * np.exp(-1j * np.arccos(pf))))
        return v, mult * self.i_rated_a * v

    def start_torque_pu(self, method: str, v_min: float) -> float:
        """Par de arranque disponible [pu del par nominal] con la tensión v_min."""
        return self.LOCKED_TORQUE_PU * self.TORQUE_FACTOR[method] * v_min ** 2


class DailyLoadProfile:
    """Generador de curvas de carga diarias por estación."""

    # (base, pico mañana, pico mediodía, pico noche)
    SEASONS: ClassVar[Dict[str, Tuple[float, float, float, float]]] = {
        "Verano": (0.38, 0.25, 0.45, 0.80),
        "Invierno": (0.42, 0.40, 0.15, 1.00),
        "Primavera/Otoño": (0.35, 0.30, 0.20, 0.75),
    }

    @classmethod
    def generate(cls, season: str, p_peak_mw: float,
                 step_h: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
        """Devuelve (horas, P [MW]) con el pico normalizado a ``p_peak_mw``."""
        if season not in cls.SEASONS:
            raise ParameterError(f"Estación desconocida: {season}")
        require_range("Potencia pico", p_peak_mw, 1.0, 1000.0)
        h = np.arange(0.0, 24.0, step_h)
        base, am, mid, night = cls.SEASONS[season]

        def gauss(mu: float, sig: float) -> np.ndarray:
            return np.exp(-0.5 * ((h - mu) / sig) ** 2)

        shape = base + am * gauss(8, 1.8) + mid * gauss(13, 2.5) + night * gauss(20, 2.0)
        return h, p_peak_mw * shape / shape.max()


def reactive_from_pf(p_mw: np.ndarray, pf: float, inductive: bool = True) -> np.ndarray:
    """Potencia reactiva [MVAr] de una carga con factor de potencia ``pf``."""
    require_range("Factor de potencia", pf, 0.5, 1.0)
    q = p_mw * math.tan(math.acos(pf))
    return q if inductive else -q


@dataclass(frozen=True)
class FeederModel:
    """Alimentador/subestación con carga PQ, banco de capacitores y OLTC.

    Resuelve V² - Vs·V + (R·P + X·Q_neta) = 0 de forma iterativa.

    Attributes:
        s_base_mva: Base de potencia [MVA].
        r_pu: Resistencia serie [pu].
        x_pu: Reactancia serie [pu].
        qc_pu: Carga capacitiva de la línea a 1 pu (efecto Ferranti) [pu].
        v_source_pu: Tensión de la fuente [pu].
    """

    s_base_mva: float = 100.0
    r_pu: float = 0.04
    x_pu: float = 0.18
    qc_pu: float = 0.12
    v_source_pu: float = 1.0

    def __post_init__(self) -> None:
        require_range("R", self.r_pu, 0.0, 1.0)
        require_range("X", self.x_pu, 0.0, 1.0)
        require_range("Qc línea", self.qc_pu, 0.0, 2.0)
        require_range("V fuente", self.v_source_pu, 0.9, 1.1)

    def receiving_voltage(self, p_mw: float, q_mvar: float, tap_steps: int = 0,
                          q_cap_mvar: float = 0.0) -> float:
        """Tensión en barra de carga [pu]; ``nan`` si hay colapso de tensión."""
        require_range("Tap", tap_steps, -8, 8)
        vs = self.v_source_pu * (1.0 + 0.0125 * tap_steps)
        p, q_load = p_mw / self.s_base_mva, q_mvar / self.s_base_mva
        q_shunt = q_cap_mvar / self.s_base_mva + self.qc_pu
        v = vs
        for _ in range(50):
            q_net = q_load - q_shunt * v * v
            disc = vs * vs - 4.0 * (self.r_pu * p + self.x_pu * q_net)
            if disc < 0:
                return float("nan")
            v_new = 0.5 * (vs + math.sqrt(disc))
            if abs(v_new - v) < 1e-7:
                return v_new
            v = v_new
        return v

    def profile(self, p_mw: np.ndarray, q_mvar: np.ndarray, tap_steps: int = 0,
                q_cap_mvar: np.ndarray | None = None) -> np.ndarray:
        """Perfil de tensión para series de P, Q y banco de capacitores."""
        caps = np.zeros_like(p_mw) if q_cap_mvar is None else q_cap_mvar
        return np.array([self.receiving_voltage(p, q, tap_steps, c)
                         for p, q, c in zip(p_mw, q_mvar, caps)])


@dataclass(frozen=True)
class FrequencyModel:
    """Respuesta frecuencia-potencia de un área (ecuación de oscilación).

    2H·dΔω/dt = ΔPm − ΔPL − D·Δω ;  Tg·dPv/dt = Pref − Δω/R − Pv ;
    Tt·dΔPm/dt = Pv − ΔPm ;  dPref/dt = −Ki·Δω (control secundario/AGC).

    Attributes:
        f0: Frecuencia nominal [Hz].
        s_base_mw: Base del sistema [MW].
        h: Constante de inercia [s].
        d: Autorregulación de carga [pu/pu].
        r: Estatismo del gobernador [pu].
        tg: Constante del gobernador [s].
        tt: Constante de la turbina [s].
        ki: Ganancia integral del AGC [pu/(pu·s)].
        reserve_mw: Reserva rodante disponible (límite de la válvula) [MW].
        governor_on: Activa la regulación primaria.
    """

    f0: float = 60.0
    s_base_mw: float = 1000.0
    h: float = 5.0
    d: float = 1.0
    r: float = 0.05
    tg: float = 0.5
    tt: float = 2.0
    ki: float = 0.0
    reserve_mw: float = 1000.0
    governor_on: bool = True

    def __post_init__(self) -> None:
        require_range("Inercia H", self.h, 1.0, 12.0)
        require_range("Amortiguamiento D", self.d, 0.0, 5.0)
        require_range("Estatismo R", self.r, 0.01, 0.12)
        require_range("Ki", self.ki, 0.0, 3.0)
        require_range("Reserva", self.reserve_mw, 0.0, self.s_base_mw)

    def _deriv(self, x: Sequence[float], dpl: float) -> List[float]:
        dw, pm, pv, pref = x
        droop = dw / self.r if self.governor_on else 0.0
        return [(pm - dpl - self.d * dw) / (2 * self.h),
                (pv - pm) / self.tt,
                (pref - droop - pv) / self.tg,
                -self.ki * dw]

    def simulate(self, delta_load_mw: float, t_event: float = 5.0,
                 t_end: float = 120.0, dt: float = 0.05) -> Dict[str, np.ndarray]:
        """Integra con RK4 un escalón de déficit (>0) o exceso (<0) de potencia.

        Returns:
            Diccionario con ``t`` [s], ``f`` [Hz], ``pm`` [MW] y ``pref`` [MW].
        """
        require_range("Magnitud del evento", abs(delta_load_mw), 0.0, self.s_base_mw)
        dpl_pu = delta_load_mw / self.s_base_mw
        lim = self.reserve_mw / self.s_base_mw
        n = int(t_end / dt) + 1
        t, out, x = np.arange(n) * dt, np.zeros((n, 4)), [0.0] * 4
        for k in range(n):
            out[k] = x
            dpl = dpl_pu if t[k] >= t_event else 0.0
            k1 = self._deriv(x, dpl)
            k2 = self._deriv([a + 0.5 * dt * b for a, b in zip(x, k1)], dpl)
            k3 = self._deriv([a + 0.5 * dt * b for a, b in zip(x, k2)], dpl)
            k4 = self._deriv([a + dt * b for a, b in zip(x, k3)], dpl)
            x = [a + dt / 6 * (b + 2 * c + 2 * e + g)
                 for a, b, c, e, g in zip(x, k1, k2, k3, k4)]
            x[2], x[3] = (min(max(v, -lim), lim) for v in (x[2], x[3]))  # saturación
        return {"t": t, "f": self.f0 * (1 + out[:, 0]),
                "pm": out[:, 1] * self.s_base_mw, "pref": out[:, 3] * self.s_base_mw}

    def rocof_hz_s(self, delta_load_mw: float) -> float:
        """Tasa inicial de cambio de frecuencia [Hz/s]."""
        return -(delta_load_mw / self.s_base_mw) * self.f0 / (2 * self.h)

    def static_deviation_hz(self, delta_load_mw: float) -> float:
        """Desviación de régimen permanente sin AGC [Hz]."""
        beta = self.d + (1.0 / self.r if self.governor_on else 0.0)
        return -self.f0 * (delta_load_mw / self.s_base_mw) / max(beta, 1e-9)


class EquipmentImpactModel:
    """Curvas didácticas de impacto de la tensión sobre equipos (Módulo 5)."""

    @staticmethod
    def motor_torque(v: float) -> float:
        """Par de un motor de inducción ∝ V² [pu]."""
        return v ** 2

    @staticmethod
    def motor_current(v: float) -> float:
        """Corriente a plena carga (potencia constante) ∝ 1/V [pu]."""
        return 1.0 / v

    @staticmethod
    def lamp_flux(v: float) -> float:
        """Flujo luminoso de lámpara incandescente ∝ V^3.4 [pu]."""
        return v ** 3.4

    @staticmethod
    def lamp_life(v: float) -> float:
        """Vida útil de lámpara incandescente ∝ V^-13 [pu]."""
        return v ** -13.0


class OverexcitationModel:
    """Relación V/f, protecciones de frecuencia y límites de turbina (Módulo 8)."""

    # (nombre, desplazamiento respecto a f0 [Hz], acción) - valores ilustrativos
    PROTECTION_STAGES: ClassVar[List[Tuple[str, float, str]]] = [
        ("81U-1", -0.7, "Deslastre de carga etapa 1"),
        ("81U-2", -1.2, "Deslastre de carga etapa 2"),
        ("81U-3", -1.8, "Deslastre etapa 3 / separación de islas"),
        ("81O-1", +1.0, "Reducción de generación"),
        ("81O-2", +1.5, "Disparo de unidad generadora"),
    ]

    def __init__(self, f0: float = 60.0) -> None:
        self.f0 = require_range("f0", f0, 45.0, 65.0)

    def vf_pu(self, v_pu: float, f_hz: float) -> float:
        """Relación V/f en por unidad."""
        require_range("Frecuencia", f_hz, 0.5 * self.f0, 1.5 * self.f0)
        return v_pu / (f_hz / self.f0)

    def triggered_stages(self, f_hz: float) -> List[str]:
        """Etapas 81U/81O cuyo ajuste se ha superado."""
        return [f"{n} ({a})" for n, off, a in self.PROTECTION_STAGES
                if (off < 0 and f_hz <= self.f0 + off) or (off > 0 and f_hz >= self.f0 + off)]

    def turbine_limit(self, f_hz: float) -> str:
        """Tiempo permitido de operación de turbina (curva ILUSTRATIVA)."""
        dev = abs(f_hz - self.f0) * 60.0 / self.f0
        for limit, text in ((0.5, "Continuo"), (1.0, "≈ 30 min acumulados"),
                            (1.5, "≈ 3 min"), (2.0, "≈ 30 s")):
            if dev <= limit:
                return text
        return "Disparo inmediato (resonancia de álabes)"

    def pump_power_pu(self, f_hz: float) -> float:
        """Potencia de bomba/ventilador ∝ n³ (leyes de afinidad) [pu]."""
        return (f_hz / self.f0) ** 3


# =============================================================================
# 3. CAPA DE INTERFAZ
# =============================================================================


@dataclass
class AppContext:
    """Contexto compartido por todos los módulos."""

    f0: float
    template: str
    dark: bool


@dataclass(frozen=True)
class Question:
    """Pregunta de opción múltiple del modo evaluación."""

    prompt: str
    options: Tuple[str, ...]
    answer: int
    explanation: str


def render_quiz(key: str, questions: Sequence[Question]) -> None:
    """Muestra preguntas con retroalimentación inmediata."""
    st.markdown("#### Modo evaluación")
    for i, q in enumerate(questions):
        choice = st.radio(q.prompt, q.options, index=None, key=f"{key}_q{i}")
        if choice is not None:
            if q.options.index(choice) == q.answer:
                st.success("✅ Correcto. " + q.explanation)
            else:
                st.error("❌ Incorrecto. " + q.explanation)


def show(fig: go.Figure) -> None:
    """Dibuja una figura Plotly adaptada al ancho (compatible entre versiones)."""
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def style_fig(fig: go.Figure, ctx: AppContext, title: str, xlab: str = "",
              ylab: str = "", height: int = 380) -> go.Figure:
    """Aplica estilo común (tema, leyenda, hover unificado, zoom)."""
    fig.update_layout(template=ctx.template, title=title, height=height,
                      margin=dict(l=10, r=10, t=50, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=-0.22))
    if xlab:
        fig.update_xaxes(title_text=xlab)
    if ylab:
        fig.update_yaxes(title_text=ylab)
    return fig


def kpi_row(items: Sequence[Tuple[str, str, str | None]]) -> None:
    """Fila de tarjetas métricas: (etiqueta, valor, delta opcional)."""
    for col, (label, value, delta) in zip(st.columns(len(items)), items):
        col.metric(label, value, delta, delta_color="off")


def md_table(rows: Sequence[Dict[str, object]]) -> None:
    """Tabla Markdown simple (evita depender de pandas para mostrar tablas)."""
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    st.markdown("\n".join(lines))


def inject_css(dark: bool) -> None:
    """Estilo industrial claro/oscuro.

    Fuerza fondo y texto en TODA la página (barra lateral, cabecera, listas
    desplegables y campos), de modo que el tema elegido no dependa del tema
    nativo del navegador/Streamlit.
    """
    bg, side, card, border, fg = (
        ("#0e141b", "#141c26", "#1b2430", "#2f3b4c", "#e6edf3") if dark
        else ("#f4f6f9", "#e6ebf1", "#ffffff", "#c9d2dd", "#1c2733"))
    st.markdown(f"""<style>
    html, body, .stApp, .stApp [data-testid="stAppViewContainer"],
    .stApp [data-testid="stMain"], .stApp .main {{ background:{bg} !important; }}
    .stApp header[data-testid="stHeader"], .stApp [data-testid="stToolbar"],
    .stApp [data-testid="stDecoration"] {{ background:{bg} !important; }}
    .stApp section[data-testid="stSidebar"],
    .stApp section[data-testid="stSidebar"] > div,
    .stApp [data-testid="stSidebarContent"] {{ background:{side} !important; }}
    .stApp, .stApp *, [data-baseweb="popover"] * {{ color:{fg} !important; }}
    .stApp svg, [data-baseweb="popover"] svg {{ fill:{fg}; }}
    .stApp [data-baseweb="select"] > div, .stApp [data-baseweb="input"],
    .stApp [data-baseweb="base-input"], [data-baseweb="popover"],
    [data-baseweb="popover"] ul, [data-baseweb="menu"],
    .stApp [data-testid="stNumberInput"] button,
    .stApp [data-testid="stExpander"] {{
        background:{card} !important; border-color:{border} !important; }}
    .stApp [data-testid="stMetric"] {{ background:{card} !important; border:1px solid {border};
        border-left:4px solid #f5a623; padding:12px 14px; border-radius:10px; }}
    .stApp [data-testid="stVerticalBlockBorderWrapper"] {{ border-color:{border}; }}
    .stApp table, .stApp th, .stApp td {{ border-color:{border} !important; }}
    </style>""", unsafe_allow_html=True)


class LabModule(ABC):
    """Clase base de un módulo del laboratorio.

    EXTENSIÓN: para añadir un tema nuevo cree una subclase, defina los
    atributos de clase y decórela con ``@register_module``. No hay que tocar
    ``main()`` ni los demás módulos.
    """

    module_id: ClassVar[str] = "base"
    title: ClassVar[str] = "Módulo"
    icon: ClassVar[str] = "⚡"
    objective: ClassVar[str] = ""

    def __init__(self, ctx: AppContext) -> None:
        self.ctx = ctx

    def key(self, name: str) -> str:
        """Clave única de widget (evita colisiones entre pestañas)."""
        return f"{self.module_id}_{name}"

    @abstractmethod
    def render(self) -> None:
        """Dibuja el contenido del módulo."""

    def run(self) -> None:
        """Encabezado común + manejo robusto de errores."""
        st.header(f"{self.icon} {self.title}")
        if self.objective:
            st.caption(f"Objetivo: {self.objective}")
        try:
            self.render()
        except ParameterError as exc:
            st.error(f"Parámetro inválido: {exc}")
        except Exception as exc:  # noqa: BLE001 - la UI nunca debe caerse
            st.error(f"Error inesperado en el módulo: {exc}")


MODULE_REGISTRY: List[Type[LabModule]] = []


def register_module(cls: Type[LabModule]) -> Type[LabModule]:
    """Decorador que añade un módulo al laboratorio (orden de registro = pestañas)."""
    MODULE_REGISTRY.append(cls)
    return cls


# ----------------------------- Módulo 1 --------------------------------------
@register_module
class ControlImportanceModule(LabModule):
    """Importancia del control del sistema de potencia."""

    module_id, title, icon = "m1", "1. Control del SEP", "🎛️"
    objective = "Entender por qué P rige la frecuencia y Q rige la tensión."

    def render(self) -> None:
        c1, c2 = st.columns(2)
        c1.markdown("""**¿Por qué controlar?** La energía no se almacena en la red:
generación = demanda + pérdidas en todo instante. Si se rompe el balance, el sistema
reacciona con su energía cinética (inercia) y las protecciones actúan.""")
        with c2:
            md_table([
                {"Variable": "Frecuencia (f)", "Se controla con": "Potencia activa P",
                 "Naturaleza": "Global", "Actuadores": "Gobernadores, AGC, reserva"},
                {"Variable": "Tensión (V)", "Se controla con": "Potencia reactiva Q",
                 "Naturaleza": "Local", "Actuadores": "AVR, capacitores, taps, SVC"}])
        st.markdown("#### Panel de balance generación vs. demanda")
        a, b = st.columns(2)
        pg = a.slider("Generación programada P_g [MW]", 700, 1300, 1000, key=self.key("pg"))
        pd_ = a.slider("Demanda P_d [MW]", 700, 1300, 1000, key=self.key("pd"))
        qg = b.slider("Reactivo generado Q_g [MVAr]", 0, 600, 300, key=self.key("qg"))
        qd = b.slider("Reactivo demandado Q_d [MVAr]", 0, 600, 300, key=self.key("qd"))
        r_pct = a.slider("Estatismo R [%]", 1.0, 12.0, 5.0, 0.5, key=self.key("r")) / 100
        scc = b.slider("Potencia de cortocircuito [MVA]", 500, 10000, 3000, 100, key=self.key("scc"))
        model = PowerBalanceModel(f0=self.ctx.f0, droop=r_pct, s_sc_mva=float(scc))
        f, v = model.steady_frequency(pg, pd_), model.steady_voltage(qg, qd)
        kpi_row([("Frecuencia", f"{f:.3f} Hz", f"{f - self.ctx.f0:+.3f} Hz"),
                 ("Tensión", f"{v:.3f} pu", f"{v - 1:+.3f} pu"),
                 ("ΔP", f"{pg - pd_:+d} MW", None), ("ΔQ", f"{qg - qd:+d} MVAr", None)])
        if abs(f - self.ctx.f0) > 0.5:
            st.warning("Frecuencia fuera de ±0.5 Hz: riesgo de actuación de protecciones 81U/81O.")
        if not 0.95 <= v <= 1.05:
            st.warning("Tensión fuera de la banda 0.95–1.05 pu.")
        dp = np.linspace(-300, 300, 121)
        left, right = st.columns(2)
        fig = go.Figure(go.Scatter(x=dp, y=[model.steady_frequency(d + 1000, 1000) for d in dp],
                                   name="f(ΔP)"))
        fig.add_trace(go.Scatter(x=[pg - pd_], y=[f], mode="markers", name="Punto actual",
                                 marker=dict(size=12, color="orange")))
        with left:
            show(style_fig(fig, self.ctx, "Característica P–f", "ΔP = P_g − P_d [MW]", "f [Hz]"))
        dq = np.linspace(-300, 300, 121)
        fig2 = go.Figure(go.Scatter(x=dq, y=[model.steady_voltage(d, 0) for d in dq], name="V(ΔQ)"))
        fig2.add_trace(go.Scatter(x=[qg - qd], y=[v], mode="markers", name="Punto actual",
                                  marker=dict(size=12, color="orange")))
        with right:
            show(style_fig(fig2, self.ctx, "Característica Q–V", "ΔQ = Q_g − Q_d [MVAr]", "V [pu]"))
        render_quiz(self.module_id, [Question(
            "¿Qué variable refleja el balance GLOBAL de potencia activa?",
            ("La tensión", "La frecuencia", "El factor de potencia"), 1,
            "La frecuencia es común a todo el sistema síncrono; la tensión es local.")])


# ----------------------------- Módulo 2 --------------------------------------
@register_module
class MotorStartModule(LabModule):
    """Problemas de tensión durante el arranque de motores."""

    module_id, title, icon = "m2", "2. Arranque de motores", "⚙️"
    objective = "Cuantificar la caída de tensión e irrupción de corriente según el método."

    def render(self) -> None:
        a, b, c = st.columns(3)
        unit = a.radio("Unidad", ["kW", "HP"], horizontal=True, key=self.key("unit"))
        val = a.number_input(f"Potencia [{unit}]", 1.0, 3000.0, 75.0, key=self.key("p"))
        p_kw = val * (0.7457 if unit == "HP" else 1.0)
        v_ll = b.selectbox("Tensión de línea [V]", [208, 220, 380, 440, 480, 600], 3, key=self.key("v"))
        scc = b.slider("Potencia de cortocircuito S_cc [kVA]", 500, 50000, 3000, 100, key=self.key("scc"))
        t_acc = c.slider("Tiempo de aceleración [s]", 1.0, 30.0, 8.0, key=self.key("t"))
        load_t = c.slider("Par resistente [pu]", 0.0, 1.5, 0.5, 0.05, key=self.key("tl"))
        method = st.radio("Método de arranque", MotorStartingModel.METHODS, horizontal=True,
                          key=self.key("m"))
        model = MotorStartingModel(p_kw, float(v_ll), s_sc_kva=float(scc), t_acc_s=t_acc)
        t = np.linspace(0, 1.5 * t_acc, 300)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=("Tensión en el PCC", "Corriente de línea"))
        rows = []
        for m in MotorStartingModel.METHODS:
            v, i = model.simulate(m, t)
            w = 4 if m == method else 1.5
            fig.add_trace(go.Scatter(x=t, y=v, name=m, line=dict(width=w), legendgroup=m), 1, 1)
            fig.add_trace(go.Scatter(x=t, y=i, name=m, line=dict(width=w), legendgroup=m,
                                     showlegend=False), 2, 1)
            rows.append({"Método": m, "I_pico [A]": round(float(i.max()), 1),
                         "V_mín [pu]": round(float(v.min()), 3),
                         "Par arranque [pu]": round(model.start_torque_pu(m, float(v.min())), 2)})
        fig.add_hline(y=0.9, line_dash="dash", line_color="red", row=1, col=1,
                      annotation_text="Límite 0.90 pu")
        fig.update_xaxes(title_text="Tiempo [s]", row=2, col=1)
        fig.update_yaxes(title_text="V [pu]", row=1, col=1)
        fig.update_yaxes(title_text="I [A]", row=2, col=1)
        show(style_fig(fig, self.ctx, "Respuesta durante el arranque", height=560))
        sel = next(r for r in rows if r["Método"] == method)
        kpi_row([("I nominal", f"{model.i_rated_a:.1f} A", None),
                 ("I de arranque", f"{sel['I_pico [A]']:.0f} A",
                  f"{sel['I_pico [A]'] / model.i_rated_a:.1f} × I_n"),
                 ("Caída de tensión", f"{(1 - sel['V_mín [pu]']) * 100:.1f} %", None),
                 ("Par de arranque", f"{sel['Par arranque [pu]']:.2f} pu", None)])
        if sel["V_mín [pu]"] < 0.9:
            st.error("Caída > 10 %: riesgo de disparo de contactores, parpadeo (flicker) y fallo de arranque.")
        else:
            st.success("Caída de tensión admisible (≥ 0.90 pu).")
        if sel["Par arranque [pu]"] < load_t:
            st.error("El par disponible es menor que el par resistente: el motor NO arranca.")
        md_table(rows)
        render_quiz(self.module_id, [Question(
            "El arranque estrella-triángulo reduce la corriente de arranque aproximadamente a:",
            ("1/√3", "1/3", "1/9"), 1,
            "La tensión de fase baja a 1/√3, la corriente de fase a 1/√3 y la de línea a 1/3.")])


# ----------------------------- Módulo 3 --------------------------------------
@register_module
class LoadChangeModule(LabModule):
    """Problemas de tensión por cambios de carga en subestaciones."""

    module_id, title, icon = "m3", "3. Cambios de carga", "📈"
    objective = "Relacionar la curva de carga con la tensión (caída en pico, Ferranti en valle)."

    def render(self) -> None:
        a, b, c = st.columns(3)
        season = a.selectbox("Estación", list(DailyLoadProfile.SEASONS), key=self.key("s"))
        peak = a.slider("Demanda pico [MW]", 10, 150, 70, key=self.key("pk"))
        pf = b.slider("Factor de potencia", 0.70, 1.0, 0.90, 0.01, key=self.key("pf"))
        kind = b.radio("Tipo de carga", ["Inductiva", "Capacitiva"], horizontal=True, key=self.key("k"))
        qc = c.slider("Carga capacitiva de la línea [pu]", 0.0, 1.0, 0.25, 0.01, key=self.key("qc"))
        vs = c.slider("Tensión de la fuente [pu]", 0.95, 1.05, 1.0, 0.005, key=self.key("vs"))
        h, p = DailyLoadProfile.generate(season, float(peak))
        q = reactive_from_pf(p, pf, kind == "Inductiva")
        v = FeederModel(qc_pu=qc, v_source_pu=vs).profile(p, q)
        if np.isnan(v).any():
            st.error("Colapso de tensión: reduzca la demanda o mejore el factor de potencia.")
            return
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            subplot_titles=("Curva de carga", "Tensión en la barra de carga"))
        fig.add_trace(go.Scatter(x=h, y=p, name="P [MW]", fill="tozeroy"), 1, 1)
        fig.add_trace(go.Scatter(x=h, y=q, name="Q [MVAr]"), 1, 1)
        fig.add_trace(go.Scatter(x=h, y=v, name="V [pu]", line=dict(color="orange")), 2, 1)
        fig.add_hrect(y0=0.95, y1=1.05, fillcolor="green", opacity=0.12, line_width=0, row=2, col=1)
        fig.update_xaxes(title_text="Hora del día [h]", row=2, col=1)
        fig.update_yaxes(title_text="MW / MVAr", row=1, col=1)
        fig.update_yaxes(title_text="V [pu]", row=2, col=1)
        show(style_fig(fig, self.ctx, f"Día típico de {season}", height=560))
        kpi_row([("V mínima", f"{v.min():.3f} pu", f"a las {h[v.argmin()]:.1f} h"),
                 ("V máxima", f"{v.max():.3f} pu", f"a las {h[v.argmax()]:.1f} h"),
                 ("Demanda mínima", f"{p.min():.1f} MW", None),
                 ("Horas fuera de banda", f"{np.mean((v < 0.95) | (v > 1.05)) * 24:.1f} h", None)])
        if v.max() > 1.05:
            st.warning("Sobretensión en horas valle: efecto Ferranti (carga capacitiva de la línea "
                       "o carga capacitiva con poca demanda inductiva que la compense).")
        if v.min() < 0.95:
            st.warning("Subtensión en horas pico: la corriente de carga aumenta la caída R·P + X·Q.")
        render_quiz(self.module_id, [Question(
            "¿Cuándo es más probable el efecto Ferranti en una línea larga?",
            ("En el pico de demanda", "En horas valle con poca carga", "Durante un cortocircuito"), 1,
            "Con poca carga, la corriente capacitiva de la línea sube la tensión en el extremo receptor.")])


# ----------------------------- Módulo 4 --------------------------------------
@register_module
class VoltagePracticeModule(LabModule):
    """Práctica interactiva: variaciones de tensión."""

    module_id, title, icon = "m4", "4. Práctica: tensión", "🛠️"
    objective = "Llevar el perfil de tensión de 24 h a la banda 0.95–1.05 pu con OLTC y capacitores."

    def render(self) -> None:
        st.info("**Caso:** alimentador largo en invierno (70 MW pico, FP 0.88). "
                "Reto: ≥ 95 % de las horas dentro de 0.95–1.05 pu sin horas fuera de 0.90–1.10 pu.")
        feeder = FeederModel(r_pu=0.04, x_pu=0.18, qc_pu=0.12)
        h, p = DailyLoadProfile.generate("Invierno", 70.0)
        q = reactive_from_pf(p, 0.88)
        left, right = st.columns([1, 2])
        with left:
            tap = st.slider("Tap OLTC (pasos de 1.25 %)", -8, 8, 0, key=self.key("tap"))
            steps = st.slider("Pasos de capacitor (5 MVAr c/u)", 0, 8, 0, key=self.key("n"))
            t_on = st.slider("Conectar banco a las [h]", 0, 23, 17, key=self.key("on"))
            t_off = st.slider("Desconectar banco a las [h]", 1, 24, 22, key=self.key("off"))
        mask = (h >= t_on) & (h < t_off) if t_on <= t_off else (h >= t_on) | (h < t_off)
        cap = np.where(mask, 5.0 * steps, 0.0)
        v0 = feeder.profile(p, q)
        v = feeder.profile(p, q, tap, cap)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=h, y=v0, name="Sin control", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=h, y=v, name="Con control", line=dict(color="orange", width=3)))
        fig.add_hrect(y0=0.95, y1=1.05, fillcolor="green", opacity=0.12, line_width=0)
        with right:
            show(style_fig(fig, self.ctx, "Perfil de tensión de 24 h", "Hora [h]", "V [pu]"))
        if np.isnan(v).any():
            st.error("Colapso de tensión con esta configuración.")
            return
        ok_band = float(np.mean((v >= 0.95) & (v <= 1.05)))
        severe = bool(((v < 0.90) | (v > 1.10)).any())
        kpi_row([("Horas en banda", f"{ok_band * 100:.0f} %", None),
                 ("V mín / máx", f"{v.min():.3f} / {v.max():.3f} pu", None),
                 ("Q capacitiva", f"{5 * steps} MVAr", None)])
        if ok_band >= 0.95 and not severe:
            st.success("🏆 Reto superado: perfil dentro de norma.")
        else:
            if v.min() < 0.95:
                st.warning("Subtensión: suba el tap o conecte capacitores durante el pico.")
            if v.max() > 1.05:
                st.warning("Sobretensión en valle: baje el tap o desconecte el banco en horas valle.")
        render_quiz(self.module_id, [Question(
            "¿Qué efecto tiene dejar un banco de capacitores conectado todo el día?",
            ("Siempre mejora la tensión", "Puede causar sobretensión en el valle",
             "No afecta a la tensión"), 1,
            "En baja demanda el exceso de reactivo capacitivo eleva la tensión.")])


# ----------------------------- Módulo 5 --------------------------------------
@register_module
class VoltageEquipmentModule(LabModule):
    """Equipos afectados por cambios de tensión."""

    module_id, title, icon = "m5", "5. Equipos y tensión", "🔌"
    objective = "Ver cómo subtensión y sobretensión degradan transformadores, motores, luminarias y electrónica."

    def render(self) -> None:
        v = st.slider("Tensión de operación [pu]", 0.80, 1.20, 1.00, 0.01, key=self.key("v"))
        m = EquipmentImpactModel
        i_mag = magnetizing_current_pu(v) * 100
        if v < 0.85:
            elec = "⛔ Disparo por subtensión"
        elif v < 0.90:
            elec = "⚠️ Ride-through limitado"
        elif v <= 1.10:
            elec = "✅ Normal"
        else:
            elec = "⚠️ Estrés en bus DC / disparo OV"
        cards = [
            ("🧲 Transformador", f"I_mag = {i_mag:.1f} %",
             "↓V: sin riesgo mayor, regulación pobre.\n\n↑V: saturación del núcleo, calentamiento, armónicos."),
            ("⚙️ Motor de inducción", f"Par = {m.motor_torque(v) * 100:.0f} %  |  I = {m.motor_current(v) * 100:.0f} %",
             "↓V: par ∝ V², más deslizamiento y corriente → calentamiento.\n\n↑V: saturación, más pérdidas."),
            ("💡 Iluminación incandescente", f"Flujo = {m.lamp_flux(v) * 100:.0f} %  |  Vida = {min(m.lamp_life(v), 99) * 100:.0f} %",
             "↓V: luz muy reducida.\n\n↑V: vida útil se desploma (∝ V⁻¹³)."),
            ("🖥️ Electrónica de potencia", elec,
             "↓V: reinicios, pérdida de regulación del bus DC.\n\n↑V: sobretensión en semiconductores y capacitores."),
        ]
        for col, (name, value, text) in zip(st.columns(4), cards):
            with col.container(border=True):
                st.markdown(f"**{name}**")
                st.markdown(f"`{value}`")
                st.caption(text)
        vs = np.linspace(0.8, 1.2, 81)
        fig = go.Figure()
        for name, fn in (("Par del motor (V²)", m.motor_torque), ("Flujo lámpara (V^3.4)", m.lamp_flux),
                         ("Vida lámpara (V^-13)", m.lamp_life), ("Corriente motor (1/V)", m.motor_current),
                         ("I_mag transformador", lambda x: math.exp(22 * (x - 1)))):
            fig.add_trace(go.Scatter(x=vs, y=[fn(x) for x in vs], name=name))
        fig.add_vline(x=v, line_dash="dash", line_color="orange")
        fig.update_yaxes(type="log")
        show(style_fig(fig, self.ctx, "Impacto relativo (1.0 = valor a 1 pu)", "V [pu]", "Relativo (log)", 420))
        render_quiz(self.module_id, [Question(
            "Si la tensión cae al 90 %, el par del motor de inducción cae aproximadamente al:",
            ("90 %", "81 %", "73 %"), 1, "El par es proporcional a V²: 0.9² = 0.81.")])


# ----------------------------- Módulo 6 --------------------------------------
@register_module
class FrequencyProblemsModule(LabModule):
    """Problemas relacionados con la variación de frecuencia."""

    module_id, title, icon = "m6", "6. Variación de frecuencia", "〰️"
    objective = "Analizar la respuesta f(t) ante un desbalance usando la ecuación de oscilación."

    def render(self) -> None:
        a, b, c = st.columns(3)
        event = a.radio("Evento", ["Pérdida de generación", "Entrada masiva de demanda",
                                   "Rechazo de carga"], key=self.key("e"))
        size = a.slider("Magnitud [MW]", 10, 400, 150, key=self.key("mw"))
        h = b.slider("Inercia H [s]", 1.0, 10.0, 5.0, 0.5, key=self.key("h"))
        d = b.slider("Autorregulación D [pu/pu]", 0.0, 3.0, 1.0, 0.1, key=self.key("d"))
        r = c.slider("Estatismo R [%]", 1.0, 12.0, 5.0, 0.5, key=self.key("r")) / 100
        ref = c.checkbox("Mostrar caso sin gobernador (referencia)", key=self.key("ref"))
        delta = float(-size if event == "Rechazo de carga" else size)
        f0 = self.ctx.f0
        model = FrequencyModel(f0=f0, h=h, d=d, r=r)
        res = model.simulate(delta, t_end=60.0)
        fig = go.Figure(go.Scatter(x=res["t"], y=res["f"], name="Con gobernador",
                                   line=dict(width=3, color="orange")))
        if ref:
            r0 = FrequencyModel(f0=f0, h=h, d=d, r=r, governor_on=False).simulate(delta, t_end=60.0)
            fig.add_trace(go.Scatter(x=r0["t"], y=r0["f"], name="Sin gobernador", line=dict(dash="dot")))
        for name, off, _ in OverexcitationModel.PROTECTION_STAGES[:2] + OverexcitationModel.PROTECTION_STAGES[3:4]:
            fig.add_hline(y=f0 + off, line_dash="dash", line_color="red", annotation_text=name)
        show(style_fig(fig, self.ctx, "Frecuencia del sistema", "Tiempo [s]", "f [Hz]", 420))
        extreme = res["f"].min() if delta > 0 else res["f"].max()
        kpi_row([("RoCoF inicial", f"{model.rocof_hz_s(delta):+.3f} Hz/s", None),
                 ("Nadir/Cresta", f"{extreme:.3f} Hz", f"{extreme - f0:+.3f} Hz"),
                 ("f final (60 s)", f"{res['f'][-1]:.3f} Hz", None),
                 ("Δf estático teórico", f"{model.static_deviation_hz(delta):+.3f} Hz", None)])
        if abs(extreme - f0) >= 0.7:
            st.warning("Se supera el primer ajuste 81U/81O: habría deslastre/reducción automática.")
        render_quiz(self.module_id, [Question(
            "Si el sistema tiene MENOS inercia, ante la misma pérdida de generación:",
            ("El RoCoF es menor", "El RoCoF es mayor", "No cambia el RoCoF"), 1,
            "RoCoF = −ΔP·f0/(2H): a menor H, caída inicial más rápida.")])


# ----------------------------- Módulo 7 --------------------------------------
@register_module
class FrequencyPracticeModule(LabModule):
    """Práctica interactiva: regulación primaria y secundaria."""

    module_id, title, icon = "m7", "7. Práctica: frecuencia", "🎚️"
    objective = "Ajustar droop, AGC y reserva para recuperar la frecuencia nominal tras una contingencia."

    CONTINGENCIES: ClassVar[Dict[str, float]] = {
        "Disparo de unidad (60 MW)": 60.0, "Disparo de central (100 MW)": 100.0,
        "Pérdida de interconexión (150 MW)": 150.0}

    def render(self) -> None:
        f0 = self.ctx.f0
        st.info(f"**Metas:** (1) nadir > {f0 - 0.7:.1f} Hz (sin deslastre), (2) |f − {f0:.0f}| ≤ 0.05 Hz "
                "a los 150 s, (3) reserva suficiente para cubrir la contingencia.")
        ev = st.selectbox("Contingencia", list(self.CONTINGENCIES), key=self.key("ev"))
        a, b, c = st.columns(3)
        r = a.slider("Estatismo R [%] (control primario)", 1.0, 12.0, 5.0, 0.5, key=self.key("r")) / 100
        ki = b.slider("Ganancia AGC Ki (control secundario)", 0.0, 2.0, 0.0, 0.05, key=self.key("ki"))
        res_mw = c.slider("Reserva rodante [MW]", 0, 300, 200, 10, key=self.key("res"))
        delta = self.CONTINGENCIES[ev]
        model = FrequencyModel(f0=f0, h=5.0, r=r, ki=ki, reserve_mw=float(res_mw))
        out = model.simulate(delta, t_end=150.0)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            subplot_titles=("Frecuencia", "Potencia mecánica añadida y referencia AGC"))
        fig.add_trace(go.Scatter(x=out["t"], y=out["f"], name="f [Hz]", line=dict(color="orange", width=3)), 1, 1)
        fig.add_hrect(y0=f0 - 0.05, y1=f0 + 0.05, fillcolor="green", opacity=0.15, line_width=0, row=1, col=1)
        fig.add_hline(y=f0 - 0.7, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_trace(go.Scatter(x=out["t"], y=out["pm"], name="ΔP_m [MW]"), 2, 1)
        fig.add_trace(go.Scatter(x=out["t"], y=out["pref"], name="P_ref AGC [MW]", line=dict(dash="dot")), 2, 1)
        fig.add_hline(y=delta, line_dash="dash", line_color="gray", row=2, col=1,
                      annotation_text="Contingencia")
        fig.update_xaxes(title_text="Tiempo [s]", row=2, col=1)
        fig.update_yaxes(title_text="f [Hz]", row=1, col=1)
        fig.update_yaxes(title_text="MW", row=2, col=1)
        show(style_fig(fig, self.ctx, "Resultado de la simulación", height=560))
        nadir, f_end = float(out["f"].min()), float(out["f"][-1])
        goals = {"Nadir sobre ajuste 81U": nadir > f0 - 0.7,
                 "Frecuencia restaurada (±0.05 Hz)": abs(f_end - f0) <= 0.05,
                 "Reserva suficiente": res_mw >= delta}
        kpi_row([("Nadir", f"{nadir:.3f} Hz", f"{nadir - f0:+.3f} Hz"),
                 ("f final", f"{f_end:.3f} Hz", f"{f_end - f0:+.3f} Hz"),
                 ("Metas cumplidas", f"{sum(goals.values())} / 3", None)])
        for name, ok in goals.items():
            st.write(("✅ " if ok else "❌ ") + name)
        if all(goals.values()):
            st.success("🏆 Reto superado: regulación primaria y secundaria bien coordinadas.")
        else:
            with st.expander("💡 Pista"):
                st.write("El droop solo limita la caída (queda error en régimen permanente); "
                         "el AGC (Ki) integra el error hasta devolver f a nominal, pero con Ki muy alto "
                         "aparecen oscilaciones. Si la reserva es menor que la contingencia, ninguna "
                         "ganancia restituye la frecuencia.")
        render_quiz(self.module_id, [Question(
            "¿Qué acción elimina el error de frecuencia remanente del control primario?",
            ("Aumentar la inercia", "Control secundario (AGC)", "Subir la tensión"), 1,
            "El AGC desplaza la referencia de carga-velocidad con acción integral hasta Δf = 0.")])


# ----------------------------- Módulo 8 --------------------------------------
@register_module
class FrequencyEquipmentModule(LabModule):
    """Equipos afectados por cambios de frecuencia."""

    module_id, title, icon = "m8", "8. Equipos y frecuencia", "🌀"
    objective = "Analizar sobreflujo V/f, protecciones 81 y efectos en turbinas, motores y bombas."

    def render(self) -> None:
        f0 = self.ctx.f0
        model = OverexcitationModel(f0)
        a, b = st.columns(2)
        v = a.slider("Tensión [pu]", 0.8, 1.3, 1.0, 0.01, key=self.key("v"))
        f = b.slider("Frecuencia [Hz]", f0 - 5.0, f0 + 5.0, f0, 0.05, key=self.key("f"))
        vf = model.vf_pu(v, f)
        rpm = 120 * f / 4
        kpi_row([("V/f", f"{vf:.3f} pu", "SOBREEXCITACIÓN" if vf > 1.1 else None),
                 ("I magnetizante", f"{magnetizing_current_pu(vf) * 100:.1f} %", None),
                 ("Velocidad motor 4 polos", f"{rpm:.0f} rpm", None),
                 ("Potencia bomba/ventilador", f"{model.pump_power_pu(f) * 100:.0f} %", None)])
        if vf > 1.1:
            st.error("V/f > 1.10 pu: saturación del núcleo, calentamiento y disparo de la protección 24 (V/Hz).")
        elif vf > 1.05:
            st.warning("V/f > 1.05 pu: límite continuo a plena carga (IEEE C57.12.00).")
        stages = model.triggered_stages(f)
        st.write("**Protecciones 81O/81U activadas:** " + (", ".join(stages) if stages else "ninguna"))
        st.write(f"**Tiempo permitido de turbina (ilustrativo):** {model.turbine_limit(f)}")
        fs = np.linspace(f0 - 5, f0 + 5, 200)
        fig = go.Figure(go.Scatter(x=fs, y=[model.vf_pu(v, x) for x in fs], name=f"V/f con V = {v:.2f} pu"))
        fig.add_hline(y=1.1, line_dash="dash", line_color="red", annotation_text="Límite 1.10")
        fig.add_hline(y=1.05, line_dash="dot", line_color="orange", annotation_text="1.05")
        fig.add_trace(go.Scatter(x=[f], y=[vf], mode="markers", name="Punto actual",
                                 marker=dict(size=12, color="orange")))
        show(style_fig(fig, self.ctx, "Relación V/f vs. frecuencia", "f [Hz]", "V/f [pu]"))
        st.markdown("**Ajustes ilustrativos de protección** (use los del código de red local):")
        md_table([{"Función": n, "Ajuste [Hz]": round(f0 + off, 2), "Acción": act}
                  for n, off, act in OverexcitationModel.PROTECTION_STAGES])
        render_quiz(self.module_id, [Question(
            "Un transformador a tensión nominal y frecuencia reducida 10 % queda con V/f ≈:",
            ("0.90 pu", "1.11 pu", "1.00 pu"), 1,
            "V/f = 1/0.9 ≈ 1.11 pu: sobreexcitación, el flujo es proporcional a V/f.")])

# =============================================================================
# 2-bis. FÍSICA: RED DE 6 BARRAS (flujo de potencia, fallas, renovable, control)
# -----------------------------------------------------------------------------
# PEGAR ESTE BLOQUE COMPLETO justo ANTES de la sección
#     "# 4. PUNTO DE ENTRADA"   (es decir, después del Módulo 8).
# No hay que tocar main(): el módulo nuevo se registra solo con @register_module.
# =============================================================================
import dataclasses as _dc  # noqa: E402  (se usa _dc.replace para variar escenarios)

REN_Q_MODES: Tuple[str, ...] = ("FP unitario", "Inyecta Q (FP 0.95)", "Absorbe Q (FP 0.95)")


@dataclass(frozen=True)
class GridScenario:
    """Escenario de operación de la red de 6 barras.

    Attributes:
        load_pct: Nivel de carga respecto al pico [%].
        ren_bus: Índice (0-5) de la barra donde se conecta la renovable.
        ren_mw: Potencia activa entregada por la renovable [MW].
        ren_q_mode: Modo de control de reactivo del inversor.
        tap_steps: Pasos del OLTC (+ sube la tensión del lado de carga).
        cap_bus: Barra del banco de capacitores.
        cap_mvar: Banco de capacitores [MVAr] (inyecta Q).
        reactor_bus: Barra del reactor shunt.
        reactor_mvar: Reactor [MVAr] (absorbe Q).
    """

    load_pct: float = 100.0
    ren_bus: int = 4
    ren_mw: float = 0.0
    ren_q_mode: str = REN_Q_MODES[0]
    tap_steps: int = 0
    cap_bus: int = 5
    cap_mvar: float = 0.0
    reactor_bus: int = 1
    reactor_mvar: float = 0.0

    def __post_init__(self) -> None:
        require_range("Nivel de carga", self.load_pct, 0.0, 200.0)
        require_range("Potencia renovable", self.ren_mw, 0.0, 200.0)
        require_range("Tap", self.tap_steps, -8, 8)
        require_range("Capacitor", self.cap_mvar, 0.0, 100.0)
        require_range("Reactor", self.reactor_mvar, 0.0, 100.0)
        for name, bus in (("Barra renovable", self.ren_bus), ("Barra capacitor", self.cap_bus),
                          ("Barra reactor", self.reactor_bus)):
            require_range(name, bus, 0, 5)
        if self.ren_q_mode not in REN_Q_MODES:
            raise ParameterError(f"Modo Q desconocido: {self.ren_q_mode}")


class VoltageSagTolerance:
    """Curvas ILUSTRATIVAS de tolerancia a huecos de tensión por tipo de carga.

    Cada tupla es (V_límite [pu], ciclos tolerados): si la tensión residual es
    menor que V_límite, la carga soporta como máximo esos ciclos.
    """

    CURVES: ClassVar[Dict[str, Tuple[Tuple[float, float], ...]]] = {
        "res": ((0.20, 1.0), (0.50, 3.0), (0.70, 6.0), (0.80, 12.0), (0.90, 30.0)),
        "ind": ((0.20, 1.0), (0.50, 2.0), (0.70, 4.0), (0.80, 8.0), (0.90, 20.0)),
    }

    @classmethod
    def tolerated_cycles(cls, kind: str, v_pu: float) -> float:
        """Ciclos máximos que la carga tolera con tensión residual ``v_pu``."""
        if kind not in cls.CURVES:
            return math.inf
        for v_lim, cycles in cls.CURVES[kind]:
            if v_pu < v_lim:
                return cycles
        return math.inf


class SixBusNetwork:
    """Red didáctica de 6 barras (base 100 MVA, 13.8 kV).

    B1 Red (slack) - B2 Subestación (trafo OLTC) - B3/B5 residenciales -
    B4/B6 industriales. Admite renovable, capacitor, reactor, motor y fallas.
    """

    N: ClassVar[int] = 6
    NAMES: ClassVar[Tuple[str, ...]] = ("B1 Red", "B2 Subestación", "B3 Residencial",
                                        "B4 Industrial", "B5 Residencial", "B6 Industrial")
    KINDS: ClassVar[Tuple[str, ...]] = ("grid", "sub", "res", "ind", "res", "ind")
    KIND_LABEL: ClassVar[Dict[str, str]] = {"grid": "Red (slack)", "sub": "Subestación",
                                            "res": "Residencial", "ind": "Industrial"}
    COORDS: ClassVar[Tuple[Tuple[float, float], ...]] = (
        (0.0, 1.0), (1.2, 1.0), (2.4, 2.0), (2.4, 0.0), (3.8, 2.0), (3.8, 0.0))
    LOAD_PEAK_MW: ClassVar[Tuple[float, ...]] = (0.0, 0.0, 12.0, 25.0, 10.0, 20.0)
    LOAD_PF: ClassVar[Dict[str, float]] = {"res": 0.95, "ind": 0.85}
    # (desde, hasta, R [pu], X [pu], B total [pu], capacidad [MVA]); la rama 0 lleva el OLTC
    BRANCHES: ClassVar[Tuple[Tuple[int, int, float, float, float, float], ...]] = (
        (0, 1, 0.010, 0.080, 0.00, 100.0),
        (1, 2, 0.040, 0.140, 0.02, 40.0),
        (1, 3, 0.030, 0.100, 0.02, 60.0),
        (2, 3, 0.050, 0.160, 0.02, 25.0),
        (2, 4, 0.050, 0.170, 0.02, 25.0),
        (3, 5, 0.040, 0.130, 0.02, 40.0),
        (4, 5, 0.060, 0.180, 0.02, 25.0),
    )
    LOAD_BUSES: ClassVar[Tuple[int, ...]] = (2, 3, 4, 5)

    def __init__(self, s_base_mva: float = 100.0, kv_base: float = 13.8,
                 grid_scc_mva: float = 2500.0) -> None:
        self.s_base_mva = require_range("S base", s_base_mva, 1.0, 1e4)
        self.kv_base = require_range("kV base", kv_base, 0.4, 500.0)
        self.grid_scc_mva = require_range("S_cc de la red", grid_scc_mva, 100.0, 1e6)

    # ---- datos de entrada -------------------------------------------------
    @staticmethod
    def branch_label(i: int, j: int) -> str:
        """Etiqueta corta de una rama, p. ej. ``B1→B2``."""
        return f"B{i + 1}→B{j + 1}"

    def loads(self, load_pct: float) -> Tuple[np.ndarray, np.ndarray]:
        """Cargas P [MW] y Q [MVAr] por barra para un nivel de carga dado."""
        p = np.array(self.LOAD_PEAK_MW) * load_pct / 100.0
        pf = np.array([self.LOAD_PF.get(k, 1.0) for k in self.KINDS])
        return p, p * np.tan(np.arccos(pf))

    def injections(self, sc: GridScenario) -> Tuple[np.ndarray, np.ndarray]:
        """Inyecciones netas P, Q [MW, MVAr] (generación renovable − carga)."""
        p_l, q_l = self.loads(sc.load_pct)
        p, q = -p_l, -q_l
        p[sc.ren_bus] += sc.ren_mw
        sign = {REN_Q_MODES[0]: 0.0, REN_Q_MODES[1]: 1.0, REN_Q_MODES[2]: -1.0}[sc.ren_q_mode]
        q[sc.ren_bus] += sign * sc.ren_mw * math.tan(math.acos(0.95))
        return p, q

    def shunt_mvar(self, sc: GridScenario) -> np.ndarray:
        """Compensación shunt por barra [MVAr]: capacitor (+) y reactor (−) a 1 pu."""
        s = np.zeros(self.N)
        s[sc.cap_bus] += sc.cap_mvar
        s[sc.reactor_bus] -= sc.reactor_mvar
        return s

    @staticmethod
    def tap_ratio(tap_steps: int) -> float:
        """Relación del OLTC; paso positivo => sube la tensión del lado de carga."""
        return 1.0 - 0.0125 * tap_steps

    # ---- red --------------------------------------------------------------
    def admittance(self, tap_steps: int = 0) -> np.ndarray:
        """Matriz de admitancias de barra (sin shunts de compensación)."""
        a = self.tap_ratio(tap_steps)
        y = np.zeros((self.N, self.N), dtype=complex)
        for k, (i, j, r, x, b, _) in enumerate(self.BRANCHES):
            ys, t = 1.0 / complex(r, x), (a if k == 0 else 1.0)
            y[i, i] += ys / t ** 2 + 0.5j * b
            y[j, j] += ys + 0.5j * b
            y[i, j] -= ys / t
            y[j, i] -= ys / t
        return y

    def solve(self, sc: GridScenario, y_extra: np.ndarray | None = None) -> Dict[str, object]:
        """Flujo de potencia Newton-Raphson (Jacobiano numérico, barra 1 = slack).

        Args:
            sc: Escenario de operación.
            y_extra: Admitancias shunt adicionales por barra [pu] (p. ej. motor).

        Returns:
            Diccionario con ``v`` (fasores [pu]), ``y`` (Ybus), ``converged`` e ``iterations``.
        """
        n, m = self.N, self.N - 1
        p_inj, q_inj = self.injections(sc)
        ysh = 1j * self.shunt_mvar(sc) / self.s_base_mva
        if y_extra is not None:
            ysh = ysh + y_extra
        y = self.admittance(sc.tap_steps) + np.diag(ysh)
        p_s, q_s = p_inj / self.s_base_mva, q_inj / self.s_base_mva

        def build(x: np.ndarray) -> np.ndarray:
            return (np.concatenate(([1.0], x[m:])) *
                    np.exp(1j * np.concatenate(([0.0], x[:m]))))

        def mismatch(x: np.ndarray) -> np.ndarray:
            v = build(x)
            s = v * np.conj(y @ v)
            return np.concatenate((s.real[1:] - p_s[1:], s.imag[1:] - q_s[1:]))

        x = np.concatenate((np.zeros(m), np.ones(m)))
        converged, it = False, 0
        for it in range(1, 41):
            f = mismatch(x)
            if np.max(np.abs(f)) < 1e-8:
                converged = True
                break
            jac = np.empty((2 * m, 2 * m))
            for k in range(2 * m):
                xp = x.copy()
                xp[k] += 1e-6
                jac[:, k] = (mismatch(xp) - f) / 1e-6
            try:
                x = x + np.linalg.solve(jac, -f)
            except np.linalg.LinAlgError:
                break
            if not np.all(np.isfinite(x)) or np.any(x[m:] < 0.2):
                break
        return {"v": build(x), "y": y, "converged": converged, "iterations": it}

    def slack_power(self, case: Dict[str, object]) -> complex:
        """Potencia compleja entregada por la red [MW + j MVAr]."""
        v, y = case["v"], case["y"]
        return complex((v * np.conj(y @ v))[0]) * self.s_base_mva

    def branch_flows(self, v: np.ndarray, tap_steps: int = 0) -> List[Dict[str, float]]:
        """Flujos P/Q por rama en ambos extremos, pérdidas y % de carga."""
        a, rows = self.tap_ratio(tap_steps), []
        for k, (i, j, r, x, b, rating) in enumerate(self.BRANCHES):
            ys, t = 1.0 / complex(r, x), (a if k == 0 else 1.0)
            i_ij = (ys / t ** 2 + 0.5j * b) * v[i] - ys / t * v[j]
            i_ji = (ys + 0.5j * b) * v[j] - ys / t * v[i]
            s_ij = v[i] * np.conj(i_ij) * self.s_base_mva
            s_ji = v[j] * np.conj(i_ji) * self.s_base_mva
            rows.append({"from": i, "to": j, "p_from": float(s_ij.real), "q_from": float(s_ij.imag),
                         "p_to": float(s_ji.real), "q_to": float(s_ji.imag),
                         "loss_mw": float((s_ij + s_ji).real),
                         "loading_pct": float(max(abs(s_ij), abs(s_ji)) / rating * 100.0),
                         "rating": rating})
        return rows

    def fault_voltages(self, sc: GridScenario, case: Dict[str, object], bus: int,
                       z_f_pu: float = 0.0) -> Tuple[np.ndarray, float]:
        """Falla trifásica en ``bus`` con el método de la matriz Zbus.

        Cargas como impedancia constante; la red se modela con su S_cc (X/R = 10);
        el aporte de la renovable se desprecia (inversores limitan la corriente).

        Returns:
            (|V| de todas las barras durante la falla [pu], corriente de falla [kA]).
        """
        require_range("Barra en falla", bus, 0, self.N - 1)
        require_range("Z de falla", z_f_pu, 0.0, 1.0)
        v0 = case["v"]
        p_l, q_l = self.loads(sc.load_pct)
        y_load = np.conj((p_l + 1j * q_l) / self.s_base_mva) / np.abs(v0) ** 2
        y = self.admittance(sc.tap_steps) + np.diag(
            1j * self.shunt_mvar(sc) / self.s_base_mva + y_load)
        z_src = (self.s_base_mva / self.grid_scc_mva) * cmath.rect(1.0, math.atan(10.0))
        y[0, 0] += 1.0 / z_src
        zbus = np.linalg.inv(y)
        i_f = v0[bus] / (zbus[bus, bus] + z_f_pu)
        v_fault = v0 - zbus[:, bus] * i_f
        i_base_ka = self.s_base_mva / (math.sqrt(3.0) * self.kv_base)
        return np.abs(v_fault), float(abs(i_f) * i_base_ka)


@dataclass(frozen=True)
class EquipmentItem:
    """Equipo de una central o subestación con sus límites de tensión [pu] (ILUSTRATIVOS)."""

    location: str
    name: str
    v_dmg_low: float | None
    v_warn_low: float | None
    v_warn_high: float
    v_dmg_high: float
    effect_low: str
    effect_high: str


class PlantEquipmentCatalog:
    """Catálogo didáctico de equipos afectados por sub/sobretensión sostenida."""

    CENTRAL, SUBSTATION = "Central generadora", "Subestación"
    AXIS: ClassVar[Tuple[float, float]] = (0.60, 1.40)
    _E = EquipmentItem
    ITEMS: ClassVar[Tuple[EquipmentItem, ...]] = (
        _E("Central generadora", "Generador síncrono (estator y rotor)", 0.85, 0.95, 1.05, 1.10,
           "Sube la corriente de estator y el calentamiento; riesgo de pérdida de sincronismo.",
           "Sobreexcitación (V/Hz alto): calentamiento del núcleo y del rotor, disparo por protección 24."),
        _E("Central generadora", "Excitatriz / AVR y rectificadores", 0.70, 0.85, 1.10, 1.25,
           "La excitación pierde techo y no sostiene la tensión: colapso de reactivo.",
           "Sobretensión en tiristores/diodos del puente y falla de aislamiento."),
        _E("Central generadora", "Transformador elevador (GSU)", None, 0.90, 1.05, 1.10,
           "Sin daño directo: más corriente, más pérdidas y pobre regulación.",
           "Saturación del núcleo, corriente de magnetización alta, calentamiento y gases en el aceite."),
        _E("Central generadora", "Transformador de servicios auxiliares", None, 0.90, 1.05, 1.10,
           "Caída de tensión en toda la barra de auxiliares.",
           "Sobreexcitación y sobretensión en motores y contactores aguas abajo."),
        _E("Central generadora", "Motores de servicios auxiliares (bombas, ventiladores, molinos)", 0.75, 0.90,
           1.10, 1.20,
           "Par ∝ V²: se frenan o se calan, corriente alta y calentamiento; puede disparar la unidad.",
           "Saturación, más pérdidas y sobrecalentamiento del aislamiento."),
        _E("Central generadora", "Contactores y arrancadores (CCM)", 0.70, 0.85, 1.10, 1.20,
           "Se sueltan (drop-out) y paran bombas/ventiladores: posible disparo de caldera o turbina.",
           "Sobrecalentamiento de bobinas y falla de aislamiento."),
        _E("Central generadora", "Variadores de frecuencia (VFD)", 0.80, 0.90, 1.10, 1.20,
           "Disparo por subtensión del bus DC y paro del proceso.",
           "Sobretensión en el bus DC: falla de IGBT y capacitores."),
        _E("Central generadora", "Baterías y cargadores DC", 0.80, 0.90, 1.10, 1.20,
           "El cargador no sostiene el bus DC: se descargan las baterías y se pierde control/protección.",
           "Sobrecarga y gaseo de baterías, falla del cargador."),
        _E("Central generadora", "Sistemas de control (DCS/PLC) y UPS", 0.80, 0.90, 1.10, 1.20,
           "Reinicios y pérdida de señales; el UPS pasa a baterías.",
           "Fuentes de alimentación dañadas y pérdida de tarjetas."),
        _E("Central generadora", "Cables y aislamiento de media tensión", None, None, 1.10, 1.20,
           "Sin efecto directo.",
           "Envejecimiento acelerado y descargas parciales en el aislamiento."),
        _E("Subestación", "Transformador de potencia", None, 0.90, 1.05, 1.10,
           "Sin daño directo: más corriente, más pérdidas y pobre regulación.",
           "Sobreexcitación: aumentan las pérdidas en vacío, calentamiento, gases en el aceite y armónicos."),
        _E("Subestación", "Regulador de tap (OLTC) y su motor", 0.70, 0.85, 1.10, 1.20,
           "El accionamiento falla al conmutar y el control pierde alimentación.",
           "Mayor esfuerzo en el selector y en los contactos."),
        _E("Subestación", "Pararrayos (óxido metálico)", None, None, 1.05, 1.15,
           "Sin efecto directo.",
           "Opera cerca de su tensión máxima continua: fuga térmica y posible explosión."),
        _E("Subestación", "Aisladores y boquillas", None, None, 1.10, 1.25,
           "Sin efecto directo.", "Descargas superficiales (flashover) y contornamiento."),
        _E("Subestación", "Interruptores y seccionadores", None, None, 1.10, 1.20,
           "Sin efecto directo.",
           "Menor margen de aislamiento y de recuperación de tensión al interrumpir."),
        _E("Subestación", "Transformadores de medida (TP y TC)", None, 0.90, 1.10, 1.20,
           "Errores de medición: los relés reciben información incorrecta.",
           "Saturación y ferrorresonancia en TP, calentamiento."),
        _E("Subestación", "Bancos de capacitores", None, 0.90, 1.10, 1.20,
           "Entregan menos reactivo (Q ∝ V²): menos soporte justo cuando se necesita.",
           "Sobrecarga de kVAr ∝ V², falla de unidades e inflado de las latas."),
        _E("Subestación", "Reactores shunt", None, None, 1.10, 1.20,
           "Absorben menos reactivo (Q ∝ V²).", "Saturación, calentamiento y armónicos."),
        _E("Subestación", "Relés de protección y medición digital", 0.60, 0.80, 1.10, 1.25,
           "Pueden reiniciar o no operar: protección sin respuesta.",
           "Daño de fuentes y de entradas de tensión."),
        _E("Subestación", "Baterías y cargadores de la subestación", 0.80, 0.90, 1.10, 1.20,
           "Se descargan las baterías: riesgo de perder disparo y control.",
           "Sobrecarga y gaseo de baterías, falla del cargador."),
        _E("Subestación", "Cables de potencia", None, None, 1.10, 1.20,
           "Sin efecto directo.",
           "Envejecimiento acelerado y descargas parciales en el aislamiento."),
    )

    @classmethod
    def status(cls, item: EquipmentItem, v_pu: float) -> Tuple[str, str]:
        """Devuelve (estado, efecto) del equipo a tensión sostenida ``v_pu``."""
        if item.v_dmg_low is not None and v_pu < item.v_dmg_low:
            return "⛔ Falla / daño", item.effect_low
        if item.v_warn_low is not None and v_pu < item.v_warn_low:
            return "⚠️ En riesgo", item.effect_low
        if v_pu > item.v_dmg_high:
            return "⛔ Falla / daño", item.effect_high
        if v_pu > item.v_warn_high:
            return "⚠️ En riesgo", item.effect_high
        return "✅ Normal", "Opera dentro de su rango de tensión."

    @classmethod
    def evaluate(cls, v_pu: float, location: str | None = None) -> List[Tuple[EquipmentItem, str, str]]:
        """Evalúa los equipos de ``location`` (o todos si es ``None``)."""
        require_range("Tensión", v_pu, 0.0, 2.0)
        return [(it, *cls.status(it, v_pu)) for it in cls.ITEMS
                if location is None or it.location == location]


# =============================================================================
# 3-bis. INTERFAZ: MÓDULO 9
# =============================================================================


@register_module
class SixBusNetworkModule(LabModule):
    """Red de 6 barras: flujo de potencia, fallas, renovable, motor y control de tensión."""

    module_id, title, icon = "m9", "9. Red de 6 barras", "🕸️"
    objective = ("Ver cómo el flujo de potencia, las fallas (≤ 12 ciclos), la generación renovable, "
                 "el arranque de motores y el control de tensión afectan a cargas residenciales e industriales.")

    def render(self) -> None:
        net = SixBusNetwork()
        sc = self._scenario_controls()
        case = net.solve(sc)
        if not case["converged"]:
            st.error("El flujo de potencia no converge (posible colapso de tensión). "
                     "Reduzca la carga o añada compensación / tap.")
            return
        t1, t2, t3, t4 = st.tabs(["🔀 Flujo de potencia", "⚡ Fallas (≤ 12 ciclos)",
                                  "⚙️ Motor en la red", "🎚️ Control de tensión"])
        with t1:
            self._tab_flow(net, sc, case)
        with t2:
            self._tab_fault(net, sc, case)
        with t3:
            self._tab_motor(net, sc, case)
        with t4:
            self._tab_control(net, sc, case)
        render_quiz(self.module_id, [
            Question("Un hueco de 0.60 pu dura 8 ciclos. Según las curvas del modelo, ¿qué cargas se afectan?",
                     ("Solo la residencial", "Solo la industrial", "Ambas"), 2,
                     "A 0.60 pu la residencial tolera 6 ciclos y la industrial 4: 8 ciclos supera ambos límites."),
            Question("¿Qué dispositivo conviene para bajar la tensión en horas de poca carga?",
                     ("Banco de capacitores", "Reactor shunt", "Más generación renovable con FP unitario"), 1,
                     "El reactor absorbe reactivo y reduce la tensión; el capacitor la sube.")])

    # ------------------------------------------------------------------ UI --
    def _scenario_controls(self) -> GridScenario:
        names = SixBusNetwork.NAMES
        with st.container(border=True):
            st.markdown("**⚙️ Escenario de la red**")
            a, b, c = st.columns(3)
            load = a.slider("Nivel de carga [% del pico]", 40, 130, 100, 5, key=self.key("load"))
            cap_inst = a.slider("Renovable instalada [MW]", 0, 40, 20, 5, key=self.key("rcap"))
            factor = a.slider("Recurso solar/eólico disponible [%]", 0, 100, 60, 5, key=self.key("rf"))
            ren_bus = b.selectbox("Barra de la renovable", [1, 2, 3, 4, 5], 3,
                                  format_func=lambda i: names[i], key=self.key("rbus"))
            ren_q = b.selectbox("Control de Q del inversor", REN_Q_MODES, key=self.key("rq"))
            tap = b.slider("Tap OLTC (pasos de 1.25 %)", -8, 8, 0, key=self.key("tap"))
            cap_bus = c.selectbox("Barra del capacitor", [1, 2, 3, 4, 5], 4,
                                  format_func=lambda i: names[i], key=self.key("cbus"))
            cap = c.slider("Capacitor [MVAr]", 0, 30, 0, 5, key=self.key("cap"))
            rea_bus = c.selectbox("Barra del reactor", [1, 2, 3, 4, 5], 0,
                                  format_func=lambda i: names[i], key=self.key("rebus"))
            rea = c.slider("Reactor [MVAr]", 0, 30, 0, 5, key=self.key("rea"))
        return GridScenario(load_pct=float(load), ren_bus=int(ren_bus),
                            ren_mw=cap_inst * factor / 100.0, ren_q_mode=ren_q, tap_steps=int(tap),
                            cap_bus=int(cap_bus), cap_mvar=float(cap), reactor_bus=int(rea_bus),
                            reactor_mvar=float(rea))

    @staticmethod
    def _default_notes(net: SixBusNetwork) -> Dict[int, str]:
        """Rótulos por barra: generación/red, ciudad (residencial) e industria."""
        icons = {"grid": "🏭 Central / Red", "res": "🏘️ Ciudad", "ind": "🏗️ Industria"}
        return {i: icons[k] for i, k in enumerate(net.KINDS) if k in icons}

    def _diagram(self, net: SixBusNetwork, vm: np.ndarray, flows: List[Dict[str, float]],
                 sc: GridScenario, notes: Dict[int, str] | None = None,
                 title: str = "Diagrama unifilar: tensión por barra y flujo de P entre barras") -> go.Figure:
        """Diagrama unifilar con generación/red, ciudad, motores, flechas de P y color por tensión."""
        fig = go.Figure()
        for fl in flows:
            (x0, y0), (x1, y1) = net.COORDS[fl["from"]], net.COORDS[fl["to"]]
            load = fl["loading_pct"]
            color = "#e74c3c" if load > 100 else ("#f5a623" if load > 70 else "#2ecc71")
            fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", hoverinfo="skip",
                                     line=dict(width=4, color=color), showlegend=False))
            if fl["p_from"] >= 0:
                tail, head = (x0, y0), (x1, y1)
            else:
                tail, head = (x1, y1), (x0, y0)
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            length = math.hypot(head[0] - tail[0], head[1] - tail[1])
            ux, uy = (head[0] - tail[0]) / length, (head[1] - tail[1]) / length
            fig.add_annotation(x=mx + 0.18 * ux, y=my + 0.18 * uy, ax=mx - 0.18 * ux, ay=my - 0.18 * uy,
                               xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                               arrowhead=3, arrowsize=1.6, arrowwidth=3, arrowcolor=color)
            fig.add_trace(go.Scatter(
                x=[mx - 0.2 * uy], y=[my + 0.2 * ux], mode="text", showlegend=False, hoverinfo="skip",
                text=[f"{abs(fl['p_from']):.1f} MW<br>{load:.0f} %"], textfont=dict(size=11)))
        fig.add_trace(go.Scatter(
            x=[c[0] for c in net.COORDS], y=[c[1] for c in net.COORDS], mode="markers+text",
            text=[f"{n}<br>{v:.3f} pu" for n, v in zip(net.NAMES, vm)], textposition="bottom center",
            showlegend=False, hovertext=[f"{n}: {v:.3f} pu" for n, v in zip(net.NAMES, vm)],
            marker=dict(size=34, color=vm, colorscale="RdYlGn", cmin=0.90, cmax=1.05,
                        line=dict(width=2, color="#333"),
                        colorbar=dict(title="V [pu]", thickness=12, len=0.6))))
        rx, ry = net.COORDS[sc.ren_bus]
        if sc.ren_mw > 0:
            fig.add_annotation(x=rx, y=ry, ax=rx, ay=ry + 0.55, xref="x", yref="y", axref="x", ayref="y",
                               text=f"🌞 {sc.ren_mw:.1f} MW", showarrow=True, arrowhead=3, arrowwidth=2,
                               arrowcolor="#f5a623")
        for bus, qty, sym in ((sc.cap_bus, sc.cap_mvar, "🔋 C"), (sc.reactor_bus, sc.reactor_mvar, "🧲 L")):
            if qty > 0:
                cx, cy = net.COORDS[bus]
                fig.add_annotation(x=cx + 0.35, y=cy - (0.0 if sym.endswith("C") else 0.25),
                                   text=f"{sym} {qty:.0f} MVAr", showarrow=False)
        for bus, text in (notes if notes is not None else self._default_notes(net)).items():
            nx, ny = net.COORDS[bus]
            fig.add_annotation(x=nx, y=ny - 0.78, text=text, showarrow=False, font=dict(size=12))
        fig.update_xaxes(visible=False, range=[-0.5, 4.5])
        fig.update_yaxes(visible=False, range=[-1.4, 3.0])
        fig.update_layout(template=self.ctx.template, height=500, margin=dict(l=10, r=10, t=50, b=10),
                          title=title)
        return fig

    # -------------------------------------------------------- pestaña 1 -----
    def _tab_flow(self, net: SixBusNetwork, sc: GridScenario, case: Dict[str, object]) -> None:
        v = case["v"]
        vm, ang = np.abs(v), np.degrees(np.angle(v))
        flows = net.branch_flows(v, sc.tap_steps)
        p_l, q_l = net.loads(sc.load_pct)
        s_grid = net.slack_power(case)
        losses = sum(f["loss_mw"] for f in flows)
        kpi_row([("P desde la red", f"{s_grid.real:.1f} MW", f"Q = {s_grid.imag:+.1f} MVAr"),
                 ("Pérdidas", f"{losses:.2f} MW", f"{losses / max(p_l.sum(), 1e-9) * 100:.1f} % de la carga"),
                 ("V mínima", f"{vm.min():.3f} pu", net.NAMES[int(vm.argmin())]),
                 ("V máxima", f"{vm.max():.3f} pu", net.NAMES[int(vm.argmax())])])
        show(self._diagram(net, vm, flows, sc))
        st.caption("Flecha = sentido del flujo de P. Color de la línea: verde < 70 %, naranja 70–100 %, "
                   "rojo > 100 % de su capacidad.")
        over = [net.branch_label(f["from"], f["to"]) for f in flows if f["loading_pct"] > 100]
        if over:
            st.error("Sobrecarga en: " + ", ".join(over))
        low = [net.NAMES[i] for i in range(net.N) if vm[i] < 0.95]
        high = [net.NAMES[i] for i in range(net.N) if vm[i] > 1.05]
        if low:
            st.warning("Subtensión (< 0.95 pu) en: " + ", ".join(low))
        if high:
            st.warning("Sobretensión (> 1.05 pu) en: " + ", ".join(high))
        fig = go.Figure(go.Bar(x=list(net.NAMES), y=vm, marker_color=[
            "#2ecc71" if 0.95 <= x <= 1.05 else "#e74c3c" for x in vm], name="V [pu]"))
        fig.add_hrect(y0=0.95, y1=1.05, fillcolor="green", opacity=0.10, line_width=0)
        fig.update_yaxes(range=[min(0.80, float(vm.min()) - 0.02), max(1.10, float(vm.max()) + 0.02)])
        show(style_fig(fig, self.ctx, "Tensión en las 6 barras", "", "V [pu]", 330))
        md_table([{"Barra": net.NAMES[i], "Tipo": net.KIND_LABEL[net.KINDS[i]],
                   "V [pu]": round(float(vm[i]), 3), "Ángulo [°]": round(float(ang[i]), 2),
                   "P carga [MW]": round(float(p_l[i]), 1), "Q carga [MVAr]": round(float(q_l[i]), 1),
                   "Renovable [MW]": round(sc.ren_mw, 1) if i == sc.ren_bus else 0.0}
                  for i in range(net.N)])
        st.markdown("**Flujo de potencia entre barras** (P, Q medidos en el extremo emisor)")
        md_table([{"Rama": net.branch_label(f["from"], f["to"]) + (" (OLTC)" if k == 0 else ""),
                   "P [MW]": round(f["p_from"], 2), "Q [MVAr]": round(f["q_from"], 2),
                   "Pérdidas [MW]": round(f["loss_mw"], 3), "Carga [%]": round(f["loading_pct"], 0)}
                  for k, f in enumerate(flows)])

    # -------------------------------------------------------- pestaña 2 -----
    def _tab_fault(self, net: SixBusNetwork, sc: GridScenario, case: Dict[str, object]) -> None:
        f0 = self.ctx.f0
        cycle_ms = 1000.0 / f0
        a, b, c = st.columns(3)
        fbus = a.selectbox("Barra en falla", list(range(net.N)), 3,
                           format_func=lambda i: net.NAMES[i], key=self.key("fbus"))
        cycles = b.slider("Duración de la falla [ciclos] (máx. 12)", 1, 12, 6, key=self.key("fcyc"))
        rf = c.slider("Impedancia de falla R_f [pu]", 0.0, 0.05, 0.0, 0.005, key=self.key("frf"))
        v_pre = np.abs(case["v"])
        v_fault, i_ka = net.fault_voltages(sc, case, int(fbus), float(rf))
        rows, affected = [], 0
        for i in net.LOAD_BUSES:
            kind = net.KINDS[i]
            tol = VoltageSagTolerance.tolerated_cycles(kind, float(v_fault[i]))
            hit = cycles > tol
            affected += int(hit)
            rows.append({"Barra": net.NAMES[i], "Tipo": net.KIND_LABEL[kind],
                         "V durante falla [pu]": round(float(v_fault[i]), 3),
                         "Tolera [ciclos]": "∞" if math.isinf(tol) else f"{tol:g}",
                         "Se afecta si dura más de [ms]": "—" if math.isinf(tol) else f"{tol * cycle_ms:.0f}",
                         "Estado": "⛔ Afectada" if hit else "✅ Soporta"})
        kpi_row([("Corriente de falla", f"{i_ka:.1f} kA", None),
                 ("Duración", f"{cycles * cycle_ms:.0f} ms", f"{cycles} ciclos a {f0:.0f} Hz"),
                 ("V mín en cargas", f"{min(v_fault[i] for i in net.LOAD_BUSES):.3f} pu", None),
                 ("Cargas afectadas", f"{affected} / {len(net.LOAD_BUSES)}", None)])
        if affected:
            st.error(f"La falla en {net.NAMES[int(fbus)]} con {cycles} ciclos afecta a {affected} carga(s).")
        else:
            st.success("Todas las cargas soportan el hueco de tensión con esta duración.")
        t0 = 2.0
        t_end = t0 + cycles + 6.0
        fig = go.Figure()
        for i in range(net.N):
            fig.add_trace(go.Scatter(
                x=[0, t0, t0, t0 + cycles, t0 + cycles, t_end],
                y=[v_pre[i], v_pre[i], v_fault[i], v_fault[i], v_pre[i], v_pre[i]],
                name=net.NAMES[i], mode="lines", line=dict(width=4 if i == fbus else 2)))
        fig.add_hline(y=0.9, line_dash="dash", line_color="red", annotation_text="0.90 pu")
        show(style_fig(fig, self.ctx, "Hueco de tensión por barra", "Tiempo [ciclos]", "V [pu]", 380))
        md_table(rows)
        st.markdown("**Mapa de sensibilidad:** ciclos que tolera cada carga según la barra fallada "
                    "(⛔ = se afecta con la duración elegida)")
        mat = []
        for fb in range(net.N):
            vf, _ = net.fault_voltages(sc, case, fb, float(rf))
            row: Dict[str, object] = {"Falla en": net.NAMES[fb]}
            for i in net.LOAD_BUSES:
                tol = VoltageSagTolerance.tolerated_cycles(net.KINDS[i], float(vf[i]))
                row[net.NAMES[i]] = ("∞" if math.isinf(tol) else f"{tol:g}") + (" ⛔" if cycles > tol else "")
            mat.append(row)
        md_table(mat)
        with st.expander("Curvas de tolerancia usadas (ilustrativas)"):
            md_table([{"Si V residual <": f"{lim:.2f} pu",
                       "Residencial tolera [ciclos]": f"{VoltageSagTolerance.CURVES['res'][k][1]:g}",
                       "Industrial tolera [ciclos]": f"{VoltageSagTolerance.CURVES['ind'][k][1]:g}"}
                      for k, (lim, _) in enumerate(VoltageSagTolerance.CURVES["res"])])
            st.caption("Residencial: electrónica e iluminación. Industrial: contactores, variadores y procesos "
                       "sensibles. Falla trifásica, cargas como impedancia constante, aporte renovable "
                       "despreciado. Use las curvas de su norma (ITIC / SEMI F47) en estudios reales.")

    # -------------------------------------------------------- pestaña 3 -----
    def _tab_motor(self, net: SixBusNetwork, sc: GridScenario, case: Dict[str, object]) -> None:
        st.markdown("Generación/red → subestación → **motores industriales** en paralelo con la "
                    "**ciudad** (cargas residenciales). Observe cuánto cae la tensión de la ciudad "
                    "cuando arrancan los motores.")
        a, b, c = st.columns(3)
        p4 = a.slider("Motor en B4 Industrial [MW]", 0.0, 5.0, 3.0, 0.5, key=self.key("mp4"))
        p6 = a.slider("Motor en B6 Industrial [MW]", 0.0, 5.0, 2.0, 0.5, key=self.key("mp6"))
        t_acc = b.slider("Tiempo de aceleración [s]", 2.0, 20.0, 8.0, 1.0, key=self.key("mt"))
        load_t = b.slider("Par resistente [pu]", 0.0, 1.5, 0.5, 0.05, key=self.key("mtl"))
        method = c.radio("Método de arranque", MotorStartingModel.METHODS, key=self.key("mm"))
        powers = {bus: p for bus, p in ((3, p4), (5, p6)) if p >= 0.5}
        if not powers:
            st.info("Active al menos un motor (≥ 0.5 MW).")
            return
        models = {bus: MotorStartingModel(p * 1000.0, 13800.0, t_acc_s=t_acc) for bus, p in powers.items()}
        ref = next(iter(models.values()))
        t = np.linspace(0.0, 1.2 * t_acc, 31)
        res_buses = [i for i in net.LOAD_BUSES if net.KINDS[i] == "res"]
        v_base = np.abs(case["v"])

        def run(m: str):
            mult, pf = ref._shape(m, t)
            vmat = np.zeros((len(t), net.N))
            vph = np.zeros((len(t), net.N), dtype=complex)
            cur = {bus: np.zeros(len(t)) for bus in models}
            for k in range(len(t)):
                y_ex = np.zeros(net.N, dtype=complex)
                for bus, mm in models.items():
                    s_pu = mult[k] * mm.s_rated_kva / 1000.0 / net.s_base_mva * complex(
                        pf[k], math.sqrt(max(1.0 - pf[k] ** 2, 0.0)))
                    y_ex[bus] += np.conj(s_pu)
                out = net.solve(sc, y_ex)
                if not out["converged"]:
                    return None
                vph[k], vmat[k] = out["v"], np.abs(out["v"])
                for bus, mm in models.items():
                    cur[bus][k] = mult[k] * mm.i_rated_a * vmat[k, bus]
            return vmat, vph, cur

        def summary(m: str, out) -> Dict[str, object]:
            vm_, _, cu_ = out
            v_motor = min(float(vm_[:, bus].min()) for bus in models)
            drop = max((v_base[i] - float(vm_[:, i].min())) / v_base[i] * 100.0 for i in res_buses)
            torque = min(models[bus].start_torque_pu(m, float(vm_[:, bus].min())) for bus in models)
            return {"Método": m, "I pico total [A]": round(float(sum(cu_.values()).max()), 0),
                    "V mín motores [pu]": round(v_motor, 3),
                    "V mín ciudad [pu]": round(float(vm_[:, res_buses].min()), 3),
                    "Caída en ciudad [%]": round(drop, 1), "Par arranque mín [pu]": round(torque, 2)}

        results = {m: run(m) for m in MotorStartingModel.METHODS}
        rows = [summary(m, o) if o is not None else
                {"Método": m, "I pico total [A]": "—", "V mín motores [pu]": "colapso",
                 "V mín ciudad [pu]": "colapso", "Caída en ciudad [%]": "—", "Par arranque mín [pu]": "—"}
                for m, o in results.items()]
        sel = results[method]
        if sel is None:
            st.error("Colapso de tensión con estos motores y método: use arranque suave/VFD, reduzca la "
                     "potencia o añada compensación.")
            md_table(rows)
            return
        vmat, vph, cur = sel
        dt = float(t[1] - t[0])
        t_show = st.slider("Instante mostrado en el diagrama [s] (0 s = máximo arranque)", 0.0,
                           float(t[-1]), 0.0, dt, key=self.key("mts"))
        k = int(np.argmin(np.abs(t - t_show)))
        notes = self._default_notes(net)
        for bus, p in powers.items():
            notes[bus] = f"⚙️ Motor {p:.1f} MW<br>I = {cur[bus][k]:.0f} A"
        flows = net.branch_flows(vph[k], sc.tap_steps)
        show(self._diagram(net, vmat[k], flows, sc, notes,
                           f"Arranque {method}: t = {t[k]:.1f} s (generación → motores y ciudad)"))
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            subplot_titles=("Tensión en las barras (línea punteada = ciudad)",
                                            "Corriente de línea de los motores"))
        for i in range(net.N):
            dash = "dot" if i in res_buses else "solid"
            fig.add_trace(go.Scatter(x=t, y=vmat[:, i], name=net.NAMES[i],
                                     line=dict(width=4 if i in powers else 2, dash=dash)), 1, 1)
        fig.add_hline(y=0.9, line_dash="dash", line_color="red", row=1, col=1)
        for bus in models:
            fig.add_trace(go.Scatter(x=t, y=cur[bus], name=f"I motor {net.NAMES[bus][:2]} [A]"), 2, 1)
        fig.add_vline(x=float(t[k]), line_dash="dash", line_color="orange")
        fig.update_xaxes(title_text="Tiempo [s]", row=2, col=1)
        fig.update_yaxes(title_text="V [pu]", row=1, col=1)
        fig.update_yaxes(title_text="I [A]", row=2, col=1)
        show(style_fig(fig, self.ctx, "Respuesta de la red durante el arranque", height=560))
        info = summary(method, sel)
        kpi_row([("I de arranque total", f"{info['I pico total [A]']:.0f} A", None),
                 ("V mín motores", f"{info['V mín motores [pu]']:.3f} pu", None),
                 ("V mín ciudad", f"{info['V mín ciudad [pu]']:.3f} pu", None),
                 ("Caída en la ciudad", f"{info['Caída en ciudad [%]']:.1f} %", None),
                 ("Par arranque mín", f"{info['Par arranque mín [pu]']:.2f} pu", None)])
        if vmat.min() < 0.90:
            st.error("Caída < 0.90 pu en alguna barra: disparo de contactores y pérdida de procesos.")
        if info["Caída en ciudad [%]"] > 3.0 or info["V mín ciudad [pu]"] < 0.95:
            st.warning("La ciudad sufre una caída apreciable (> 3 % o < 0.95 pu): parpadeo (flicker) "
                       "en iluminación y subtensión en electrodomésticos.")
        if info["Par arranque mín [pu]"] < load_t:
            st.error("El par disponible es menor que el par resistente: algún motor NO arranca.")
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=list(net.NAMES), y=v_base, name="Antes del arranque"))
        fig2.add_trace(go.Bar(x=list(net.NAMES), y=vmat.min(axis=0), name="Mínimo durante el arranque"))
        fig2.update_layout(barmode="group")
        fig2.update_yaxes(range=[min(0.6, float(vmat.min()) - 0.05), 1.1])
        show(style_fig(fig2, self.ctx, "Efecto de los motores sobre cada barra", "", "V [pu]", 330))
        md_table(rows)
        st.caption("Los motores se modelan como impedancia constante sumada a la carga existente; "
                   "t = 0 es el máximo desarrollo de la corriente de arranque.")

    # -------------------------------------------------------- pestaña 4 -----
    def _tab_control(self, net: SixBusNetwork, sc: GridScenario, case: Dict[str, object]) -> None:
        vm = np.abs(case["v"])
        base_sc = _dc.replace(sc, tap_steps=0, cap_mvar=0.0, reactor_mvar=0.0)
        base = net.solve(base_sc)
        fig = go.Figure()
        if base["converged"]:
            fig.add_trace(go.Bar(x=list(net.NAMES), y=np.abs(base["v"]), name="Sin control"))
        else:
            st.warning("Sin control la red colapsa: la compensación es indispensable.")
        fig.add_trace(go.Bar(x=list(net.NAMES), y=vm, name="Con control"))
        fig.add_hrect(y0=0.95, y1=1.05, fillcolor="green", opacity=0.10, line_width=0)
        fig.update_layout(barmode="group")
        fig.update_yaxes(range=[min(0.80, float(vm.min()) - 0.02), 1.10])
        show(style_fig(fig, self.ctx, "Perfil de tensión: sin control vs. con control", "", "V [pu]", 360))
        loss_now = sum(f["loss_mw"] for f in net.branch_flows(case["v"], sc.tap_steps))
        if base["converged"]:
            loss_0 = sum(f["loss_mw"] for f in net.branch_flows(base["v"], 0))
            kpi_row([("V mín", f"{vm.min():.3f} pu", f"{vm.min() - np.abs(base['v']).min():+.3f} vs sin control"),
                     ("V máx", f"{vm.max():.3f} pu", f"{vm.max() - np.abs(base['v']).max():+.3f} vs sin control"),
                     ("Pérdidas", f"{loss_now:.2f} MW", f"{loss_now - loss_0:+.2f} MW")])
        with st.expander("🏭 Equipos de la subestación en riesgo con estas tensiones"):
            for label, volt in (("tensión mínima", float(vm.min())), ("tensión máxima", float(vm.max()))):
                bad = [(it, stt, eff) for it, stt, eff in PlantEquipmentCatalog.evaluate(volt, "Subestación")
                       if not stt.startswith("✅")]
                if bad:
                    st.markdown(f"**Con la {label} ({volt:.3f} pu):**")
                    for it, stt, eff in bad:
                        st.write(f"{stt} — **{it.name}**: {eff}")
                else:
                    st.write(f"✅ Con la {label} ({volt:.3f} pu) ningún equipo de la subestación está en riesgo.")
        taps = list(range(-8, 9))
        vmin_t, vmax_t = [], []
        for tp in taps:
            r = net.solve(_dc.replace(sc, tap_steps=tp))
            vv = np.abs(r["v"]) if r["converged"] else np.full(net.N, np.nan)
            vmin_t.append(float(np.nanmin(vv)) if r["converged"] else float("nan"))
            vmax_t.append(float(np.nanmax(vv)) if r["converged"] else float("nan"))
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=taps, y=vmin_t, name="V mínima"))
        fig2.add_trace(go.Scatter(x=taps, y=vmax_t, name="V máxima"))
        fig2.add_hrect(y0=0.95, y1=1.05, fillcolor="green", opacity=0.10, line_width=0)
        fig2.add_vline(x=sc.tap_steps, line_dash="dash", line_color="orange")
        show(style_fig(fig2, self.ctx, "Barrido del tap del OLTC (con la compensación actual)",
                       "Paso del tap", "V [pu]", 330))
        if st.button("🔎 Sugerir ajuste (tap + capacitor + reactor)", key=self.key("sug")):
            with st.spinner("Buscando la mejor combinación…"):
                best = None
                for tp in range(-8, 9):
                    for cp in range(0, 31, 5):
                        for rc in (0, 5, 10):
                            cand = _dc.replace(sc, tap_steps=tp, cap_mvar=float(cp), reactor_mvar=float(rc))
                            r = net.solve(cand)
                            if not r["converged"]:
                                continue
                            v = np.abs(r["v"])
                            score = (float(np.sum(np.maximum(0.0, 0.95 - v) + np.maximum(0.0, v - 1.05))),
                                     abs(tp) + cp / 5 + rc / 5)
                            if best is None or score < best[0]:
                                best = (score, tp, cp, rc, float(v.min()), float(v.max()))
            if best is None:
                st.error("No se encontró una combinación factible.")
            else:
                _, tp, cp, rc, v_lo, v_hi = best
                st.success(f"Sugerencia: tap {tp:+d}, capacitor {cp} MVAr en {net.NAMES[sc.cap_bus]}, "
                           f"reactor {rc} MVAr en {net.NAMES[sc.reactor_bus]} → V entre "
                           f"{v_lo:.3f} y {v_hi:.3f} pu.")
        md_table([
            {"Dispositivo": "Capacitor", "Acción": "Inyecta Q (∝ V²)", "Efecto en V": "Sube (local)",
             "Cuándo": "Horas pico / subtensión"},
            {"Dispositivo": "Reactor", "Acción": "Absorbe Q", "Efecto en V": "Baja (local)",
             "Cuándo": "Valle / Ferranti / exceso de renovable"},
            {"Dispositivo": "Regulador de tap (OLTC)", "Acción": "Cambia la relación del trafo",
             "Efecto en V": "Sube/baja todo aguas abajo", "Cuándo": "Ajuste global; no crea Q"},
            {"Dispositivo": "Inversor renovable", "Acción": "Inyecta o absorbe Q (FP 0.95)",
             "Efecto en V": "Sube/baja en su barra", "Cuándo": "Control fino de tensión"}])


# ----------------------------- Módulo 10 -------------------------------------
@register_module
class PlantEquipmentModule(LabModule):
    """Equipos de la central y de la subestación afectados por la tensión."""

    module_id, title, icon = "m10", "10. Equipos de central y subestación", "🏭"
    objective = ("Obtener la lista específica de equipos que quedan en riesgo o se dañan cuando "
                 "la tensión sube o baja en una central o en una subestación.")

    def render(self) -> None:
        a, b = st.columns([1, 2])
        loc = a.radio("Instalación", ["Central generadora", "Subestación", "Ambas"], key=self.key("loc"))
        v = b.slider("Tensión de operación sostenida [pu]", 0.60, 1.40, 1.12, 0.01, key=self.key("v"))
        scope = None if loc == "Ambas" else loc
        res = PlantEquipmentCatalog.evaluate(float(v), scope)
        dmg = [(it, e) for it, s, e in res if s.startswith("⛔")]
        risk = [(it, e) for it, s, e in res if s.startswith("⚠️")]
        kpi_row([("Tensión", f"{v:.2f} pu", "Sobretensión" if v > 1.05 else ("Subtensión" if v < 0.95 else None)),
                 ("⛔ Fallan / se dañan", str(len(dmg)), None),
                 ("⚠️ En riesgo", str(len(risk)), None),
                 ("✅ Normales", str(len(res) - len(dmg) - len(risk)), None)])
        tag = (lambda it: f"**{it.name}**" + ("" if scope else f" ({it.location})"))
        if dmg:
            st.error("**⛔ Equipos que se dañan o fallan:**\n\n" +
                     "\n".join(f"- {tag(it)}: {e}" for it, e in dmg))
        if risk:
            st.warning("**⚠️ Equipos en riesgo (operación sostenida no recomendada):**\n\n" +
                       "\n".join(f"- {tag(it)}: {e}" for it, e in risk))
        if not dmg and not risk:
            st.success("Todos los equipos operan dentro de su rango de tensión.")
        lo, hi = PlantEquipmentCatalog.AXIS
        fig = go.Figure()
        segs = (("Falla / daño", "#e74c3c"), ("Riesgo", "#f5a623"), ("Normal", "#2ecc71"),
                ("Riesgo ", "#f5a623"), ("Falla / daño ", "#e74c3c"))
        items = [it for it, _, _ in res]
        widths: List[List[float]] = [[] for _ in segs]
        for it in items:
            d_lo = it.v_dmg_low if it.v_dmg_low is not None else lo
            w_lo = it.v_warn_low if it.v_warn_low is not None else d_lo
            for k, w in enumerate((d_lo - lo, w_lo - d_lo, it.v_warn_high - w_lo,
                                   it.v_dmg_high - it.v_warn_high, hi - it.v_dmg_high)):
                widths[k].append(max(w, 0.0))
        names = [it.name for it in items]
        fig.add_trace(go.Bar(y=names, x=[lo] * len(names), orientation="h", showlegend=False, marker_color="rgba(0,0,0,0)", hoverinfo="skip"))
        for (label, color), w in zip(segs, widths):
            fig.add_trace(go.Bar(y=names, x=w, base=None, orientation="h", name=label.strip(),
                                 marker_color=color, showlegend=not label.endswith(" "),
                                 hoverinfo="skip"))
        fig.update_layout(barmode="stack")
        fig.update_xaxes(range=[lo, hi])
        fig.update_yaxes(autorange="reversed")
        fig.add_vline(x=float(v), line_width=3, line_dash="dash", line_color="white" if self.ctx.dark else "black")
        fig.update_layout(hovermode=False)
        show(style_fig(fig, self.ctx, "Rango seguro de cada equipo (la línea marca la tensión elegida)",
                       "V [pu]", "", 90 + 30 * len(items)))
        md_table([{"Equipo": it.name, "Instalación": it.location, "Estado": s,
                   "Efecto": e} for it, s, e in res])
        st.caption("Límites ILUSTRATIVOS para operación sostenida. Para estudios reales use los datos de placa "
                   "y las normas aplicables a cada equipo.")
        render_quiz(self.module_id, [
            Question("Un transformador de potencia opera sostenido a 1.12 pu (frecuencia nominal). Su estado es:",
                     ("Normal", "En riesgo", "Falla / daño"), 2,
                     "Supera 1.10 pu: sobreexcitación, saturación del núcleo, calentamiento y gases en el aceite."),
            Question("Al bajar la tensión, ¿qué pasa con un banco de capacitores?",
                     ("Entrega más reactivo", "Entrega menos reactivo (Q ∝ V²)", "No cambia"), 1,
                     "La potencia reactiva de un capacitor es proporcional a V², justo cuando más se necesita.")])

# =============================================================================
# 4. PUNTO DE ENTRADA
# =============================================================================


def main() -> None:
    """Ensambla el dashboard: barra lateral, tema y pestañas por módulo."""
    st.set_page_config(page_title="Laboratorio Virtual SEP", page_icon="⚡", layout="wide")
    with st.sidebar:
        st.title("⚡ Lab Virtual SEP")
        theme = st.radio("Tema", ["Oscuro industrial", "Claro"], key="theme")
        f0 = float(st.radio("Frecuencia nominal [Hz]", [60, 50], horizontal=True, key="f0"))
        st.caption("Modelos didácticos simplificados. Valores en por unidad (pu).")
        # EXTENSIÓN: añada aquí controles globales (p. ej., base de potencia del sistema).
    dark = theme.startswith("Oscuro")
    ctx = AppContext(f0=f0, template="plotly_dark" if dark else "plotly_white", dark=dark)
    inject_css(dark)
    st.title("Laboratorio Virtual de Sistemas Eléctricos de Potencia")
    # EXTENSIÓN: los módulos nuevos se registran con @register_module y aparecen aquí solos.
    modules = [cls(ctx) for cls in MODULE_REGISTRY]
    for tab, module in zip(st.tabs([f"{m.icon} {m.title}" for m in modules]), modules):
        with tab:
            module.run()


if __name__ == "__main__":
    main()

# Additional module clusters:
# using ${path.separator} (';' on Windows or ':' on Unix):
#
#netbeans_extraclusters="/absolute/path/to/cluster1:/absolute/path/to/cluster2"
