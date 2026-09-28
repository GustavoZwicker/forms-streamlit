# -*- coding: utf-8 -*-
"""
Data-collection app for the deepfake-detection study (Supabase backend).

Groups (balanced in real time):
  - "Controle"  (group 1): no training. Watches the videos and classifies them.
  - "Checklist" (group 2): checklist training + learning check, then classifies.
  - "XAI"       (group 3): classifies videos shown together with the model's
                           deepfake probability and an XAI/LLM explanation.

Flow:
  intro -> consent (TCLE) -> [GROUP ASSIGNMENT] -> demographics
        -> (Checklist) training -> (Checklist) learning check
        -> classification task (videos from Supabase Storage) -> final -> end

Assignment: a single Postgres function (assign_group) picks the group furthest
below its target proportion and increments its count inside one transaction
guarded by an advisory lock, so it is race-free even with many simultaneous
participants and across app restarts.

Storage & media (Supabase):
  - responses / counts / stimuli  -> Postgres tables.
  - video stimuli                 -> a public Storage bucket ("videos").
  Without credentials the app falls back to local SQLite + placeholder stimuli
  (testing only).

NOTE: participant-facing text is Portuguese on purpose; the code is English.
"""

import collections
import io
import json
import os
import random
import re
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone

import streamlit as st

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None

# =============================================================================
# CONFIGURATION
# =============================================================================

GROUPS = ["Controle", "Checklist", "XAI"]
WEIGHTS = {"Controle": 1, "Checklist": 1, "XAI": 1}  # 1:1:1 => same-size samples

GROUPS_WITH_MATERIAL = {"Checklist", "Todos"}
GROUPS_WITH_AI_EXPLANATION = {"XAI", "Todos"}

# Researcher-only preview options. "Todos" enables every group-specific feature.
DEBUG_GROUP_OPTIONS = ["Normal", "Controle", "Checklist", "XAI", "Todos"]

# Demographic variables kept balanced across groups (minimization). All Section 1
# questions are considered. Optional ones (gender, social_media_freq, used_ai) are
# skipped per-participant when left blank. Must be collected BEFORE assignment.
BALANCE_FACTORS = ["age_range", "gender", "education", "ai_familiarity",
                   "deepfake_knowledge", "social_media_freq", "used_ai"]

LABEL_AUTENTICO = "Legítimo"
LABEL_DEEPFAKE = "Deepfake"
TASK_OPTIONS = [LABEL_AUTENTICO, LABEL_DEEPFAKE]

# Fields stored per participant. Note: "group" maps to DB column "grp".
RESPONSE_HEADERS = [
    "timestamp", "participant_id", "group", "consented",
    "age_range", "gender", "education",
    "ai_familiarity", "deepfake_knowledge", "social_media_freq", "used_ai",
    "check_2_1", "check_2_2", "check_2_3", "check_score",
    "task_json", "task_timings_json", "task_score", "final_json", "complete",
]

# Section 2 answer key (NOT shown to the participant).
ANSWER_2_1 = "Transições ou bordas não naturais entre o rosto e o fundo"
ANSWER_2_2 = "Falso"
ANSWER_2_3 = "Verificar a fonte e o contexto da mídia"

OPTIONS_2_1 = [
    "Transições ou bordas não naturais entre o rosto e o fundo",
    "A imagem estar em alta resolução",
    "A pessoa estar sorrindo",
    "O arquivo ser grande",
]
OPTIONS_2_3 = [
    "Verificar a fonte e o contexto da mídia",
    "Confiar apenas no número de curtidas",
    "Aumentar o brilho da tela",
    "Compartilhar antes de checar",
]

# -----------------------------------------------------------------------------
# Consent text (Portuguese). Fill in the bracketed fields before collecting data.
# -----------------------------------------------------------------------------
CONSENT_TEXT = """
**TERMO DE CONSENTIMENTO LIVRE E ESCLARECIDO**

Você está sendo convidado(a) a participar da pesquisa ITT - Vision,
conduzida por Gustavo Zwicker, vinculada à UTFPR-CP,
sob orientação de Rogério Pozza e Robson Bonidia.

- **Objetivo:** avaliar como orientações de letramento digital e explicações de
  inteligência artificial ajudam pessoas a identificar vídeos faciais autênticos
  ou manipulados (*deepfakes*).
- **Procedimentos:** você responderá a um questionário inicial, poderá receber um
  breve material educativo, assistirá a alguns vídeos e os classificará como
  legítimos ou deepfakes, e responderá a questionários finais. Duração
  estimada: **cerca de 15 minutos**.
- **Riscos:** mínimos, limitados a eventual desconforto ou cansaço ao analisar os
  vídeos. Você pode interromper a participação a qualquer momento.
- **Benefícios:** contribuir para o desenvolvimento de ferramentas de combate à
  desinformação e ampliar sua percepção sobre mídias manipuladas.
- **Voluntariedade:** a participação é **voluntária e não remunerada**. Você pode
  desistir a qualquer momento, sem qualquer prejuízo.
- **Confidencialidade e dados (LGPD – Lei nº 13.709/2018):** **não** serão coletados
  dados que identifiquem você pessoalmente. As respostas serão armazenadas de forma
  anonimizada e usadas apenas para fins acadêmicos e científicos, de forma agregada.
- **Contatos:** Pesquisador(a) responsável — Gustavo Zwicker, gustavogzwicker@gmail.com.
"""

# -----------------------------------------------------------------------------
# Certificate (participant-facing). Adjust the study/institution text if needed.
# -----------------------------------------------------------------------------
CERT_STUDY_TITLE = "ITT-Vision: identificação de vídeos autênticos e deepfakes"
CERT_INSTITUTION = ("Universidade Tecnológica Federal do Paraná (UTFPR) "
                    "– Campus Cornélio Procópio")
_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
SEAL_PATH = os.path.join(_ASSETS, "selobg.png")            # transparent InteliGente seal
WATERMARK_PATH = os.path.join(_ASSETS, "logo_wm.png")      # faint centered watermark

# =============================================================================
# ASSIGNMENT HELPERS (used by the local SQLite fallback)
# =============================================================================


@st.cache_resource
def get_lock() -> threading.Lock:
    return threading.Lock()


def choose_group(counts: dict) -> str:
    """Group that minimizes (n+1)/weight; ties broken at random."""
    best_val = None
    candidates = []
    for g in GROUPS:
        val = (counts.get(g, 0) + 1) / WEIGHTS[g]
        if best_val is None or val < best_val - 1e-9:
            best_val = val
            candidates = [g]
        elif abs(val - best_val) <= 1e-9:
            candidates.append(g)
    return random.choice(candidates)


# =============================================================================
# STORAGE BACKENDS
# =============================================================================

@st.cache_resource
def _supabase_client():
    from supabase import create_client
    return create_client(
        st.secrets["supabase"]["url"],
        st.secrets["supabase"]["service_key"],
    )


class SupabaseStorage:
    def __init__(self):
        self.client = _supabase_client()
        self.bucket = st.secrets["supabase"].get("bucket", "videos")

    def counts(self) -> dict:
        res = self.client.table("counts").select("grp, n").execute()
        return {r["grp"]: int(r["n"]) for r in (res.data or [])}

    def assign_group(self, factors=None) -> str:
        # Atomic + race-free: minimization runs inside the Postgres function.
        res = self.client.rpc("assign_group_min", {"p_factors": factors or {}}).execute()
        data = res.data
        if isinstance(data, list):
            data = data[0] if data else None
        return data

    def strata(self) -> list:
        res = self.client.table("strata_counts").select("*").execute()
        return res.data or []

    def get_stimuli(self) -> list:
        res = self.client.table("stimuli").select("*").order("sort_order").execute()
        return res.data or []

    def signed_url(self, path: str, expires: int = 7200) -> str:
        """Time-limited URL for an object in a PRIVATE bucket (service_role signs it)."""
        r = self.client.storage.from_(self.bucket).create_signed_url(path, expires)
        url = r.get("signedURL") or r.get("signedUrl") or r.get("signed_url")
        if url and url.startswith("/"):
            url = st.secrets["supabase"]["url"].rstrip("/") + url
        return url

    def save_response(self, row: dict):
        data = {h: str(row.get(h, "")) for h in RESPONSE_HEADERS}
        data["grp"] = data.pop("group")  # DB column is "grp"
        self.client.table("responses").insert(data).execute()

    def get_responses(self) -> list:
        """Return completed participant responses for the researcher dashboard."""
        res = (
            self.client
            .table("responses")
            .select(
                "participant_id, grp, task_json, task_score, "
                "complete, timestamp"
            )
            .eq("complete", "Sim")
            .execute()
        )
        return res.data or []


class SQLiteStorage:
    """Local fallback for testing only."""

    def __init__(self, path="responses.db"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        c = self.conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS counts (grp TEXT PRIMARY KEY, n INTEGER)")
        for g in GROUPS:
            c.execute("INSERT OR IGNORE INTO counts (grp, n) VALUES (?, 0)", (g,))
        cols = ", ".join(f'"{h}" TEXT' for h in RESPONSE_HEADERS)
        c.execute(f"CREATE TABLE IF NOT EXISTS responses ({cols})")
        self.conn.commit()

    def counts(self) -> dict:
        return {g: n for g, n in self.conn.execute("SELECT grp, n FROM counts")}

    def assign_group(self, factors=None) -> str:
        with get_lock():
            counts = self.counts()
            group = choose_group(counts)
            self.conn.execute("UPDATE counts SET n = n + 1 WHERE grp = ?", (group,))
            self.conn.commit()
            return group

    def strata(self) -> list:
        return []

    def get_stimuli(self) -> list:
        return []

    def save_response(self, row: dict):
        ph = ", ".join("?" * len(RESPONSE_HEADERS))
        self.conn.execute(
            f"INSERT INTO responses VALUES ({ph})",
            [str(row.get(h, "")) for h in RESPONSE_HEADERS],
        )
        self.conn.commit()

    def get_responses(self) -> list:
        """Return completed participant responses for the researcher dashboard."""
        columns = [
            "participant_id", "group", "task_json", "task_score",
            "complete", "timestamp"
        ]
        rows = self.conn.execute(
            "SELECT participant_id, group, task_json, task_score, "
            "complete, timestamp FROM responses WHERE complete = ?",
            ("Sim",)
        ).fetchall()
        return [dict(zip(columns, row)) for row in rows]


def _has_supabase() -> bool:
    try:
        return "supabase" in st.secrets
    except Exception:
        return False


@st.cache_resource
def get_storage():
    if _has_supabase():
        return SupabaseStorage(), "supabase"
    return SQLiteStorage(), "sqlite"


# =============================================================================
# VIDEO / STIMULI HELPERS
# =============================================================================

@st.cache_data(show_spinner=False, ttl=3000)
def _signed_url(path: str):
    storage, _ = get_storage()
    try:
        return storage.signed_url(path)
    except Exception:
        return None


def video_url(video_value: str):
    """A time-limited signed Storage URL (works with a PRIVATE bucket), or the
    value itself if it is already a full URL."""
    v = str(video_value).strip()
    if not v:
        return None
    if v.startswith("http://") or v.startswith("https://"):
        return v
    storage, mode = get_storage()
    if mode != "supabase":
        return None
    return _signed_url(v)


def parse_prob(value):
    """Return the deepfake probability as a float in [0, 1], or None."""
    try:
        x = float(str(value).replace("%", "").replace(",", ".").strip())
    except (TypeError, ValueError):
        return None
    return x / 100.0 if x > 1 else x


def model_verdict(prob_deepfake_raw):
    """Return (verdict, confidence_in_verdict) or None from a deepfake probability.
    Verdict is 'Deepfake' when P(deepfake) >= 0.5, else 'Real'; confidence is the
    probability of the chosen verdict."""
    p = parse_prob(prob_deepfake_raw)
    if p is None:
        return None
    return ("Deepfake", p) if p >= 0.5 else ("Real", 1 - p)


def label_matches(ground_truth: str, answer: str) -> bool:
    gt = str(ground_truth).strip().lower()
    if gt in {"real", "autentico", "autêntico", "authentic", "genuino", "genuíno"}:
        return answer == LABEL_AUTENTICO
    if gt in {"deepfake", "fake", "ia", "gerado por ia", "manipulado", "falso"}:
        return answer == LABEL_DEEPFAKE
    return False


def load_stimuli():
    """Return (list_of_stimulus_dicts, mode). In test mode, returns placeholders."""
    storage, mode = get_storage()
    if mode == "supabase":
        try:
            return storage.get_stimuli(), "supabase"
        except Exception:
            return [], "supabase"
    dummies = [
        {"sort_order": i, "video": "", "label": "",
         "prob_deepfake": "0.5",
         "explanation": f"〔explicação XAI de exemplo para o vídeo {i}〕"}
        for i in range(1, 4)
    ]
    return dummies, "test"


# =============================================================================
# CERTIFICATE  (generated on the fly; the participant's NAME is never stored)
# =============================================================================

def _emitido_em() -> str:
    now = datetime.now(ZoneInfo("America/Sao_Paulo")) if ZoneInfo else datetime.now()
    return now.strftime("%d/%m/%Y")


def _ano() -> str:
    now = datetime.now(ZoneInfo("America/Sao_Paulo")) if ZoneInfo else datetime.now()
    return str(now.year)


def _slug(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", (name or "").strip()).strip("-").lower()
    return s[:40] or "participacao"


def build_certificate_pdf(name: str) -> bytes:
    """Build the participation certificate PDF in memory and return its bytes.
    Requires reportlab (raises ImportError if unavailable)."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.colors import HexColor
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.pdfgen import canvas
    from reportlab.platypus import Paragraph

    navy = HexColor("#1e3a8a")       # InteliGente blue
    gray = HexColor("#374151")
    name_dark = HexColor("#111827")
    ano = _ano()

    buf = io.BytesIO()
    W, H = A4
    cx = W / 2.0
    c = canvas.Canvas(buf, pagesize=A4)

    # faint centered watermark (drawn first, behind everything)
    if os.path.exists(WATERMARK_PATH):
        try:
            wm_w = 130 * mm
            wm_h = wm_w * 992.0 / 1403.0   # logo.png aspect ratio
            c.drawImage(WATERMARK_PATH, cx - wm_w / 2.0, H / 2.0 - wm_h / 2.0,
                        width=wm_w, height=wm_h, mask="auto")
        except Exception:
            pass

    # elegant single border
    c.setStrokeColor(navy)
    c.setLineWidth(2.5)
    c.rect(10 * mm, 10 * mm, W - 20 * mm, H - 20 * mm)

    def para(text, size, color, leading=None, bold=False, align=TA_CENTER):
        style = ParagraphStyle(
            "s", fontName=("Times-Bold" if bold else "Times-Roman"),
            fontSize=size, leading=leading or size * 1.35,
            textColor=color, alignment=align)
        return Paragraph(text, style)

    def draw_centered(p, cx, top_y, max_w):
        w, h = p.wrapOn(c, max_w, H)
        p.drawOn(c, cx - w / 2.0, top_y - h)
        return h

    content_w = W - 50 * mm
    y = H - 30 * mm

    y -= draw_centered(para("CERTIFICADO", 32, navy, bold=True), cx, y, content_w)
    y -= 2 * mm
    y -= draw_centered(para("Declaração de Participação", 13, gray), cx, y, content_w)

    y -= 16 * mm
    y -= draw_centered(para("Certificamos que", 12, gray), cx, y, content_w)
    y -= 5 * mm
    y -= draw_centered(para((name or "Participante").strip(), 24, name_dark, bold=True),
                       cx, y, content_w)

    y -= 9 * mm
    body = (
        f'participou, na condição de <b>voluntário(a)</b>, da pesquisa intitulada '
        f'<b>“{CERT_STUDY_TITLE}”</b>, durante o ano de <b>{ano}</b>, contribuindo '
        f'para o desenvolvimento das atividades previstas no projeto.'
    )
    y -= draw_centered(para(body, 12, gray, leading=18), cx, y, content_w)

    y -= 5 * mm
    body2 = (
        f'A participação é válida para o <b>ano letivo de {ano}</b> e corresponde a '
        f'<b>5 (cinco) pontos para fins de Atividades Complementares, no Grupo 2</b>, '
        f'conforme regulamentação institucional vigente.'
    )
    y -= draw_centered(para(body2, 12, gray, leading=18), cx, y, content_w)

    y -= 8 * mm
    y -= draw_centered(para("<b>Responsáveis pela pesquisa:</b>", 11.5, gray), cx, y, content_w)
    y -= 1.5 * mm
    y -= draw_centered(para("<b>Prof. Dr. Rogério Pozza</b> – Orientador/UTFPR-CP",
                            11.5, gray), cx, y, content_w)
    y -= 1 * mm
    y -= draw_centered(para("<b>Prof. Dr. Robson Bonidia</b> – Co-Orientador/UTFPR-CP",
                            11.5, gray), cx, y, content_w)

    y -= 7 * mm
    y -= draw_centered(
        para("Por ser verdade, firmamos o presente certificado para os devidos fins.",
             11.5, gray), cx, y, content_w)

    y -= 6 * mm
    y -= draw_centered(para(f"Emitido em {_emitido_em()}", 10.5, gray), cx, y, content_w)

    # seal (centered near the bottom, slight stamp-like rotation)
    if os.path.exists(SEAL_PATH):
        try:
            seal_w = 40 * mm
            c.saveState()
            c.translate(cx, 42 * mm)      # seal center
            c.rotate(-5)
            c.drawImage(SEAL_PATH, -seal_w / 2.0, -seal_w / 2.0,
                        width=seal_w, height=seal_w,
                        mask="auto", preserveAspectRatio=True)
            c.restoreState()
        except Exception:
            pass

    c.showPage()
    c.save()
    return buf.getvalue()


def _certificate_block():
    """Name field + PDF download. The name is used only to render the certificate
    in-session and is NEVER added to the response row or saved anywhere."""
    ss = st.session_state
    st.divider()
    st.subheader("Certificado de participação")
    st.write(
        "Se desejar, gere um certificado de participação. O nome informado é usado "
        "**apenas** para gerar o PDF e **não é armazenado** pela pesquisa."
    )
    name = st.text_input("Nome completo (como deve aparecer no certificado)",
                         key="cert_name_input")
    if st.button("Gerar certificado"):
        if not name.strip():
            st.error("Digite seu nome para gerar o certificado.")
        else:
            try:
                ss.cert_bytes = build_certificate_pdf(name)
                ss.cert_file = f"certificado_{_slug(name)}.pdf"
            except ImportError:
                st.error("A geração de certificado requer a biblioteca 'reportlab'. "
                         "Adicione 'reportlab' ao requirements.txt.")
            except Exception as e:  # noqa: BLE001
                st.error("Não foi possível gerar o certificado.")
                st.caption(f"Detalhe técnico: {e}")

    if ss.get("cert_bytes"):
        st.download_button(
            "⬇️ Baixar certificado (PDF)",
            data=ss.cert_bytes,
            file_name=ss.get("cert_file", "certificado.pdf"),
            mime="application/pdf",
            type="primary",
        )


# =============================================================================
# STATE AND NAVIGATION
# =============================================================================

def init_state():
    ss = st.session_state
    ss.setdefault("step", "intro")
    ss.setdefault("pid", str(uuid.uuid4()))
    ss.setdefault("group", None)
    ss.setdefault("data", {})
    ss.setdefault("saved", False)
    # Per-video task state: one video is shown and submitted at a time.
    ss.setdefault("task_index", 0)
    ss.setdefault("task_started_at", None)
    ss.setdefault("task_answers", {})
    ss.setdefault("task_timings", {})
    ss.setdefault("task_order", None)  # per-participant randomized video order


def go_to(step):
    st.session_state.step = step
    st.rerun()


def effective_group():
    """Return the group whose features should be active in this session.

    The researcher can temporarily override the assigned group from the admin
    panel. The override is session-local and does not affect database assignment.
    """
    debug_group = st.session_state.get("debug_group", "Normal")
    if debug_group != "Normal":
        return debug_group
    return st.session_state.get("group")


def next_after_demographics():
    return "material" if effective_group() in GROUPS_WITH_MATERIAL else "task"


# =============================================================================
# SCREENS  (headings, questions and buttons are Portuguese on purpose)
# =============================================================================

def screen_intro(mode):
    st.title("Pesquisa: identificação de vídeos autênticos e deepfakes")
    st.write(
        "Obrigado pelo seu interesse. Nesta pesquisa você responderá a algumas "
        "perguntas e assistirá a vídeos de rostos, indicando se são autênticos ou "
        "gerados por inteligência artificial. A participação é anônima e leva "
        "cerca de **15 minutos**."
    )
    if mode == "sqlite":
        st.warning(
            "⚠️ **Modo de teste local (SQLite).** Configure o Supabase antes de "
            "coletar dados reais."
        )
    if st.button("Começar", type="primary"):
        go_to("consent")


def screen_consent():
    st.header("Termo de Consentimento Livre e Esclarecido (TCLE)")
    st.markdown(CONSENT_TEXT)

    choice = st.radio(
        "Declaro que li e compreendi o TCLE acima e concordo em participar da pesquisa. *",
        ["Sim, li, compreendi e concordo em participar.", "Não desejo participar."],
        index=None,
    )
    if st.button("Continuar", type="primary"):
        if choice is None:
            st.error("Selecione uma opção para continuar.")
        elif choice.startswith("Não"):
            go_to("declined")
        else:
            st.session_state.data["consented"] = "Sim"
            go_to("demographics")


def screen_declined():
    st.header("Participação não iniciada")
    st.write("Tudo bem. Obrigado pelo seu tempo — você pode fechar esta janela.")


def screen_demographics():
    st.header("Seção 1 — Questionário sociodemográfico")
    st.caption("Campos com * são obrigatórios.")
    with st.form("demographics"):
        age = st.radio("1.1 Qual é a sua faixa etária? *",
                       ["<18", "18–24", "25–34", "35–44", "45–54", "55 ou mais"], index=None)
        gender = st.radio("1.2 Com qual gênero você se identifica?",
                          ["Feminino", "Masculino", "Não-binário", "Agênero / Gênero Fluido", "Prefiro descrever de outra forma", "Prefiro não responder"], index=None)
        education = st.radio("1.3 Qual seu grau de escolaridade? *",
                             ["Ensino fundamental incompleto", "Ensino fundamental completo",
                              "Ensino médio incompleto", "Ensino médio completo", "Ensino técnico ou profissionalizante", "Ensino superior incompleto", "Ensino superior completo", "Pós-graduação"], index=None)
        ai_familiarity = st.radio(
            "1.4 Familiaridade com Inteligência Artificial? *  (1 = Nenhuma … 5 = Especialista)",
            [1, 2, 3, 4, 5], index=None, horizontal=True)
        deepfake_knowledge = st.radio("1.5 Conhecimento prévio sobre *deepfakes*? *",
                                      ["Nenhum", "Algum", "Bastante"], index=None)
        social_media_freq = st.radio("1.6 Com que frequência você usa redes sociais?",
                                     ["Raramente", "Semanalmente", "Diariamente", "Várias vezes ao dia"],
                                     index=None)
        used_ai = st.radio("1.7 Você já usou alguma ferramenta baseada em IA?",
                           ["Sim", "Não"], index=None)
        submit = st.form_submit_button("Continuar", type="primary")

    if submit:
        missing = [q for q, v in [("1.1", age), ("1.3", education),
                                  ("1.4", ai_familiarity), ("1.5", deepfake_knowledge)] if v is None]
        if missing:
            st.error("Responda às perguntas obrigatórias: " + ", ".join(missing))
        else:
            st.session_state.data.update({
                "age_range": age, "gender": gender, "education": education,
                "ai_familiarity": ai_familiarity, "deepfake_knowledge": deepfake_knowledge,
                "social_media_freq": social_media_freq, "used_ai": used_ai,
            })
            # Assign AFTER demographics so the groups stay balanced on them.
            if st.session_state.group is None:
                storage, _ = get_storage()
                factors = {f: str(st.session_state.data.get(f))
                           for f in BALANCE_FACTORS
                           if st.session_state.data.get(f) is not None}
                st.session_state.group = storage.assign_group(factors)
            go_to(next_after_demographics())


def screen_material():
    st.header("Como identificar um vídeo manipulado (deepfake)")
    st.write(
        "Deepfakes de alta qualidade quase sempre alteram o **rosto** — é ali que "
        "ficam a maioria das pistas. Assista ao vídeo com calma e, se possível, em "
        "tela cheia e mais de uma vez. Use a lista abaixo como um guia do que observar."
    )
    st.markdown(
        "**1. Comece pelo rosto**\n"
        "- É onde a maioria das manipulações acontece. Olhe primeiro para lá.\n\n"
        "**2. Pele das bochechas e da testa**\n"
        "- A pele parece lisa demais ou enrugada demais?\n"
        "- A \"idade\" da pele combina com a do cabelo e dos olhos? (Ex.: rosto muito "
        "liso com cabelo grisalho é suspeito.)\n\n"
        "**3. Olhos e sobrancelhas**\n"
        "- As sombras aparecem onde você esperaria, de acordo com a luz da cena?\n"
        "- O reflexo de luz nos **dois** olhos é parecido? Reflexos diferentes ou "
        "ausentes chamam atenção.\n"
        "- O olhar parece natural?\n\n"
        "**4. Piscadas**\n"
        "- A pessoa pisca de forma natural — nem demais, nem de menos, nem de modo "
        "estranho?\n\n"
        "**5. Óculos**\n"
        "- Há reflexo (brilho) nas lentes? É reflexo de mais ou de menos?\n"
        "- Quando a pessoa move a cabeça, o reflexo muda de forma natural? Deepfakes "
        "costumam errar a física da luz.\n\n"
        "**6. Barba, bigode e costeletas**\n"
        "- Os pelos do rosto parecem reais?\n"
        "- Parece que foram adicionados ou removidos? As bordas dos pelos estão "
        "naturais?\n\n"
        "**7. Pintas e marcas na pele**\n"
        "- As pintas parecem reais e continuam no mesmo lugar durante todo o vídeo? "
        "Marcas que somem, aparecem ou \"tremem\" são um alerta.\n\n"
        "**8. Boca, lábios e dentes**\n"
        "- Os lábios acompanham o áudio? Muitos deepfakes fazem \"dublagem\" "
        "(lip sync) e erram esse encaixe.\n"
        "- Os dentes estão bem definidos ou parecem uma mancha branca? Dentes "
        "borrados ou sem separação são suspeitos.\n\n"
        "**9. Orelhas e brincos**\n"
        "- As orelhas são simétricas e bem formadas? Brincos que deformam ou mudam "
        "ao longo do vídeo chamam atenção.\n\n"
        "**10. Onde o rosto encontra o cabelo, o pescoço e o fundo**\n"
        "- Há bordas borradas, \"costuras\" ou tremores nessas junções?\n"
        "- O tom de pele do rosto combina com o do pescoço e das mãos?\n\n"
        "**11. Iluminação e sombras**\n"
        "- A luz no rosto combina com a luz do ambiente?\n"
        "- As sombras seguem a mesma direção da fonte de luz?\n\n"
        "**12. Movimento ao longo do vídeo**\n"
        "- A imagem \"pisca\", treme ou muda de cor/nitidez de um instante para outro?\n"
        "- A qualidade piora quando a pessoa vira o rosto de lado? Deepfakes costumam "
        "falhar em perfil e em movimentos rápidos.\n\n"
        "**13. Som e expressão** (se o vídeo tiver áudio)\n"
        "- A voz soa natural e combina com o movimento da boca?\n"
        "- A emoção do rosto combina com o tom de voz e com o que está sendo dito?"
    )
    st.warning(
        "Lembre-se: **nenhum sinal isolado prova** que o vídeo é falso. Alguns vídeos "
        "reais têm imperfeições e alguns deepfakes são muito convincentes. Considere "
        "o **conjunto** das pistas."
    )
    st.info("Agora, você verá alguns vídeos e precisará distinguir se são "
            "**legítimos** ou **deepfakes**.")
    if st.button("Concluí o treinamento e desejo continuar", type="primary"):
        go_to("check")


def screen_check():
    st.header("Seção 2 — Verificação de aprendizagem")
    with st.form("check"):
        q1 = st.radio("2.1 Qual é um sinal comum de que um vídeo pode ter sido manipulado por IA? *",
                      OPTIONS_2_1, index=None)
        q2 = st.radio("2.2 A inspeção visual, sozinha, é suficiente para garantir que um vídeo é autêntico. *",
                      ["Verdadeiro", "Falso"], index=None)
        q3 = st.radio("2.3 Ao avaliar um possível *deepfake*, além de observar o vídeo, também é importante: *",
                      OPTIONS_2_3, index=None)
        submit = st.form_submit_button("Continuar", type="primary")

    if submit:
        if None in (q1, q2, q3):
            st.error("Responda a todas as perguntas para continuar.")
        else:
            score = int(q1 == ANSWER_2_1) + int(q2 == ANSWER_2_2) + int(q3 == ANSWER_2_3)
            st.session_state.data.update({
                "check_2_1": q1, "check_2_2": q2, "check_2_3": q3, "check_score": score,
            })
            go_to("task")


def screen_task():
    """Show one stimulus at a time and collect classification, confidence, and timing."""
    ss = st.session_state
    group = effective_group()
    show_ai = group in GROUPS_WITH_AI_EXPLANATION

    st.header("Tarefa — assista e classifique os vídeos")
    st.write(
        "Assista a cada vídeo e indique se você o considera **legítimo** ou "
        "**deepfake**. Após a classificação, informe sua confiança."
    )

    stimuli, source = load_stimuli()
    if not stimuli:
        st.warning(
            "Nenhum vídeo configurado. Preencha a tabela **stimuli** no Supabase "
            "(colunas: sort_order, video, label, prob_deepfake, explanation) e envie "
            "os arquivos para o bucket de Storage."
        )
        return

    if source == "test":
        st.caption("〔Modo de teste: vídeos indisponíveis; exibindo apenas a estrutura.〕")

    # Safety check if the session contains an invalid index.
    if ss.task_index >= len(stimuli):
        go_to("final")
        return

    # Random per-participant order (controls for order effects). The video KEY
    # stays tied to the stimulus's sort_order position, so answers remain
    # comparable across participants and the admin/score logic keeps working.
    if not ss.task_order or len(ss.task_order) != len(stimuli):
        order = list(range(len(stimuli)))
        random.shuffle(order)
        ss.task_order = order

    pos = ss.task_index                 # 0-based display position (what the person sees)
    real_idx = ss.task_order[pos]       # index into the sort_order stimuli list
    stimulus = stimuli[real_idx]
    video_number = real_idx + 1         # STABLE id (sort_order position) -> stable key
    video_key = f"vid_{video_number}"

    st.progress((pos + 1) / len(stimuli), text=f"Vídeo {pos + 1} de {len(stimuli)}")
    st.subheader(f"Vídeo {pos + 1}")

    url = video_url(stimulus.get("video", ""))
    if url:
        st.video(url)
    else:
        st.markdown(
            "<div style='width:100%;max-width:480px;height:240px;background:#eee;"
            "border-radius:8px;display:flex;align-items:center;justify-content:center;"
            f"color:#888'>vídeo {pos + 1}</div>",
            unsafe_allow_html=True,
        )

    if show_ai:
        mv = model_verdict(stimulus.get("prob_deepfake"))
        if mv:
            verdict, model_confidence = mv
            st.info(
                f"🔎 **Resultado do modelo de detecção:** "
                f"{verdict} ({model_confidence:.0%})"
            )

        explanation = str(stimulus.get("explanation", "")).strip()
        if explanation:
            st.markdown(f"**Explicação da IA:** {explanation}")

    # Start the timer once, when this video is first rendered.
    if ss.task_started_at is None:
        ss.task_started_at = time.perf_counter()

    with st.form(f"task_video_{video_number}"):
        answer = st.radio(
            "Classificação do vídeo *",
            TASK_OPTIONS,
            index=None,
            horizontal=True,
        )
        confidence = st.radio(
            "Qual é o seu nível de confiança nesta classificação? "
            "(1 = Nada confiante … 5 = Muito confiante) *",
            [1, 2, 3, 4, 5],
            index=None,
            horizontal=True,
        )
        submit_label = (
            "Enviar classificação e avançar"
            if pos + 1 < len(stimuli)
            else "Enviar classificação e finalizar tarefa"
        )
        submit = st.form_submit_button(submit_label, type="primary")

    if submit:
        if answer is None or confidence is None:
            st.error("Selecione a classificação e o nível de confiança.")
            return

        elapsed_seconds = round(time.perf_counter() - ss.task_started_at, 3)
        ground_truth = str(stimulus.get("label", "")).strip()

        ss.task_answers[video_key] = {
            "answer": answer,
            "confidence": confidence,
        }
        ss.task_timings[video_key] = {
            "response_time_seconds": elapsed_seconds,
            "video_number": video_number,      # stable id (sort_order position)
            "display_position": pos + 1,        # where it appeared for this participant
        }

        if ground_truth:
            ss.task_answers[video_key]["correct"] = label_matches(
                ground_truth, answer
            )

        # Keep the collected values in the existing participant data structure.
        ss.data["task_json"] = json.dumps(ss.task_answers, ensure_ascii=False)
        ss.data["task_timings_json"] = json.dumps(
            ss.task_timings, ensure_ascii=False
        )

        if all(str(s.get("label", "")).strip() for s in stimuli):
            ss.data["task_score"] = sum(
                1
                for i, s in enumerate(stimuli, start=1)
                if ss.task_answers.get(f"vid_{i}", {}).get("correct") is True
            )

        ss.task_index += 1
        ss.task_started_at = None

        if ss.task_index >= len(stimuli):
            go_to("final")
        else:
            st.rerun()

def screen_final():
    st.header("Questionários finais")
    st.caption("〔Placeholder — insira aqui os questionários finais da sua pesquisa.〕")
    with st.form("final"):
        confidence = st.radio("Quão confiante você ficou nas suas classificações? (1 = Nada … 5 = Muito)",
                              [1, 2, 3, 4, 5], index=None, horizontal=True)
        difficulty = st.radio("Quão difícil foi a tarefa? (1 = Muito fácil … 5 = Muito difícil)",
                              [1, 2, 3, 4, 5], index=None, horizontal=True)
        comments = st.text_area("Comentários (opcional)")
        submit = st.form_submit_button("Enviar respostas", type="primary")

    if submit:
        st.session_state.data["final_json"] = json.dumps(
            {"confidence": confidence, "difficulty": difficulty, "comments": comments},
            ensure_ascii=False,
        )
        go_to("end")


def screen_end():
    ss = st.session_state
    if not ss.saved:
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "participant_id": ss.pid,
            "group": ss.group,
            "complete": "Sim",
            **ss.data,
        }
        try:
            storage, _ = get_storage()
            storage.save_response(row)
            ss.saved = True
        except Exception as e:  # noqa: BLE001
            st.error("Não foi possível salvar suas respostas. Tente novamente em instantes.")
            st.caption(f"Detalhe técnico: {e}")
            if st.button("Tentar novamente"):
                st.rerun()
            return

    st.header("Obrigado por participar! ✅")
    st.write("Suas respostas foram registradas de forma anônima.")
    st.success(f"Grupo atribuído: **{ss.group}**")
    if ss.get("debug_group", "Normal") != "Normal":
        st.caption(f"Configuração de depuração aplicada: **{ss.debug_group}**")
    st.caption("Anote esta informação caso precise informá-la à equipe da pesquisa.")
    st.write("Você pode fechar esta janela.")
    if ss.get("debug_group", "Normal") != "Normal":
        st.info(
            "Modo de depuração ativo: "
            f"as características exibidas foram **{ss.debug_group}**. "
            "Essa execução não deve ser usada como resposta de participante."
        )

    _certificate_block()


def screen_admin():
    """Password-protected researcher dashboard."""
    st.header("Researcher panel")

    # -------------------------------------------------------------------------
    # Researcher-only feature debugging
    # -------------------------------------------------------------------------
    st.subheader("Feature debugging / participant preview")
    st.caption(
        "Choose which group-specific features should be active in a preview. "
        "This override is local to the current session and does not change the "
        "participant's assigned group or the balancing counters."
    )

    current_debug_group = st.session_state.get("debug_group", "Normal")
    debug_group = st.selectbox(
        "Feature configuration",
        options=DEBUG_GROUP_OPTIONS,
        index=DEBUG_GROUP_OPTIONS.index(current_debug_group)
        if current_debug_group in DEBUG_GROUP_OPTIONS else 0,
        key="admin_debug_group",
        help=(
            "Normal uses the assigned group. Todos enables Checklist material, "
            "the learning check, model probability, and XAI explanation."
        ),
    )
    st.session_state["debug_group"] = debug_group

    debug_features = {
        "Assigned group used": st.session_state.get("group") or "Not assigned yet",
        "Active feature group": debug_group,
        "Checklist material": "Enabled" if debug_group in GROUPS_WITH_MATERIAL else "Disabled",
        "Learning check": "Enabled" if debug_group in GROUPS_WITH_MATERIAL else "Disabled",
        "Model probability": "Enabled" if debug_group in GROUPS_WITH_AI_EXPLANATION else "Disabled",
        "XAI explanation": "Enabled" if debug_group in GROUPS_WITH_AI_EXPLANATION else "Disabled",
    }
    st.table([{"Feature": key, "Status": value} for key, value in debug_features.items()])

    if st.button("Launch participant preview", type="primary"):
        # Start a fresh preview while preserving the selected debug configuration.
        st.session_state["step"] = "intro"
        st.session_state["pid"] = "DEBUG-" + str(uuid.uuid4())
        st.session_state["group"] = None
        st.session_state["data"] = {}
        st.session_state["saved"] = False
        st.session_state["task_index"] = 0
        st.session_state["task_started_at"] = None
        st.session_state["task_answers"] = {}
        st.session_state["task_timings"] = {}
        st.session_state["task_order"] = None
        st.query_params.clear()
        st.rerun()

    st.divider()

    try:
        storage, mode = get_storage()
        counts = storage.counts()
    except Exception as e:  # noqa: BLE001
        st.error(f"Failed to read counts: {e}")
        return

    st.subheader("Participant distribution")
    metric_cols = st.columns(len(GROUPS))
    for col, group in zip(metric_cols, GROUPS):
        col.metric(group, counts.get(group, 0))

    st.metric("Total assigned", sum(counts.values()))
    st.caption(f"Backend: {mode} · target weights: {WEIGHTS}")
    st.divider()

    # Demographic balance
    try:
        rows = storage.strata() if hasattr(storage, "strata") else []
    except Exception:
        rows = []

    if rows:
        st.subheader("Demographic balance")
        by_factor = collections.defaultdict(
            lambda: collections.defaultdict(dict)
        )
        for row in rows:
            by_factor[row["factor"]][str(row["level"])][row["grp"]] = row["n"]

        for factor in sorted(by_factor):
            st.caption(factor)
            table = [
                {
                    "level": level,
                    **{group: levels.get(group, 0) for group in GROUPS},
                }
                for level, levels in sorted(by_factor[factor].items())
            ]
            st.table(table)

    st.divider()
    st.subheader("Video inference comparison")

    try:
        responses = storage.get_responses()
    except Exception as e:  # noqa: BLE001
        st.error(f"Failed to read participant responses: {e}")
        return

    if not responses:
        st.info("No completed participant responses found.")
        return

    stimuli, _ = load_stimuli()
    if not stimuli:
        st.warning("No stimuli configured.")
        return

    selected_groups = st.multiselect(
        "Groups to compare",
        options=GROUPS,
        default=GROUPS,
        key="admin_selected_groups",
    )
    if not selected_groups:
        st.info("Select at least one group to compare.")
        return

    group_answers = {
        group: collections.defaultdict(list)
        for group in GROUPS
    }

    for response in responses:
        group = response.get("grp") or response.get("group")
        if group not in GROUPS:
            continue

        try:
            task = json.loads(response.get("task_json") or "{}")
        except (TypeError, json.JSONDecodeError):
            continue

        if not isinstance(task, dict):
            continue

        for video_key, answer_data in task.items():
            # New format: {"answer": "...", "confidence": 1, ...}
            if isinstance(answer_data, dict):
                answer = answer_data.get("answer")
            else:
                # Backward compatibility with responses from the old app.
                answer = answer_data

            if answer in TASK_OPTIONS:
                group_answers[group][video_key].append(answer)

    show_videos = st.checkbox(
        "Show videos in the Admin panel",
        value=True,
        key="admin_show_videos",
    )
    show_researcher_info = st.checkbox(
        "Show ground truth, model probability, and XAI explanation",
        value=True,
        key="admin_show_researcher_info",
    )

    st.caption(
        "Percentages use submitted answers for each video and group. "
        "Missing answers are excluded."
    )

    for idx, stimulus in enumerate(stimuli, start=1):
        video_key = f"vid_{idx}"
        st.markdown(f"## Video {idx}")

        if show_videos:
            url = video_url(stimulus.get("video", ""))
            if url:
                st.video(url)
            else:
                st.warning("Video unavailable for this stimulus.")

        if show_researcher_info:
            with st.expander("Researcher information", expanded=False):
                st.write("Ground truth:", stimulus.get("label", ""))
                st.write(
                    "Model deepfake probability:",
                    stimulus.get("prob_deepfake", ""),
                )
                explanation = str(stimulus.get("explanation", "")).strip()
                if explanation:
                    st.markdown("**XAI explanation:**")
                    st.write(explanation)

        comparison = []
        for group in selected_groups:
            answers = group_answers[group][video_key]
            n = len(answers)
            n_real = answers.count(LABEL_AUTENTICO)
            n_fake = answers.count(LABEL_DEEPFAKE)

            comparison.append({
                "Group": group,
                "Responses": n,
                "Legitimate (n)": n_real,
                "Legitimate (%)": round(n_real / n * 100, 2) if n else 0.0,
                "Deepfake (n)": n_fake,
                "Deepfake (%)": round(n_fake / n * 100, 2) if n else 0.0,
            })

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True,
        )
        st.divider()

def render_admin_gate():
    """Password-gated researcher panel. The password lives in secrets, never in the URL."""
    ss = st.session_state
    try:
        expected = st.secrets.get("admin", {}).get("key")
    except Exception:
        expected = None
    if not expected:
        st.error("Painel indisponível: defina admin.key nos secrets.")
        return
    if not ss.get("admin_ok"):
        st.header("Acesso restrito")
        pwd = st.text_input("Senha do pesquisador", type="password")
        if st.button("Entrar", type="primary"):
            if pwd == expected:
                ss.admin_ok = True
                st.rerun()
            else:
                st.error("Senha incorreta.")
        return
    screen_admin()


# =============================================================================
# ROUTER
# =============================================================================

def main():
    st.set_page_config(page_title="Pesquisa deepfakes", page_icon="🔎")
    init_state()
    _, mode = get_storage()

    if "admin" in st.query_params:  # researcher panel: ?admin  (password asked on the page)
        render_admin_gate()
        return

    screens = {
        "intro": lambda: screen_intro(mode),
        "consent": screen_consent,
        "declined": screen_declined,
        "demographics": screen_demographics,
        "material": screen_material,
        "check": screen_check,
        "task": screen_task,
        "final": screen_final,
        "end": screen_end,
    }
    render = screens.get(st.session_state.step)
    if render is None:
        st.session_state.step = "intro"
        st.rerun()
    else:
        render()


if __name__ == "__main__":
    main()
