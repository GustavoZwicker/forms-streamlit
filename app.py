# -*- coding: utf-8 -*-
"""
Data-collection app for the deepfake-detection study.

Groups (balanced in real time):
  - "Controle"  (group 1): no training. Watches the videos and classifies them.
  - "Checklist" (group 2): checklist training + learning check, then classifies.
  - "XAI"       (group 3): classifies videos shown together with the model's
                           deepfake probability and an XAI/LLM explanation.

Flow:
  intro -> consent (TCLE) -> [GROUP ASSIGNMENT] -> demographics
        -> (Checklist) training -> (Checklist) learning check
        -> classification task (videos from Google Drive) -> final questionnaires -> end

Assignment: each consenting participant goes to the group furthest below its target
proportion (WEIGHTS); equal weights (1:1:1) keep the three samples the same size.
The read-choose-increment step is serialized by a process-wide lock (race-free on
Streamlit Community Cloud's single instance).

Storage & media:
  - Responses + running counts -> Google Sheets (durable).
  - Video stimuli              -> Google Drive folder, fetched via the service account.
  - Stimulus metadata (which video, model probability, XAI explanation, ground-truth
    label) -> a "stimuli" tab in the same spreadsheet, filled in by the researcher.
  Without credentials the app falls back to local SQLite + placeholder stimuli
  (testing only; Community Cloud wipes local files).

NOTE: participant-facing text is Portuguese on purpose; the code is English.
"""

import io
import json
import os
import random
import sqlite3
import tempfile
import threading
import uuid
from datetime import datetime, timezone

import streamlit as st

# =============================================================================
# CONFIGURATION
# =============================================================================

# Internal group codes (also stored in the Sheet). See module docstring.
GROUPS = ["Controle", "Checklist", "XAI"]

# Target proportion between groups. Equal (1:1:1) => same-size samples.
WEIGHTS = {"Controle": 1, "Checklist": 1, "XAI": 1}

# Only the Checklist group gets the training material + learning check.
GROUPS_WITH_MATERIAL = {"Checklist"}
# Only the XAI group sees the model probability + explanation during the task.
GROUPS_WITH_AI_EXPLANATION = {"XAI"}

# Classification options shown for each video.
LABEL_AUTENTICO = "Autêntico"
LABEL_DEEPFAKE = "Gerado por IA"
TASK_OPTIONS = [LABEL_AUTENTICO, LABEL_DEEPFAKE]

# Columns stored per participant (English schema).
RESPONSE_HEADERS = [
    "timestamp", "participant_id", "group", "consented",
    "age_range", "gender", "education",
    "ai_familiarity", "deepfake_knowledge", "social_media_freq", "used_ai",
    "check_2_1", "check_2_2", "check_2_3", "check_score",
    "task_json", "task_score", "final_json", "complete",
]

# Columns of the "stimuli" tab (researcher-filled).
#   order         : display order (number)
#   video         : file NAME in the Drive folder (or a Drive file id)
#   label         : ground truth: "real" / "deepfake" (optional; enables task_score)
#   prob_deepfake : model probability, e.g. 0.87 or 87% (shown to XAI group)
#   explanation   : XAI/LLM text explaining why it is real/deepfake (shown to XAI group)
STIMULI_HEADERS = ["order", "video", "label", "prob_deepfake", "explanation"]

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

Você está sendo convidado(a) a participar da pesquisa **“[TÍTULO DO PROJETO]”**,
conduzida por **[NOME DO PESQUISADOR]**, vinculada à **[INSTITUIÇÃO / PROGRAMA]**,
sob orientação de **[NOME DO ORIENTADOR]**.

- **Objetivo:** avaliar como orientações de letramento digital e explicações de
  inteligência artificial ajudam pessoas a identificar vídeos faciais autênticos
  ou manipulados (*deepfakes*).
- **Procedimentos:** você responderá a um questionário inicial, poderá receber um
  breve material educativo, assistirá a alguns vídeos e os classificará como
  autênticos ou gerados por IA, e responderá a questionários finais. Duração
  estimada: **cerca de [X] minutos**.
- **Riscos:** mínimos, limitados a eventual desconforto ou cansaço ao analisar os
  vídeos. Você pode interromper a participação a qualquer momento.
- **Benefícios:** contribuir para o desenvolvimento de ferramentas de combate à
  desinformação e ampliar sua percepção sobre mídias manipuladas.
- **Voluntariedade:** a participação é **voluntária e não remunerada**. Você pode
  desistir a qualquer momento, sem qualquer prejuízo.
- **Confidencialidade e dados (LGPD – Lei nº 13.709/2018):** **não** serão coletados
  dados que identifiquem você pessoalmente. As respostas serão armazenadas de forma
  anonimizada e usadas apenas para fins acadêmicos e científicos, de forma agregada.
- **Contatos:** Pesquisador(a) responsável — [NOME], [E-MAIL/TELEFONE].
  Comitê de Ética em Pesquisa (CEP) — [NOME DO CEP], [ENDEREÇO], [TELEFONE/E-MAIL].
"""

# =============================================================================
# INFRASTRUCTURE (lock + assignment)
# =============================================================================


@st.cache_resource
def get_lock() -> threading.Lock:
    """Single lock shared across every session in this instance."""
    return threading.Lock()


def choose_group(counts: dict) -> str:
    """Return the group that minimizes (n+1)/weight; ties broken at random."""
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

class SheetsStorage:
    def __init__(self):
        import gspread
        from google.oauth2.service_account import Credentials

        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        info = dict(st.secrets["gcp_service_account"])
        creds = Credentials.from_service_account_info(info, scopes=scopes)
        self._gspread = gspread
        self.client = gspread.authorize(creds)
        self.sh = self.client.open_by_key(st.secrets["spreadsheet"]["spreadsheet_key"])
        self.responses_ws = self._ws("responses", RESPONSE_HEADERS)
        self.counts_ws = self._ws("counts", ["group", "n"])
        self._ensure_counts()

    def _ws(self, name, headers):
        try:
            ws = self.sh.worksheet(name)
        except self._gspread.WorksheetNotFound:
            ws = self.sh.add_worksheet(title=name, rows=2000, cols=max(12, len(headers)))
        if not ws.row_values(1):
            ws.append_row(headers)
        return ws

    def _ensure_counts(self):
        existing = {r["group"] for r in self.counts_ws.get_all_records()}
        for g in GROUPS:
            if g not in existing:
                self.counts_ws.append_row([g, 0])

    def counts(self) -> dict:
        return {r["group"]: int(r["n"]) for r in self.counts_ws.get_all_records()}

    def _set_count(self, group, n):
        for i, r in enumerate(self.counts_ws.get_all_records(), start=2):  # row 1 = header
            if r["group"] == group:
                self.counts_ws.update_cell(i, 2, n)
                return

    def assign_group(self) -> str:
        with get_lock():
            counts = self.counts()
            group = choose_group(counts)
            self._set_count(group, counts.get(group, 0) + 1)
            return group

    def get_stimuli(self) -> list:
        ws = self._ws("stimuli", STIMULI_HEADERS)
        rows = ws.get_all_records()

        def _key(r):
            try:
                return float(r.get("order", 0) or 0)
            except (TypeError, ValueError):
                return 0.0
        return sorted(rows, key=_key)

    def save_response(self, row: dict):
        self.responses_ws.append_row(
            [str(row.get(h, "")) for h in RESPONSE_HEADERS],
            value_input_option="RAW",
        )


class SQLiteStorage:
    """Local fallback for testing only (Community Cloud wipes local files)."""

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

    def assign_group(self) -> str:
        with get_lock():
            counts = self.counts()
            group = choose_group(counts)
            self.conn.execute("UPDATE counts SET n = n + 1 WHERE grp = ?", (group,))
            self.conn.commit()
            return group

    def get_stimuli(self) -> list:
        return []  # no stimuli sheet locally; placeholders are used instead

    def save_response(self, row: dict):
        ph = ", ".join("?" * len(RESPONSE_HEADERS))
        self.conn.execute(
            f"INSERT INTO responses VALUES ({ph})",
            [str(row.get(h, "")) for h in RESPONSE_HEADERS],
        )
        self.conn.commit()


def _has_sheets() -> bool:
    try:
        return "gcp_service_account" in st.secrets and "spreadsheet" in st.secrets
    except Exception:
        return False


@st.cache_resource
def get_storage():
    if _has_sheets():
        return SheetsStorage(), "sheets"
    return SQLiteStorage(), "sqlite"


# =============================================================================
# GOOGLE DRIVE (video stimuli)
# =============================================================================

@st.cache_resource
def _drive_service():
    from googleapiclient.discovery import build
    from google.oauth2.service_account import Credentials

    scopes = ["https://www.googleapis.com/auth/drive.readonly"]
    creds = Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]), scopes=scopes)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


@st.cache_data(show_spinner=False, ttl=300)
def drive_name_map() -> dict:
    """Map {filename: file_id} for the configured Drive folder (refreshes every 5 min)."""
    try:
        folder_id = st.secrets["drive"]["folder_id"]
    except Exception:
        return {}
    service = _drive_service()
    files, page_token = {}, None
    while True:
        resp = service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields="nextPageToken, files(id, name)",
            pageToken=page_token,
        ).execute()
        for f in resp.get("files", []):
            files[f["name"]] = f["id"]
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return files


@st.cache_resource(show_spinner=False)
def get_video_path(file_id: str) -> str:
    """Download a Drive video once to a local temp file; return its path."""
    from googleapiclient.http import MediaIoBaseDownload

    path = os.path.join(tempfile.gettempdir(), f"stimulus_{file_id}.mp4")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        service = _drive_service()
        request = service.files().get_media(fileId=file_id)
        fh = io.FileIO(path, "wb")
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        fh.close()
    return path


def safe_video_path(file_id: str):
    try:
        return get_video_path(file_id)
    except Exception:
        return None


def fmt_prob(value) -> str:
    """Format a probability (0.87, '0,87', '87%', 87) as a percentage string."""
    try:
        v = float(str(value).replace("%", "").replace(",", ".").strip())
        if v > 1:
            v /= 100.0
        return f"{v * 100:.0f}%"
    except (TypeError, ValueError):
        return str(value)


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
    if mode == "sheets":
        try:
            return storage.get_stimuli(), "sheets"
        except Exception:
            return [], "sheets"
    dummies = [
        {"order": i, "video": "", "label": "",
         "prob_deepfake": "0.5",
         "explanation": f"〔explicação XAI de exemplo para o vídeo {i}〕"}
        for i in range(1, 4)
    ]
    return dummies, "test"


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


def go_to(step):
    st.session_state.step = step
    st.rerun()


def next_after_demographics():
    return "material" if st.session_state.group in GROUPS_WITH_MATERIAL else "task"


# =============================================================================
# SCREENS  (headings, questions and buttons are Portuguese on purpose)
# =============================================================================

def screen_intro(mode):
    st.title("Pesquisa: identificação de vídeos autênticos e deepfakes")
    st.write(
        "Obrigado pelo seu interesse. Nesta pesquisa você responderá a algumas "
        "perguntas e assistirá a vídeos de rostos, indicando se são autênticos ou "
        "gerados por inteligência artificial. A participação é anônima e leva "
        "cerca de **[X] minutos**."
    )
    if mode == "sqlite":
        st.warning(
            "⚠️ **Modo de teste local (SQLite).** Configure o Google Sheets e o Google "
            "Drive antes de coletar dados reais — no Streamlit Community Cloud o "
            "armazenamento local é apagado a cada reinício do app."
        )
    if st.button("Começar", type="primary"):
        go_to("consent")


def screen_consent():
    st.header("Termo de Consentimento Livre e Esclarecido (TCLE)")
    st.info(
        "Pesquisa com seres humanos no Brasil normalmente exige aprovação de um "
        "Comitê de Ética (CEP) via Plataforma Brasil. Insira CAAE/parecer e contatos "
        "no texto abaixo antes de coletar dados."
    )
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
            if st.session_state.group is None:  # assign exactly once per session
                storage, _ = get_storage()
                st.session_state.group = storage.assign_group()
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
                       ["18–24", "25–34", "35–44", "45–54", "55 ou mais"], index=None)
        gender = st.radio("1.2 Com qual gênero você se identifica?",
                          ["Feminino", "Masculino", "Outro", "Prefiro não responder"], index=None)
        education = st.radio("1.3 Nível de escolaridade mais alto já concluído? *",
                             ["Ensino fundamental", "Ensino médio",
                              "Ensino superior (graduação)", "Pós-graduação"], index=None)
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
            go_to(next_after_demographics())


def screen_material():
    st.header("Treinamento — checklist para identificar deepfakes")
    # TODO: replace with the real checklist training for group "Checklist".
    st.write("Antes de classificar os vídeos, revise este checklist de verificação:")
    st.markdown(
        "1. **Bordas e transições** — o rosto se mistura de forma natural ao fundo, ao "
        "cabelo e ao pescoço?\n"
        "2. **Olhos e piscadas** — o olhar e a frequência de piscadas parecem naturais?\n"
        "3. **Boca e fala** — os lábios acompanham o áudio? Há dentes/língua estranhos?\n"
        "4. **Iluminação e sombras** — a luz no rosto é coerente com o ambiente?\n"
        "5. **Textura de pele/cabelo** — há áreas borradas, cerosas ou artificiais?\n"
        "6. **Fonte e contexto** — de onde vem o vídeo? A situação faz sentido?\n\n"
        "A inspeção visual **não** basta sozinha: sempre considere a **fonte** e o "
        "**contexto** da mídia."
    )
    st.caption("〔Placeholder — substitua pelo material de treinamento definitivo.〕")
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
    group = st.session_state.group
    show_ai = group in GROUPS_WITH_AI_EXPLANATION

    st.header("Tarefa — assista e classifique os vídeos")
    st.write("Para cada vídeo, indique se você o considera **autêntico** ou **gerado por IA**.")

    stimuli, source = load_stimuli()
    if not stimuli:
        st.warning(
            "Nenhum vídeo configurado. Preencha a aba **stimuli** da planilha "
            "(colunas: order, video, label, prob_deepfake, explanation) e coloque os "
            "arquivos na pasta do Google Drive."
        )
        return
    if source == "test":
        st.caption("〔Modo de teste: vídeos indisponíveis; exibindo apenas a estrutura.〕")

    name_map = drive_name_map()

    with st.form("task"):
        answers, labels = {}, {}
        for idx, s in enumerate(stimuli, start=1):
            st.markdown(f"**Vídeo {idx}**")

            video_value = str(s.get("video", "")).strip()
            file_id = name_map.get(video_value, video_value)  # filename -> id, else assume id
            path = safe_video_path(file_id) if file_id else None
            if path:
                st.video(path)
            else:
                st.markdown(
                    "<div style='width:100%;max-width:480px;height:240px;background:#eee;"
                    "border-radius:8px;display:flex;align-items:center;justify-content:center;"
                    f"color:#888'>vídeo {idx}</div>", unsafe_allow_html=True)

            if show_ai:
                st.info(f"🤖 Probabilidade estimada de ser deepfake (modelo): "
                        f"**{fmt_prob(s.get('prob_deepfake'))}**")
                explanation = str(s.get("explanation", "")).strip()
                if explanation:
                    st.markdown(f"**Explicação da IA:** {explanation}")

            key = f"vid_{idx}"
            answers[key] = st.radio(f"Classificação do vídeo {idx} *", TASK_OPTIONS,
                                    index=None, horizontal=True, key=key,
                                    label_visibility="collapsed")
            labels[key] = str(s.get("label", "")).strip()
            st.divider()
        submit = st.form_submit_button("Continuar", type="primary")

    if submit:
        if any(v is None for v in answers.values()):
            st.error("Classifique todos os vídeos antes de continuar.")
        else:
            st.session_state.data["task_json"] = json.dumps(answers, ensure_ascii=False)
            if all(labels.values()):  # score only if every stimulus has a ground-truth label
                score = sum(label_matches(labels[k], answers[k]) for k in answers)
                st.session_state.data["task_score"] = score
            go_to("final")


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
    st.success(f"Você participou do grupo: **{ss.group}**")
    st.caption("Anote esta informação caso precise informá-la à equipe da pesquisa.")
    st.write("Você pode fechar esta janela.")


def screen_admin():
    st.header("Researcher panel")
    try:
        storage, mode = get_storage()
        counts = storage.counts()
    except Exception as e:  # noqa: BLE001
        st.error(f"Failed to read counts: {e}")
        return
    st.metric("Total assigned", sum(counts.values()))
    st.write({g: counts.get(g, 0) for g in GROUPS})
    st.caption(f"Backend: {mode} · target weights: {WEIGHTS}")


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
