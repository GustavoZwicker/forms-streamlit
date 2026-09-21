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
import json
import random
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone

import streamlit as st

# =============================================================================
# CONFIGURATION
# =============================================================================

GROUPS = ["Controle", "Checklist", "XAI"]
WEIGHTS = {"Controle": 1, "Checklist": 1, "XAI": 1}  # 1:1:1 => same-size samples

GROUPS_WITH_MATERIAL = {"Checklist"}
GROUPS_WITH_AI_EXPLANATION = {"XAI"}

# Demographic variables kept balanced across groups (minimization). Fewer factors
# => stronger balance on each. Must be collected BEFORE assignment (Section 1).
BALANCE_FACTORS = ["ai_familiarity", "deepfake_knowledge", "education", "age_range"]

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
            # Assign AFTER demographics so the groups stay balanced on them.
            if st.session_state.group is None:
                storage, _ = get_storage()
                factors = {f: str(st.session_state.data.get(f))
                           for f in BALANCE_FACTORS
                           if st.session_state.data.get(f) is not None}
                st.session_state.group = storage.assign_group(factors)
            go_to(next_after_demographics())


def screen_material():
    st.header("Como inspecionar um vídeo")
    st.write(
        "Observe o vídeo com calma e verifique cada item abaixo. Nenhum sinal "
        "isolado prova que o vídeo é falso; considere o conjunto."
    )
    st.markdown(
        "**Bordas e contornos**\n"
        "- Transições não naturais entre o rosto e o fundo\n"
        "- Mistura estranha entre regiões do rosto (linha do queixo ou do cabelo)\n"
        "- Bordas borradas ou distorcidas ao redor de olhos, nariz e boca\n"
        "- Fios de cabelo misturados de forma não natural ou cortados abruptamente\n\n"
        "**Pele e nitidez**\n"
        "- Pele lisa demais ou com manchas/texturas irregulares\n"
        "- Diferença de nitidez entre o rosto e o restante da imagem\n\n"
        "**Formato e simetria do rosto**\n"
        "- Traços faciais distorcidos ou assimétricos\n"
        "- Padrões estranhos em dentes, olhos ou orelhas\n\n"
        "**Iluminação e sombras**\n"
        "- Iluminação inconsistente entre o rosto e o fundo\n"
        "- Sombras que não seguem a direção da luz\n"
        "- Reflexos nos olhos que não combinam com a cena\n\n"
        "**Fundo e artefatos**\n"
        "- Regiões do fundo deformadas, duplicadas ou geometricamente inconsistentes\n"
        "- Pequenos artefatos: borrões, \"fantasmas\" (ghosting) ou padrões de pixel estranhos"
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
    group = ss.group
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

    idx = ss.task_index
    stimulus = stimuli[idx]
    video_number = idx + 1
    video_key = f"vid_{video_number}"

    st.progress(video_number / len(stimuli), text=f"Vídeo {video_number} de {len(stimuli)}")
    st.subheader(f"Vídeo {video_number}")

    url = video_url(stimulus.get("video", ""))
    if url:
        st.video(url)
    else:
        st.markdown(
            "<div style='width:100%;max-width:480px;height:240px;background:#eee;"
            "border-radius:8px;display:flex;align-items:center;justify-content:center;"
            f"color:#888'>vídeo {video_number}</div>",
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
            if video_number < len(stimuli)
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
            "video_number": video_number,
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
    st.success(f"Você participou do grupo: **{ss.group}**")
    st.caption("Anote esta informação caso precise informá-la à equipe da pesquisa.")
    st.write("Você pode fechar esta janela.")


def screen_admin():
    """Password-protected researcher dashboard."""
    st.header("Researcher panel")

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
