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

import json
import random
import sqlite3
import threading
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

LABEL_AUTENTICO = "Autêntico"
LABEL_DEEPFAKE = "Gerado por IA"
TASK_OPTIONS = [LABEL_AUTENTICO, LABEL_DEEPFAKE]

# Fields stored per participant. Note: "group" maps to DB column "grp".
RESPONSE_HEADERS = [
    "timestamp", "participant_id", "group", "consented",
    "age_range", "gender", "education",
    "ai_familiarity", "deepfake_knowledge", "social_media_freq", "used_ai",
    "check_2_1", "check_2_2", "check_2_3", "check_score",
    "task_json", "task_score", "final_json", "complete",
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

    def counts(self) -> dict:
        res = self.client.table("counts").select("grp, n").execute()
        return {r["grp"]: int(r["n"]) for r in (res.data or [])}

    def assign_group(self) -> str:
        # Atomic + race-free: all logic lives in the Postgres function.
        res = self.client.rpc("assign_group", {}).execute()
        data = res.data
        if isinstance(data, list):
            data = data[0] if data else None
        return data

    def get_stimuli(self) -> list:
        res = self.client.table("stimuli").select("*").order("sort_order").execute()
        return res.data or []

    def save_response(self, row: dict):
        data = {h: str(row.get(h, "")) for h in RESPONSE_HEADERS}
        data["grp"] = data.pop("group")  # DB column is "grp"
        self.client.table("responses").insert(data).execute()


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

    def assign_group(self) -> str:
        with get_lock():
            counts = self.counts()
            group = choose_group(counts)
            self.conn.execute("UPDATE counts SET n = n + 1 WHERE grp = ?", (group,))
            self.conn.commit()
            return group

    def get_stimuli(self) -> list:
        return []

    def save_response(self, row: dict):
        ph = ", ".join("?" * len(RESPONSE_HEADERS))
        self.conn.execute(
            f"INSERT INTO responses VALUES ({ph})",
            [str(row.get(h, "")) for h in RESPONSE_HEADERS],
        )
        self.conn.commit()


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

def video_url(video_value: str):
    """Full URL for a stimulus: a public Storage URL from the filename, or the
    value itself if it is already a URL."""
    v = str(video_value).strip()
    if not v:
        return None
    if v.startswith("http://") or v.startswith("https://"):
        return v
    base = st.secrets["supabase"]["url"].rstrip("/")
    bucket = st.secrets["supabase"].get("bucket", "videos")
    return f"{base}/storage/v1/object/public/{bucket}/{v}"


def fmt_prob(value) -> str:
    try:
        x = float(str(value).replace("%", "").replace(",", ".").strip())
        if x > 1:
            x /= 100.0
        return f"{x * 100:.0f}%"
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
            "⚠️ **Modo de teste local (SQLite).** Configure o Supabase antes de "
            "coletar dados reais."
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
            "Nenhum vídeo configurado. Preencha a tabela **stimuli** no Supabase "
            "(colunas: sort_order, video, label, prob_deepfake, explanation) e envie "
            "os arquivos para o bucket de Storage."
        )
        return
    if source == "test":
        st.caption("〔Modo de teste: vídeos indisponíveis; exibindo apenas a estrutura.〕")

    with st.form("task"):
        answers, labels = {}, {}
        for idx, s in enumerate(stimuli, start=1):
            st.markdown(f"**Vídeo {idx}**")

            url = video_url(s.get("video", ""))
            if url:
                st.video(url)
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
            if all(labels.values()):
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
