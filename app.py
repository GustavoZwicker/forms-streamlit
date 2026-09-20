# -*- coding: utf-8 -*-
"""
Aplicativo de coleta para a pesquisa sobre identificação de deepfakes.

Fluxo:
  intro -> TCLE -> [ATRIBUIÇÃO DE GRUPO] -> sociodemográfico
        -> (G1/G2) material educativo -> (G1/G2) verificação de aprendizagem
        -> tarefa de classificação (12 imagens) -> questionários finais -> fim

Atribuição balanceada em tempo real:
  Cada novo participante que consente é alocado ao grupo que está mais "atrasado"
  em relação à sua proporção-alvo (PESOS). Com pesos iguais (1:1:1) isso mantém as
  três amostras do mesmo tamanho ao longo de toda a coleta ("minimização").
  A decisão ler-contar-incrementar é serializada por um Lock de processo, o que
  elimina condições de corrida no Streamlit Community Cloud (instância única).

Armazenamento:
  - Google Sheets (durável) se houver credenciais em st.secrets  -> use para coleta real.
  - SQLite local (efêmero) como fallback                          -> apenas para testes.
"""

import json
import random
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

GRUPOS = ["Controle", "G1", "G2"]

# Proporção-alvo entre os grupos. Iguais (1:1:1) => amostras do mesmo tamanho.
# Para dar "necessidade proporcional" diferente a algum grupo, altere aqui
# (ex.: {"Controle": 1, "G1": 1, "G2": 2} atribui o dobro ao G2).
PESOS = {"Controle": 1, "G1": 1, "G2": 1}

# Grupos que recebem material educativo + verificação (Seção 2 do formulário).
GRUPOS_COM_MATERIAL = {"G1", "G2"}
# Grupos que veem explicações da IA durante a tarefa de classificação.
GRUPOS_COM_EXPLICACAO_IA = {"G2"}

N_IMAGENS = 12  # tarefa de classificação (imagens não estão no MD -> placeholder)

# Colunas gravadas por participante.
RESP_HEADERS = [
    "timestamp", "participante_id", "grupo", "consentiu",
    "faixa_etaria", "genero", "escolaridade",
    "familiaridade_ia", "conhecimento_deepfake", "freq_redes", "usou_ia",
    "verif_2_1", "verif_2_2", "verif_2_3", "verif_score",
    "tarefa_json", "final_json", "completo",
]

# Gabarito da Seção 2 (NÃO exibido ao participante).
GAB_2_1 = "Transições ou bordas não naturais entre o rosto e o fundo"
GAB_2_2 = "Falso"
GAB_2_3 = "Verificar a fonte e o contexto da mídia"

OPCOES_2_1 = [
    "Transições ou bordas não naturais entre o rosto e o fundo",
    "A imagem estar em alta resolução",
    "A pessoa estar sorrindo",
    "O arquivo ser grande",
]
OPCOES_2_3 = [
    "Verificar a fonte e o contexto da mídia",
    "Confiar apenas no número de curtidas",
    "Aumentar o brilho da tela",
    "Compartilhar antes de checar",
]

# -----------------------------------------------------------------------------
# TCLE — preencha os campos entre colchetes antes de coletar dados.
# -----------------------------------------------------------------------------
TCLE_TEXTO = """
**TERMO DE CONSENTIMENTO LIVRE E ESCLARECIDO**

Você está sendo convidado(a) a participar da pesquisa **“[TÍTULO DO PROJETO]”**,
conduzida por **[NOME DO PESQUISADOR]**, vinculada à **[INSTITUIÇÃO / PROGRAMA]**,
sob orientação de **[NOME DO ORIENTADOR]**.

- **Objetivo:** avaliar como orientações de letramento digital e explicações de
  inteligência artificial ajudam pessoas a identificar imagens faciais autênticas
  ou manipuladas (*deepfakes*).
- **Procedimentos:** você responderá a um questionário inicial, poderá receber um
  breve material educativo, classificará 12 imagens como autênticas ou geradas por
  IA e responderá a questionários finais. Duração estimada: **cerca de [X] minutos**.
- **Riscos:** mínimos, limitados a eventual desconforto ou cansaço ao analisar as
  imagens. Você pode interromper a participação a qualquer momento.
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
# INFRAESTRUTURA (lock + armazenamento)
# =============================================================================


@st.cache_resource
def get_lock() -> threading.Lock:
    """Lock único e compartilhado por todas as sessões da instância."""
    return threading.Lock()


def escolher_grupo(counts: dict) -> str:
    """Grupo que minimiza (n+1)/peso; empate resolvido aleatoriamente."""
    melhor_val = None
    candidatos = []
    for g in GRUPOS:
        val = (counts.get(g, 0) + 1) / PESOS[g]
        if melhor_val is None or val < melhor_val - 1e-9:
            melhor_val = val
            candidatos = [g]
        elif abs(val - melhor_val) <= 1e-9:
            candidatos.append(g)
    return random.choice(candidatos)


# ----- Backend: Google Sheets ------------------------------------------------

class SheetsStorage:
    def __init__(self):
        import gspread
        from google.oauth2.service_account import Credentials

        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        info = dict(st.secrets["gcp_service_account"])
        creds = Credentials.from_service_account_info(info, scopes=scopes)
        self._gspread = gspread
        self.client = gspread.authorize(creds)
        self.sh = self.client.open_by_key(st.secrets["planilha"]["spreadsheet_key"])
        self.resp = self._ws("respostas", RESP_HEADERS)
        self.cont = self._ws("contagem", ["grupo", "n"])
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
        existentes = {r["grupo"] for r in self.cont.get_all_records()}
        for g in GRUPOS:
            if g not in existentes:
                self.cont.append_row([g, 0])

    def counts(self) -> dict:
        return {r["grupo"]: int(r["n"]) for r in self.cont.get_all_records()}

    def _set_count(self, grupo, n):
        for i, r in enumerate(self.cont.get_all_records(), start=2):  # linha 1 = cabeçalho
            if r["grupo"] == grupo:
                self.cont.update_cell(i, 2, n)
                return

    def assign_group(self) -> str:
        with get_lock():
            counts = self.counts()
            grupo = escolher_grupo(counts)
            self._set_count(grupo, counts.get(grupo, 0) + 1)
            return grupo

    def save_response(self, row: dict):
        self.resp.append_row(
            [str(row.get(h, "")) for h in RESP_HEADERS],
            value_input_option="RAW",
        )


# ----- Backend: SQLite (fallback local, efêmero) -----------------------------

class SQLiteStorage:
    def __init__(self, path="respostas.db"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        c = self.conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS contagem (grupo TEXT PRIMARY KEY, n INTEGER)")
        for g in GRUPOS:
            c.execute("INSERT OR IGNORE INTO contagem (grupo, n) VALUES (?, 0)", (g,))
        cols = ", ".join(f'"{h}" TEXT' for h in RESP_HEADERS)
        c.execute(f"CREATE TABLE IF NOT EXISTS respostas ({cols})")
        self.conn.commit()

    def counts(self) -> dict:
        cur = self.conn.execute("SELECT grupo, n FROM contagem")
        return {g: n for g, n in cur.fetchall()}

    def assign_group(self) -> str:
        with get_lock():
            counts = self.counts()
            grupo = escolher_grupo(counts)
            self.conn.execute("UPDATE contagem SET n = n + 1 WHERE grupo = ?", (grupo,))
            self.conn.commit()
            return grupo

    def save_response(self, row: dict):
        ph = ", ".join("?" * len(RESP_HEADERS))
        self.conn.execute(
            f"INSERT INTO respostas VALUES ({ph})",
            [str(row.get(h, "")) for h in RESP_HEADERS],
        )
        self.conn.commit()


def _tem_sheets() -> bool:
    try:
        return "gcp_service_account" in st.secrets and "planilha" in st.secrets
    except Exception:
        return False


@st.cache_resource
def get_storage():
    if _tem_sheets():
        return SheetsStorage(), "sheets"
    return SQLiteStorage(), "sqlite"


# =============================================================================
# ESTADO E NAVEGAÇÃO
# =============================================================================

def init_state():
    ss = st.session_state
    ss.setdefault("step", "intro")
    ss.setdefault("pid", str(uuid.uuid4()))
    ss.setdefault("grupo", None)
    ss.setdefault("dados", {})
    ss.setdefault("enviado", False)


def ir_para(step):
    st.session_state.step = step
    st.rerun()


def proximo_apos_socio():
    return "material" if st.session_state.grupo in GRUPOS_COM_MATERIAL else "tarefa"


# =============================================================================
# TELAS
# =============================================================================

def tela_intro(modo):
    st.title("Pesquisa: identificação de imagens autênticas e deepfakes")
    st.write(
        "Obrigado pelo seu interesse. Nesta pesquisa você responderá a algumas "
        "perguntas e analisará imagens de rostos, indicando se são autênticas ou "
        "geradas por inteligência artificial. A participação é anônima e leva "
        "cerca de **[X] minutos**."
    )
    if modo == "sqlite":
        st.warning(
            "⚠️ **Modo de teste local (SQLite).** Configure o Google Sheets antes de "
            "coletar dados reais — no Streamlit Community Cloud o armazenamento local "
            "é apagado a cada reinício do app."
        )
    if st.button("Começar", type="primary"):
        ir_para("tcle")


def tela_tcle():
    st.header("Termo de Consentimento Livre e Esclarecido (TCLE)")
    st.info(
        "Pesquisa com seres humanos no Brasil normalmente exige aprovação de um "
        "Comitê de Ética (CEP) via Plataforma Brasil. Insira CAAE/parecer e contatos "
        "no texto abaixo antes de coletar dados."
    )
    st.markdown(TCLE_TEXTO)

    escolha = st.radio(
        "Declaro que li e compreendi o TCLE acima e concordo em participar da pesquisa. *",
        ["Sim, li, compreendi e concordo em participar.", "Não desejo participar."],
        index=None,
    )
    if st.button("Continuar", type="primary"):
        if escolha is None:
            st.error("Selecione uma opção para continuar.")
        elif escolha.startswith("Não"):
            ir_para("recusou")
        else:
            # ---- ATRIBUIÇÃO BALANCEADA (ocorre uma única vez por sessão) ----
            if st.session_state.grupo is None:
                storage, _ = get_storage()
                st.session_state.grupo = storage.assign_group()
            st.session_state.dados["consentiu"] = "Sim"
            ir_para("socio")


def tela_recusou():
    st.header("Participação não iniciada")
    st.write("Tudo bem. Obrigado pelo seu tempo — você pode fechar esta janela.")


def tela_socio():
    st.header("Seção 1 — Questionário sociodemográfico")
    st.caption("Campos com * são obrigatórios.")
    with st.form("socio"):
        faixa = st.radio("1.1 Qual é a sua faixa etária? *",
                         ["18–24", "25–34", "35–44", "45–54", "55 ou mais"], index=None)
        genero = st.radio("1.2 Com qual gênero você se identifica?",
                          ["Feminino", "Masculino", "Outro", "Prefiro não responder"], index=None)
        escol = st.radio("1.3 Nível de escolaridade mais alto já concluído? *",
                         ["Ensino fundamental", "Ensino médio",
                          "Ensino superior (graduação)", "Pós-graduação"], index=None)
        fam = st.radio("1.4 Familiaridade com Inteligência Artificial? *  (1 = Nenhuma … 5 = Especialista)",
                       [1, 2, 3, 4, 5], index=None, horizontal=True)
        conhec = st.radio("1.5 Conhecimento prévio sobre *deepfakes*? *",
                          ["Nenhum", "Algum", "Bastante"], index=None)
        freq = st.radio("1.6 Com que frequência você usa redes sociais?",
                        ["Raramente", "Semanalmente", "Diariamente", "Várias vezes ao dia"], index=None)
        usou = st.radio("1.7 Você já usou alguma ferramenta baseada em IA?",
                        ["Sim", "Não"], index=None)
        enviar = st.form_submit_button("Continuar", type="primary")

    if enviar:
        faltando = [q for q, v in [("1.1", faixa), ("1.3", escol), ("1.4", fam), ("1.5", conhec)] if v is None]
        if faltando:
            st.error("Responda às perguntas obrigatórias: " + ", ".join(faltando))
        else:
            st.session_state.dados.update({
                "faixa_etaria": faixa, "genero": genero, "escolaridade": escol,
                "familiaridade_ia": fam, "conhecimento_deepfake": conhec,
                "freq_redes": freq, "usou_ia": usou,
            })
            ir_para(proximo_apos_socio())


def tela_material():
    st.header("Material educativo — letramento digital")
    # TODO: inserir aqui o conteúdo real do módulo educativo (texto/vídeo/imagens).
    st.write(
        "Leia com atenção antes de prosseguir. Alguns sinais de que uma imagem pode "
        "ter sido manipulada por IA:"
    )
    st.markdown(
        "- Transições ou bordas não naturais entre o rosto e o fundo\n"
        "- Assimetrias em olhos, dentes, orelhas ou acessórios\n"
        "- Texturas de pele/cabelo artificiais e iluminação inconsistente\n\n"
        "A inspeção visual **não** é suficiente sozinha: verifique também a **fonte** e "
        "o **contexto** da mídia."
    )
    st.caption("〔Placeholder — substitua pelo material definitivo da sua pesquisa.〕")
    if st.button("Li o material e desejo continuar", type="primary"):
        ir_para("verificacao")


def tela_verificacao():
    st.header("Seção 2 — Verificação de aprendizagem")
    with st.form("verif"):
        q1 = st.radio("2.1 Qual é um sinal comum de que uma imagem pode ter sido manipulada por IA? *",
                      OPCOES_2_1, index=None)
        q2 = st.radio("2.2 A inspeção visual, sozinha, é suficiente para garantir que uma imagem é autêntica. *",
                      ["Verdadeiro", "Falso"], index=None)
        q3 = st.radio("2.3 Ao avaliar uma possível *deepfake*, além de observar a imagem, também é importante: *",
                      OPCOES_2_3, index=None)
        enviar = st.form_submit_button("Continuar", type="primary")

    if enviar:
        if None in (q1, q2, q3):
            st.error("Responda a todas as perguntas para continuar.")
        else:
            score = int(q1 == GAB_2_1) + int(q2 == GAB_2_2) + int(q3 == GAB_2_3)
            st.session_state.dados.update({
                "verif_2_1": q1, "verif_2_2": q2, "verif_2_3": q3, "verif_score": score,
            })
            ir_para("tarefa")


def tela_tarefa():
    grupo = st.session_state.grupo
    st.header("Tarefa — classifique as imagens")
    st.write("Para cada imagem, indique se você a considera **autêntica** ou **gerada por IA**.")
    st.caption(
        "〔Placeholder — coloque as 12 imagens em `imagens/1.jpg … 12.jpg`. "
        "Substitua também a explicação da IA (mostrada apenas ao G2) pela real.〕"
    )

    with st.form("tarefa"):
        respostas = {}
        for i in range(1, N_IMAGENS + 1):
            st.markdown(f"**Imagem {i}**")
            caminho = Path(f"imagens/{i}.jpg")
            if caminho.exists():
                st.image(str(caminho), width=320)
            else:
                st.markdown(
                    "<div style='width:320px;height:180px;background:#eee;border-radius:8px;"
                    "display:flex;align-items:center;justify-content:center;color:#888'>"
                    f"imagem {i}</div>", unsafe_allow_html=True)
            if grupo in GRUPOS_COM_EXPLICACAO_IA:
                st.info("🤖 Explicação da IA: 〔texto/heatmap da análise automática desta imagem〕")
            respostas[f"img_{i}"] = st.radio(
                f"Classificação da imagem {i} *",
                ["Autêntica", "Gerada por IA"], index=None, horizontal=True, key=f"img_{i}",
                label_visibility="collapsed",
            )
            st.divider()
        enviar = st.form_submit_button("Continuar", type="primary")

    if enviar:
        if any(v is None for v in respostas.values()):
            st.error("Classifique todas as imagens antes de continuar.")
        else:
            st.session_state.dados["tarefa_json"] = json.dumps(respostas, ensure_ascii=False)
            ir_para("final")


def tela_final():
    st.header("Questionários finais")
    st.caption("〔Placeholder — insira aqui os questionários finais da sua pesquisa.〕")
    with st.form("final"):
        confianca = st.radio("Quão confiante você ficou nas suas classificações? (1 = Nada … 5 = Muito)",
                             [1, 2, 3, 4, 5], index=None, horizontal=True)
        dificuldade = st.radio("Quão difícil foi a tarefa? (1 = Muito fácil … 5 = Muito difícil)",
                               [1, 2, 3, 4, 5], index=None, horizontal=True)
        comentarios = st.text_area("Comentários (opcional)")
        enviar = st.form_submit_button("Enviar respostas", type="primary")

    if enviar:
        st.session_state.dados["final_json"] = json.dumps(
            {"confianca": confianca, "dificuldade": dificuldade, "comentarios": comentarios},
            ensure_ascii=False,
        )
        ir_para("fim")


def tela_fim():
    ss = st.session_state
    if not ss.enviado:
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "participante_id": ss.pid,
            "grupo": ss.grupo,
            "completo": "Sim",
            **ss.dados,
        }
        try:
            storage, _ = get_storage()
            storage.save_response(row)
            ss.enviado = True
        except Exception as e:  # noqa: BLE001
            st.error("Não foi possível salvar suas respostas. Tente novamente em instantes.")
            st.caption(f"Detalhe técnico: {e}")
            if st.button("Tentar novamente"):
                st.rerun()
            return

    st.header("Obrigado por participar! ✅")
    st.write("Suas respostas foram registradas de forma anônima. Você pode fechar esta janela.")


def tela_admin():
    st.header("Painel do pesquisador")
    try:
        storage, modo = get_storage()
        counts = storage.counts()
    except Exception as e:  # noqa: BLE001
        st.error(f"Erro ao ler contagens: {e}")
        return
    total = sum(counts.values())
    st.metric("Total atribuído", total)
    st.write({g: counts.get(g, 0) for g in GRUPOS})
    st.caption(f"Backend: {modo} · pesos-alvo: {PESOS}")


# =============================================================================
# ROTEADOR
# =============================================================================

def main():
    st.set_page_config(page_title="Pesquisa deepfakes", page_icon="🔎")
    init_state()
    _, modo = get_storage()

    # Painel do pesquisador: acesse com ?admin=SUA_CHAVE
    admin_key = st.query_params.get("admin")
    if admin_key is not None:
        try:
            esperado = st.secrets.get("admin", {}).get("key")
        except Exception:
            esperado = None
        if esperado and admin_key == esperado:
            tela_admin()
        else:
            st.error("Chave de administrador inválida.")
        return

    step = st.session_state.step
    if step == "intro":
        tela_intro(modo)
    elif step == "tcle":
        tela_tcle()
    elif step == "recusou":
        tela_recusou()
    elif step == "socio":
        tela_socio()
    elif step == "material":
        tela_material()
    elif step == "verificacao":
        tela_verificacao()
    elif step == "tarefa":
        tela_tarefa()
    elif step == "final":
        tela_final()
    elif step == "fim":
        tela_fim()
    else:
        st.session_state.step = "intro"
        st.rerun()


if __name__ == "__main__":
    main()
