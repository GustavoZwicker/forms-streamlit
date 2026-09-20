# Coleta — Pesquisa sobre deepfakes

App em Streamlit que aplica o questionário e **distribui cada participante, em
tempo real, para um de três grupos** (Controle, G1, G2), mantendo as amostras
equilibradas.

## Como funciona a distribuição

Quando alguém consente no TCLE, o app olha as contagens atuais e aloca a pessoa
ao grupo mais "atrasado" em relação à sua proporção-alvo — minimiza `(n+1)/peso`,
com desempate aleatório. Com pesos iguais (`1:1:1`, padrão), isso mantém os três
grupos **do mesmo tamanho** durante toda a coleta.

- Para dar peso diferente a um grupo ("necessidade proporcional"), edite `PESOS`
  em `app.py`. Ex.: `{"Controle": 1, "G1": 1, "G2": 2}` aloca o dobro ao G2.
- A decisão *ler → escolher → incrementar* é serializada por um `Lock` de
  processo. No **Streamlit Community Cloud** (instância única) isso elimina
  qualquer condição de corrida entre participantes simultâneos. Se um dia rodar
  em várias réplicas, troque o backend por um banco com incremento atômico
  (ex.: Postgres/Supabase RPC).
- A atribuição é contada no **consentimento**. Quem desistir depois deixa uma
  pequena diferença entre grupos; para analisar só quem terminou, filtre pela
  coluna `completo`.

## Rodar localmente (teste)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Sem credenciais, o app usa **SQLite** (`respostas.db`) só para teste. **Não use
SQLite para coleta real**: no Cloud o disco é apagado a cada reinício.

## Publicar (Streamlit Community Cloud — grátis)

1. Suba esta pasta para um repositório no **GitHub**.
2. Crie uma **planilha** no Google Sheets e copie o ID (parte da URL entre `/d/`
   e `/edit`).
3. No **Google Cloud Console**: crie um projeto → ative a **Google Sheets API**
   → crie uma **Service Account** → gere uma **chave JSON**.
4. **Compartilhe a planilha** com o `client_email` da service account, como
   **Editor**.
5. Em [share.streamlit.io](https://share.streamlit.io), clique **New app**,
   aponte para o repositório e `app.py`, e faça deploy.
6. Em **Settings → Secrets**, cole o conteúdo de `.streamlit/secrets.toml.example`
   preenchido (dados do JSON + `spreadsheet_key` + `admin.key`).
7. Abra o app e teste. O app cria sozinho as abas `respostas` e `contagem`.

## Painel de contagens

Acesse `SUA_URL/?admin=SUA_CHAVE` (a `admin.key` dos secrets) para ver quantos
participantes há em cada grupo.

## O que ainda falta preencher

O formulário base (MD) cobre TCLE, sociodemográfico e verificação de aprendizagem.
Estão como *placeholder* em `app.py`, marcados com `TODO`/`〔...〕`:

- **Campos do TCLE** entre colchetes (título, pesquisador, CEP/CAAE, duração).
- **Material educativo** (`tela_material`).
- **12 imagens** da tarefa (coloque em `imagens/1.jpg … 12.jpg`) e a
  **explicação da IA** mostrada ao G2 (`tela_tarefa`).
- **Questionários finais** (`tela_final`).
