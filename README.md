# Coleta — Pesquisa sobre deepfakes (vídeos) · backend Supabase

App em Streamlit que aplica o questionário, **distribui cada participante em tempo
real** para um de três grupos (mantendo as amostras equilibradas) e apresenta
**vídeos hospedados no Supabase Storage** para classificação. Todos os dados ficam
no **Postgres do Supabase** — sem Google Cloud.

## Os três grupos

- **Controle** — sem treinamento. Assiste aos vídeos e classifica.
- **Checklist** — treinamento com checklist + verificação (Seção 2) e depois classifica.
- **XAI** — classifica os vídeos exibidos junto com a **probabilidade de deepfake do
  modelo** e uma **explicação (XAI/LLM)** de por que o vídeo é real ou manipulado.

## Distribuição balanceada

Ao consentir, o participante é alocado por uma função Postgres (`assign_group`) que
escolhe o grupo mais "atrasado" em relação ao peso e incrementa a contagem numa única
transação protegida por advisory lock — **atômica e sem condição de corrida**, mesmo
com muitos participantes simultâneos. Pesos iguais (1:1:1) mantêm os grupos do mesmo
tamanho. Para mudar a proporção: `update counts set weight = 2 where grp = 'XAI';`.

## Probabilidade e explicação da IA (grupo XAI)

O app **não** roda o detector nem o LLM ao vivo — cada participante deve ver a mesma
explicação para o mesmo vídeo (controle experimental). Você fornece esses valores
**pré-calculados** na tabela `stimuli` (colunas `prob_deepfake` e `explanation`). Só o
grupo XAI os vê.

## Configurar o Supabase (uma vez)

1. **Tabelas + função:** Dashboard -> **SQL Editor** -> New query -> cole o conteúdo de
   `supabase_setup.sql` -> **Run**. Isso cria `counts`, `responses`, `stimuli` e a
   função `assign_group`.
2. **Bucket de vídeos:** Dashboard -> **Storage** -> **New bucket** -> nome `videos`,
   marque **Public bucket** -> Create. Faça **Upload** dos seus `.mp4` nesse bucket.
   (Bucket público serve os vídeos direto ao navegador; os dados no banco continuam
   privados.)
3. **Estímulos:** Dashboard -> **Table Editor** -> `stimuli` -> Insert row, uma por
   vídeo:

   | sort_order | video       | label    | prob_deepfake | explanation                   |
   |------------|-------------|----------|---------------|-------------------------------|
   | 1          | video01.mp4 | real     | 0.12          | Bordas e iluminacao naturais… |
   | 2          | video02.mp4 | deepfake | 0.88          | Piscadas irregulares e…       |

   - `video`: nome do arquivo no bucket (ou uma URL completa).
   - `label`: `real`/`deepfake` — opcional; se preenchido em todos, calcula `task_score`.
   - `prob_deepfake` / `explanation`: mostrados **apenas ao grupo XAI**.
4. **Chaves:** Dashboard -> **Project Settings -> API**. Copie a **Project URL** e a
   chave **`service_role`** (a secreta, não a `anon`).

## Secrets

Preencha `.streamlit/secrets.toml.example` (a `url` já vem preenchida com seu projeto):

```toml
[supabase]
url = "https://zmfhaffmytoefywtpbpn.supabase.co"
service_key = "SUA_CHAVE_SERVICE_ROLE"
bucket = "videos"

[admin]
key = "uma-senha-forte"
```

A chave `service_role` roda **só no servidor** (Streamlit) e ignora o RLS, então os
dados ficam acessíveis apenas por quem tem a chave. **Nunca** a exponha nem faça commit.

## Rodar localmente (teste)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Sem os secrets do Supabase, o app usa **SQLite** (`responses.db`) e stimuli de exemplo,
só para percorrer o fluxo.

## Publicar (Streamlit Community Cloud — grátis)

1. Suba esta pasta para um repositório no **GitHub**.
2. Em [share.streamlit.io](https://share.streamlit.io) -> **New app**, aponte para o
   repositório e `app.py`, e faça deploy.
3. Em **Settings -> Secrets**, cole o conteúdo do `secrets.toml` preenchido.
4. Abra o app e faça um teste de ponta a ponta.

## Painel do pesquisador

Acesse `SUA_URL/?admin` e informe a senha (`admin.key`) para ver as contagens por
grupo. A senha **não** vai na URL. Os dados completos estão no Table Editor do Supabase
(exportáveis em CSV).

## O que ainda falta preencher

Marcados com `TODO`/`〔...〕` em `app.py`:

- **Campos do TCLE** entre colchetes (título, pesquisador, CEP/CAAE, duração).
- **Treinamento com checklist** do grupo Checklist (`screen_material`).
- **Vídeos** (bucket) e a tabela **`stimuli`** com `prob_deepfake`/`explanation`.
- **Questionários finais** (`screen_final`).
