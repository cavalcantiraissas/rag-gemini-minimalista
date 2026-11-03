"""
RAG minimalista com Gemini: interface Streamlit.

Mesmo pipeline do rag.py (chunking, embeddings, busca por cosseno e resposta
com fontes), com upload de arquivos .pdf/.txt/.md e conversa no navegador.

Uso:
    export GEMINI_API_KEY="sua-chave"   # obtida em https://aistudio.google.com
    streamlit run app.py
"""

from __future__ import annotations

import os

import streamlit as st
from google.genai import errors

from rag import (
    DOCS_DIR,
    EMBEDDING_MODEL,
    EXTENSOES_SUPORTADAS,
    GEMINI_MODEL,
    TOP_K,
    ErroDeArquivo,
    ErroDeConfiguracao,
    IndiceVetorial,
    buscar,
    carregar_documentos,
    criar_cliente,
    criar_indice,
    descrever_erro_api,
    extrair_texto,
    gerar_chunks,
    montar_contexto,
    perguntar_ao_gemini,
)

PERGUNTAS_EXEMPLO = [
    "Quantos dias posso trabalhar de casa?",
    "Qual é o auxílio para internet e energia?",
    "Quantos dias de férias eu tenho por ano?",
]


def md(texto: str) -> str:
    """Escapa o cifrão, que o Streamlit interpretaria como fórmula (ex.: "R$ 150,00")."""
    return texto.replace("$", r"\$")


# ---------------------------------------------------------------------------
# Indexação (com cache: só gera embeddings de novo se os documentos mudarem)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, max_entries=10)
def indexar(documentos: dict[str, str], modelo: str, _chave: str) -> IndiceVetorial:
    # O parâmetro com "_" fica fora da chave do cache (a chave da API não é guardada nele).
    return criar_indice(gerar_chunks(documentos), criar_cliente(_chave))


def ler_uploads(arquivos: list) -> dict[str, str]:
    """Extrai o texto dos arquivos enviados; avisa e pula os que não puderem ser lidos."""
    documentos: dict[str, str] = {}
    for arquivo in arquivos:
        try:
            documentos[arquivo.name] = extrair_texto(arquivo.name, arquivo.getvalue())
        except ErroDeArquivo as e:
            st.sidebar.warning(f"Ignorando {e}")
    return documentos


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------

def barra_lateral() -> tuple[str, dict[str, str], int]:
    """Mostra chave, upload e ajustes. Devolve (chave, documentos, top_k)."""
    with st.sidebar:
        chave = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
        if not chave:
            chave = st.text_input(
                "Chave da API do Gemini",
                type="password",
                help="Gere uma chave gratuita em https://aistudio.google.com. Ela fica só nesta sessão.",
            )

        st.subheader("Documentos")
        arquivos = st.file_uploader(
            "Envie PDFs ou arquivos de texto",
            type=sorted(ext.lstrip(".") for ext in EXTENSOES_SUPORTADAS),
            accept_multiple_files=True,
        )
        usar_pasta = st.checkbox(
            "Incluir a pasta documentos/",
            value=not arquivos,
            help="Inclui os arquivos que estão na pasta documentos/ do projeto, como o exemplo de política de home office.",
        )

        documentos = ler_uploads(arquivos or [])
        if usar_pasta:
            try:
                documentos = carregar_documentos(DOCS_DIR) | documentos
            except ErroDeConfiguracao as e:
                st.warning(str(e))

        st.subheader("Ajustes")
        top_k = st.slider(
            "Trechos por pergunta",
            min_value=1,
            max_value=10,
            value=TOP_K,
            help="Quantos trechos dos documentos são enviados ao Gemini junto com a pergunta.",
        )
        if st.button("Limpar conversa", width="stretch"):
            st.session_state.mensagens = []

        st.caption(f"Respostas: `{GEMINI_MODEL}` · Embeddings: `{EMBEDDING_MODEL}`")
        st.caption(
            "No plano gratuito, o Google pode usar o texto enviado para melhorar os produtos dele. "
            "Não envie documentos confidenciais."
        )
    return chave, documentos, top_k


# ---------------------------------------------------------------------------
# Conversa
# ---------------------------------------------------------------------------

def mostrar_fontes(fontes: list[dict]) -> None:
    with st.expander(f"Trechos recuperados ({len(fontes)})"):
        for fonte in fontes:
            st.markdown(f"**{md(fonte['referencia'])}** · similaridade {fonte['similaridade']:.3f}")
            st.text(fonte["texto"])


def responder(pergunta: str, indice: IndiceVetorial, chave: str, top_k: int) -> dict:
    """Busca os trechos, pergunta ao Gemini e devolve a mensagem do assistente."""
    cliente = criar_cliente(chave)
    resultados = buscar(pergunta, indice, cliente, k=top_k)
    resposta = perguntar_ao_gemini(cliente, pergunta, montar_contexto(resultados))
    fontes = [
        {"referencia": chunk.referencia, "similaridade": score, "texto": chunk.texto}
        for chunk, score in resultados
    ]
    return {"papel": "assistant", "texto": resposta, "fontes": fontes}


def main() -> None:
    st.set_page_config(page_title="RAG com Gemini", page_icon="📄")
    st.title("Pergunte aos seus documentos")
    st.caption(
        "Envie PDFs ou textos e faça perguntas. As respostas usam só o que está nos documentos "
        "e citam de onde vieram. Cada pergunta é independente: o modelo não vê as anteriores."
    )

    st.session_state.setdefault("mensagens", [])
    chave, documentos, top_k = barra_lateral()

    if not chave:
        st.info("Informe a chave da API do Gemini na barra lateral para começar.")
        return
    if not documentos:
        st.info("Envie um arquivo na barra lateral ou marque **Incluir a pasta documentos/**.")
        return

    try:
        with st.spinner("Gerando embeddings dos documentos..."):
            indice = indexar(documentos, EMBEDDING_MODEL, chave)
    except errors.APIError as e:
        st.error(descrever_erro_api(e))
        return

    nomes = ", ".join(sorted(documentos))
    st.success(f"{len(documentos)} documento(s), {len(indice.chunks)} trecho(s) indexados: {md(nomes)}")

    for mensagem in st.session_state.mensagens:
        with st.chat_message(mensagem["papel"]):
            st.markdown(md(mensagem["texto"]))
            if mensagem.get("fontes"):
                mostrar_fontes(mensagem["fontes"])

    pergunta = st.chat_input("Faça uma pergunta sobre os documentos")
    if not st.session_state.mensagens and not pergunta:
        st.caption("Sugestões para o documento de exemplo:")
        for coluna, exemplo in zip(st.columns(len(PERGUNTAS_EXEMPLO)), PERGUNTAS_EXEMPLO):
            if coluna.button(exemplo, width="stretch"):
                pergunta = exemplo
    if not pergunta:
        return

    st.session_state.mensagens.append({"papel": "user", "texto": pergunta})
    with st.chat_message("user"):
        st.markdown(md(pergunta))

    with st.chat_message("assistant"):
        try:
            with st.spinner("Buscando nos documentos..."):
                mensagem = responder(pergunta, indice, chave, top_k)
        except errors.APIError as e:
            st.error(descrever_erro_api(e))
            st.session_state.mensagens.pop()  # permite repetir a pergunta
            return
        st.markdown(md(mensagem["texto"]))
        mostrar_fontes(mensagem["fontes"])
    st.session_state.mensagens.append(mensagem)
    st.rerun()  # redesenha a conversa sem os botões de sugestão


if __name__ == "__main__":
    main()
