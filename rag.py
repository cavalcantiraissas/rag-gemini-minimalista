"""
RAG minimalista com Gemini.

RAG (Retrieval-Augmented Generation) = "geração aumentada por recuperação":
antes de perguntar ao modelo, buscamos nos seus documentos os trechos mais
relevantes e os enviamos junto com a pergunta. Assim o modelo responde com
base no seu conteúdo, e não no que "acha" que sabe.

Fluxo:
    1. Carrega arquivos .txt/.md/.pdf da pasta ./documentos (ao lado deste script)
    2. Divide o texto em chunks (pedaços) com sobreposição
    3. Gera embeddings (vetores numéricos que representam o significado) via API do Gemini
    4. Busca os chunks mais parecidos com a pergunta por similaridade de cosseno (NumPy, em memória)
    5. Envia os chunks como contexto para o Gemini responder estritamente com base neles

Instalação:
    pip install -r requirements.txt

Uso:
    export GEMINI_API_KEY="sua-chave"   # obtida em https://aistudio.google.com
    python rag.py            # no terminal
    streamlit run app.py     # interface web, com upload de arquivos
"""

from __future__ import annotations

import io
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from google import genai
from google.genai import errors, types
from pypdf import PdfReader
from pypdf.errors import PyPdfError

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

# Resolvida a partir do local do script, para funcionar de qualquer diretório.
DOCS_DIR = Path(__file__).resolve().parent / "documentos"
EXTENSOES_SUPORTADAS = {".txt", ".md", ".pdf"}

# Modelos com camada gratuita; ambos podem ser trocados por variáveis de ambiente.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

EMBEDDING_BATCH_SIZE = 100  # máximo de textos aceitos por requisição de embeddings
MAX_OUTPUT_TOKENS = 4096    # limite de tamanho da resposta do modelo

# Chunks pequenos deixam a busca mais precisa; a sobreposição evita que uma
# informação seja cortada ao meio entre dois chunks e se perca na busca.
CHUNK_SIZE = 800      # tamanho máximo de cada chunk, em caracteres
CHUNK_OVERLAP = 150   # sobreposição entre chunks consecutivos, em caracteres
TOP_K = 4             # quantidade de chunks enviados como contexto

SYSTEM_PROMPT = """Você é um assistente que responde perguntas usando APENAS o contexto fornecido.

Regras:
- Baseie sua resposta exclusivamente nos trechos dentro de <contexto>.
- Se a resposta não estiver no contexto, diga claramente: "Não encontrei essa informação nos documentos fornecidos."
- Não use conhecimento externo nem invente informações.
- Ao final, cite as fontes usadas no formato [arquivo#chunk]. Se não encontrar a informação, não cite fontes.
- Responda em português, de forma clara e objetiva."""


class ErroDeConfiguracao(Exception):
    """Problema que o usuário resolve ajustando o ambiente (chave, pasta, parâmetros)."""


class ErroDeArquivo(Exception):
    """Arquivo que não pôde ser lido; ele é ignorado e os demais seguem."""


# ---------------------------------------------------------------------------
# Estruturas de dados
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """Um pedaço de texto com a referência de origem."""
    texto: str
    fonte: str
    indice: int

    @property
    def referencia(self) -> str:
        """Identificador legível do chunk, ex.: 'manual.md#3'."""
        return f"{self.fonte}#{self.indice}"


@dataclass
class IndiceVetorial:
    """Índice em memória: chunks + matriz de embeddings normalizados."""
    chunks: list[Chunk]
    embeddings: np.ndarray  # shape: (n_chunks, dimensão); linha i ↔ chunks[i]


# ---------------------------------------------------------------------------
# Carregamento e chunking
# ---------------------------------------------------------------------------

def extrair_texto(nome: str, dados: bytes) -> str:
    """Extrai o texto de um arquivo .txt, .md ou .pdf a partir dos seus bytes."""
    if not nome.lower().endswith(".pdf"):
        try:
            return dados.decode("utf-8").strip()
        except UnicodeDecodeError:
            raise ErroDeArquivo(f"'{nome}' não está em UTF-8.") from None

    try:
        leitor = PdfReader(io.BytesIO(dados))
        # PDFs com restrição de cópia costumam ter senha vazia; os que pedem senha de verdade são recusados.
        if leitor.is_encrypted and not leitor.decrypt(""):
            raise ErroDeArquivo(f"'{nome}' é protegido por senha.")
        paginas = [pagina.extract_text() or "" for pagina in leitor.pages]
    except PyPdfError as e:
        raise ErroDeArquivo(f"'{nome}' não pôde ser lido como PDF ({e}).") from None

    texto = "\n\n".join(p.strip() for p in paginas if p.strip())
    if not texto:
        raise ErroDeArquivo(f"'{nome}' não tem texto selecionável (provavelmente é digitalizado e precisa de OCR).")
    return texto


def carregar_documentos(pasta: Path) -> dict[str, str]:
    """Lê todos os arquivos .txt/.md/.pdf da pasta (e subpastas) e retorna {caminho_relativo: conteúdo}."""
    if not pasta.is_dir():
        raise ErroDeConfiguracao(f"Pasta '{pasta}' não encontrada. Crie-a e adicione arquivos .txt, .md ou .pdf.")

    documentos: dict[str, str] = {}
    for arquivo in sorted(pasta.rglob("*")):
        if not (arquivo.is_file() and arquivo.suffix.lower() in EXTENSOES_SUPORTADAS):
            continue
        try:
            conteudo = extrair_texto(arquivo.name, arquivo.read_bytes())
        except ErroDeArquivo as e:
            print(f"[aviso] Ignorando {e}")
            continue
        if conteudo:
            documentos[str(arquivo.relative_to(pasta))] = conteudo

    if not documentos:
        raise ErroDeConfiguracao(f"Nenhum arquivo .txt, .md ou .pdf com conteúdo encontrado em '{pasta}'.")
    return documentos


def dividir_em_chunks(texto: str, tamanho: int = CHUNK_SIZE, sobreposicao: int = CHUNK_OVERLAP) -> list[str]:
    """
    Divide o texto em chunks de até `tamanho` caracteres, com sobreposição.
    Tenta cortar em quebras de parágrafo/frase para não partir ideias no meio.
    """
    if not 0 <= sobreposicao < tamanho:
        raise ErroDeConfiguracao("CHUNK_OVERLAP deve ser maior ou igual a 0 e menor que CHUNK_SIZE.")

    chunks: list[str] = []
    inicio = 0
    while inicio < len(texto):
        fim = min(inicio + tamanho, len(texto))

        # Se não chegamos ao fim do texto, recua até o separador mais "forte"
        # encontrado na segunda metade do chunk (parágrafo > linha > frase > palavra).
        if fim < len(texto):
            for separador in ("\n\n", "\n", ". ", " "):
                corte = texto.rfind(separador, inicio + tamanho // 2, fim)
                if corte != -1:
                    fim = corte + len(separador)
                    break

        pedaco = texto[inicio:fim].strip()
        if pedaco:
            chunks.append(pedaco)

        if fim >= len(texto):
            break
        # Volta `sobreposicao` caracteres; o max() garante que sempre avançamos.
        inicio = max(fim - sobreposicao, inicio + 1)

    return chunks


def gerar_chunks(documentos: dict[str, str]) -> list[Chunk]:
    """Aplica o chunking a todos os documentos, preservando a origem de cada pedaço."""
    return [
        Chunk(texto=pedaco, fonte=nome, indice=i)
        for nome, conteudo in documentos.items()
        for i, pedaco in enumerate(dividir_em_chunks(conteudo))
    ]


# ---------------------------------------------------------------------------
# Embeddings e busca vetorial
# ---------------------------------------------------------------------------

def gerar_embeddings(cliente: genai.Client, textos: list[str], tipo_tarefa: str) -> np.ndarray:
    """
    Gera embeddings via API do Gemini, em lotes, e normaliza cada vetor (norma L2 = 1).

    `tipo_tarefa` informa ao modelo como o texto será usado, o que melhora a busca:
    "RETRIEVAL_DOCUMENT" para os chunks e "RETRIEVAL_QUERY" para a pergunta.
    """
    vetores: list[list[float]] = []
    for i in range(0, len(textos), EMBEDDING_BATCH_SIZE):
        resultado = cliente.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=textos[i:i + EMBEDDING_BATCH_SIZE],
            config=types.EmbedContentConfig(task_type=tipo_tarefa),
        )
        vetores.extend(e.values for e in resultado.embeddings)

    matriz = np.array(vetores, dtype=np.float32)
    normas = np.linalg.norm(matriz, axis=1, keepdims=True)
    return matriz / np.maximum(normas, 1e-12)  # evita divisão por zero


def criar_indice(chunks: list[Chunk], cliente: genai.Client) -> IndiceVetorial:
    """Gera os embeddings de todos os chunks e monta o índice em memória."""
    embeddings = gerar_embeddings(cliente, [c.texto for c in chunks], "RETRIEVAL_DOCUMENT")
    return IndiceVetorial(chunks=chunks, embeddings=embeddings)


def buscar(pergunta: str, indice: IndiceVetorial, cliente: genai.Client, k: int = TOP_K) -> list[tuple[Chunk, float]]:
    """
    Retorna os k chunks mais similares à pergunta, do mais para o menos parecido.
    Como os vetores estão normalizados, o produto escalar é a similaridade de cosseno
    (1 = mesmo significado, 0 = sem relação).
    """
    vetor_pergunta = gerar_embeddings(cliente, [pergunta], "RETRIEVAL_QUERY")[0]
    similaridades = indice.embeddings @ vetor_pergunta
    melhores = np.argsort(similaridades)[::-1][:k]
    return [(indice.chunks[i], float(similaridades[i])) for i in melhores]


# ---------------------------------------------------------------------------
# Integração com o Gemini
# ---------------------------------------------------------------------------

def criar_cliente(api_key: str | None = None) -> genai.Client:
    """Cria o cliente do Gemini, validando a presença da chave de API (argumento ou variável de ambiente)."""
    api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ErroDeConfiguracao(
            "Variável GEMINI_API_KEY não definida. "
            "Gere uma chave em https://aistudio.google.com e execute: export GEMINI_API_KEY='sua-chave'"
        )
    return genai.Client(api_key=api_key)


def montar_contexto(resultados: list[tuple[Chunk, float]]) -> str:
    """Formata os chunks recuperados em blocos XML, para o modelo distinguir contexto e pergunta."""
    blocos = [
        f'<trecho fonte="{chunk.referencia}">\n{chunk.texto}\n</trecho>'
        for chunk, _ in resultados
    ]
    return "<contexto>\n" + "\n\n".join(blocos) + "\n</contexto>"


def perguntar_ao_gemini(cliente: genai.Client, pergunta: str, contexto: str) -> str:
    """Envia pergunta + contexto ao Gemini e retorna o texto da resposta."""
    resposta = cliente.models.generate_content(
        model=GEMINI_MODEL,
        contents=f"{contexto}\n\nPergunta: {pergunta}",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        ),
    )

    # response.text é None quando a resposta foi bloqueada ou veio vazia.
    if not resposta.text:
        motivo = resposta.prompt_feedback.block_reason if resposta.prompt_feedback else None
        return f"O modelo não retornou resposta{f' (bloqueio: {motivo})' if motivo else ''}."
    return resposta.text.strip()


def descrever_erro_api(erro: errors.APIError) -> str:
    """Traduz erros da API do Gemini em mensagens com a ação que o usuário deve tomar."""
    if erro.code in (401, 403):
        return "Chave de API inválida ou sem permissão. Verifique GEMINI_API_KEY."
    if erro.code == 404:
        return (f"Modelo não encontrado ('{GEMINI_MODEL}' ou '{EMBEDDING_MODEL}'). "
                "Ajuste GEMINI_MODEL ou GEMINI_EMBEDDING_MODEL.")
    if erro.code == 429:
        return "Limite da camada gratuita atingido. Aguarde um pouco e tente novamente."
    if isinstance(erro, errors.ServerError):
        return f"Falha temporária no servidor do Gemini ({erro.code}). Tente novamente."
    return f"Erro da API ({erro.code}): {erro.message}"


# ---------------------------------------------------------------------------
# Loop interativo
# ---------------------------------------------------------------------------

def loop_interativo(indice: IndiceVetorial, cliente: genai.Client) -> None:
    """Lê perguntas do terminal até o usuário digitar 'sair' (ou Ctrl+C / Ctrl+D)."""
    print("\nPronto! Faça sua pergunta (ou digite 'sair' para encerrar).\n")
    while True:
        try:
            pergunta = input("Pergunta> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAté logo!")
            return

        if not pergunta:
            continue
        if pergunta.lower() in {"sair", "exit", "quit"}:
            print("Até logo!")
            return

        # Um erro em uma pergunta não encerra o programa: mostra a mensagem e segue.
        try:
            resultados = buscar(pergunta, indice, cliente)
            print("\nChunks recuperados:")
            for chunk, score in resultados:
                print(f"  - {chunk.referencia}  (similaridade: {score:.3f})")

            resposta = perguntar_ao_gemini(cliente, pergunta, montar_contexto(resultados))
            print(f"\nResposta:\n{resposta}\n")
        except errors.APIError as e:
            print(f"\n[erro] {descrever_erro_api(e)}\n")


def main() -> int:
    """Ponto de entrada. Retorna o código de saída do processo (0 = sucesso)."""
    try:
        cliente = criar_cliente()

        print(f"Carregando documentos de '{DOCS_DIR}'...")
        documentos = carregar_documentos(DOCS_DIR)
        chunks = gerar_chunks(documentos)
        print(f"{len(documentos)} documento(s), {len(chunks)} chunk(s).")

        print(f"Gerando embeddings com '{EMBEDDING_MODEL}'...")
        indice = criar_indice(chunks, cliente)

        loop_interativo(indice, cliente)
        return 0
    except ErroDeConfiguracao as e:
        print(f"[erro] {e}", file=sys.stderr)
    except errors.APIError as e:
        print(f"[erro] {descrever_erro_api(e)}", file=sys.stderr)
    except KeyboardInterrupt:
        print("\nInterrompido.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
