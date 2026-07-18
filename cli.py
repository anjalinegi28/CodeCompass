"""
CodeCompass CLI.

    python cli.py ingest /path/to/your/project      # give it the WHOLE project folder
    python cli.py ingest ./my_repo --reset            # wipe index first, then ingest
    python cli.py ask "How does auth work?"
    python cli.py ask "How does auth work?" --provider claude
    python cli.py eval                                # run the RAGAS gate locally
    python cli.py stale-docs README.md src/auth.py
"""
from __future__ import annotations

import click

from app.eval.ragas_eval import main as run_eval_main
from app.ingestion.chunker import chunk_files
from app.ingestion.loader import load_folder
from app.rag.agent import ask as rag_ask
from app.stale_docs.watcher import check_doc_staleness
from app.vectorstore import get_store


@click.group()
def cli():
    """CodeCompass — agentic RAG over your codebase, with an eval-gated CI/CD pipeline."""


@cli.command()
@click.argument("folder_path", type=click.Path(exists=True, file_okay=False))
@click.option("--reset", is_flag=True, help="Clear the existing index before ingesting.")
def ingest(folder_path: str, reset: bool):
    """Ingest an ENTIRE project folder (not just a single file)."""
    click.echo(f"Walking project folder: {folder_path}")
    loaded_files = load_folder(folder_path)

    if not loaded_files:
        click.echo("No ingestible files found (check allowed_extensions in app/config.py).")
        return

    click.echo(f"Found {len(loaded_files)} files. Chunking...")
    chunks = chunk_files(loaded_files)
    click.echo(f"Produced {len(chunks)} chunks. Embedding + indexing...")

    store = get_store()
    if reset:
        store.reset()
    store.add(chunks)

    click.echo(f"Done. Indexed {len(chunks)} chunks from {len(loaded_files)} files.")


@cli.command()
@click.argument("question")
@click.option("--k", default=5, help="Number of chunks to retrieve.")
@click.option("--provider", default=None, help="Override LLM_PROVIDER for this call (openai|gemini|claude).")
def ask(question: str, k: int, provider: str | None):
    """Ask a question about the ingested codebase."""
    result = rag_ask(question, k=k, provider=provider)
    click.echo(f"\n{result.text}\n")
    if result.citations:
        click.echo("Sources:")
        for c in result.citations:
            click.echo(f"  - {c}")


@cli.command(name="eval")
def eval_cmd():
    """Run the RAGAS eval gate locally (same thing CI runs on every PR)."""
    raise SystemExit(run_eval_main())


@cli.command(name="stale-docs")
@click.argument("doc_path", type=click.Path(exists=True))
@click.argument("code_path", type=click.Path(exists=True))
def stale_docs_cmd(doc_path: str, code_path: str):
    """Check whether a doc file has gone stale relative to a code file."""
    doc_text = open(doc_path).read()
    code_text = open(code_path).read()

    result = check_doc_staleness(doc_path, doc_text, code_text)

    click.echo(f"\nDoc: {doc_path}")
    click.echo(f"Similarity to current code: {result.similarity:.3f}")
    click.echo(f"Stale: {'YES' if result.is_stale else 'no'}")

    if result.is_stale and result.drafted_update:
        click.echo("\n--- Drafted update ---\n")
        click.echo(result.drafted_update)


if __name__ == "__main__":
    cli()
