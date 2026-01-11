"""CLI interface for Evidence Coverage Evaluator.

Author: Goutam Adwant (gadwant)

This module provides a command-line interface using Click for running
evidence coverage evaluation from the terminal. It supports all evaluation
parameters and can output results as JSON or HTML reports.
"""

import json
import sys
from pathlib import Path
from typing import Optional
import click
from ece.evaluator import EvidenceCoverageEvaluator
from ece.models import Context, Passage
from ece.visualizer import HTMLVisualizer


@click.group()
@click.version_option(version="0.1.0")
def main():
    """Evidence Coverage Evaluator - Evaluate RAG answer grounding."""
    pass


@main.command()
@click.option(
    "--answer",
    "-a",
    required=True,
    type=click.Path(exists=True),
    help="Path to answer text file"
)
@click.option(
    "--context",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to context JSON file"
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output JSON file path (default: stdout)"
)
@click.option(
    "--html",
    type=click.Path(),
    help="Generate HTML report at specified path"
)
@click.option(
    "--threshold",
    "-t",
    type=float,
    default=0.7,
    help="Minimum support score threshold (0-1)"
)
@click.option(
    "--retrieval-method",
    type=click.Choice(["bm25", "embedding"], case_sensitive=False),
    default="bm25",
    help="Evidence retrieval method"
)
@click.option(
    "--retrieval-top-k",
    type=int,
    default=3,
    help="Number of evidence candidates per claim"
)
@click.option(
    "--nli-model",
    default="roberta-large-mnli",
    help="NLI model name"
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Verbose output"
)
def evaluate(
    answer: str,
    context: str,
    output: Optional[str],
    html: Optional[str],
    threshold: float,
    retrieval_method: str,
    retrieval_top_k: int,
    nli_model: str,
    verbose: bool
):
    """Evaluate evidence coverage for a RAG answer.
    
    This command runs the complete ECE evaluation pipeline on the provided
    answer and context files, outputting coverage metrics and claim analyses.
    """
    # Load answer file
    try:
        with open(answer, "r", encoding="utf-8") as f:
            answer_text = f.read()
    except Exception as e:
        click.echo(f"Error reading answer file: {e}", err=True)
        sys.exit(1)
    
    # Load and parse context JSON
    try:
        with open(context, "r", encoding="utf-8") as f:
            context_data = json.load(f)
        
        if "passages" not in context_data:
            click.echo("Error: context JSON must have 'passages' key", err=True)
            sys.exit(1)
        
        passages = [
            Passage(id=p["id"], text=p["text"])
            for p in context_data["passages"]
        ]
        context_obj = Context(passages=passages)
    except json.JSONDecodeError as e:
        click.echo(f"Error parsing context JSON: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error reading context file: {e}", err=True)
        sys.exit(1)
    
    # Initialize evaluator with specified parameters
    try:
        evaluator = EvidenceCoverageEvaluator(
            retrieval_method=retrieval_method,
            retrieval_top_k=retrieval_top_k,
            nli_model=nli_model,
            threshold=threshold
        )
    except Exception as e:
        click.echo(f"Error initializing evaluator: {e}", err=True)
        sys.exit(1)
    
    # Run evaluation
    try:
        if verbose:
            click.echo("Running evaluation in lightweight mode (NLI-based)...")
        
        result = evaluator.evaluate(answer_text, context_obj)
        
        # Convert to dict for JSON serialization
        result_dict = result.model_dump()
        
        # Output results
        output_json = json.dumps(result_dict, indent=2, ensure_ascii=False)
        
        if output:
            with open(output, "w", encoding="utf-8") as f:
                f.write(output_json)
            click.echo(f"Results saved to {output}")
        else:
            click.echo(output_json)
        
        # Generate HTML report if requested
        if html:
            try:
                visualizer = HTMLVisualizer()
                html_path = visualizer.generate_report(
                    result, answer_text, html, "Evidence Coverage Evaluation Report"
                )
                click.echo(f"HTML report generated: {html_path}")
            except Exception as e:
                click.echo(f"Error generating HTML report: {e}", err=True)
                if verbose:
                    import traceback
                    traceback.print_exc()
        
        # Exit with error code if coverage is critically low
        if result.coverage_score < 0.5:
            sys.exit(1)
        
    except Exception as e:
        click.echo(f"Error during evaluation: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
