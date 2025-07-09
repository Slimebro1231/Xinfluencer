#!/usr/bin/env python3
"""
Xinfluencer AI CLI
A comprehensive command-line interface for the AI system
"""

import click
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model.generate import TextGenerator

@click.group()
def cli():
    """A CLI for interacting with the Xinfluencer AI."""
    pass

@cli.command()
@click.option('--prompt', required=True, help='The prompt to generate text from.')
@click.option('--max-length', default=150, help='The maximum length of the generated text.')
@click.option('--temperature', default=0.7, help='The temperature for sampling.')
def generate(prompt, max_length, temperature):
    """Generate text from a prompt."""
    click.echo("Loading model...")
    generator = TextGenerator()
    click.echo("Generating text...")
    response = generator.generate(prompt, max_length=max_length, temperature=temperature)
    click.echo(f"\nResponse:\n{response}")

@cli.command()
def interactive():
    """Enter an interactive session with the AI."""
    click.echo("Loading model...")
    generator = TextGenerator()
    click.echo("Entering interactive mode. Type 'quit' or 'exit' to end.")
    while True:
        prompt = click.prompt("Your prompt")
        if prompt.lower() in ['quit', 'exit']:
            break
        response = generator.generate(prompt)
        click.echo(f"\nResponse:\n{response}")

@cli.command()
def status():
    """Get the status of the AI model and GPU."""
    click.echo("Getting status...")
    generator = TextGenerator()
    mem_usage = generator.get_memory_usage()
    if "error" in mem_usage:
        click.echo("Could not get GPU memory usage. Is CUDA available?")
    else:
        click.echo(f"GPU Memory: {mem_usage['allocated_gb']:.2f}GB allocated, {mem_usage['free_gb']:.2f}GB free.")
    click.echo(f"Model: {generator.model_name}")

if __name__ == '__main__':
    cli()