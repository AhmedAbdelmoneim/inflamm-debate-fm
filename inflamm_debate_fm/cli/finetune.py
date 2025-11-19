"""CLI commands for fine-tuning."""

from pathlib import Path

from loguru import logger
import typer

from inflamm_debate_fm.finetuning.train import train_lora_model

app = typer.Typer(help="Fine-tuning commands for LoRA models")


@app.command()
def train(
    species: str = typer.Option(
        ...,
        "--species",
        "-s",
        help="Species to train on: 'human', 'mouse', or 'combined'",
    ),
    n_inflammation: int = typer.Option(
        32, "--n-inflammation", "-ni", help="Number of inflammation samples"
    ),
    n_control: int = typer.Option(32, "--n-control", "-nc", help="Number of control samples"),
    n_epochs: int = typer.Option(
        50, "--epochs", "-e", help="Number of training epochs (default: 50)"
    ),
    batch_size: int = typer.Option(8, "--batch-size", "-b", help="Batch size"),
    learning_rate: float = typer.Option(1e-4, "--lr", help="Learning rate"),
    weight_decay: float = typer.Option(0.01, "--weight-decay", help="Weight decay"),
    device: str = typer.Option("cuda", "--device", "-d", help="Device: 'cuda' or 'cpu'"),
    output_dir: Path = typer.Option(
        None, "--output-dir", "-o", help="Output directory for checkpoints"
    ),
    random_seed: int = typer.Option(42, "--seed", help="Random seed"),
    use_wandb: bool = typer.Option(False, "--use-wandb", help="Log to Weights & Biases"),
    early_stopping_patience: int = typer.Option(
        7, "--early-stopping-patience", "-p", help="Early stopping patience (default: 7)"
    ),
):
    """Train a LoRA fine-tuned model for inflammation classification.

    This command fine-tunes the BulkFormer model using LoRA (Low-Rank Adaptation)
    for inflammation vs control classification. It uses 32+32 samples by default
    and tracks which samples were used for fine-tuning to exclude them from
    downstream evaluation.

    Examples:
        # Train on human data
        python -m inflamm_debate_fm.cli finetune train --species human

        # Train on combined human+mouse data with custom settings
        python -m inflamm_debate_fm.cli finetune train \\
            --species combined \\
            --n-inflammation 32 \\
            --n-control 32 \\
            --epochs 20 \\
            --batch-size 16 \\
            --lr 5e-5 \\
            --use-wandb
    """
    if species not in ["human", "mouse", "combined"]:
        raise typer.BadParameter(
            f"species must be 'human', 'mouse', or 'combined', got '{species}'"
        )

    logger.info(f"Starting LoRA fine-tuning for {species}")
    logger.info(
        f"Configuration: {n_inflammation} inflammation + {n_control} control samples, "
        f"{n_epochs} epochs, batch_size={batch_size}, lr={learning_rate}, "
        f"early_stopping_patience={early_stopping_patience}"
    )

    try:
        output_path = train_lora_model(
            species=species,
            n_inflammation=n_inflammation,
            n_control=n_control,
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=device,
            output_dir=output_dir,
            random_seed=random_seed,
            use_wandb=use_wandb,
            early_stopping_patience=early_stopping_patience,
        )
        logger.success(f"Fine-tuning complete! Checkpoints saved to {output_path}")
        typer.echo("\n✓ Fine-tuning complete!")
        typer.echo(f"  Checkpoints: {output_path}")
        typer.echo(f"  Sample metadata: {output_path / f'finetuning_samples_{species}.csv'}")
        typer.echo(f"  Summary: {output_path / f'finetuning_summary_{species}.json'}")
    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
