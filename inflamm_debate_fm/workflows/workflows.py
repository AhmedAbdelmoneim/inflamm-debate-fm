"""Flyte workflow definitions for pipeline orchestration.

This module provides Flyte workflows for orchestrating the data preprocessing,
embedding generation, probing experiments, and analysis pipeline.

Note: Flyte workflows are optional. The project can be run using the CLI
or Python API directly. These workflows are provided for users who want
to run the pipeline on a Flyte cluster.
"""

from pathlib import Path
from typing import Dict

from flytekit import Resources, task, workflow

from inflamm_debate_fm.config.config import get_config


# Resource specifications from config
def get_task_resources(resource_type: str = "cpu_task") -> Resources:
    """Get resource specifications from config.

    Args:
        resource_type: Type of resource specification ('cpu_task', 'gpu_task', 'embedding_task').

    Returns:
        Flyte Resources object.
    """
    config = get_config()
    hpc_config = config.get("hpc", {})
    task_config = hpc_config.get(resource_type, {})

    cpu = task_config.get("cpu", "2")
    memory = task_config.get("memory", "4Gi")
    gpu = task_config.get("gpu", "0")
    storage = task_config.get("storage", "10Gi")

    return Resources(
        cpu=cpu,
        mem=memory,
        gpu=gpu,
        storage=storage,
    )


@task(requests=get_task_resources("cpu_task"))
def preprocess_data_task(
    ann_data_dir: str,
    embeddings_dir: str,
    combined_data_dir: str,
    load_embeddings: bool = True,
) -> str:
    """Flyte task for data preprocessing.

    Args:
        ann_data_dir: Directory containing AnnData files.
        embeddings_dir: Directory containing embedding files.
        combined_data_dir: Directory to save combined datasets.
        load_embeddings: Whether to load and add embeddings.

    Returns:
        Path to combined data directory.
    """
    from inflamm_debate_fm.dataset import main as preprocess_main

    preprocess_main(
        ann_data_dir=Path(ann_data_dir),
        embeddings_dir=Path(embeddings_dir),
        combined_data_dir=Path(combined_data_dir),
        load_embeddings=load_embeddings,
        skip_preprocessing=False,
        skip_combining=False,
    )

    return combined_data_dir


@task(requests=get_task_resources("embedding_task"))
def generate_embeddings_task(
    dataset_name: str,
    ann_data_dir: str,
    output_dir: str,
    model_name: str = "bulkformer",
    flavor: str = "default",
    batch_size: int = 32,
    device: str = "cuda",
) -> str:
    """Flyte task for embedding generation.

    Args:
        dataset_name: Name of the dataset to process.
        ann_data_dir: Directory containing AnnData files.
        output_dir: Directory to save embeddings.
        model_name: Name of the embedding model.
        flavor: Embedding flavor identifier.
        batch_size: Batch size for embedding generation.
        device: Device to use ('cpu' or 'cuda').

    Returns:
        Path to output directory.
    """
    from inflamm_debate_fm.cli import embed_generate

    embed_generate(
        dataset_name=dataset_name,
        ann_data_dir=Path(ann_data_dir),
        output_dir=Path(output_dir),
        model_name=model_name,
        flavor=flavor,
        batch_size=batch_size,
        device=device,
    )

    return output_dir


@task(requests=get_task_resources("gpu_task"))
def probe_within_species_task(
    species: str,
    combined_data_dir: str,
    output_dir: str,
    n_cv_folds: int = 10,
    use_wandb: bool = True,
) -> str:
    """Flyte task for within-species probing experiments.

    Args:
        species: Species to probe ('human' or 'mouse').
        combined_data_dir: Directory containing combined AnnData files.
        output_dir: Directory to save results.
        n_cv_folds: Number of CV folds.
        use_wandb: Whether to log to wandb.

    Returns:
        Path to output directory.
    """
    from inflamm_debate_fm.cli import probe_within_species

    probe_within_species(
        species=species,
        combined_data_dir=Path(combined_data_dir),
        output_dir=Path(output_dir),
        n_cv_folds=n_cv_folds,
        use_wandb=use_wandb,
        wandb_project=None,
        wandb_entity=None,
        wandb_tags=None,
    )

    return output_dir


@task(requests=get_task_resources("gpu_task"))
def probe_cross_species_task(
    combined_data_dir: str,
    output_dir: str,
    use_wandb: bool = True,
) -> str:
    """Flyte task for cross-species probing experiments.

    Args:
        combined_data_dir: Directory containing combined AnnData files.
        output_dir: Directory to save results.
        use_wandb: Whether to log to wandb.

    Returns:
        Path to output directory.
    """
    from inflamm_debate_fm.cli import probe_cross_species

    probe_cross_species(
        combined_data_dir=Path(combined_data_dir),
        output_dir=Path(output_dir),
        use_wandb=use_wandb,
        wandb_project=None,
        wandb_entity=None,
        wandb_tags=None,
    )

    return output_dir


@task(requests=get_task_resources("cpu_task"))
def analyze_coefficients_task(
    combined_data_dir: str,
    output_dir: str,
    use_wandb: bool = True,
) -> str:
    """Flyte task for coefficient analysis.

    Args:
        combined_data_dir: Directory containing combined AnnData files.
        output_dir: Directory to save coefficients.
        use_wandb: Whether to log to wandb.

    Returns:
        Path to output directory.
    """
    from inflamm_debate_fm.cli import analyze_coefficients

    analyze_coefficients(
        combined_data_dir=Path(combined_data_dir),
        output_dir=Path(output_dir),
        use_wandb=use_wandb,
    )

    return output_dir


@task(requests=get_task_resources("cpu_task"))
def analyze_gsea_task(
    coeff_dir: str,
    output_dir: str,
) -> str:
    """Flyte task for GSEA analysis.

    Args:
        coeff_dir: Directory containing coefficient files.
        output_dir: Directory to save GSEA results.

    Returns:
        Path to output directory.
    """
    from inflamm_debate_fm.cli import analyze_gsea

    analyze_gsea(
        coeff_dir=Path(coeff_dir),
        output_dir=Path(output_dir),
    )

    return output_dir


@workflow
def preprocessing_workflow(
    ann_data_dir: str,
    embeddings_dir: str,
    combined_data_dir: str,
    load_embeddings: bool = True,
) -> str:
    """Flyte workflow for data preprocessing.

    Args:
        ann_data_dir: Directory containing AnnData files.
        embeddings_dir: Directory containing embedding files.
        combined_data_dir: Directory to save combined datasets.
        load_embeddings: Whether to load and add embeddings.

    Returns:
        Path to combined data directory.
    """
    return preprocess_data_task(
        ann_data_dir=ann_data_dir,
        embeddings_dir=embeddings_dir,
        combined_data_dir=combined_data_dir,
        load_embeddings=load_embeddings,
    )


@workflow
def embedding_workflow(
    dataset_name: str,
    ann_data_dir: str,
    output_dir: str,
    model_name: str = "bulkformer",
    flavor: str = "default",
    batch_size: int = 32,
    device: str = "cuda",
) -> str:
    """Flyte workflow for embedding generation.

    Args:
        dataset_name: Name of the dataset to process.
        ann_data_dir: Directory containing AnnData files.
        output_dir: Directory to save embeddings.
        model_name: Name of the embedding model.
        flavor: Embedding flavor identifier.
        batch_size: Batch size for embedding generation.
        device: Device to use ('cpu' or 'cuda').

    Returns:
        Path to output directory.
    """
    return generate_embeddings_task(
        dataset_name=dataset_name,
        ann_data_dir=ann_data_dir,
        output_dir=output_dir,
        model_name=model_name,
        flavor=flavor,
        batch_size=batch_size,
        device=device,
    )


@workflow
def probing_workflow(
    combined_data_dir: str,
    within_species_output_dir: str,
    cross_species_output_dir: str,
    n_cv_folds: int = 10,
    use_wandb: bool = True,
) -> Dict[str, str]:
    """Flyte workflow for probing experiments.

    Args:
        combined_data_dir: Directory containing combined AnnData files.
        within_species_output_dir: Directory to save within-species results.
        cross_species_output_dir: Directory to save cross-species results.
        n_cv_folds: Number of CV folds.
        use_wandb: Whether to log to wandb.

    Returns:
        Dictionary with output directories for human, mouse, and cross-species results.
    """
    human_output = probe_within_species_task(
        species="human",
        combined_data_dir=combined_data_dir,
        output_dir=f"{within_species_output_dir}/human",
        n_cv_folds=n_cv_folds,
        use_wandb=use_wandb,
    )

    mouse_output = probe_within_species_task(
        species="mouse",
        combined_data_dir=combined_data_dir,
        output_dir=f"{within_species_output_dir}/mouse",
        n_cv_folds=n_cv_folds,
        use_wandb=use_wandb,
    )

    cross_species_output = probe_cross_species_task(
        combined_data_dir=combined_data_dir,
        output_dir=cross_species_output_dir,
        use_wandb=use_wandb,
    )

    return {
        "human": human_output,
        "mouse": mouse_output,
        "cross_species": cross_species_output,
    }


@workflow
def analysis_workflow(
    combined_data_dir: str,
    coeff_dir: str,
    gsea_output_dir: str,
    use_wandb: bool = True,
) -> Dict[str, str]:
    """Flyte workflow for analysis (coefficients and GSEA).

    Args:
        combined_data_dir: Directory containing combined AnnData files.
        coeff_dir: Directory to save coefficients.
        gsea_output_dir: Directory to save GSEA results.
        use_wandb: Whether to log to wandb.

    Returns:
        Dictionary with output directories for coefficients and GSEA results.
    """
    coeff_output = analyze_coefficients_task(
        combined_data_dir=combined_data_dir,
        output_dir=coeff_dir,
        use_wandb=use_wandb,
    )

    gsea_output = analyze_gsea_task(
        coeff_dir=coeff_output,
        output_dir=gsea_output_dir,
    )

    return {
        "coefficients": coeff_output,
        "gsea": gsea_output,
    }


@workflow
def full_pipeline_workflow(
    ann_data_dir: str,
    embeddings_dir: str,
    combined_data_dir: str,
    within_species_output_dir: str,
    cross_species_output_dir: str,
    coeff_dir: str,
    gsea_output_dir: str,
    load_embeddings: bool = True,
    n_cv_folds: int = 10,
    use_wandb: bool = True,
) -> Dict[str, str]:
    """Full pipeline workflow orchestrating all stages.

    Args:
        ann_data_dir: Directory containing AnnData files.
        embeddings_dir: Directory containing embedding files.
        combined_data_dir: Directory to save combined datasets.
        within_species_output_dir: Directory to save within-species results.
        cross_species_output_dir: Directory to save cross-species results.
        coeff_dir: Directory to save coefficients.
        gsea_output_dir: Directory to save GSEA results.
        load_embeddings: Whether to load and add embeddings.
        n_cv_folds: Number of CV folds.
        use_wandb: Whether to log to wandb.

    Returns:
        Dictionary with all output directories.
    """
    # Step 1: Preprocess data
    combined_data = preprocessing_workflow(
        ann_data_dir=ann_data_dir,
        embeddings_dir=embeddings_dir,
        combined_data_dir=combined_data_dir,
        load_embeddings=load_embeddings,
    )

    # Step 2: Run probing experiments
    probing_results = probing_workflow(
        combined_data_dir=combined_data,
        within_species_output_dir=within_species_output_dir,
        cross_species_output_dir=cross_species_output_dir,
        n_cv_folds=n_cv_folds,
        use_wandb=use_wandb,
    )

    # Step 3: Run analysis
    analysis_results = analysis_workflow(
        combined_data_dir=combined_data,
        coeff_dir=coeff_dir,
        gsea_output_dir=gsea_output_dir,
        use_wandb=use_wandb,
    )

    return {
        "combined_data": combined_data,
        **probing_results,
        **analysis_results,
    }
