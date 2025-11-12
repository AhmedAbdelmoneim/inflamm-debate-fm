"""Flyte workflows for inflamm-debate-fm."""

from pathlib import Path

from flytekit import task

from inflamm_debate_fm.utils.resources import get_task_resources


@task(requests=get_task_resources("cpu_task"))
def preprocess_data_task(
    ann_data_dir: str,
    embeddings_dir: str,
    combined_data_dir: str,
    load_embeddings: bool = True,
) -> str:
    """Flyte task for data preprocessing and combining."""
    from inflamm_debate_fm.data.load import combine_adatas, load_adatas
    from inflamm_debate_fm.data.preprocessing import preprocess_all_datasets

    ann_data_dir = Path(ann_data_dir)
    embeddings_dir = Path(embeddings_dir)
    combined_data_dir = Path(combined_data_dir)
    combined_data_dir.mkdir(parents=True, exist_ok=True)

    adatas = load_adatas(ann_data_dir, embeddings_dir, load_embeddings=load_embeddings)
    preprocess_all_datasets(adatas)

    combine_adatas(adatas, "human").write_h5ad(combined_data_dir / "human_combined.h5ad")
    combine_adatas(adatas, "mouse").write_h5ad(combined_data_dir / "mouse_combined.h5ad")

    return combined_data_dir


@task(requests=get_task_resources("embedding_task"))
def generate_embeddings_task(
    dataset_name: str,
    ann_data_dir: str,
    output_dir: str,
    batch_size: int = 256,
    device: str = "cuda",
) -> str:
    """Flyte task for generating embeddings."""
    import anndata as ad
    import numpy as np

    from inflamm_debate_fm.bulkformer.generate import generate_bulkformer_embeddings

    ann_data_dir = Path(ann_data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(ann_data_dir / f"{dataset_name}_orthologs.h5ad")
    embeddings = generate_bulkformer_embeddings(adata=adata, batch_size=batch_size, device=device)

    output_path = output_dir / f"{dataset_name}_transcriptome_embeddings.npy"
    np.save(output_path, embeddings)

    return output_dir


@task(requests=get_task_resources("cpu_task"))
def probe_within_species_task(
    species: str,
    combined_data_dir: str,
    output_dir: str,
    n_cv_folds: int = 10,
) -> str:
    """Flyte task for within-species probing."""
    import pickle

    from inflamm_debate_fm.cli.utils import get_setup_transforms
    from inflamm_debate_fm.data.load import load_combined_adatas
    from inflamm_debate_fm.modeling.evaluation import evaluate_within_species

    combined_data_dir = Path(combined_data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_adatas = load_combined_adatas(combined_data_dir)
    setups = get_setup_transforms()

    all_results, all_roc_data = evaluate_within_species(
        adata=combined_adatas[species],
        setups=setups,
        species_name=species.capitalize(),
        n_cv_folds=n_cv_folds,
    )

    results_path = output_dir / f"{species}_results.pkl"
    with results_path.open("wb") as f:
        pickle.dump({"results": all_results, "roc_data": all_roc_data}, f)

    return output_dir


@task(requests=get_task_resources("cpu_task"))
def probe_cross_species_task(
    combined_data_dir: str,
    output_dir: str,
    n_cv_folds: int = 10,
) -> str:
    """Flyte task for cross-species probing."""
    import pickle

    from inflamm_debate_fm.cli.utils import get_setup_transforms
    from inflamm_debate_fm.data.load import load_combined_adatas
    from inflamm_debate_fm.modeling.evaluation import evaluate_cross_species

    combined_data_dir = Path(combined_data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_adatas = load_combined_adatas(combined_data_dir)
    setups = get_setup_transforms()

    all_results, all_roc_data = evaluate_cross_species(
        human_adata=combined_adatas["human"],
        mouse_adata=combined_adatas["mouse"],
        setups=setups,
        n_cv_folds=n_cv_folds,
    )

    results_path = output_dir / "cross_species_results.pkl"
    with results_path.open("wb") as f:
        pickle.dump({"results": all_results, "roc_data": all_roc_data}, f)

    return output_dir


@task(requests=get_task_resources("cpu_task"))
def analyze_coefficients_task(
    combined_data_dir: str,
    output_dir: str,
) -> str:
    """Flyte task for coefficient analysis."""
    from inflamm_debate_fm.cli.utils import get_setup_transforms
    from inflamm_debate_fm.data.load import load_combined_adatas
    from inflamm_debate_fm.modeling.coefficients import evaluate_linear_models

    combined_data_dir = Path(combined_data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_adatas = load_combined_adatas(combined_data_dir)
    setups = get_setup_transforms()

    evaluate_linear_models(
        human_adata=combined_adatas["human"],
        mouse_adata=combined_adatas["mouse"],
        setups=setups,
        output_dir=output_dir,
    )

    return output_dir


@task(requests=get_task_resources("cpu_task"))
def analyze_gsea_task(
    coeff_dir: str,
    output_dir: str,
) -> str:
    """Flyte task for GSEA analysis."""
    from inflamm_debate_fm.analysis.gsea import post_analysis_from_coeffs

    coeff_dir = Path(coeff_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    post_analysis_from_coeffs(coeff_dir, output_dir)

    return output_dir
