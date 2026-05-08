from collections import defaultdict
from dataclasses import dataclass
from loguru import logger
import numpy as np
import pandas as pd
from pathlib import Path
from plotnine import (
    ggplot,
    aes,
    facet_grid,
    geom_tile,
    geom_text,
    labs,
    theme,
    scale_y_discrete,
    coord_equal,
    scale_fill_continuous,
    facet_wrap,
    scale_color_manual,
    element_text,
)
from typing import Literal, List

from transformers import HfArgumentParser
from transformers.hf_argparser import HfArg


@dataclass
class Arguments:
    option: Literal["default", "larger_vec"] = HfArg(default="default")
    cfg: Literal[
        "2b_l10",
        "9b_l20",
        "q25_32b_l32",
    ] = HfArg(default="2b_l10")
    pos: Literal[
        "all",
        "all_prompt",
        "f4+l4",
        "f8",
        "l8",
        "f2+l2",
        "f4",
        "l4",
        "f1+l1",
        "f2",
        "l2",
    ] = HfArg(default="all")
    svs: List[str] = HfArg(default_factory=list)
    score_types: List[str] = HfArg(default_factory=list)
    seed_list: List[int] = HfArg(default_factory=list)
    factor_lr_list: List[str] = HfArg(default_factory=list)
    factor_init_size_list: List[int] = HfArg(default_factory=list)
    # output_base_dir = Path.home() / "reft_data" / f"concept_sweep_grid_{option}" / cfg
    output_base_dir: str = HfArg(default="reft_data/")

    def __post_init__(self):
        self.output_base_dir = Path(
            self.output_base_dir, f"concept_sweep_grid_{self.option}", self.cfg
        )

def harmonic_mean(scores):
    # Return 0 if any score is 0 to maintain strict evaluation
    if 0 in scores:
        return 0.0
    return len(scores) / sum(1 / s for s in scores)


def get_overall_score_concept_df(eval_df):
    harmonic_scores = []
    for _, row in eval_df.iterrows():
        mean = harmonic_mean(
            [row["concept_score"], row["relevance_score"], row["fluency_score"]]
        )
        harmonic_scores.append(mean)

    return np.mean(harmonic_scores).item()


def read_data(
    seed_list, factor_lr_list, factor_init_size_list, output_base_dir: Path, positions
):
    res_df = defaultdict(list)

    for k in ("add_free", "clamp_free"):
        for seed in seed_list:
            for factor_lr in factor_lr_list:
                for factor_init in factor_init_size_list:
                    for concept_id in range(10):
                        eval_path = (
                            output_base_dir
                            / f"outputs_{k}"
                            / f"{positions}"
                            / f"seed={seed}_scale={factor_init}_lr={factor_lr}"
                            / f"{concept_id}"
                            / "eval.parquet"
                        )
                        if not eval_path.exists():
                            continue

                        eval_df = pd.read_parquet(eval_path)
                        overall_score = get_overall_score_concept_df(eval_df)
                        concept_score = eval_df["concept_score"].mean()
                        instruct_score = eval_df["relevance_score"].mean()
                        fluency_score = eval_df["fluency_score"].mean()

                        res_df["concept_id"].append(concept_id)
                        res_df["method"].append(k)
                        res_df["positions"].append(positions)
                        res_df["seed"].append(seed)
                        res_df["factor_lr"].append(factor_lr)
                        res_df["factor_init"].append(factor_init)

                        res_df["overall_score"].append(overall_score)
                        res_df["concept_score"].append(concept_score)
                        res_df["instruct_score"].append(instruct_score)
                        res_df["fluency_score"].append(fluency_score)

    res_df = pd.DataFrame(res_df)
    if len(res_df) == 0:
        return None

    res_df["method"] = res_df["method"].replace(
        ["add_free", "clamp_free"], ["AddInv", "ClampInv"]
    )
    return res_df


def plot_side_by_side(res_df: pd.DataFrame, score_type, option, cfg, positions):
    plot_df = (
        res_df.groupby(["method", "factor_lr", "factor_init", "seed"])
        .agg(
            overall_score=("overall_score", "mean"),
            concept_score=("concept_score", "mean"),
            instruct_score=("instruct_score", "mean"),
            fluency_score=("fluency_score", "mean"),
        )
        .reset_index()
    )
    plot_df = (
        plot_df.groupby(["method", "factor_lr", "factor_init"])
        .agg(
            overall_score=("overall_score", "mean"),
            std=("overall_score", "std"),
            concept_score=("concept_score", "mean"),
            instruct_score=("instruct_score", "mean"),
            fluency_score=("fluency_score", "mean"),
        )
        .reset_index()
    )

    plot_df["text"] = plot_df[f"{score_type}_score"].apply(lambda x: f"{x:.2f}")
    # plot_df['text'] = plot_df[f'{score_type}_score'].apply(lambda x: f"{x:.2f}" if x > 0.77 else "")

    text_colors = []
    for _, row in plot_df.iterrows():
        if (
            abs(
                row[f"{score_type}_score"]
                - plot_df.groupby("method")[f"{score_type}_score"].max()[row["method"]]
            )
            < 1e-3
        ):
            text_colors.append("red")
        else:
            text_colors.append("black")
    plot_df["text_color"] = text_colors

    plot_df["factor_lr"] = plot_df["factor_lr"].apply(lambda s: float(s))
    plot_df["factor_lr"] = pd.Categorical(
        plot_df["factor_lr"],
        categories=sorted(plot_df["factor_lr"].unique()),
        ordered=True,
    )
    plot_df["factor_init"] = pd.Categorical(
        plot_df["factor_init"],
        categories=sorted(plot_df["factor_init"].unique()),
        ordered=True,
    )

    plot = (
        ggplot(plot_df, aes(x="factor_init", y="factor_lr", fill=f"{score_type}_score"))
        + facet_wrap("method")
        + geom_tile()
        + geom_text(aes(label="text", color="text_color"), size=8)
        # + geom_text(aes(label="text"), size=8)
        + scale_color_manual({"black": "black", "red": "red"}, guide=None)
        + labs(
            x="Factor initialization size ($\\beta$)",
            y="Factor learning rate ($\\eta_\\alpha$)",
            fill="Score",
            title=f"{score_type.capitalize()} score$\\uparrow$",
            # title="Concept score$\\uparrow$",
        )
        + theme(figure_size=(5, 3))
        + scale_y_discrete(
            labels=lambda lst: [
                f"{x:.1e}".replace("e-0", "e-").replace(".0", "") if x < 1 else f"{x}"
                for x in lst
            ],
        )
        # + scale_fill_gradient(low="steelblue", high="white")
        + scale_fill_continuous("summer")
        + coord_equal()
        # + guides(stroke=None)
    )

    save_dir = Path("figures", f"{option}")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{cfg}_{positions}_{score_type}.pdf"
    plot.save(save_path, dpi=100)
    # logger.warning(f"Saved to {save_path}")

    return plot


def plot_one_score_one_method(
    res_df: pd.DataFrame, method, score_type, option, cfg, positions
):
    if len(res_df[res_df['method']==method]) == 0:
        logger.warning(f"No results for `{method}`; skipping")
        return

    plot_df = (
        res_df.groupby(["method", "factor_lr", "factor_init", "seed"])
        .agg(
            overall_score=("overall_score", "mean"),
            concept_score=("concept_score", "mean"),
            instruct_score=("instruct_score", "mean"),
            fluency_score=("fluency_score", "mean"),
        )
        .reset_index()
    )
    plot_df = (
        plot_df.groupby(["method", "factor_lr", "factor_init"])
        .agg(
            overall_score=("overall_score", "mean"),
            std=("overall_score", "std"),
            concept_score=("concept_score", "mean"),
            instruct_score=("instruct_score", "mean"),
            fluency_score=("fluency_score", "mean"),
        )
        .reset_index()
    )
    plot_df = plot_df[plot_df["method"] == method]

    plot_df["text"] = plot_df[f"{score_type}_score"].apply(lambda x: f"{x:.2f}")
    # plot_df['text'] = plot_df[f'{score_type}_score'].apply(lambda x: f"{x:.2f}" if x > 0.77 else "")

    text_colors = []
    for _, row in plot_df.iterrows():
        if (
            abs(
                row[f"{score_type}_score"]
                - plot_df.groupby("method")[f"{score_type}_score"].max()[row["method"]]
            )
            < 1e-3
        ):
            text_colors.append("red")
        else:
            text_colors.append("black")
    plot_df["text_color"] = text_colors

    plot_df["factor_lr"] = plot_df["factor_lr"].apply(lambda s: float(s))
    plot_df["factor_lr"] = pd.Categorical(
        plot_df["factor_lr"],
        categories=sorted(plot_df["factor_lr"].unique()),
        ordered=True,
    )
    plot_df["factor_init"] = pd.Categorical(
        plot_df["factor_init"],
        categories=sorted(plot_df["factor_init"].unique()),
        ordered=True,
    )

    plot = (
        ggplot(plot_df, aes(x="factor_init", y="factor_lr", fill=f"{score_type}_score"))
        + geom_tile()
        + geom_text(aes(label="text", color="text_color"), size=10)
        # + geom_text(aes(label="text"), size=8)
        + scale_color_manual({"black": "black", "red": "red"}, guide=None)
        + labs(
            x="Factor initialization size ($\\beta$)",
            y="Factor learning rate ($\\eta_\\alpha$)",
            fill="Score",
            # title=f"{score_type.capitalize()} score$\\uparrow$",
            # title="Concept score$\\uparrow$",
        )
        + theme(
            figure_size=(3.5, 3),
            axis_title=element_text(size=14),
            axis_text=element_text(size=11),
            legend_text=element_text(size=11),
            legend_title=element_text(size=12),
        )
        + scale_y_discrete(
            labels=lambda lst: [
                f"{x:.1e}".replace("e-0", "e-").replace(".0", "") if x < 1 else f"{x}"
                for x in lst
            ],
        )
        # + scale_fill_gradient(low="white", high="steelblue")
        + scale_fill_continuous("summer")
        + coord_equal()
    )

    save_dir = Path("figures", f"{option}")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{cfg}_{positions}_{method}_{score_type}.pdf"
    plot.save(save_path, dpi=100)
    # logger.warning(f"Saved to {save_path}")

    return plot


def plot_score_diff(res_df: pd.DataFrame, score_type, option, cfg, positions):
    plot_df = (
        res_df.groupby(["method", "factor_lr", "factor_init", "seed"])
        .agg(
            overall_score=("overall_score", "mean"),
            concept_score=("concept_score", "mean"),
            instruct_score=("instruct_score", "mean"),
            fluency_score=("fluency_score", "mean"),
        )
        .reset_index()
    )
    plot_df = (
        plot_df.groupby(["method", "factor_lr", "factor_init"])
        .agg(
            overall_score=("overall_score", "mean"),
            std=("overall_score", "std"),
            concept_score=("concept_score", "mean"),
            instruct_score=("instruct_score", "mean"),
            fluency_score=("fluency_score", "mean"),
        )
        .reset_index()
    )

    plot_df["factor_lr"] = plot_df["factor_lr"].apply(lambda s: float(s))
    plot_df["factor_lr"] = pd.Categorical(
        plot_df["factor_lr"],
        categories=sorted(plot_df["factor_lr"].unique()),
        ordered=True,
    )
    plot_df["factor_init"] = pd.Categorical(
        plot_df["factor_init"],
        categories=sorted(plot_df["factor_init"].unique()),
        ordered=True,
    )

    add_free = plot_df[plot_df["method"] == "AddInv"].copy()
    clamp_free = plot_df[plot_df["method"] == "ClampInv"].copy()

    merged = add_free.merge(
        clamp_free, on=["factor_init", "factor_lr"], suffixes=("_add", "_clamp")
    )
    merged["Delta"] = (
        merged[f"{score_type}_score_clamp"] - merged[f"{score_type}_score_add"]
    )
    # merged["Delta_text"] = merged["Delta"].apply(lambda x: f"{x:.2f}")
    merged["Delta_text"] = merged["Delta"].apply(lambda x: f"{x:.2f}" if x > 0 else "")

    plot = (
        ggplot(merged, aes(x="factor_init", y="factor_lr", fill="Delta"))
        + geom_tile()
        + geom_text(aes(label="Delta_text"), size=9)
        # + scale_fill_gradient2(low="darkred", mid="white", high="steelblue", midpoint=0)
        + scale_fill_continuous("summer")
        + labs(
            x="Factor initialization size ($\\beta$)",
            y="Factor learning rate ($\\eta_\\alpha$)",
            title=f"ClampInv - AddInv ({score_type})",
        )
        + theme(figure_size=(4, 3))
        + coord_equal()
    )
    save_dir = Path("figures", f"{option}")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{cfg}_{positions}_diff.pdf"
    plot.save(save_path, dpi=100)
    # logger.warning(f"Saved to {save_path}")

    return plot


def plot_score_breakdown(res_df: pd.DataFrame, option, cfg, positions):
    # score_df = res_df[res_df["method"] == sv_key + "|" + positions]
    score_df = (
        res_df.groupby(["method", "factor_lr", "factor_init", "seed"])
        .agg(
            overall_score=("overall_score", "mean"),
            concept_score=("concept_score", "mean"),
            instruct_score=("instruct_score", "mean"),
            fluency_score=("fluency_score", "mean"),
        )
        .reset_index()
    )
    score_df = (
        score_df.groupby(["method", "factor_lr", "factor_init"])
        .agg(
            overall_score=("overall_score", "mean"),
            concept_score=("concept_score", "mean"),
            instruct_score=("instruct_score", "mean"),
            fluency_score=("fluency_score", "mean"),
        )
        .reset_index()
    )
    plot_df = score_df.melt(
        id_vars=["method", "factor_lr", "factor_init"],
        value_vars=[
            "overall_score",
            "concept_score",
            "instruct_score",
            "fluency_score",
        ],
        var_name="score",
    )
    plot_df["score"] = plot_df["score"].replace(
        {
            "overall_score": "Overall score",
            "concept_score": "Concept score",
            "instruct_score": "Instruct score",
            "fluency_score": "Fluency score",
        }
    )

    plot_df["text"] = plot_df["value"].apply(lambda x: f"{x:.2f}")

    plot_df["factor_lr"] = plot_df["factor_lr"].apply(lambda s: float(s))
    plot_df["factor_lr"] = pd.Categorical(
        plot_df["factor_lr"],
        categories=sorted(plot_df["factor_lr"].unique()),
        ordered=True,
    )
    plot_df["factor_init"] = pd.Categorical(
        plot_df["factor_init"],
        categories=sorted(plot_df["factor_init"].unique()),
        ordered=True,
    )

    plots = []
    for k in ("Overall", "Concept", "Instruct", "Fluency"):
        plot = (
            ggplot(
                plot_df[plot_df["score"] == f"{k} score"],
                aes(x="factor_init", y="factor_lr", fill="value"),
            )
            + geom_tile()
            + facet_wrap("method")
            + geom_text(aes(label="text"), size=9)
            + labs(
                x="Factor initialization size ($\\beta$)",
                y="Factor learning rate ($\\eta_\\alpha$)",
                fill="Score",
                title=f"{k} score",
            )
            + scale_y_discrete(
                labels=lambda lst: [
                    f"{x:.1e}".replace("e-0", "e-").replace(".0", "")
                    if x < 1
                    else f"{x}"
                    for x in lst
                ],
            )
            # + scale_fill_gradient(low="white", high="steelblue")
            + scale_fill_continuous("summer")
            + coord_equal()
            + theme(figure_size=(11, 7))
        )
        plots.append(plot)

    plot = (plots[0] / plots[2]) | (plots[1] / plots[3])

    save_dir = Path("figures", f"{option}")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{cfg}_{positions}_all_scores.pdf"
    plot.save(save_path, dpi=100)
    # logger.warning(f"Saved to {save_path}")

    return plot


def standard_deviation(res_df: pd.DataFrame, option, cfg, positions):
    plot_df = (
        res_df.groupby(["method", "factor_lr", "factor_init", "seed"])
        .agg(
            overall_score=("overall_score", "mean"),
            concept_score=("concept_score", "mean"),
            instruct_score=("instruct_score", "mean"),
            fluency_score=("fluency_score", "mean"),
        )
        .reset_index()
    )
    plot_df = (
        plot_df.groupby(["method", "factor_lr", "factor_init"])
        .agg(
            overall_std=("overall_score", "std"),
            concept_std=("concept_score", "std"),
            instruct_std=("instruct_score", "std"),
            fluency_std=("fluency_score", "std"),
        )
        .reset_index()
    )
    plot_df = plot_df.melt(
        id_vars=["method", "factor_lr", "factor_init"],
        value_vars=["overall_std", "concept_std", "instruct_std", "fluency_std"],
        var_name="score",
    )
    plot_df["score"] = plot_df["score"].replace(
        ["overall_std", "concept_std", "instruct_std", "fluency_std"],
        ["Overall", "Concept", "Instruct", "Fluency"],
    )
    plot_df["score"] = pd.Categorical(
        plot_df["score"],
        categories=["Overall", "Concept", "Instruct", "Fluency"],
        ordered=True,
    )

    plot_df["text"] = plot_df["value"].apply(lambda x: f"{x:.2f}")

    plot_df["factor_lr"] = plot_df["factor_lr"].apply(lambda s: float(s))
    plot_df["factor_lr"] = pd.Categorical(
        plot_df["factor_lr"],
        categories=sorted(plot_df["factor_lr"].unique()),
        ordered=True,
    )
    plot_df["factor_init"] = pd.Categorical(
        plot_df["factor_init"],
        categories=sorted(plot_df["factor_init"].unique()),
        ordered=True,
    )

    plot = (
        ggplot(plot_df, aes(x="factor_init", y="factor_lr", fill="value"))
        + geom_tile()
        + geom_text(aes(label="text"), size=9)
        + facet_grid("method ~ score")
        + labs(
            x="Factor initialization size ($\\beta$)",
            y="Factor learning rate ($\\eta_\\alpha$)",
            fill="Std",
            title="Standard deviation$\\downarrow$",
        )
        + theme(figure_size=(10, 6))
        + scale_y_discrete(
            labels=lambda lst: [
                f"{x:.1e}".replace("e-0", "e-").replace(".0", "") if x < 1 else f"{x}"
                for x in lst
            ],
        )
        # + scale_fill_gradient(low='steelblue', high='white')
        + scale_fill_continuous("summer")
        + coord_equal()
    )
    save_dir = Path("figures", f"{option}")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{cfg}_{positions}_std.pdf"
    plot.save(save_path, dpi=100)
    # logger.warning(f"Saved to {save_path}")
    return plot


def main(args: Arguments):
    logger.warning(args)

    res_df = read_data(
        seed_list=args.seed_list,
        factor_lr_list=args.factor_lr_list,
        factor_init_size_list=args.factor_init_size_list,
        output_base_dir=args.output_base_dir,
        positions=args.pos,
    )
    if res_df is None:
        logger.warning("No results; exiting")
        exit(0)

    for score_type in args.score_types:
        plot_side_by_side(
            res_df=res_df,
            score_type=score_type,
            option=args.option,
            cfg=args.cfg,
            positions=args.pos,
        )
        for sv in args.svs:
            plot_one_score_one_method(
                res_df=res_df,
                method=sv,
                score_type=score_type,
                option=args.option,
                cfg=args.cfg,
                positions=args.pos,
            )
        plot_score_diff(
            res_df=res_df,
            score_type=score_type,
            option=args.option,
            cfg=args.cfg,
            positions=args.pos,
        )

    plot_score_breakdown(
        res_df=res_df, option=args.option, cfg=args.cfg, positions=args.pos
    )
    standard_deviation(res_df=res_df, option=args.option, cfg=args.cfg, positions=args.pos)


if __name__ == "__main__":
    parser = HfArgumentParser(Arguments)
    ns = parser.parse_args()
    args = Arguments(**vars(ns))
    main(args)
