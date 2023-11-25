import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy


def plot_width_error(model, model_name, ax, index_confidence):

    errors = model["errors"]
    widths = model["width"][:, index_confidence]
    color = model["color"]
    ax.scatter(
        scipy.stats.rankdata(widths),
        scipy.stats.rankdata(errors),
        s=25,
        c=color
    )

    ax.set_xlim(-len(errors) * .2, len(errors) * 1.2)
    ax.set_ylim(-len(errors) * .2, len(errors) * 1.2)

    ci_spearman_correlation = model[
        "spearman_correlation_to_error"
    ][index_confidence].confidence_interval
    mean_spearman_correlation = np.mean(
        model[
            "spearman_correlation_to_error"
        ][index_confidence].bootstrap_distribution
    )

    ax.set_xlabel("Rank of the width of the prediction interval\n", size=20)
    ax.set_title(
        str(model_name[0]) + "\n" +
        r"$r_s \in [$" +
        r"$" + str(round(ci_spearman_correlation[0], 2)) +
        ", " + str(round(ci_spearman_correlation[1], 2)) + r"], \;$" +
        r"$\overline{r_s}= " + str(round(mean_spearman_correlation, 3)) + "$",
        fontsize=23
    )


def plot_boxplot(models, alpha, color1, color2, color3, objective="correlation"):
    if objective == "width":
        item_dict = "boostrapped_width"
        y_label = "Bootstrapped width"
    elif objective == "correlation":
        item_dict = "spearman_correlation_to_error"
        y_label = "Bootstrapped Spearman correlation"
    index_90 = np.argmax(np.isclose(alpha, 1 - .90))
    index_95 = np.argmax(np.isclose(alpha, 1 - .95))
    index_99 = np.argmax(np.isclose(alpha, 1 - .99))
    df = pd.concat(
        [
            pd.DataFrame(
                {
                    "model": model_name[0],
                    "spearman_distrib": model[
                        item_dict
                    ][index].bootstrap_distribution,
                    "1 - alpha": 1 - alpha[index]
                }
            ) for index in [
                index_90, index_95, index_99
            ] for model_name, model in models.items()
        ]
    )
    palette = {.9: color1, .95: color2, .99: color3}
    _ = plt.figure(figsize=(18, 14))
    sns.set_palette("bright")
    sns.boxplot(
        data=df, x="model", y="spearman_distrib",
        hue="1 - alpha", palette=palette, fill=False
    )
    plt.legend(
        title=r"$1 - \alpha$", fontsize=30,
        title_fontsize=30, loc="lower right"
    )
    plt.xlabel("Model", fontsize=25)
    plt.ylabel(y_label, fontsize=25)
    plt.yticks(fontsize=25)
    plt.xticks(fontsize=25)
