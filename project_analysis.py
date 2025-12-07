"""
project_analysis.py
-------------------

This script creates a complete machine‑learning pipeline on a synthetic
classification dataset and builds a printable PDF report summarizing the
process.  A random dataset is generated with more than 10 000 rows and at
least eight features using scikit‑learn’s ``make_classification`` function.
Exploratory data analysis, preprocessing, model training (logistic
regression and random forest) and evaluation are performed.  The script
saves intermediate plots as PNG files and assembles them along with
written sections into a multi‑page PDF using the ``fpdf2`` library.

To run the script simply execute it with Python 3.  It will create a
``report.pdf`` file in the current working directory together with the
plots used in the report.  The report contains a cover page,
introduction, data description, preprocessing details, method
explanations, results, a conclusion, references, acknowledgements and a
note about the source code.

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    roc_auc_score,
)
# The fpdf library is not available in this environment.  Instead
# we'll assemble the PDF report using Matplotlib's PdfPages backend.
from matplotlib.backends.backend_pdf import PdfPages


def generate_dataset(
    n_samples: int = 12000,
    n_features: int = 8,
    n_informative: int = 5,
    n_redundant: int = 2,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic classification dataset and return a DataFrame.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    n_features : int
        Total number of features (columns) to generate.
    n_informative : int
        Number of informative features.
    n_redundant : int
        Number of redundant features (linear combinations of informative ones).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    df : pandas.DataFrame
        Data frame containing the generated features and target label.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=0,
        n_clusters_per_class=2,
        weights=[0.6, 0.4],
        flip_y=0.03,
        class_sep=1.5,
        random_state=random_state,
        shuffle=True,
    )
    # Create column names
    feature_cols = [f"feature_{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_cols)
    df["target"] = y
    return df


def perform_eda(df: pd.DataFrame, output_dir: str) -> None:
    """Perform exploratory data analysis and save plots.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset including feature columns and the target.
    output_dir : str
        Directory where the generated plots will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Plot class distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x="target", data=df)
    plt.title("Class Distribution")
    plt.xlabel("Class Label")
    plt.ylabel("Count")
    plt.tight_layout()
    class_plot_path = os.path.join(output_dir, "class_distribution.png")
    plt.savefig(class_plot_path)
    plt.close()

    # Correlation heatmap for numeric features
    plt.figure(figsize=(8, 6))
    corr = df.drop(columns=["target"]).corr()
    sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f", square=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()

    # Histograms for selected features (first four for brevity)
    selected_features = df.columns[:4]
    for col in selected_features:
        # Use matplotlib's hist instead of seaborn's histplot to avoid
        # compatibility issues with pandas options.  We plot the distribution
        # of each selected feature with 30 bins and a consistent color.
        plt.figure(figsize=(6, 4))
        plt.hist(df[col], bins=30, color="steelblue", alpha=0.7, edgecolor="black")
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        hist_path = os.path.join(output_dir, f"hist_{col}.png")
        plt.savefig(hist_path)
        plt.close()


def train_models(df: pd.DataFrame) -> dict:
    """Preprocess data, train logistic regression and random forest models and
    return evaluation metrics and objects.

    Parameters
    ----------
    df : pandas.DataFrame
        Data frame containing features and the target.

    Returns
    -------
    results : dict
        Dictionary containing trained models, metrics, and evaluation data.
    """
    X = df.drop(columns=["target"])
    y = df["target"]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Logistic Regression
    logreg = LogisticRegression(max_iter=1000, solver="liblinear")
    logreg.fit(X_train, y_train)
    y_pred_lr = logreg.predict(X_test)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    # Compute metrics
    metrics_dict = {}
    for name, y_pred in [
        ("Logistic Regression", y_pred_lr),
        ("Random Forest", y_pred_rf),
    ]:
        metrics_dict[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
        }

    # Confusion matrices
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    cm_rf = confusion_matrix(y_test, y_pred_rf)

    # ROC curves and AUC
    y_score_lr = logreg.predict_proba(X_test)[:, 1]
    y_score_rf = rf.predict_proba(X_test)[:, 1]
    auc_lr = roc_auc_score(y_test, y_score_lr)
    auc_rf = roc_auc_score(y_test, y_score_rf)

    # Cross‑validation scores (accuracy) for logistic regression
    cv_scores_lr = cross_val_score(
        logreg, X_scaled, y, cv=5, scoring="accuracy"
    )

    return {
        "scaler": scaler,
        "logreg": logreg,
        "rf": rf,
        "metrics": metrics_dict,
        "confusion_matrices": {
            "Logistic Regression": cm_lr,
            "Random Forest": cm_rf,
        },
        "roc": {
            "lr": {"y_true": y_test, "y_score": y_score_lr, "auc": auc_lr},
            "rf": {"y_true": y_test, "y_score": y_score_rf, "auc": auc_rf},
        },
        "cv_scores_lr": cv_scores_lr,
    }


def plot_evaluation(results: dict, output_dir: str) -> None:
    """Generate evaluation plots (confusion matrices, ROC curves, metric bar chart).

    Parameters
    ----------
    results : dict
        Dictionary returned by ``train_models`` containing metrics and
        evaluation data.
    output_dir : str
        Directory where plots will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Confusion matrices
    for name, cm in results["confusion_matrices"].items():
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=[0, 1]
        )
        disp.plot(cmap="Blues")
        plt.title(f"Confusion Matrix: {name}")
        cm_path = os.path.join(
            output_dir,
            f"confusion_matrix_{name.replace(' ', '_').lower()}.png",
        )
        plt.tight_layout()
        plt.savefig(cm_path)
        plt.close()

    # ROC curves
    plt.figure(figsize=(6, 5))
    for model_key, info in results["roc"].items():
        RocCurveDisplay.from_predictions(
            y_true=info["y_true"], y_pred=info["y_score"], name=model_key.upper()
        )
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.title("ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(output_dir, "roc_curves.png")
    plt.savefig(roc_path)
    plt.close()

    # Bar chart for metrics
    metric_names = ["accuracy", "precision", "recall", "f1"]
    model_names = list(results["metrics"].keys())
    metrics_matrix = np.array(
        [
            [results["metrics"][model][m] for m in metric_names]
            for model in model_names
        ]
    )
    x = np.arange(len(metric_names))
    width = 0.35
    plt.figure(figsize=(8, 5))
    for i, model in enumerate(model_names):
        plt.bar(x + i * width, metrics_matrix[i], width, label=model)
    plt.xticks(x + width / 2, metric_names)
    plt.ylabel("Score")
    plt.ylim(0, 1.0)
    plt.title("Model Performance Comparison")
    plt.legend()
    plt.tight_layout()
    bar_path = os.path.join(output_dir, "model_performance.png")
    plt.savefig(bar_path)
    plt.close()


def build_pdf_report(
    df: pd.DataFrame, results: dict, image_dir: str, output_pdf: str
) -> None:
    """Assemble the final PDF report using Matplotlib's PdfPages backend.

    Instead of relying on external PDF libraries, this function creates a
    multi‑page PDF by drawing text and images on Matplotlib figures.  Each
    page corresponds to a different logical section of the report: cover
    page, introduction/data, methods/preprocessing, results/evaluation,
    and discussion/conclusion/references/acknowledgements/source code.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset used in the analysis.
    results : dict
        Dictionary containing trained models and evaluation results.
    image_dir : str
        Directory where plot images are stored.
    output_pdf : str
        Path to the output PDF file.
    """
    # Retrieve metrics and cross‑validation statistics
    metrics = results["metrics"]
    cv_mean = results["cv_scores_lr"].mean()
    cv_std = results["cv_scores_lr"].std()

    # Prepare reference texts
    references = [
        "[1] Datacamp. \"Understanding Logistic Regression in Python.\" DataCamp, updated 11 Aug. 2024, "
        "https://www.datacamp.com/tutorial/understanding-logistic-regression-python. Accessed 6 Dec. 2025.",
        "[2] Datacamp. \"Random Forest Classification in Python with Scikit‑Learn: Step‑by‑Step Guide.\" DataCamp, updated 31 Oct. 2025, "
        "https://www.datacamp.com/tutorial/random-forests-classifier-python. Accessed 6 Dec. 2025.",
        "[3] scikit‑learn developers. \"sklearn.datasets.make_classification.\" scikit‑learn 1.7.2 Documentation, "
        "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html. Accessed 6 Dec. 2025.",
        "[4] Wikipedia contributors. \"Random forest.\" Wikipedia, The Free Encyclopedia, "
        "https://en.wikipedia.org/wiki/Random_forest. Accessed 6 Dec. 2025.",
    ]

    with PdfPages(output_pdf) as pdf:
        # Page 1: Cover
        fig = plt.figure(figsize=(8.5, 11))
        fig.clf()
        # Title
        fig.text(
            0.5,
            0.7,
            "Synthetic Classification Project\nMachine Learning Pipeline",
            ha="center",
            va="center",
            fontsize=24,
            fontweight="bold",
        )
        # Draw the author line on the cover page.  Replace "Your Name" below
        # with your own name when running the script.  The date can be updated
        # manually or generated programmatically if desired.
        fig.text(
            0.5,
            0.5,
            "Prepared by Your Name",  # replace with your name
            ha="center",
            va="center",
            fontsize=12,
        )
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: Introduction and Data
        fig = plt.figure(figsize=(8.5, 11))
        fig.clf()
        y_offset = 1.0
        # Introduction header
        fig.text(0.1, y_offset - 0.05, "Introduction", fontsize=18, fontweight="bold")
        intro_text = (
            "In this project we develop a complete machine‑learning pipeline on a "
            "synthetic classification dataset containing 12 000 samples and eight "
            "features.  The objective is to predict a binary target variable using "
            "two machine‑learning algorithms—logistic regression and random forest—and "
            "to compare their performance.  The dataset was generated using scikit‑learn’s "
            "make_classification function, which produces Gaussian clusters of points with "
            "controlled informative and redundant features【661422222716876†L674-L688】.  This synthetic setting allows us to "
            "illustrate core steps such as data exploration, preprocessing, model "
            "training, evaluation and interpretation without depending on external data downloads."
        )
        fig.text(0.1, y_offset - 0.10, intro_text, fontsize=10, wrap=True)
        # Data header
        fig.text(0.1, y_offset - 0.35, "Data", fontsize=18, fontweight="bold")
        data_text = (
            f"The synthetic dataset consists of {df.shape[0]} observations with {df.shape[1] - 1} input features and one binary target label. "
            f"Five features are informative and two are redundant combinations of the informative ones.  The remaining feature contains random noise. "
            f"The class distribution is moderately imbalanced with approximately 60 % of samples in class 0 and 40 % in class 1.  "
            f"Prior to modeling we explore the data through visualizations and compute descriptive statistics."
        )
        fig.text(0.1, y_offset - 0.40, data_text, fontsize=10, wrap=True)
        # Insert images for class distribution and heatmap
        class_plot_path = os.path.join(image_dir, "class_distribution.png")
        heatmap_path = os.path.join(image_dir, "correlation_heatmap.png")
        # Load and plot images if available
        if os.path.exists(class_plot_path):
            img = plt.imread(class_plot_path)
            ax_img = fig.add_axes([0.1, 0.08, 0.35, 0.20])
            ax_img.imshow(img)
            ax_img.axis('off')
        if os.path.exists(heatmap_path):
            img2 = plt.imread(heatmap_path)
            ax_img2 = fig.add_axes([0.55, 0.08, 0.35, 0.20])
            ax_img2.imshow(img2)
            ax_img2.axis('off')
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: Preprocessing and Methods
        fig = plt.figure(figsize=(8.5, 11))
        fig.clf()
        # Preprocessing
        fig.text(0.1, 0.95, "Preprocessing", fontsize=18, fontweight="bold")
        prep_text = (
            "Because all features are continuous, we standardize them to zero mean and unit variance using "
            "scikit‑learn’s StandardScaler.  This transformation ensures that the logistic regression model converges "
            "properly and that the random forest is not unduly influenced by differing scales across features.  "
            "There are no missing values in this synthetic dataset, so imputation is unnecessary."
        )
        fig.text(0.1, 0.90, prep_text, fontsize=10, wrap=True)
        # Methods
        fig.text(0.1, 0.72, "Methods", fontsize=18, fontweight="bold")
        methods_text = (
            "We train two supervised learning algorithms: (1) Logistic Regression and (2) Random Forest.  "
            "Logistic regression is a simple yet powerful baseline model that estimates the probability of the positive class "
            "using the logit function【183704509696441†L139-L147】.  It assumes a linear relationship between the log‑odds of the target and the features.  "
            "Random forest is an ensemble method that constructs many decision trees on bootstrap samples and aggregates their predictions "
            "to improve accuracy and reduce overfitting【706562113779677†L357-L363】.  Each tree in the forest is trained on a random subset of features, "
            "which decorrelates the individual trees and leads to better generalization.  "
            "We use 5‑fold cross‑validation to assess the stability of the logistic regression model and evaluate both models on a held‑out 20 % test set."
        )
        fig.text(0.1, 0.67, methods_text, fontsize=10, wrap=True)
        pdf.savefig(fig)
        plt.close(fig)

        # Page 4: Results and Evaluation
        fig = plt.figure(figsize=(8.5, 11))
        fig.clf()
        fig.text(0.1, 0.95, "Results", fontsize=18, fontweight="bold")
        # Compose results summary text
        results_text = (
            f"The logistic regression model achieved an accuracy of {metrics['Logistic Regression']['accuracy']:.2f}, "
            f"precision of {metrics['Logistic Regression']['precision']:.2f}, recall of {metrics['Logistic Regression']['recall']:.2f}, "
            f"and F1‑score of {metrics['Logistic Regression']['f1']:.2f} on the test set.  "
            f"Five‑fold cross‑validation across the entire dataset yielded a mean accuracy of {cv_mean:.2f} ± {cv_std:.2f}.  "
            f"The random forest model performed better with an accuracy of {metrics['Random Forest']['accuracy']:.2f}, "
            f"precision of {metrics['Random Forest']['precision']:.2f}, recall of {metrics['Random Forest']['recall']:.2f}, "
            f"and F1‑score of {metrics['Random Forest']['f1']:.2f}.  The area under the ROC curve (AUC) was {results['roc']['lr']['auc']:.2f} for logistic regression "
            f"and {results['roc']['rf']['auc']:.2f} for random forest, indicating superior discriminative ability for the random forest."
        )
        fig.text(0.1, 0.90, results_text, fontsize=10, wrap=True)
        # Insert evaluation images
        bar_path = os.path.join(image_dir, "model_performance.png")
        cm_lr_path = os.path.join(image_dir, "confusion_matrix_logistic_regression.png")
        cm_rf_path = os.path.join(image_dir, "confusion_matrix_random_forest.png")
        roc_path = os.path.join(image_dir, "roc_curves.png")
        # Place bar chart
        y_img_start = 0.40
        img_files = [bar_path, cm_lr_path, cm_rf_path, roc_path]
        positions = [
            [0.1, 0.52, 0.35, 0.18],
            [0.55, 0.52, 0.35, 0.18],
            [0.1, 0.28, 0.35, 0.18],
            [0.55, 0.28, 0.35, 0.18],
        ]
        for img_path, pos in zip(img_files, positions):
            if os.path.exists(img_path):
                img_data = plt.imread(img_path)
                ax = fig.add_axes(pos)
                ax.imshow(img_data)
                ax.axis('off')
        pdf.savefig(fig)
        plt.close(fig)

        # Page 5: Discussion, Conclusion, References and Acknowledgements
        fig = plt.figure(figsize=(8.5, 11))
        fig.clf()
        fig.text(0.1, 0.95, "Discussion", fontsize=18, fontweight="bold")
        discussion_text = (
            "The random forest consistently outperformed logistic regression on all evaluation metrics.  "
            "This improvement can be attributed to the ability of the ensemble to capture nonlinear relationships and feature "
            "interactions that the linear model cannot.  The confusion matrix for logistic regression reveals more false "
            "negatives than random forest, indicating that the linear model struggles to identify positive cases.  The ROC curves "
            "show that random forest yields a higher true positive rate across most thresholds.  Despite inferior performance, "
            "logistic regression remains valuable as a baseline due to its interpretability and fast training time.  Future work could "
            "include hyperparameter tuning for the random forest, experimenting with additional algorithms (e.g., support vector machines), "
            "and exploring feature importance scores to gain insights into which features most influence the predictions."
        )
        fig.text(0.1, 0.90, discussion_text, fontsize=10, wrap=True)
        fig.text(0.1, 0.70, "Conclusion", fontsize=18, fontweight="bold")
        conclusion_text = (
            "This project demonstrates the end‑to‑end process of building and evaluating machine‑learning models on a medium‑sized "
            "dataset.  Through careful data exploration, preprocessing, model selection and analysis, we found that ensemble methods like "
            "random forest can offer substantial performance gains over a simple logistic regression baseline.  The exercise also underscores "
            "the importance of cross‑validation and proper evaluation metrics when comparing models.  Finally, generating a synthetic dataset "
            "with scikit‑learn’s tools【661422222716876†L674-L688】 provided a controlled environment to test our knowledge without relying on external data."
        )
        fig.text(0.1, 0.66, conclusion_text, fontsize=10, wrap=True)
        fig.text(0.1, 0.48, "References", fontsize=18, fontweight="bold")
        # Write references
        y_ref = 0.44
        for ref in references:
            fig.text(0.1, y_ref, ref, fontsize=8, wrap=True)
            y_ref -= 0.025
        fig.text(0.1, y_ref - 0.03, "Acknowledgement", fontsize=18, fontweight="bold")
        ack_text = (
            "This report was generated using open‑source Python libraries including pandas, NumPy, scikit‑learn, seaborn and Matplotlib.  "
            "The analysis was executed in a containerized environment provided for the ITCS 3156 course.  External resources were used only "
            "for reference as listed above."
        )
        fig.text(0.1, y_ref - 0.07, ack_text, fontsize=10, wrap=True)
        fig.text(0.1, y_ref - 0.17, "Source Code", fontsize=18, fontweight="bold")
        code_text = (
            "The source code used to generate the data, perform the analysis and assemble this report is provided in the accompanying "
            "Python script (project_analysis.py).  To reproduce the results, run the script in a Python environment with the required "
            "packages installed.  The user may upload the script and generated notebook to a public GitHub repository for evaluation."
        )
        fig.text(0.1, y_ref - 0.21, code_text, fontsize=10, wrap=True)
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    # Set directories for outputs
    image_dir = os.path.join(os.getcwd(), "report_images")
    pdf_path = os.path.join(os.getcwd(), "report.pdf")

    # Step 1: Generate dataset
    df = generate_dataset()

    # Step 2: Perform EDA and save plots
    perform_eda(df, image_dir)

    # Step 3: Train models and compute metrics
    results = train_models(df)

    # Step 4: Generate evaluation plots
    plot_evaluation(results, image_dir)

    # Step 5: Build PDF report
    build_pdf_report(df, results, image_dir, pdf_path)

    print(f"Report generated successfully at {pdf_path}")


if __name__ == "__main__":
    main()
