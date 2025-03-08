import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

# ============================
# Helper Functions
# ============================

def get_categories_list(csv_file_path):
    """
    Extracts the list of disease categories from the CSV file.

    Parameters:
    - csv_file_path: Path to the CSV file containing true labels.

    Returns:
    - List of category names.
    """
    df = pd.read_csv(csv_file_path, nrows=0)
    columns = df.columns.tolist()[7:]
    result = [f"{col}" for col in columns]
    return result

def make_true_labels(cxr_true_labels_path: str, cutlabels: bool = True):
    """
    Loads true labels from the CSV file.

    Parameters:
    - cxr_true_labels_path: Path to the CSV file containing true labels.
    - cutlabels: Whether to select specific label columns.

    Returns:
    - Numpy array of true labels.
    """
    full_labels = pd.read_csv(cxr_true_labels_path)
    cxr_labels = get_categories_list(cxr_true_labels_path)
    if cutlabels:
        full_labels = full_labels.loc[:, cxr_labels]
    else:
        full_labels.drop(full_labels.columns[0], axis=1, inplace=True)
    y_true = full_labels.to_numpy()
    return y_true

def load_true_labels(file):
    """
    Loads true labels and category names.

    Parameters:
    - file: Path to the CSV file containing true labels.

    Returns:
    - Tuple of (true_labels, categories).
    """
    category = get_categories_list(file)
    true_matrix = make_true_labels(file)
    true_labels = np.argmax(true_matrix, axis=1)
    return true_labels, category

def calculate_accuracy_metrics(true_labels, pred_labels):
    """
    Calculates accuracy, precision, recall, and F1 score.

    Parameters:
    - true_labels: Ground truth labels.
    - pred_labels: Predicted labels.

    Returns:
    - Tuple of (accuracy, precision, recall, f1).
    """
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(
        true_labels, pred_labels, average="macro", zero_division=1
    )
    recall = recall_score(true_labels, pred_labels, average="macro")
    f1 = f1_score(true_labels, pred_labels, average="macro")
    return accuracy, precision, recall, f1

def calculate_group_accuracy(df, true_labels, pred_labels, group_column, group_values):
    """
    Calculates accuracy for different groups.

    Parameters:
    - df: DataFrame containing demographic information.
    - true_labels: Ground truth labels.
    - pred_labels: Predicted labels.
    - group_column: Column name to group by.
    - group_values: List of group values or tuples for ranges.

    Returns:
    - Dictionary of accuracies per group.
    """
    accuracies = {}
    for value in group_values:
        if isinstance(value, tuple):
            indices = df[
                (df[group_column] >= value[0]) & (df[group_column] < value[1])
            ].index.tolist()
        else:
            indices = df[df[group_column] == value].index.tolist()
        if len(indices) == 0:
            accuracies[str(value)] = None
            continue
        accuracy = accuracy_score(true_labels[indices], pred_labels[indices])
        accuracies[str(value)] = accuracy
    return accuracies

def calculate_group_auroc(
    df, true_labels, pred_probs, group_column, group_values, num_classes
):
    """
    Calculates AUROC for different groups.

    Parameters:
    - df: DataFrame containing demographic information.
    - true_labels: Ground truth labels.
    - pred_probs: Predicted probabilities.
    - group_column: Column name to group by.
    - group_values: List of group values or tuples for ranges.
    - num_classes: Number of classes.

    Returns:
    - Dictionary of AUROC scores per group.
    """
    aurocs = {}
    # Binarize the true labels for multiclass AUROC calculation
    true_labels_bin = label_binarize(true_labels, classes=list(range(num_classes)))

    for value in group_values:
        if isinstance(value, tuple):
            indices = df[
                (df[group_column] >= value[0]) & (df[group_column] < value[1])
            ].index.tolist()
        else:
            indices = df[df[group_column] == value].index.tolist()

        if len(indices) == 0:
            aurocs[str(value)] = None
            continue

        true_labels_group = true_labels_bin[indices]
        pred_probs_group = pred_probs[indices]

        # Calculate AUROC using OvR and macro-average
        try:
            auroc = roc_auc_score(
                true_labels_group,
                pred_probs_group,
                average="macro",
                multi_class="ovr",
            )
        except ValueError:
            auroc = None
        aurocs[str(value)] = auroc

    return aurocs

def calculate_eq_odds(df, true_labels, pred_labels, attribute, values):
    """
    Calculates Equal Odds difference for a given attribute.

    Parameters:
    - df: DataFrame containing demographic information.
    - true_labels: Ground truth labels.
    - pred_labels: Predicted labels.
    - attribute: Attribute to calculate Equal Odds for.
    - values: List of attribute values or tuples for ranges.

    Returns:
    - Average Equal Odds difference.
    """
    eq_odds_results = []
    for val in values:
        if isinstance(val, tuple):
            mask = (df[attribute] >= val[0]) & (df[attribute] <= val[1])
        else:
            mask = df[attribute] == val
        tp = np.sum((pred_labels[mask] == 1) & (true_labels[mask] == 1))
        fn = np.sum((pred_labels[mask] == 0) & (true_labels[mask] == 1))
        fp = np.sum((pred_labels[mask] == 1) & (true_labels[mask] == 0))
        tn = np.sum((pred_labels[mask] == 0) & (true_labels[mask] == 0))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        eq_odds_results.append((tpr, fpr))
    if len(eq_odds_results) < 2:
        return None
    tpr_diff = abs(eq_odds_results[0][0] - eq_odds_results[1][0])
    fpr_diff = abs(eq_odds_results[0][1] - eq_odds_results[1][1])
    return (tpr_diff + fpr_diff) / 2

def calculate_multiclass_eq_odds(
    df, true_labels, pred_labels, attribute, categories, values
):
    """
    Calculates average Equal Odds difference across multiple classes.

    Parameters:
    - df: DataFrame containing demographic information.
    - true_labels: Ground truth labels.
    - pred_labels: Predicted labels.
    - attribute: Attribute to calculate Equal Odds for.
    - categories: List of category names.
    - values: List of attribute values or tuples for ranges.

    Returns:
    - Mean Equal Odds difference across all categories.
    """
    true_labels_oh = label_binarize(true_labels, classes=list(range(len(categories))))
    pred_labels_oh = label_binarize(pred_labels, classes=list(range(len(categories))))
    eq_odds_results = []
    for i in range(len(categories)):
        true_labels_cat = true_labels_oh[:, i]
        pred_labels_cat = pred_labels_oh[:, i]
        eq_odds_cat = calculate_eq_odds(
            df, true_labels_cat, pred_labels_cat, attribute, values
        )
        if eq_odds_cat is not None:
            eq_odds_results.append(eq_odds_cat)
    if len(eq_odds_results) == 0:
        return None
    return np.mean(eq_odds_results)

def calculate_ece(probabilities, true_labels):
    """
    Calculates Expected Calibration Error (ECE).

    Parameters:
    - probabilities: Predicted probabilities.
    - true_labels: Ground truth labels.

    Returns:
    - ECE value.
    """
    num_samples = len(true_labels)
    ece = np.sum(np.abs(probabilities - true_labels)) / num_samples
    return ece

def calculate_multiclass_ece_difference(true_label_csv, saved_npy_path, categories):
    """
    Calculates the difference in ECE between demographic groups.

    Parameters:
    - true_label_csv: Path to the CSV file containing true labels.
    - saved_npy_path: Path to the directory containing .npy files.
    - categories: List of category names.

    Returns:
    - DataFrame containing ECE differences.
    """
    # Load true labels from CSV
    df = pd.read_csv(true_label_csv)

    # One-hot encode the true labels for each category
    true_labels_oh = np.array([df[cat].values for cat in categories]).T

    results = []

    # Separate indices for the two groups (Male and Female)
    male_indices = df[df["Patient Gender"] == "M"].index
    female_indices = df[df["Patient Gender"] == "F"].index

    # Separate indices for the two groups (young and old)
    young_indices = df[df["Patient Age"] < 60].index
    old_indices = df[df["Patient Age"] >= 60].index

    for root, dirs, files in os.walk(saved_npy_path):
        for file in sorted(files):
            if file.endswith(".npy"):
                npy_path = os.path.join(root, file)
                probs = np.load(npy_path)

                # Initialize lists to store ECEs for each class
                male_ece_list = []
                female_ece_list = []
                young_ece_list = []
                old_ece_list = []

                for i in range(len(categories)):
                    # Get the probabilities and true labels for the current class (One-vs-Rest)
                    male_probs = probs[male_indices, i]
                    female_probs = probs[female_indices, i]
                    young_probs = probs[young_indices, i]
                    old_probs = probs[old_indices, i]

                    male_true = true_labels_oh[male_indices, i]
                    female_true = true_labels_oh[female_indices, i]
                    young_true = true_labels_oh[young_indices, i]
                    old_true = true_labels_oh[old_indices, i]

                    # Calculate ECE for each class
                    male_ece = calculate_ece(male_probs, male_true)
                    female_ece = calculate_ece(female_probs, female_true)
                    young_ece = calculate_ece(young_probs, young_true)
                    old_ece = calculate_ece(old_probs, old_true)

                    male_ece_list.append(male_ece)
                    female_ece_list.append(female_ece)
                    young_ece_list.append(young_ece)
                    old_ece_list.append(old_ece)

                # Calculate macro-average ECE for both groups
                male_macro_ece = np.mean(male_ece_list)
                female_macro_ece = np.mean(female_ece_list)
                young_macro_ece = np.mean(young_ece_list)
                old_macro_ece = np.mean(old_ece_list)

                # Calculate ECE Differences
                ece_difference_gender = abs(male_macro_ece - female_macro_ece)
                ece_difference_age = abs(young_macro_ece - old_macro_ece)

                # Append the result
                results.append(
                    {
                        "filename": file,
                        "ECE Difference for Patient Gender": ece_difference_gender,
                        "ECE Difference for Patient Age": ece_difference_age,
                    }
                )

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df

def insert_average_rows(df, interval):
    """
    Inserts average rows every 'interval' rows.

    Parameters:
    - df: DataFrame to process.
    - interval: Number of rows between average rows.

    Returns:
    - New DataFrame with average rows inserted.
    """
    if interval > 0:
        results = []
        averages = []
        for i in range(0, len(df), interval):
            subset = df.iloc[i : i + interval]
            if not subset.empty:
                avg_values = subset.iloc[:, 1:].mean()
                first_file = subset.iloc[0, 0]
                label_prefix = first_file.split("_")[0]
                avg_row = [None] * len(subset.columns)
                avg_row[0] = "average_" + label_prefix
                avg_row[1:] = avg_values.tolist()
                results.extend(subset.values.tolist())
                averages.append(avg_row)
        results.extend(averages)
        new_df = pd.DataFrame(results, columns=df.columns)
        return new_df
    else:
        return df

def add_variance_column(df, columns_to_include=None, new_column_name="variance"):
    """
    Adds a variance column to the DataFrame based on specified columns.

    Parameters:
    - df: DataFrame to process.
    - columns_to_include: List of columns to calculate variance on. If None, use all numeric columns except 'npy_file' or 'file'.
    - new_column_name: Name of the variance column to add.

    Returns:
    - DataFrame with an added variance column.
    """
    if columns_to_include is None:
        # Exclude non-numeric columns like 'npy_file' or 'file'
        columns_to_include = df.select_dtypes(include=[np.number]).columns.tolist()
    variance_values = df[columns_to_include].var(axis=1)
    df[new_column_name] = variance_values
    return df

def extract_and_add_file_info(df):
    """
    Extracts information from filenames and adds them as new columns.

    Filename format: “method_modelname_modeltype_batchsize_rank_alpha_time.npy”

    Parameters:
    - df: DataFrame containing filenames in the first column.

    Returns:
    - DataFrame with additional columns extracted from filenames.
    """
    columns = ["method", "modelname", "modeltype", "batchsize", "seed", "epoches", "time"]

    def extract_info(filename):
        parts = filename.replace(".npy", "").split("_")
        info = {col: (parts[i] if i < len(parts) else None) for i, col in enumerate(columns)}
        return info

    # Extract information and create new columns
    info_df = df.iloc[:, 0].apply(lambda x: pd.Series(extract_info(x)))
    df = pd.concat([df, info_df], axis=1)

    return df

# ============================
# Processing Functions
# ============================

def process_attribute_accuracy(saved_npy, true_label_path, categories):
    """
    Processes attribute accuracy metrics.

    Parameters:
    - saved_npy: Directory containing .npy files.
    - true_label_path: Path to the CSV file with true labels.
    - categories: List of disease categories.

    Returns:
    - DataFrame with attribute accuracy metrics.
    """
    results = []
    for root, dirs, files in os.walk(saved_npy):
        for file in sorted(files):
            if file.endswith(".npy"):
                npypath = os.path.join(root, file)
                probs = np.load(npypath)
                pred = np.argmax(probs, axis=1)
                df = pd.read_csv(true_label_path)
                true_labels, _ = load_true_labels(true_label_path)

                accuracy, precision, recall, f1 = calculate_accuracy_metrics(
                    true_labels, pred
                )
                gender_accuracies = calculate_group_accuracy(
                    df, true_labels, pred, "Patient Gender", ["M", "F"]
                )
                eq_odds_gender = calculate_multiclass_eq_odds(
                    df,
                    true_labels,
                    pred,
                    "Patient Gender",
                    categories,
                    ["M", "F"],
                )
                gender_aurocs = calculate_group_auroc(
                    df,
                    true_labels,
                    probs,
                    "Patient Gender",
                    ["M", "F"],
                    len(categories),
                )

                age_ranges = [(0, 60), (60, 150)]  # Adjust upper bound if needed
                age_accuracies = calculate_group_accuracy(
                    df, true_labels, pred, "Patient Age", age_ranges
                )
                eq_odds_age = calculate_multiclass_eq_odds(
                    df,
                    true_labels,
                    pred,
                    "Patient Age",
                    categories,
                    age_ranges,
                )
                age_aurocs = calculate_group_auroc(
                    df, true_labels, probs, "Patient Age", age_ranges, len(categories)
                )

                view_positions = ["AP", "PA"]
                view_accuracies = calculate_group_accuracy(
                    df, true_labels, pred, "View Position", view_positions
                )
                view_aurocs = calculate_group_auroc(
                    df,
                    true_labels,
                    probs,
                    "View Position",
                    view_positions,
                    len(categories),
                )

                results.append(
                    {
                        "file": file,
                        "eq_odds_gender": eq_odds_gender,
                        "eq_odds_age": eq_odds_age,
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1,
                        "auroc_male": gender_aurocs.get("M"),
                        "auroc_female": gender_aurocs.get("F"),
                        **{f"auroc_age_{k}": v for k, v in age_aurocs.items()},
                    }
                )
    results_df = pd.DataFrame(results)
    return results_df

def process_disease_f1_scores(saved_npy, true_label_path, categories):
    """
    Processes disease-wise F1 score metrics.

    Parameters:
    - saved_npy: Directory containing .npy files.
    - true_label_path: Path to the CSV file with true labels.
    - categories: List of disease categories.

    Returns:
    - DataFrame with disease-wise F1 score metrics.
    """
    results = []
    for root, dirs, files in os.walk(saved_npy):
        for file in sorted(files):
            if file.endswith(".npy"):
                npypath = os.path.join(root, file)
                probs = np.load(npypath)
                pred = np.argmax(probs, axis=1)
                true_labels, _ = load_true_labels(true_label_path)
                f1_scores = f1_score(true_labels, pred, average=None, zero_division=1)
                accuracy, _, _, _ = calculate_accuracy_metrics(true_labels, pred)
                f1_score_dict = {"npy_file": file, "accuracy": accuracy}
                for i, cat in enumerate(categories):
                    f1_score_dict[cat] = f1_scores[i]
                results.append(f1_score_dict)
    results_df = pd.DataFrame(results)
    return results_df

def process_attribute_f1_scores(saved_npy, true_label_path):
    """
    Processes F1 scores for different demographic groups.

    Parameters:
    - saved_npy: Directory containing .npy files.
    - true_label_path: Path to the CSV file with true labels.

    Returns:
    - DataFrame with group-wise F1 scores and differences.
    """
    results = []
    for root, dirs, files in os.walk(saved_npy):
        for file in sorted(files):
            if file.endswith(".npy"):
                npypath = os.path.join(root, file)
                probs = np.load(npypath)
                pred = np.argmax(probs, axis=1)
                df = pd.read_csv(true_label_path)
                true_labels, _ = load_true_labels(true_label_path)

                # Define age groups
                young_mask = df["Patient Age"] < 60
                old_mask = df["Patient Age"] >= 60

                # Define gender groups
                male_mask = df["Patient Gender"] == "M"
                female_mask = df["Patient Gender"] == "F"

                # Define combinations
                combinations = {
                    "f1_young_M": young_mask & male_mask,
                    "f1_young_F": young_mask & female_mask,
                    "f1_old_M": old_mask & male_mask,
                    "f1_old_F": old_mask & female_mask,
                }

                # Calculate F1 scores for each combination
                f1_scores = {}
                for combo_name, mask in combinations.items():
                    if np.sum(mask) == 0:
                        f1 = None
                    else:
                        f1 = f1_score(
                            true_labels[mask],
                            pred[mask],
                            average="macro",
                            zero_division=1,
                        )
                    f1_scores[combo_name] = f1

                # Calculate delta F1 age and delta F1 gender
                delta_f1_age = None
                delta_f1_gender = None
                if (
                    f1_scores["f1_old_M"] is not None
                    and f1_scores["f1_old_F"] is not None
                    and f1_scores["f1_young_M"] is not None
                    and f1_scores["f1_young_F"] is not None
                ):
                    # Average F1 for old and young
                    f1_old = np.mean([f1_scores["f1_old_M"], f1_scores["f1_old_F"]])
                    f1_young = np.mean(
                        [f1_scores["f1_young_M"], f1_scores["f1_young_F"]]
                    )
                    delta_f1_age = f1_old - f1_young

                    # Average F1 for male and female
                    f1_male = np.mean([f1_scores["f1_young_M"], f1_scores["f1_old_M"]])
                    f1_female = np.mean(
                        [f1_scores["f1_young_F"], f1_scores["f1_old_F"]]
                    )
                    delta_f1_gender = f1_male - f1_female

                results.append(
                    {
                        "file": file,
                        "f1_young_M": f1_scores["f1_young_M"],
                        "f1_young_F": f1_scores["f1_young_F"],
                        "f1_old_M": f1_scores["f1_old_M"],
                        "f1_old_F": f1_scores["f1_old_F"],
                        "delta_f1_age": delta_f1_age,
                        "delta_f1_gender": delta_f1_gender,
                    }
                )

    final_f1_df = pd.DataFrame(results)
    return final_f1_df

# ============================
# Main Execution
# ============================

def main():
    # Define paths
    saved_npy = "./output/pred"
    true_label_path = "./dataset/nih6x200_test.csv"

    # Define disease categories
    categories = [
        "Cardiomegaly",
        "Effusion",
        "Atelectasis",
        "Pneumothorax",
        "Edema",
        "Consolidation",
    ]

    # Set infer_times for averaging (set to 0 if not averaging)
    infer_times = 0

    # ============================
    # Load All .npy Files and Display Shapes
    # ============================
    print("Loading .npy files and displaying their shapes:")
    for root, dirs, files in os.walk(saved_npy):
        for file in sorted(files):
            if file.endswith(".npy"):
                file_path = os.path.join(root, file)
                data = np.load(file_path)
                var_name = file[:-4]  # Remove ".npy"
                # Dynamically create variables (use with caution)
                globals()[var_name] = data
                print(f"Loaded {var_name} with shape {data.shape}")
    print("Loading completed.\n")

    # ============================
    # Process Metrics
    # ============================

    print("Processing attribute accuracy metrics...")
    attribute_accuracy_df = process_attribute_accuracy(saved_npy, true_label_path, categories)
    print("Attribute accuracy processing completed.\n")

    print("Processing disease F1 scores...")
    disease_f1scores_df = process_disease_f1_scores(saved_npy, true_label_path, categories)
    print("Disease F1 scores processing completed.\n")

    print("Processing attribute F1 scores...")
    attribute_f1_df = process_attribute_f1_scores(saved_npy, true_label_path)
    print("Attribute F1 scores processing completed.\n")

    # ============================
    # DataFrame Adjustments
    # ============================

    # Modify 'file' column if necessary
    attribute_accuracy_df["file"] = attribute_accuracy_df["file"].apply(
        lambda x: x[4:] if x.startswith("200b") else x
    )
    # Rearrange file names if needed (commented out)
    # def rearrange_file_name(file_name):
    #     parts = file_name.replace('.npy', '').split('_')
    #     modelname = parts[0]
    #     seed = parts[1]
    #     prompt = parts[2]
    #     dataset = parts[3]
    #     return f'{modelname}_{prompt}_{seed}_{dataset}.npy'
    # attribute_accuracy_df['file'] = attribute_accuracy_df['file'].apply(rearrange_file_name)

    # Sort DataFrames by 'file' or 'npy_file' column
    attribute_accuracy_df = attribute_accuracy_df.sort_values("file").reset_index(drop=True)
    disease_f1scores_df = disease_f1scores_df.sort_values("npy_file").reset_index(drop=True)
    attribute_f1_df = attribute_f1_df.sort_values("file").reset_index(drop=True)

    # Insert average rows if infer_times > 0
    print("Inserting average rows...")
    attribute_accuracy_averaged = insert_average_rows(attribute_accuracy_df, infer_times)
    disease_f1scores_averaged = insert_average_rows(disease_f1scores_df, infer_times)
    attribute_f1_averaged = insert_average_rows(attribute_f1_df, infer_times)
    print("Average rows insertion completed.\n")

    # Add variance columns
    print("Adding variance columns...")
    # For attribute_accuracy_averaged, calculate variance of 'f1_score'
    # Assuming 'f1_score' is the macro F1 score
    # If you want variance across multiple metrics, adjust accordingly
    # Here, we'll calculate variance across disease F1 scores
    # But since 'process_attribute_accuracy' does not include per-disease F1, we might need to skip or adjust
    # Hence, we'll calculate variance from 'disease_f1scores_averaged'

    # Add variance to disease_f1scores_averaged based on disease F1 scores
    disease_f1scores_with_variance = add_variance_column(
        disease_f1scores_averaged, columns_to_include=categories, new_column_name="variance"
    )

    # Add variance to attribute_f1_averaged based on group F1 scores
    attribute_f1_with_variance = add_variance_column(
        attribute_f1_averaged,
        columns_to_include=["f1_young_M", "f1_young_F", "f1_old_M", "f1_old_F"],
        new_column_name="variance",
    )
    print("Variance columns added.\n")

    # Extract and add file information
    print("Extracting and adding file information...")
    # Only if needed; otherwise, skip or adjust accordingly
    # final_attribute_accuracy = extract_and_add_file_info(attribute_accuracy_averaged)
    # final_disease_f1score = extract_and_add_file_info(disease_f1scores_with_variance)
    # final_attribute_f1 = extract_and_add_file_info(attribute_f1_with_variance)
    print("File information extraction completed.\n")

    # ============================
    # Calculate ECE Differences
    # ============================

    print("Calculating ECE differences...")
    ece_diff_df = calculate_multiclass_ece_difference(
        true_label_csv=true_label_path,
        saved_npy_path=saved_npy,
        categories=categories
    )
    print("ECE differences calculation completed.\n")

    # ============================
    # Construct Final DataFrames
    # ============================

    # First DataFrame: Overall Metrics
    print("Constructing the first DataFrame (Overall Metrics)...")
    df_overall_metrics = disease_f1scores_with_variance.copy()
    # Select relevant columns: 'npy_file', 'accuracy', F1 scores for each disease, 'variance'
    overall_columns = ["npy_file", "accuracy"] + categories + ["variance"]
    # Ensure all columns exist
    missing_columns = set(overall_columns) - set(df_overall_metrics.columns)
    if missing_columns:
        print(f"Warning: The following expected columns are missing in df_overall_metrics: {missing_columns}")
    df_overall_metrics = df_overall_metrics[overall_columns]
    print("First DataFrame (Overall Metrics) constructed.\n")

    # Second DataFrame: Group Metrics
    print("Constructing the second DataFrame (Group Metrics)...")
    # Merge attribute_f1_with_variance and ece_diff_df on 'file' == 'filename'
    df_group_metrics = attribute_f1_with_variance.merge(
        ece_diff_df, left_on="file", right_on="filename", how="left"
    )

    # Merge with attribute_accuracy_df to include Equal Odds
    df_group_metrics = df_group_metrics.merge(
        attribute_accuracy_averaged[['file', 'eq_odds_gender', 'eq_odds_age']],
        on='file',
        how='left'
    )

    # Select relevant columns
    group_columns = [
        "file",
        "f1_young_M",
        "f1_old_M",
        "f1_young_F",
        "f1_old_F",
        "delta_f1_age",
        "delta_f1_gender",
        "eq_odds_age",
        "eq_odds_gender",
        "ECE Difference for Patient Age",
        "ECE Difference for Patient Gender",
        "variance"
    ]
    # Ensure all columns exist
    missing_group_columns = set(group_columns) - set(df_group_metrics.columns)
    if missing_group_columns:
        print(f"Warning: The following expected columns are missing in df_group_metrics: {missing_group_columns}")
    df_group_metrics = df_group_metrics[group_columns]
    print("Second DataFrame (Group Metrics) constructed.\n")

    # ============================
    # Save or Display Results
    # ============================

    # Save DataFrames to CSV (optional)
    df_overall_metrics.to_csv("./output/metrics/overall_metrics.csv", index=False)
    df_group_metrics.to_csv("./output/metrics/group_metrics.csv", index=False)

    # Display the DataFrames
    print("Final DataFrames:")
    print("\n--- Overall Metrics ---")
    print(df_overall_metrics.head())

    print("\n--- Group Metrics ---")
    print(df_group_metrics.head())

if __name__ == "__main__":
    main()
