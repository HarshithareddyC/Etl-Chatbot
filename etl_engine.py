import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import cleaning


def execute_etl_step(df, step):
    """
    Applies a single ETL step to the DataFrame based on the instruction.
    Delegates to cleaning.py for advanced operations.
    """
    operation = step.get("operation")

    operation_registry = {
        "drop_nulls": cleaning.drop_nulls,
        "impute_missing": cleaning.impute_missing,
        "fill_missing": cleaning.fill_missing,
        "remove_duplicates": cleaning.remove_duplicates,
        "rename_columns": cleaning.rename_columns,
        "drop_columns": cleaning.drop_columns,
        "split_column": cleaning.split_column,
        "combine_columns": cleaning.combine_columns,
        "trim_whitespace": cleaning.trim_whitespace,
        "normalize_case": cleaning.normalize_case,
        "regex_transform": cleaning.regex_transform,
        "parse_dates": cleaning.parse_dates,
        "convert_dtypes": cleaning.convert_dtypes,
        "set_index": cleaning.set_index,
        "reset_index": cleaning.reset_index,
        "filter_rows": cleaning.filter_rows,
        "select_columns": cleaning.select_columns,
        "sort_by": cleaning.sort_by,
        "join_tables": cleaning.join_tables,
        "concatenate": cleaning.concatenate,
        "pivot_table": cleaning.pivot_table,
        "melt_table": cleaning.melt_table,
        "group_and_aggregate": cleaning.group_and_aggregate,
        "apply_custom": cleaning.apply_custom,
        "one_hot_encode": cleaning.one_hot_encode,
        "label_encode": cleaning.label_encode,
        "scale_columns": cleaning.scale_columns,
        "remove_outliers_zscore": cleaning.remove_outliers_zscore,
        "remove_outliers_iqr": cleaning.remove_outliers_iqr,
        "derive_date_parts": cleaning.derive_date_parts,
        "create_ratio": cleaning.create_ratio,
        "drop_low_variance": cleaning.drop_low_variance,
        "drop_highly_correlated": cleaning.drop_highly_correlated,
        "reduce_dimensionality": cleaning.reduce_dimensionality,
        "sample_data": cleaning.sample_data,
    }

    if operation in operation_registry:
        try:
            return operation_registry[operation](df, **{k: v for k, v in step.items() if k != "operation"})
        except Exception as e:
            raise ValueError(f"Failed to execute '{operation}': {e}")
    else:
        raise ValueError(f"Unsupported operation: '{operation}'")



def summarize_etl_step(step):
    op = step.get("operation")
    col = step.get("column")

    summaries = {
        "remove_duplicates": "Removed duplicate rows.",
        "impute_missing": "Imputed missing values.",
        "fill_missing": "Filled missing values using forward/backward fill.",
        "rename_columns": f"Renamed columns: {step.get('rename_map')}",
        "drop_columns": f"Dropped columns: {step.get('columns')}",
        "split_column": f"Split column '{col}' into {step.get('into')}.",
        "combine_columns": f"Combined columns {step.get('columns')} into '{step.get('new_column')}'.",
        "normalize_case": f"Normalized text to {step.get('case')} case.",
        "trim_whitespace": "Trimmed whitespaces from all string columns.",
        "parse_dates": f"Parsed dates in columns: {step.get('columns')}.",
        "set_index": f"Set '{step.get('column')}' as index.",
        "reset_index": "Reset DataFrame index.",
        "filter_rows": f"Filtered rows where condition: {step.get('condition')}.",
        "group_and_aggregate": f"Grouped by {step.get('group_by')} and aggregated.",
        "sort_by": f"Sorted by columns: {step.get('columns')}.",
        "create_ratio": f"Created new column '{step.get('new_col')}' as ratio of '{step.get('num_col')}' and '{step.get('denom_col')}'.",
    }

    return summaries.get(op, f"Performed operation: `{op}`.")
