import json
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import os

# ----------------------------
# IO Functions
# ----------------------------
def load_results(json_file):
    """
    Load the JSON results file into a pandas DataFrame.

    Args:
        json_file (str): Path to the JSON results file.

    Returns:
        pd.DataFrame: DataFrame containing the results.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Normalize the 'constraints' dictionary into separate columns
    df = pd.json_normalize(data['results'])
    _naming = dict([(c, c.split('.')[-1]) for c in df.columns.values])
    df.rename(_naming, axis=1, inplace=True)
    
    # Parse the 'response' field to extract the decision and suggestion
    df[['decision', 'analysis', 'suggestion']] = df['response'].apply(parse_response).apply(pd.Series)
    
    return df

def parse_response(response_str):
    """
    Parse the response string to extract the decision, analysis, and message.

    Handles both base case (decision, message) and extended case (decision, analysis, message).

    Args:
        response_str (str): The response string from the LLM.

    Returns:
        tuple: (decision, analysis, message)
    """
    # Regular expression pattern to capture decision, optional analysis, and message
    pattern = r'^\{?\s*(\d+)\s*,\s*([^,]*?)\s*,?\s*(.*?)\s*\}?$'
    match = re.match(pattern, response_str)
    if match:
        decision = int(match.group(1))
        analysis = match.group(2).strip()
        message = match.group(3).strip()
        # If analysis is empty, set it to None
        if not analysis:
            analysis = None
        return decision, analysis, message
    else:
        # If the format doesn't match, return None for all
        return None, None, None
    

# ----------------------------
# Evaluation Functions
# ----------------------------
def compute_give_up_rates(df, agg_method='mean'):
    """
    Compute the give-up rates based on difficulty levels for each constraint type or type combination.

    Automatically detects constraint columns without requiring them as arguments.

    Args:
        df (pd.DataFrame): DataFrame containing the results with separate constraint columns.
        agg_method (str): Aggregation method for multi-type constraints ('mean', 'max', 'std').

    Returns:
        dict: Dictionary containing DataFrames for single-type and multi-type constraints.
    """
    # Define mapping from constraint codes to types
    constraint_type_mapping = {
        's1': 'Skill',
        's2': 'Skill',
        'i1': 'Item',
        'i2': 'Item',
        'e1': 'Environment',
        'e2': 'Environment'
    }

    # Identify constraint columns based on naming conventions (e.g., 's1', 's2', 'i1', 'i2', 'e1', 'e2')
    constraint_pattern = r'^[sie]\d+$'
    constraint_columns = [col for col in df.columns if re.match(constraint_pattern, col)]

    # Create a new column for each constraint indicating its type
    for col in constraint_columns:
        df[col + '_type'] = df[col].apply(lambda x: constraint_type_mapping.get(col, 'unknown') if pd.notnull(x) else None)

    # print(df.columns)
    # ----------------------------
    # Single-Type Constraints
    # ----------------------------
    # Melt the DataFrame to have one row per constraint per task
    single_constraints_df = df.melt(
        id_vars=['task_id', 'decision'], 
        value_vars=constraint_columns, 
        var_name='constraint_code', 
        value_name='difficulty'
    )

    # Map constraint codes to types
    single_constraints_df['constraint_type'] = single_constraints_df['constraint_code'].map(constraint_type_mapping)

    # Drop rows where difficulty is NaN
    single_constraints_df = single_constraints_df.dropna(subset=['difficulty'])

    # Group by constraint_type and difficulty to compute rates
    single_grouped = single_constraints_df.groupby(['constraint_type', 'difficulty'])

    # Compute total counts and give-up counts
    single_summary = single_grouped['decision'].agg(
        total_tasks='count', 
        give_up_tasks=lambda x: (x == 1).sum()
    ).reset_index()

    # Compute give-up rate
    single_summary['give_up_rate'] = single_summary['give_up_tasks'] / single_summary['total_tasks']

    # ----------------------------
    # Multi-Type Constraints
    # ----------------------------
    # Identify tasks with multiple constraints (2 or 3 types)
    df['constraint_count'] = df[constraint_columns].notna().sum(axis=1)
    multi_constraints_df = df[df['constraint_count'] > 1].copy()

    if not multi_constraints_df.empty:
        # Aggregate difficulties based on the specified method
        if agg_method == 'mean':
            multi_constraints_df['aggregated_difficulty'] = multi_constraints_df[constraint_columns].mean(axis=1)
        elif agg_method == 'max':
            multi_constraints_df['aggregated_difficulty'] = multi_constraints_df[constraint_columns].max(axis=1)
        elif agg_method == 'std':
            multi_constraints_df['aggregated_difficulty'] = multi_constraints_df[constraint_columns].std(axis=1)
        else:
            raise ValueError("agg_method must be one of 'mean', 'max', or 'std'.")

        # Create a sorted tuple of constraint types for grouping
        def get_constraint_types(row):
            types = []
            for col in constraint_columns:
                if pd.notnull(row[col]):
                    types.append(constraint_type_mapping.get(col, 'unknown'))
            return tuple(sorted(types))
        
        multi_constraints_df['constraint_type_combination'] = multi_constraints_df.apply(get_constraint_types, axis=1)

        # Group by constraint_type_combination and aggregated_difficulty
        multi_grouped = multi_constraints_df.groupby(['constraint_type_combination', 'aggregated_difficulty'])

        # Compute total counts and give-up counts
        multi_summary = multi_grouped['decision'].agg(
            total_tasks='count', 
            give_up_tasks=lambda x: (x == 1).sum()
        ).reset_index()

        # Compute give-up rate
        multi_summary['give_up_rate'] = multi_summary['give_up_tasks'] / multi_summary['total_tasks']
    else:
        multi_summary = pd.DataFrame(columns=['constraint_type_combination', 'aggregated_difficulty', 'total_tasks', 'give_up_tasks', 'give_up_rate'])

    results = {
        'single': single_summary,
        'multiple': multi_summary
    }

    return results


# ----------------------------
# Plotting Functions
# ----------------------------
def plot_single_type_give_up_rate(single_summary):
    """
    Plot difficulty vs give-up rate for single-type constraints with optional hue on types.
    Adds a diagonal reference line from (1, 0) to (5, 1).
    
    Args:
        single_summary (pd.DataFrame): DataFrame with give-up rates for single-type constraints.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 10))
    
    sns.set_theme(style="whitegrid")
    
    
    ax = sns.barplot(
        data=single_summary, 
        x='difficulty', 
        y='give_up_rate', 
        hue='constraint_type', 
        palette='viridis'
    )
    ax.plot([0, 4], [0, 1], linestyle='--', color='darkgray', linewidth=2, label='Ideal Progression')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, title='Constraint Types')
    plt.xlabel('Difficulty Level', fontsize=14)
    plt.ylabel('Give-Up Rate', fontsize=14)
    plt.ylim(0, 1)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    # plt.show()

def plot_multi_type_give_up_rate(multi_summary, agg_method='mean'):
    """
    Plot aggregated difficulty vs give-up rate for multi-type constraints with optional hue on type combinations.
    Adds a diagonal reference line from (1, 0) to (5, 1).
    
    Args:
        multi_summary (pd.DataFrame): DataFrame with give-up rates for multi-type constraints.
        agg_method (str): Aggregation method used ('mean', 'max', 'std').
        title (str): Title of the plot.
    """
    if multi_summary.empty:
        print("No multi-type constraints to plot.")
        return
    
    plt.figure(figsize=(10, 10))
    
    sns.set_theme(style="whitegrid")
    
    ax = sns.barplot(
        data=multi_summary, 
        x='aggregated_difficulty', 
        y='give_up_rate', 
        hue='constraint_type_combination', 
        palette='plasma'
    )
    ax.plot([0, 4], [0, 1], linestyle='--', color='darkgray', linewidth=2, label='Ideal Progression')
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, title='Constraint Type Combination')
    plt.xlabel(f'Aggregated Difficulty ({agg_method.capitalize()})', fontsize=14)
    plt.ylabel('Give-Up Rate', fontsize=14)
    plt.ylim(0, 1)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    # plt.show()
