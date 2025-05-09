import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# --- Page Configuration ---
st.set_page_config(
    page_title="LLMThinkBench Leaderboard",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Main Title and Introduction ---
st.title("LLMThinkBench Leaderboard 🏆")
st.markdown("""
A comprehensive framework designed to rigorously evaluate the reasoning capabilities of Large Language Models (LLMs).
This leaderboard presents insights from the LLMThinkBench evaluations. Navigate using the sections in the sidebar.

**GitHub:** [ctrl-gaurav/LLMThinkBench](https://github.com/ctrl-gaurav/LLMThinkBench) | **PyPI:** [llmthinkbench](https://pypi.org/project/llmthinkbench/)
""")
st.markdown("---")




# --- Global Styling (Optional - for more advanced customization) ---
# You can inject custom CSS if needed, but Streamlit's theming is often enough.
# st.markdown("""
# <style>
# /* Add custom CSS here */
# </style>
# """, unsafe_allow_html=True)

# --- Load Data ---
@st.cache_data
def load_data(file_path):
    """Loads data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        # Clean column names: remove leading/trailing spaces and replace non-alphanumeric for consistency
        df.columns = ["_".join(col.lower().strip().split()) for col in df.columns]
        df.rename(columns={'task_model': 'Task_Model'}, inplace=True) # Ensure Task_Model is consistent
        return df
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please make sure it's in the same directory as the app.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# --- Helper Functions for Parsing and Calculating ---
def parse_metric_value(value_str):
    """Extracts the mean value from a string like '75.00% ± 2.5' or '100.5 ± 10.2'."""
    if pd.isna(value_str):
        return np.nan
    if isinstance(value_str, (int, float)):
        return float(value_str)
    try:
        if '%' in value_str:
            return float(value_str.split('%')[0].strip()) / 100.0
        elif '±' in value_str:
            return float(value_str.split('±')[0].strip())
        return float(value_str)  # Fallback for plain numbers as strings
    except (ValueError, AttributeError):
        return np.nan

def parse_variance_value(value_str):
    """Extracts the variance from a string like '75.00% ± 2.5' or '100.5 ± 10.2'."""
    if pd.isna(value_str) or '±' not in str(value_str):
        return np.nan
    try:
        variance_part = str(value_str).split('±')[1].strip()
        if '%' in str(value_str): # If original value was %, variance might also need scaling if it's relative
             # Assuming variance is in the same unit as the mean part before % scaling
            original_mean_unscaled = float(str(value_str).split('%')[0].strip())
            if original_mean_unscaled != 0: # Avoid division by zero
                 # This interpretation assumes variance is an absolute value that needs to be scaled like the mean
                 # Or, if variance is already a percentage point, it should be divided by 100 too.
                 # For simplicity, let's assume variance is given in percentage points for % metrics.
                return float(variance_part) / 100.0 # if variance itself is like "2.5" meaning 2.5%
            else:
                return float(variance_part) / 100.0 # if mean is 0, treat variance as % points
        return float(variance_part)
    except (ValueError, IndexError, AttributeError):
        return np.nan


@st.cache_data
def calculate_summary_metrics(_df):
    """Calculates summary statistics for the main page."""
    df = _df.copy()
    summary_data = []

    basic_math_accuracy_cols = ['absolute_difference_accuracy', 'comparison_accuracy', 'division_accuracy', 'even_count_16_accuracy', 'even_count_32_accuracy', 'even_count_64_accuracy', 'even_count_8_accuracy', 'find_maximum_16_accuracy', 'find_maximum_32_accuracy', 'find_maximum_64_accuracy', 'find_maximum_8_accuracy', 'find_minimum_16_accuracy', 'find_minimum_32_accuracy', 'find_minimum_64_accuracy', 'find_minimum_8_accuracy', 'mean_16_accuracy', 'mean_32_accuracy', 'mean_64_accuracy', 'mean_8_accuracy', 'median_16_accuracy', 'median_32_accuracy', 'median_64_accuracy', 'median_8_accuracy', 'mode_16_accuracy', 'mode_32_accuracy', 'mode_64_accuracy', 'mode_8_accuracy', 'multiplication_2_accuracy', 'multiplication_4_accuracy', 'multiplication_8_accuracy', 'odd_count_16_accuracy', 'odd_count_32_accuracy', 'odd_count_64_accuracy', 'odd_count_8_accuracy', 'sorting_16_accuracy', 'sorting_32_accuracy', 'sorting_64_accuracy', 'sorting_8_accuracy', 'subtraction_accuracy', 'sum_16_accuracy', 'sum_32_accuracy', 'sum_64_accuracy', 'sum_8_accuracy']
    basic_math_accuracy_cols = [col for col in basic_math_accuracy_cols if col in df.columns]

    instruction_followed_cols = [col for col in df.columns if 'instruction_followed' in col]
    output_tokens_cols = [col for col in df.columns if 'output_tokens' in col]

    for index, row in df.iterrows():
        model_name = row['Task_Model']
        results = {'Model': model_name}

        for category, cols in zip(
            ["Basic Math Reasoning Accuracy", "Instruction Followed", "Output Tokens"],
            [basic_math_accuracy_cols, instruction_followed_cols, output_tokens_cols]
        ):
            if not cols: # Skip if no columns for this category
                results[f'Avg. {category}'] = np.nan
                results[f'Avg. {category} Variance'] = np.nan
                continue

            values = [parse_metric_value(row[col]) for col in cols if col in row]
            variances = [parse_variance_value(row[col]) for col in cols if col in row]

            values = [v for v in values if pd.notna(v)]
            variances = [v for v in variances if pd.notna(v)]

            results[f'Avg. {category}'] = np.mean(values) if values else np.nan
            # For average variance, it's tricky. If variances are stddevs, you can't just average them.
            # If they are confidence intervals, averaging them is also not straightforward.
            # Let's assume they represent a spread and average them for a rough idea.
            results[f'Avg. {category} Variance'] = np.mean(variances) if variances else np.nan
        summary_data.append(results)

    summary_df = pd.DataFrame(summary_data)
    return summary_df.dropna(subset=['Avg. Basic Math Reasoning Accuracy', 'Avg. Instruction Followed', 'Avg. Output Tokens'], how='all')



# --- UI Sections ---
def display_main_summary(summary_df):
    st.header("🚀 LLM Performance Highlights")
    st.markdown("""
        Key insights from the LLMThinkBench evaluations. Models are ranked based on
        average performance in mathematical reasoning, instruction adherence, and token efficiency.
    """)
    if summary_df.empty:
        st.warning("No summary data to display. Check CSV loading and column name configurations.")
        return


    # Find Highlight Models
    def get_top_model(df, metric_col, ascending=False):
        if metric_col not in df.columns or df[metric_col].isnull().all():
            return None
        return df.loc[df[metric_col].idxmax() if not ascending else df[metric_col].idxmin()]

    most_accurate_model = get_top_model(summary_df, 'Avg. Basic Math Reasoning Accuracy')
    most_instruction_model = get_top_model(summary_df, 'Avg. Instruction Followed')
    least_overthinking_model = get_top_model(summary_df, 'Avg. Output Tokens', ascending=True)

    cols = st.columns(3)
    card_style = """
        <style>
            .card {
                border: 1px solid #e6e6e6;
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                transition: all 0.3s ease-in-out;
            }
            .card:hover {
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                transform: translateY(-3px);
            }
            .card h3 { margin-top: 0; }
        </style>
    """
    st.markdown(card_style, unsafe_allow_html=True)

    def display_metric_card(column, title, icon, model_data, value_col, variance_col, help_text, is_lower_better=False):
        with column:
            st.markdown(f"<div class='card'><h3>{icon} {title}</h3>", unsafe_allow_html=True)
            if model_data is not None and pd.notna(model_data[value_col]):
                st.subheader(f"{model_data['Model']}")
                value_display = f"{model_data[value_col]:.2%}" if "Accuracy" in value_col or "Followed" in value_col else f"{model_data[value_col]:.2f}"
                delta_display = f"± {model_data[variance_col]:.2f}" if pd.notna(model_data[variance_col]) else None
                st.metric(label="Score", value=value_display, delta=delta_display, delta_color="inverse" if is_lower_better else "normal")
                st.caption(help_text)
            else:
                st.info(f"No data available for {title.lower()}.")
            st.markdown("</div>", unsafe_allow_html=True)

    display_metric_card(cols[0], "Most Accurate (Math)", "🎯", most_accurate_model,
                        'Avg. Basic Math Reasoning Accuracy', 'Avg. Basic Math Reasoning Accuracy Variance',
                        "Highest avg. accuracy in basic math.")
    display_metric_card(cols[1], "Top Instruction Follower", "✅", most_instruction_model,
                        'Avg. Instruction Followed', 'Avg. Instruction Followed Variance',
                        "Highest avg. score for instruction adherence.")
    display_metric_card(cols[2], "Least Overthinking", "💡", least_overthinking_model,
                        'Avg. Output Tokens', 'Avg. Output Tokens Variance',
                        "Lowest avg. output tokens (less verbose).", is_lower_better=True)

    st.markdown("---")
    st.subheader("📈 Overall Model Summary")
    st.dataframe(
        summary_df.style.format({
            'Avg. Basic Math Reasoning Accuracy': '{:.2%}',
            'Avg. Basic Math Reasoning Accuracy Variance': '{:.2f}',
            'Avg. Instruction Followed': '{:.2%}',
            'Avg. Instruction Followed Variance': '{:.2f}',
            'Avg. Output Tokens': '{:.2f}',
            'Avg. Output Tokens Variance': '{:.2f}'
        }, na_rep="-"),
        use_container_width=True
    )

def display_plotting_arena(df_raw):
    st.header("📊 Interactive Plotting Arena")
    st.markdown("""
        Select models and metrics to visualize their performance comparatively.
        Metrics are parsed to their mean values for plotting. Error bars represent the '±' variance if available.
    """)

    if df_raw is None or df_raw.empty:
        st.warning("No data available for plotting.")
        return

    # Prepare data for plotting: Parse all relevant columns
    df_plot = df_raw.copy()
    all_cols = df_plot.columns.tolist()
    metrics_to_parse = [col for col in all_cols if col not in ['Task_Model']] # Add other non-metric columns if any

    parsed_mean_cols = {}
    parsed_variance_cols = {}

    for col in metrics_to_parse:
        if df_plot[col].dtype == 'object': # Only try to parse string columns
            mean_col_name = f"{col}_mean"
            var_col_name = f"{col}_variance"
            df_plot[mean_col_name] = df_plot[col].apply(parse_metric_value)
            df_plot[var_col_name] = df_plot[col].apply(parse_variance_value)
            if not df_plot[mean_col_name].isnull().all():
                 parsed_mean_cols[col] = mean_col_name
            if not df_plot[var_col_name].isnull().all():
                 parsed_variance_cols[col] = var_col_name


    st.sidebar.subheader("⚙️ Plotting Controls")
    available_models = sorted(df_plot['Task_Model'].unique())
    selected_models = st.sidebar.multiselect(
        "Select Models to Plot", available_models, default=available_models[:min(3, len(available_models))]
    )

    # Let user select original metric names, we'll use parsed versions internally
    available_metrics = sorted(parsed_mean_cols.keys())
    selected_original_metrics = st.sidebar.multiselect(
        "Select Metrics to Plot", available_metrics, default=available_metrics[:min(3, len(available_metrics))]
    )

    if not selected_models or not selected_original_metrics:
        st.info("Please select at least one model and one metric to generate a plot.")
        return

    plot_df_filtered = df_plot[df_plot['Task_Model'].isin(selected_models)]

    # Prepare data for Plotly: melt the DataFrame
    metrics_for_plotting_mean = [parsed_mean_cols[orig_col] for orig_col in selected_original_metrics if orig_col in parsed_mean_cols]
    # Get corresponding variance columns if they exist
    metrics_for_plotting_var = [parsed_variance_cols.get(orig_col) for orig_col in selected_original_metrics]


    if not metrics_for_plotting_mean:
        st.warning("None of the selected metrics could be parsed into numerical values for plotting.")
        return

    # Create a display name map for prettier labels on the chart
    display_metric_names_map = {parsed_mean_cols[orig_col]: orig_col for orig_col in selected_original_metrics if orig_col in parsed_mean_cols}

    plot_data_melted = plot_df_filtered.melt(
        id_vars=['Task_Model'] + [var_col for var_col in metrics_for_plotting_var if var_col], # Keep Task_Model and variance cols
        value_vars=metrics_for_plotting_mean,
        var_name='Metric_Internal',
        value_name='Value'
    )
    plot_data_melted['Metric'] = plot_data_melted['Metric_Internal'].map(display_metric_names_map)

    # Add variance data to the melted dataframe for error bars
    # This is a bit complex because melt structure needs careful handling of corresponding variance
    error_y_map = {}
    for i, orig_metric in enumerate(selected_original_metrics):
        if metrics_for_plotting_var[i] and orig_metric in parsed_mean_cols: # Check if variance column exists for this mean metric
            internal_mean_col = parsed_mean_cols[orig_metric]
            internal_var_col = metrics_for_plotting_var[i]
            # For each model, get its variance for this specific original metric
            for model in selected_models:
                variance_value = plot_df_filtered.loc[plot_df_filtered['Task_Model'] == model, internal_var_col].values
                if len(variance_value) > 0:
                     error_y_map[(model, internal_mean_col)] = variance_value[0]


    plot_data_melted['Error_Y'] = plot_data_melted.apply(
        lambda row: error_y_map.get((row['Task_Model'], row['Metric_Internal'])), axis=1
    )


    # Plotting
    fig = px.bar(
        plot_data_melted,
        x='Metric',
        y='Value',
        color='Task_Model',
        barmode='group',
        labels={'Value': 'Parsed Metric Value', 'Metric': 'Selected Metric', 'Task_Model': 'Model'},
        title="Model Comparison on Selected Metrics",
        error_y='Error_Y' # Use the mapped error values
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        title_x=0.5,
        legend_title_text='Models',
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

    # Download Plot
    st.markdown("---")
    st.subheader("📥 Download Plot")
    plot_format = st.radio("Select plot format to download:", ('PNG', 'JPEG', 'SVG', 'HTML'), horizontal=True, key="plot_download_format")

    img_bytes = None
    file_extension = plot_format.lower()
    if plot_format != 'HTML':
        try:
            img_bytes = fig.to_image(format=file_extension, scale=2) # Scale for better resolution
        except Exception as e:
            st.warning(f"Could not generate {plot_format} image. You might need to install 'kaleido': pip install kaleido. Error: {e}")
            img_bytes = None # Ensure img_bytes is None if to_image fails
    else:
        buffer = BytesIO()
        fig.write_html(buffer)
        img_bytes = buffer.getvalue()


    if img_bytes:
        st.download_button(
            label=f"Download as {plot_format}",
            data=img_bytes,
            file_name=f"llmthinkbench_comparison.{file_extension}",
            mime=f"image/{file_extension}" if plot_format != 'HTML' else "text/html"
        )


def highlight_numerical_cols(s, props='', is_accuracy_or_instruction=False, higher_is_better=True):
    """Helper for styling: highlights max/min in numerical series."""
    if s.dtype == np.object0 or pd.api.types.is_string_dtype(s): # Check if it's object or string type
        # Attempt to parse to numeric, but only for styling, don't change original data
        s_numeric = s.apply(parse_metric_value)
    else:
        s_numeric = s

    if pd.api.types.is_numeric_dtype(s_numeric):
        s_numeric = s_numeric.astype(float) # Ensure it's float for comparison
        if s_numeric.notna().any(): # Check if there's any non-NaN value
            max_val = s_numeric.max()
            min_val = s_numeric.min()
            
            if higher_is_better:
                best_val = max_val
                worst_val = min_val
            else: # Lower is better (e.g., output tokens)
                best_val = min_val
                worst_val = max_val

            styles = []
            for v_numeric in s_numeric:
                style = ''
                if pd.notna(v_numeric):
                    if v_numeric == best_val:
                        style = 'background-color: lightgreen; color: black;'
                    elif v_numeric == worst_val:
                        style = 'background-color: #FFCCCB; color: black;' # Light red
                styles.append(style)
            return styles
    return [''] * len(s)


def display_detailed_results(df_raw):
    st.header("📚 Detailed Benchmark Results")
    st.markdown("""
        Explore the complete dataset. Sort columns by clicking headers. Use sidebar filters for focused views.
        Numerical columns are highlighted: <span style='background-color:lightgreen; color:black;'>Best</span> / <span style='background-color:#FFCCCB; color:black;'>Worst</span>.
    """, unsafe_allow_html=True)

    if df_raw is None or df_raw.empty:
        st.warning("No detailed data to display.")
        return

    df_display = df_raw.copy()

    st.sidebar.subheader("🔍 Detailed Table Filters")
    all_models = sorted(df_display['Task_Model'].unique())
    selected_models_detailed = st.sidebar.multiselect(
        "Filter Models in Table", all_models, default=all_models, key="detailed_model_filter"
    )
    if selected_models_detailed:
        df_display = df_display[df_display['Task_Model'].isin(selected_models_detailed)]
    else:
        df_display = df_display.iloc[0:0]

    # Column selection for display
    all_columns = df_raw.columns.tolist()
    default_cols = [col for col in all_columns if 'accuracy' in col or 'instruction_followed' in col or 'output_tokens' in col or 'Task_Model' in col]
    selected_columns_display = st.sidebar.multiselect(
        "Select Columns to Display in Table", all_columns, default=default_cols, key="detailed_column_select"
    )
    if selected_columns_display:
        df_display = df_display[selected_columns_display]


    # Apply styling
    styled_df = df_display.style
    for col_name in df_display.columns:
        if col_name == 'Task_Model':
            continue
        is_accuracy_or_instruction = 'accuracy' in col_name.lower() or 'instruction_followed' in col_name.lower()
        is_tokens_or_length = 'tokens' in col_name.lower() or 'length' in col_name.lower() or 'count' in col_name.lower() # word_count too

        # Default to higher is better unless it's tokens/length/count
        higher_is_better = not is_tokens_or_length

        styled_df = styled_df.apply(
            highlight_numerical_cols,
            subset=[col_name],
            is_accuracy_or_instruction=is_accuracy_or_instruction,
            higher_is_better=higher_is_better
        )

    st.dataframe(styled_df, use_container_width=True)

    st.markdown("---")
    if not df_display.empty:
        csv_download = df_display.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Table Data as CSV",
            data=csv_download,
            file_name='filtered_llmthinkbench_results.csv',
            mime='text/csv',
            key="download_detailed_csv"
        )


# --- Main Application ---
def main():
    st.sidebar.image("https://raw.githubusercontent.com/ctrl-gaurav/LLMThinkBench/main/logo/LLMThinkBench_logo_small.png", width=150) # Replace with your actual logo URL if available
    st.sidebar.title("LLMThinkBench")
    st.sidebar.markdown("Navigate through the leaderboard sections:")

    raw_df = load_data('cleaned_data.csv')

    if raw_df is None:
        st.error("Fatal Error: Could not load benchmark data. Please ensure 'cleaned_data.csv' is present and correctly formatted.")
        return

    summary_df = calculate_summary_metrics(raw_df)

    page_options = {
        "🏆 Key Performance Indicators": display_main_summary,
        "📊 Interactive Plotting Arena": display_plotting_arena,
        "📚 Detailed Benchmark Results": display_detailed_results,
    }
    selection = st.sidebar.radio("Go to", list(page_options.keys()), key="main_nav")

    # Call the selected page function
    if selection == "🏆 Key Performance Indicators":
        page_options[selection](summary_df)
    else: # For other pages that use raw_df
        page_options[selection](raw_df)


    st.sidebar.markdown("---")
    st.sidebar.markdown("Developed for [LLMThinkBench](https://github.com/ctrl-gaurav/LLMThinkBench)")
    st.sidebar.markdown(f"Current Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}")


if __name__ == "__main__":
    main()