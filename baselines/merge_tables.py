import os
import glob
import re

# Function to extract rows from the LaTeX table in a file
def extract_rows_from_file(filename):
    with open(filename, 'r') as file:
        content = file.read()
    
    # Use regex to extract table rows
    rows = re.findall(r'\\midrule\n(.*?)\\\\\n\\bottomrule', content, re.DOTALL)
    if rows:
        return [row.strip() for row in rows[0].split('\\\\')]
    return []

# Function to create a combined LaTeX table
def create_combined_table(file_rows):
    header = (
        "\\begin{tabular}{ccccccccccc}\n"
        "\\toprule\n"
        r"Dataset & Methods & Linear Model & LASSO & Random Forest & Neural Net & $\mathcal{C}^{M}$ & $\mathcal{C}^{R}$ & $\mathcal{C}^{U}$ & \textit{DECP (Single-Stage)} & DECP \\"
        "\\midrule"
    )

    metric_name = {
        "cov": "Coverage",
        "len": "Length",
    }

    body = "\n"
    for filename, rows in file_rows.items():
        for i, row in enumerate(rows):
            # Separate metric (cov, len, etc.) from the rest of the row
            metric, *values = row.split('&')
            values[-2] = r"\textit{" + values[-2] + r"}"
            values_str = ' & '.join(values).strip()
            if i == 0:
                body += f"{os.path.basename(filename).split('.')[0]} "
            body += f"& {metric_name[metric.strip()]} & {values_str} \\\\ \n"
        body += r"\hline \\" + "\n"
        body += r"\hline" + "\n"

    footer = "\\bottomrule\n\\end{tabular}"
    return header + body + footer

# Main script
def main():
    # Get all input files (assume .txt files in the current directory)
    input_files = sorted(glob.glob("results/*.txt"))

    # Extract rows from each file
    file_rows = {}
    for file in input_files:
        file_rows[file] = extract_rows_from_file(file)

    # Create the combined LaTeX table
    combined_table = create_combined_table(file_rows)

    # Save the combined table to a file
    output_file = "combined_table.tex"
    with open(output_file, 'w') as f:
        f.write(combined_table)

    print(f"Combined table saved to {output_file}")

if __name__ == "__main__":
    main()