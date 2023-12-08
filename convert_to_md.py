from nbconvert import MarkdownExporter
import nbformat
import os

# Output directory for Markdown files
output_directory = 'data'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

def convert_notebooks_in_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.ipynb'):
                notebook_file = os.path.join(root, file)
                
                # Read the notebook
                with open(notebook_file, 'r', encoding='utf-8') as f:
                    nb = nbformat.read(f, as_version=4)
                
                # Configure the exporter
                md_exporter = MarkdownExporter()
                
                # Convert notebook to Markdown
                (body, resources) = md_exporter.from_notebook_node(nb)
                
                # Extract relative path and create subdirectories in 'data' if they don't exist
                relative_path = os.path.relpath(root, folder_path)
                output_subdirectory = os.path.join(output_directory, relative_path)
                if not os.path.exists(output_subdirectory):
                    os.makedirs(output_subdirectory)
                
                # Write the Markdown content to a file in the 'data' folder
                output_file = os.path.join(output_subdirectory, file.replace('.ipynb', '.md'))
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(body)

# Specify the main folder path containing multiple subfolders with notebooks
main_folder_path = r'E:\ML Projects\llm-tensorflow-docs\raw-data'

# Process notebooks in the main folder and its subfolders
convert_notebooks_in_folder(main_folder_path)
