"""
Core visualization functionality for router agent graph.
"""

import argparse
import os
import webbrowser
from tempfile import NamedTemporaryFile

from ..graph_router import visualize_graph


def generate_html(mermaid_code: str) -> str:
    """Generate HTML with embedded Mermaid diagram."""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Router Agent Graph Visualization</title>
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 2px solid #eee;
                padding-bottom: 10px;
            }}
            .mermaid {{
                margin: 40px 0;
                overflow: auto;
            }}
            .instructions {{
                background-color: #f8f9fa;
                border-left: 4px solid #007bff;
                padding: 15px;
                margin: 20px 0;
                border-radius: 4px;
            }}
        </style>
    </head>
    <body>
        <h1>Router Agent Graph Visualization</h1>
        
        <div class="instructions">
            <p>This diagram shows the flow of the router agent graph, illustrating how queries are classified and processed through different specialized agents.</p>
        </div>
        
        <div class="mermaid">
        {mermaid_code}
        </div>
        
        <script>
            mermaid.initialize({{ startOnLoad: true, theme: 'default', securityLevel: 'loose' }});
        </script>
    </body>
    </html>
    """


def open_in_browser(html_content: str) -> None:
    """Open the generated HTML content in a browser."""
    with NamedTemporaryFile(suffix='.html', delete=False) as f:
        f.write(html_content.encode('utf-8'))
        temp_path = f.name
    
    # Open the file in the browser
    webbrowser.open(f'file://{temp_path}')
    
    print(f"Router graph visualization opened in your default browser.")
    print(f"Temporary file created at: {temp_path}")
    print("You can close the browser when done and delete the temporary file if desired.")


def save_to_file(html_content: str, path: str) -> None:
    """Save the HTML content to a specified file."""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Router graph visualization saved to: {path}")


def main():
    """Main entry point for visualization tool."""
    parser = argparse.ArgumentParser(description="Visualize the router agent graph structure")
    
    parser.add_argument(
        "--output", "-o", 
        help="Output HTML file path (if not specified, opens in browser)"
    )
    
    parser.add_argument(
        "--raw", 
        action="store_true", 
        help="Output raw Mermaid code instead of HTML"
    )
    
    args = parser.parse_args()
    
    # Get the Mermaid code for the graph
    mermaid_code = visualize_graph()
    
    if args.raw:
        print(mermaid_code)
    elif args.output:
        html_content = generate_html(mermaid_code)
        save_to_file(html_content, args.output)
    else:
        html_content = generate_html(mermaid_code)
        open_in_browser(html_content)