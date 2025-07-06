#%% packages
from mcp.server.fastmcp import FastMCP
import os
# import win32com.client as win32

#%% Create an MCP server
mcp = FastMCP("Markdown")


#%% Add an addition tool
@mcp.tool()
def list_files(folder_path: str) -> list[str]:
    """List files in a folder"""
    return os.listdir(folder_path)

@mcp.tool()
def write_markdown_file(file_path: str, content: str) -> str:
    """Write a markdown file
    Args:
        file_path: path to the file
        content: content to write to the file
    Returns:
        success message
    """
    with open(file_path, 'w') as file:
        file.write(content)
    return f"File {file_path} written successfully"

#%% install in Claude desktop
# mcp install server.py



