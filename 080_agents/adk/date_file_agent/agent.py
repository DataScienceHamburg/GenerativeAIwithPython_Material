import datetime
from google.adk.agents import Agent
import os

def get_current_date() -> dict:
    """Returns the current date.
    
    Args:

    Returns:
        dict: current date
    """

    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    if current_date:
        return {
            "status": "success",
            "current_date": current_date
        }
    else:
        return {
            "status": "error",
            "message": "Failed to get current date"
        }

def get_local_files(folder_path: str) -> dict:
    """Returns the list of files in the current directory.
    
    Args:
        folder_path (str): The path to the folder to list the files from.
    """
    try:
        files = os.listdir(folder_path)
        return {
            "status": "success",
            "files": files
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to get local files: {e}"
        }

agent = Agent(
    name="current_date_file_agent",
    model="gemini-2.0-flash",
    description="Agent to get the current date and local files",
    instruction="You are a helpful assistant that can get the current date and list files.",
    tools=[get_current_date, get_local_files]
)