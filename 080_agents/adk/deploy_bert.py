#%% packages
import os
import sys

import vertexai
from absl import app, flags
from dotenv import load_dotenv
from vertexai import agent_engines
from vertexai.preview import reasoning_engines

from multi_tool_agent.agent import root_agent
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

#%% connect to vertex ai
project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
location = os.getenv("GOOGLE_CLOUD_LOCATION")

vertexai.init(
    project=project_id,
    location=location,
    staging_bucket=os.getenv("GOOGLE_CLOUD_STAGING_BUCKET")
)

#%%
# run in terminal:  gcloud auth application-default login

app = reasoning_engines.AdkApp(
        agent=root_agent,
        enable_tracing=True,
    )


# %%
# Set up Google Cloud credentials
# run in terminal:  pip install google-cloud-aiplatform[agent_engines]
remote_app = agent_engines.create(
        agent_engine=app,
        requirements=[
            "google-cloud-aiplatform[adk,agent_engines]",
        ],
        extra_packages=["./multi_tool_agent"],
    )
print(f"Created remote app: {remote_app.resource_name}")


# %% list all deployments
next(agent_engines.list())


#%% create a session
app_name = "projects/512212709544/locations/us-central1/reasoningEngines/5515651702226681856"
remote_app = agent_engines.get(app_name)
session = remote_app.create_session(user_id="test_user")

#%% send a message
for event in remote_app.stream_query(user_id="test_user", 
                        session_id=session['id'],
                        message="What is the weather in new york?"):
    print(event)

#%% list sessions
remote_app.list_sessions(user_id="test_user")




# %%
