from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai_tools import SerperDevTool, WebsiteSearchTool

# Tools
search_tool = SerperDevTool()
website_search_tool = WebsiteSearchTool()

@CrewBase
class TravelPlanner():
    """TravelPlanner crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def city_selector_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['city_selector_agent'], # type: ignore[index]
            verbose=True
        )

    @agent
    def local_expert_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['local_expert_agent'], # type: ignore[index]
            verbose=True,
            tools=[search_tool, website_search_tool]
        )
    
    @agent
    def concierge_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['concierge_agent'], # type: ignore[index]
            verbose=True  # Add tools to concierge agent
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def identify_city_task(self) -> Task:
        return Task(
            config=self.tasks_config['identify_city_task'], # type: ignore[index]
        )

    @task
    def gather_info_task(self) -> Task:
        return Task(
            config=self.tasks_config['gather_info_task'], # type: ignore[index]
            output_file='report.md'
        )

    @task
    def detail_planning_task(self) -> Task:
        return Task(
            config=self.tasks_config['detail_planning_task'], # type: ignore[index]
            output_file='report_travel_planner.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the TravelPlanner crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            max_rpm=30
            
        )
