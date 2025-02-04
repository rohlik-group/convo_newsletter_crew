from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.llm import LLM

from convo_newsletter_crew.tools.word_counter_tool import WordCounterTool

from dotenv import load_dotenv
import os
import litellm
litellm.modify_params = True

load_dotenv()

@CrewBase
class ConvoNewsletterCrew:
    """ConvoNewsletterCrew crew"""

    def __init__(self):
        # Verify API key is loaded
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        
        model = os.getenv("MODEL", "anthropic/claude-3-sonnet-20240229-v1:0")

        # Configure the LLM once for all agents
        self.llm = LLM(
            model=model,
            temperature=0.7,
            api_key=api_key
        )

        self.agents_config = "config/agents.yaml"
        self.tasks_config = "config/tasks.yaml"

    @agent
    def synthesizer(self) -> Agent:
        return Agent(
            config=self.agents_config["synthesizer"], 
            verbose=True,
            llm=self.llm,
            )

    @agent
    def newsletter_writer(self) -> Agent:
        return Agent(
            config=self.agents_config["newsletter_writer"],
            tools=[WordCounterTool()],
            verbose=True,
            llm=self.llm,
        )

    @agent
    def newsletter_editor(self) -> Agent:
        return Agent(
            config=self.agents_config["newsletter_editor"],
            tools=[WordCounterTool()],
            verbose=True,
            llm=self.llm,
        )

    @task
    def generate_outline_task(self) -> Task:
        return Task(
            config=self.tasks_config["generate_outline_task"],
        )

    @task
    def write_newsletter_task(self) -> Task:
        return Task(
            config=self.tasks_config["write_newsletter_task"],
            output_file="newsletter_draft.md",
        )

    @task
    def review_newsletter_task(self) -> Task:
        return Task(
            config=self.tasks_config["review_newsletter_task"],
            output_file="final_newsletter.md",
        )

    @crew
    def crew(self) -> Crew:
        """Creates the ConvoNewsletterCrew crew"""

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            chat_llm=self.llm,
        )
