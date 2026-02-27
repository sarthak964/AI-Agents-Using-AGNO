from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.workflow import Step, Workflow, Parallel
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.hackernews import HackerNewsTools
from dotenv import load_dotenv

# load the api keys
load_dotenv()

# define the model
model = OpenAIChat(id="gpt-4.1-mini")

# ======================== Agents ==============================

# google search agent
google_agent = Agent(
    id="google-agent",
    name="Google Search Agent",
    instructions=["You are an expert web search agent using google search",
                  "Provide accurate and relevant information based on user input",
                  "add heading to the output Source: Google Search"],
    model=model,
    tools=[GoogleSearchTools()],
)

# duckduckgo agent
duckduckgo_agent= Agent(
    id="duckduckgo-agent",
    name="DuckDuckGo Search Agent",
    instructions=["You are an expert web search agent using duckduckgo search",
                  "Provide accurate and relevant information based on user input",
                  "add heading to the output Source: DuckDuckGo Search"],
    model=model,
    tools=[DuckDuckGoTools()],
)


# hackernews agent
hackernews_agent = Agent(
    id="hackernews-agent",
    name="HackerNews Agent",
    instructions=["You are an expert in retrieving trending topics and latest news from Hacker News",
                  "Add heading Source: HackerNews"],
    model=model,
    tools=[HackerNewsTools()],
)


# report generation agent
report_generation_agent = Agent(
    id="report-generation-agent",
    name="Report Generation Agent",
    model=model,
    instructions=["you are an expert in report generation",
                  "Compile all the information from various sources into a coherent and comprehensive report",
                  "use the information from both Google Search and DuckDuckGo search",
                  "mention the source of information in the report"
                  "The output should be in a proper format"],
    markdown=True
)


# ===================== Steps =============================

# google search step
google_search_step = Step(
    name="Google Search Step",
    agent=google_agent,
    description="Performs a web search using Google Search"
)

# duckduckgo search step
duckduckgo_search_step = Step(
    name="DuckDuckGo Search Step",
    agent=duckduckgo_agent,
    description="Performs a web search using DuckDuckgo Search"
)

# hackernews search step
hackernews_search_step = Step(
    name="Hackernews Search Step",
    agent=hackernews_agent,
    description="Performs the latest news search on HackerNews"
)

# report generation step
report_generation_step = Step(
    name="Report Generation Step",
    agent=report_generation_agent,
    description="Generates a report compiling information gathered from various sources"
)

# parallel steps
parallel_steps = Parallel(
    google_search_step, duckduckgo_search_step,
    name="Parallel Search Step",
    description="Perform searches from various sources parallely"
)



# ======================== Workflow ===================================
parallel_workflow = Workflow(
    id="parallel-workflow",
    name="Retrieval and Report Generation Workflow",
    steps=[parallel_steps, report_generation_step],
    description="A workflow that performs web searching using multiple agents in parallel and then generates a report based on the retrieved information"
)


parallel_workflow.print_response(input="topic: AI",
                                 stream=True,
                                 markdown=True)