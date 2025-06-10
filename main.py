import os
from agents import Runner, Agent, OpenAIChatCompletionsModel, AsyncOpenAI
from agents.run import RunConfig
from dotenv import load_dotenv

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    # model="gemini-2.5-flash-preview-05-20",
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

agent = Agent(
    name="Student Assistant Agent",
    instructions="An agent that helps students and answer their academic questions, \
provide study tips, summarize small text passages"
)

user_input = input("Enter your question: ")

while True:
    result = Runner.run_sync(
        agent,
        user_input,
        run_config=config
    )

    print(result.final_output)
    
    user_input = input("Your message (type exit to end chat): ")
    if user_input == 'exit':break

print('Nice to help you, bye 👋')