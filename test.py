from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    azure_deployment="your-deployment-name",
    openai_api_version="2024-02-01", # Ensure this matches your deployment's version
    azure_endpoint="https://YOUR-RESOURCE-NAME.openai.azure.com/",
    api_key="your-azure-api-key",
)

response = llm.invoke("Hello!")
print(response.content)