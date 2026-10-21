import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

models = sorted(m.id for m in client.models.list().data)
print(f"{len(models)} models available\n")
for m in models:
    print(m)