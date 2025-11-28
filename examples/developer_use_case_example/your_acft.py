# your_project/use_acft.py
from acft.adapters import OllamaLLM, OllamaEmbedder
from acft.config import load_acft_settings
from acft.engine import ACFTEngine
from developer_use_case_example.your_retriever import PostgresRetriever

# 1) Load settings (reads .env)
settings = load_acft_settings()

# 2) Build components
llm = OllamaLLM(
    model_name=settings.llama_model,
    base_url=settings.llama_base_url,
)

embedder = OllamaEmbedder(
    model=settings.embed_model,
    base_url=settings.embed_base_url,
)

# your DB connection...
conn = ...

retriever = PostgresRetriever(embedder=embedder, conn=conn)

# 3) Build engine with your own retriever
engine = ACFTEngine(
    llm=llm,
    embedder=embedder,
    retriever=retriever,
    config=settings.acft_config,
)

# 4) Run
result = engine.run("Explain how many moons Earth has.", debug=True)
print(result.decision, result.answer)