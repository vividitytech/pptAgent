CONFIGS = {
    "dbconfig": {
        "dialect": "SQL",
        "max_tokens":  14000
    },
    "faissconfig": {
        "embedding_dim": 384,
        "client": "persistent",
        "n_results_sql":  10,
        "n_results_ddl": 10,
    },
    "chatconfig": {
        "api_key": "",
        "model":  "gpt-4o-mini",
        "temperature":0.5,
    },
    "max_tries": 3,
}