# Shared project settings. Everyone imports from here; nobody hard-codes these.
CATEGORIES = ["cs.LG", "cs.CL", "cs.CV"]   # edit to match what the team agreed
YEAR_MIN, YEAR_MAX = 2018, 2024
MAX_PER_CATEGORY = 40_000                  # cap so the dataset stays manageable and balanced (None = no cap)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
SEED = 42
SAMPLE_SIZE = 10_000
