# Week One MLX

## Dev - running the app locally

### Prerequisites
Needed:
- Python (v3.9.6)
- uv for python package management (https://github.com/astral-sh/uv)
- .env file needs to be populated correctly (get this from Helen, or the environment variables in this repo). Example contents:
    ```python
    POSTGRES_USERNAME=xxx
    POSTGRES_PASSWORD=xxx
    DB_HOST=xxx
    DB_PORT=xxx
    MODEL_API_URL=xxx
    DATABASE_API_URL=xxx
    ```

### Initial python set up
To run the hackernews upvote predictor python files:
2. `source .venv/bin/activate` to use the poetry virtual environment created
    - (To deactivate virtual env if needed, run `deactivate`)
3. `uv sync` to install project requirements.
4. On Mac and VSCode, run Shift Command P and select interpreter as the poetry env created (using .venv within directory)

### Hacker news database > items info
- Column info: id, dead, type, by, time, text, parent, kids, url, score, title, descandants 