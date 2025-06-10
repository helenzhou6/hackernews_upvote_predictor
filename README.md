# Week One MLX

## Dev - running the app locally

### Prerequisites
Needed:
- Python (v3.13.3) - might work with lower versions, just try it
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
4. On Mac and VSCode, run Shift Command P and select interpreter as the poetry env created (using .venv within directory), and can run the python files

### Hacker news database > items info
- Column info: id, dead, type, by, time, text, parent, kids, url, score, title, descandants 

## How to make changes to the codebase
1. Ensure you are on the main branch `git checkout main`
2. Pull down any new changes `git pull` (you may have merge conflicts, resolve those)
3. Create a new branch - `git checkout -b <name of branch>`
4. Add changes - `git add .` - this will add all files changed
5. Commit the changes `git commit -m <commit message>`
6. Push changes to remote (github) - `git push` (you may need to do `git push --set-upstream origin <branch name>` if the branch doesn't exist on remote)
7. Go to github.com and make a pull request using the branch name