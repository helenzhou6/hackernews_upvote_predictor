# Week One MLX

## Dev - running the app locally

### Prerequisites
Needed:
- Python (v3.9.6)
- Poetry for python package management (`brew install poetry` or see https://python-poetry.org/ to install. v2.1.3 or higher)
- .env file needs to be populated correctly (get this from Helen, or the environment variables in this repo). Example contents:
    ```python
    POSTGRES_USERNAME=xxx
    POSTGRES_PASSWORD=xxx
    DB_HOST=xxx
    DB_PORT=xxx
    MODEL_API_URL=xxx
    DATABASE_API_URL=xxx
    ```

### 1. Initial python set up
To run the digit_classifier python files:
1. Ensure poetry is using python v3.9.6 (see commands listed: https://python-poetry.org/docs/managing-environments/ - e.g. using `poetry env use 3.9`)
2. `poetry env activate` to use the poetry virtual environment created
    - (To deactivate virtual env if needed, run `deactivate`)
3. `poetry install` to install project requirements.
4. On Mac and VSCode, run Shift Command P and select interpreter as the poetry env created

### 2. PostgreSQL database
Code for the postgreSQL database set up and SQL queries can be found in `src/digit_classifier/database`
A postgreSQL database is used to log the feedback of the user, along with the prediction and confidence level. All entries to the database is displayed to the user on the frontend.


### Data > items
- id, dead, type, by, time, text, parent, kids, url, score, title, descandants 