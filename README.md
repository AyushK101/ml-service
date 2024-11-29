- Create virtual env:  `uv venv`
- Install dependencies: `uv pip install -r pyproject.toml`

- DEVELOPER ONLY  create app: `python manage.py startapp app1`
- Migrate: `python manage.py migrate`.
- Create a `.env` file and add env variables required.
- Example-> env var access: `SECRET_KEY = os.getenv('SECRET_KEY')`.