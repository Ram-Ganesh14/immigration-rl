# Airport Immigration Processing Environment - Project Manual

This manual provides a detailed overview of the project structure, explaining the purpose of each file and folder, and how the entire system works together as an AI interaction environment following the OpenEnv spec.

## Project Root

### `openenv.yaml`
The core metadata configuration file mandated by the OpenEnv format. It defines:
- **Observation Space**: What the agent sees (sanitised passenger profile and hidden data elements).
- **Action Space**: Valid actions the agent can take, both terminal (`clear`, `deny`, etc.) and info-gathering (`query_interpol`, `verify_biometrics`).
- **Tasks**: Defines the 4 tasks (`task1_document_check` through `task4_adversarial`), their difficulty, queue sizes, and max steps.
- **Endpoints & Docker settings**: Defines the REST API mapping and deployment configuration (e.g., port 7860).

### `inference.py`
The baseline participant script using the OpenAI client.
- This represents how standard participants will interact with your environment.
- Uses `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`, and `ENV_URL` environment variables.
- Connects to the local FastAPI server to play through episodes, sending prompt inputs to the LLM and receiving JSON actions.
- Strictly formatted to emit `[START]`, `[STEP]`, and `[END]` stdout logs for automated grader evaluation.

### `Dockerfile`
Instructions for building the Docker image required to deploy the environment on Hugging Face Spaces.
- Uses a `python:3.11-slim` base image.
- Exposes port 7860 as required by the Hugging Face Docker Space requirements.

### `README.md`
The main documentation intended for users and participants. It provides:
- The environment's philosophy and how to interact with it.
- Comprehensive detailing of tracking, constraints, and tasks descriptions.
- A checklist validating its competition readiness.

### `client.py`
Provides synchronous and asynchronous HTTP client wrappers around the core `/step`, `/reset`, and `/grade` APIs. Useful for building more complex automated test runners outside of `inference.py`.

### `__init__.py`
An empty or near-empty initialization file marking the root directory as an importable module where needed.

### `requirements.txt` & `pyproject.toml`
The Python dependencies required to run the local API (`fastapi`, `uvicorn`, `pydantic`, `openai`, `pytest`, etc.).

---

## The `/models` Directory

### `models.py`
Contains all the strongly typed Pydantic models used to represent the environment's current state and validate requests/responses. 
- **PassengerProfile / Document**: Defines publicly visible information for passengers.
- **_PassengerInternalData**: Holds hidden information (watchlist matches, biometric discrepancies) which is NOT serialized to the agent initially and requires API queries to reveal.
- **Observation / Action**: Represents the input and expected output formats in the OpenEnv `step()` cycle.
- **Reward / State**: Used to communicate reward scores, steps remaining, and the internal progress of the environment queue.

---

## The `/server` Directory
This contains the core internal logic and API definition for the application.

### `main.py`
The FastAPI application entry point. 
- Defines the REST API endpoints corresponding to the `openenv.yaml` (`/health`, `/reset`, `/step`, `/state`, `/grade`, `/tasks`).
- Maintains an in-memory instance of the environment so concurrent testing or a single-threaded workflow can route API calls to the logic.

### `environment.py`
The `ImmigrationEnvironment` class which serves as the "game master".
- **`reset(task_id, seed)`**: Bootstraps a fresh queue of passengers.
- **`step(action)`**: The primary game loop. Takes agent actions (e.g. `clear`, `query_interpol`), processes them, charges time/costs, reveals hidden datasets when appropriate, and updates the state.
- **Internal State tracking**: Stores variables like API budget allocation and tracks demographic decisions (for bias analysis).

### `data_generator.py`
A comprehensive mock data factory that programmatically creates passengers.
- Depending on the `task_id`, it seeds a different pseudo-random scenario based dataset. 
- Uses seeded probability (so generation is deterministically reproducible) to generate edge-case passengers (e.g. expired visas paired with emergencies, partial watchlist names). 
- Returns paired `(PassengerProfile, _PassengerInternalData)` objects ensuring safe compartmentalization of what the passenger looks like vs what their background security check entails.

---

## The `/graders` Directory

### `graders.py`
Contains the evaluation logic. Instead of standard pass/fail grading, the OpenEnv format encourages fractional scoring.
- Implements targeted grading classes for Task 1 to 4 (`Task1Grader`, etc.)
- Evaluates the actions taken in a completed episode against the expected "ground truth" behaviors of the `data_generator.py`.
- **Task 4 Grader (Demographic Bias Tracking)**: Features custom logic measuring error rates segmented by generated passenger nationalities; if one nationality is systematically discriminated against for no factual reason, it penalises the agent significantly.

---

## The `/tests` Directory

### `test_environment.py`
Automated deterministic test specifications written for Python's `pytest` framework. 
- Comprises 32 individual unit and integration tests.
- Ensures features like "demographic bias punishment", "hidden APIs revealing data correctly", and "queue progression mechanics" function predictably before new commits break previous compatibility.
