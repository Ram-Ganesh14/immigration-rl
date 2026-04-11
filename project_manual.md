# Airport Immigration Processing Environment - Project Manual

This manual provides a detailed overview of the project structure, explaining the purpose of each file and folder, and how the entire system works together as an AI interaction environment following the OpenEnv spec.

## Project Root

### `openenv.yaml`
The core metadata configuration file mandated by the OpenEnv format. It defines:
- **Observation Space**: What the agent sees (sanitised passenger profile and hidden data elements).
- **Action Space**: Valid actions the agent can take, both terminal (`clear`, `deny`, etc.) and info-gathering (`query_interpol`, `verify_biometrics`).
- **Tasks**: Defines the 5 tasks (`task1_document_check` through `task5_system_disruption`), their difficulty, queue sizes, and max steps.
- **RL Dynamics**: Documents the adversarial queue escalation and API reliability degradation mechanics.
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

### `app.py`
The FastAPI application entry point. 
- Defines the REST API endpoints corresponding to the `openenv.yaml` (`/health`, `/reset`, `/step`, `/state`, `/grade`, `/tasks`, `/explain`, `/dashboard`).
- Maintains an in-memory instance of the environment so concurrent testing or a single-threaded workflow can route API calls to the logic.
- Serves the live monitoring dashboard from the `/dashboard` directory.

### `environment.py`
The `ImmigrationEnvironment` class which serves as the "game master".
- **`reset(task_id, seed)`**: Bootstraps a fresh queue of passengers.
- **`step(action)`**: The primary game loop. Takes agent actions (e.g. `clear`, `query_interpol`), processes them, charges time/costs, reveals hidden datasets when appropriate, and updates the state.
- **Internal State tracking**: Stores variables like API budget allocation and tracks demographic decisions (for bias analysis).
- **RL Mechanic 1 — Adversarial Queue Escalation**: Tracks consecutive "sloppy clears" (terminal clear without any API query). After 3 consecutive sloppy clears, dynamically injects 2 high-risk passengers (watchlist hit + forged document) at the front of the queue.
- **RL Mechanic 2 — API Reliability Degradation**: Tracks global API usage rate. When the rate exceeds 1.5 calls/passenger after 4+ passengers processed, APIs enter a degraded state: biometric face_match scores gain ±0.20 noise, and watchlist queries have a 15% chance of returning false positive fuzzy matches.

### `data_generator.py`
A comprehensive mock data factory that programmatically creates passengers.
- Depending on the `task_id`, it seeds a different pseudo-random scenario based dataset. 
- Uses seeded probability (so generation is deterministically reproducible) to generate edge-case passengers (e.g. expired visas paired with emergencies, partial watchlist names). 
- Returns paired `(PassengerProfile, _PassengerInternalData)` objects ensuring safe compartmentalization of what the passenger looks like vs what their background security check entails.

---

## The `/graders` Directory

### `graders.py`
Contains the evaluation logic. Instead of standard pass/fail grading, the OpenEnv format encourages fractional scoring.
- Implements targeted grading classes for Task 1 to 5 (`grade_task1` through `grade_task5`).
- Evaluates the actions taken in a completed episode against the expected "ground truth" behaviors of the `data_generator.py`.
- **Task 4 Grader (Intersectional Bias Tracking)**: Features custom logic measuring error rates segmented by nationality, gender, and intersectional groups (nationality × gender). Systematic discrimination triggers escalating penalties.
- **Task 5 Grader (System Disruption)**: Evaluates the agent's ability to adapt under API outages and passenger surges, scoring adaptation behavior.

---

## The `/tests` Directory

### `test_environment.py`
Automated deterministic test specifications written for Python's `pytest` framework. 
- Comprises 64 individual unit and integration tests across 10 test classes.
- Covers: lifecycle, hidden APIs, demographic bias, data generator, policy search, system disruption, new archetypes, intersectional bias, explainability, adversarial escalation (RL Mechanic 1), and API degradation (RL Mechanic 2).
- Ensures features like "adversarial queue injection", "API noise degradation", "demographic bias punishment", "hidden APIs revealing data correctly", and "queue progression mechanics" function predictably before new commits break previous compatibility.
