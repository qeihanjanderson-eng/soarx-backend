# SoarX Backend

This repository contains a FastAPI backend prepared for deployment on Render.

## Render Deployment

1. Create or connect a Render service.
2. Use the following build command:

```bash
pip install -r requirements.txt
```

3. Use the following start command:

```bash
uvicorn main:app --host 0.0.0.0 --port 10000
```

4. Render will use the included `.render.yaml` configuration to deploy the service.

> Render automatically sets `PORT=10000` for this service.

## Local Run

Install dependencies and run locally:

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 10000
```

## Notes

- The FastAPI app is exposed as `app` in `main.py`.
- CORS is enabled for all origins.
- There is no embedded `uvicorn.run(...)` call in `main.py`.
