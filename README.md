# AI Product Review Aggregator

A beginner-friendly full-stack project that analyzes product reviews with:

- React + Vite + Tailwind CSS on the frontend
- Chart.js for the sentiment pie chart
- FastAPI on the backend
- TextBlob for local sentiment analysis
- Google Gemini for summary, pros, and cons extraction

## Important Gemini note

The original request mentioned the legacy `google-generativeai` SDK and the `gemini-pro` model. By **March 31, 2026**, Google’s current Gemini docs point developers to the newer Google Gen AI SDK and current Gemini model family instead.  
To keep this project runnable today, the backend uses the current `google-genai` SDK with `gemini-2.5-flash` by default.

## Project structure

```text
backend/
  main.py
  requirements.txt
  .env.example

frontend/
  index.html
  package.json
  vite.config.js
  src/
    App.jsx
    index.css
    main.jsx
    components/
      SentimentChart.jsx
```

## Backend setup

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
Copy-Item .env.example .env
```

Update `.env` with your Gemini API key:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

Run the FastAPI server:

```powershell
uvicorn main:app --reload
```

The backend will be available at `http://127.0.0.1:8000`.

## Frontend setup

```powershell
cd frontend
npm install
npm run dev
```

The frontend will start at `http://127.0.0.1:5173`.

## API endpoint

`POST /analyze`

Example request:

```json
{
  "reviews": [
    "The battery lasts all day and the display is excellent.",
    "Good value for money, but the camera is disappointing."
  ]
}
```

Example response:

```json
{
  "summary": "Customers like the overall value and battery life, but mention camera limitations.",
  "pros": ["Good value for money", "Battery lasts all day"],
  "cons": ["Camera performance could be better"],
  "sentiment": {
    "positive": 50.0,
    "negative": 50.0
  }
}
```

## Notes

- If `GEMINI_API_KEY` is missing or Gemini fails, the backend falls back to a simple local summary/pros/cons extraction so the demo still works.
- CORS is enabled in FastAPI so the React app can call the API during development.
- Each line in the textarea is sent as one review string to the backend.
