# Doctor Face Analysis — AURA-Equivalent Skin Analysis System

## Architecture

```
face_analysis/
├── backend/                    # Python FastAPI backend (Azure T4 GPU)
│   ├── src/
│   │   ├── config.py           # Configuration management
│   │   ├── server.py           # FastAPI application & routes
│   │   ├── analyzers/          # SOLID - each analyzer is independent
│   │   │   ├── base.py         # Abstract base analyzer
│   │   │   ├── face_detector.py
│   │   │   ├── wrinkle.py
│   │   │   ├── pore.py
│   │   │   ├── pigmentation.py
│   │   │   ├── redness.py
│   │   │   ├── texture.py
│   │   │   ├── symmetry.py
│   │   │   └── measurements.py
│   │   ├── pipeline/           # Orchestration
│   │   │   ├── analysis_pipeline.py
│   │   │   └── depth_enhancer.py
│   │   ├── reports/            # Report generation
│   │   │   └── report_generator.py
│   │   └── models/             # Data models (Pydantic)
│   │       └── schemas.py
│   ├── tests/                  # Unit tests
│   │   ├── test_analyzers.py
│   │   ├── test_pipeline.py
│   │   ├── test_api.py
│   │   └── conftest.py
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/                   # React UI (integrated into desktop_app)
│   ├── FaceAnalysis.jsx        # Main analysis view
│   ├── PhotoCapture.jsx        # Multi-angle guided capture
│   ├── AnalysisResults.jsx     # Results dashboard
│   ├── HeatmapOverlay.jsx      # Heatmap visualization
│   ├── FaceMeasurements.jsx    # Measurement display
│   ├── ReportView.jsx          # Report preview & export
│   ├── BeforeAfter.jsx         # Comparison view
│   └── AnnotationEditor.jsx    # Drawing/annotation tools
└── README.md
```

## Quick Start

### Backend

```bash
cd face_analysis/backend
pip install -r requirements.txt
uvicorn src.server:app --host 0.0.0.0 --port 8766
```

### Frontend

Integrated into `desktop_app/` — see components in `face_analysis/frontend/`

## API Endpoints

| Method | Endpoint                         | Description                            |
| ------ | -------------------------------- | -------------------------------------- |
| POST   | `/analyze`                       | Full face analysis from uploaded photo |
| POST   | `/analyze/multi`                 | Multi-angle analysis (5 photos)        |
| POST   | `/analyze/depth-enhanced`        | Analysis with depth map (LiDAR)        |
| GET    | `/sessions/{id}`                 | Get session results                    |
| GET    | `/sessions/{id}/report`          | Generate PDF report                    |
| POST   | `/sessions/{id}/annotate`        | Save annotations                       |
| GET    | `/compare/{session1}/{session2}` | Before/After comparison                |
| GET    | `/health`                        | Health check                           |
