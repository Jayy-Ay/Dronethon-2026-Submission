# PlantUML Diagrams

This folder contains PlantUML sources generated from the current Drone runtime architecture.

## Files

- `01_system_overview.puml`: high-level components and data flow between Pi and PC runtime.
- `02_threaded_pipeline_sequence.puml`: sequence diagram for threaded ArUco + YOLO processing.
- `03_yolo_detection_flow.puml`: activity flow for model loading, inference, and postprocessing.
- `04_object_geolocation_flow.puml`: activity flow for converting detections into ground-plane estimates.

## Render Locally

If PlantUML is installed:

```bash
cd docs/plantuml
plantuml *.puml
```

If you use a jar directly:

```bash
cd docs/plantuml
java -jar /path/to/plantuml.jar *.puml
```

Generated images (`.png` or `.svg`) can be committed alongside these `.puml` files if needed.
