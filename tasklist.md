# Project Plan: PhotoRecognition

## Goal
Create a Dockerized web application to manage, search, and organize photos.
Key features:
1.  **Face Recognition:** Upload a reference photo to find similar faces in a local repository.
2.  **Metadata Search:** Filter by date (range) and device (camera model).
3.  **Photo Organization:** Tool to organize recovered photos based on actual EXIF creation dates (Copy/Move modes).
4.  **Person Management:** Automatic face clustering with manual correction support (Google Photos style).

## Architecture
*   **Backend:** Python with FastAPI.
    *   Libraries: `InsightFace` (ArcFace), `OpenCV`, `Pillow` (EXIF/Image processing), `SQLAlchemy` (ORM), `psycopg2`.
*   **Database:** PostgreSQL (Run as a separate service in Docker).
*   **Frontend:** HTML5, JavaScript (Vanilla), Bootstrap 5 for UI.
*   **Infrastructure:** Docker + Docker Compose.

## Task List

### Phase 1: Setup & Infrastructure
- [x] Initialize project structure (backend, frontend, config).
- [x] Create `requirements.txt` with dependencies.
- [x] Create `Dockerfile` for the Python application.
- [x] Create `docker-compose.yml` with:
    - [x] App Service definition.
    - [x] PostgreSQL Service definition.
    - [x] Volume mapping from `.env` (Source photos directory).
    - [x] Port mapping.
- [x] Create `.env` example file (Including DB credentials).

### Phase 2: Backend Core (Indexing & Search)
- [x] Implement `Database` connection (SQLAlchemy) and Models (Photo, FaceEncoding, Person, FaceExclusion).
- [x] Implement `Scanner` service:
    - [x] Recursively walk directory.
    - [x] Extract EXIF data (Date taken, Camera model).
    - [x] Detect faces and compute embeddings using `InsightFace`.
    - [x] Store/Update data in PostgreSQL.
    - [x] Automatic face clustering during scan.
- [x] Implement API Endpoints:
    - [x] `POST /scan`: Trigger a scan of selected folders.
    - [x] `POST /search/face`: Upload image -> Compare encodings -> Return matching files.
    - [x] `GET /search/metadata`: Filter by date range or device.
    - [x] `GET /scan/status`: Get current scan status.
    - [x] `POST /scan/pause`, `POST /scan/resume`, `POST /scan/stop`: Scan controls.

### Phase 3: Photo Organizer (Recovery Tool)
- [x] Implement `Organizer` logic:
    - [x] Read target directory.
    - [x] Extract "Date Taken" from EXIF.
    - [x] Handle duplicates/naming collisions.
    - [x] Perform Copy or Move to destination structure (e.g., `YYYY/MM/YYYY-MM-DD_Filename.jpg`).
- [x] Add API Endpoint: `POST /organize` with parameters (source_path, dest_path, mode=copy|move).

### Phase 4: Person Management System
- [x] Implement `Person` model with automatic naming ("Pessoa X").
- [x] Implement `FaceExclusion` model for negative feedback.
- [x] Implement clustering logic (`clustering.py`):
    - [x] Cosine similarity comparison for InsightFace embeddings.
    - [x] `find_or_create_person`: Auto-cluster faces to persons.
    - [x] `assign_face_to_person`: Assign face during scan.
    - [x] `recluster_all_faces`: Re-run clustering on all faces.
    - [x] `merge_persons`: Merge two persons into one (with name preservation).
    - [x] `exclude_face_from_person`: Negative feedback handling.
- [x] Implement Person API Endpoints:
    - [x] `GET /persons`: List all persons with representative faces.
    - [x] `GET /persons/{id}`: Get person details with all faces.
    - [x] `PUT /persons/{id}`: Update person name.
    - [x] `POST /persons/{id}/exclude`: Mark face as "not this person".
    - [x] `POST /persons/merge`: Merge two persons.
    - [x] `GET /persons/suggestions`: Get merge suggestions based on similarity.
    - [x] `POST /cluster`: Trigger full re-clustering.

### Phase 5: Frontend Development
- [x] Create main dashboard layout (Bootstrap 5, dark/light theme).
- [x] Implement "Face Search" tab:
    - [x] File upload input.
    - [x] Results grid view with face boxes.
    - [x] Click-to-search on detected faces.
- [x] Implement "Metadata Search" tab:
    - [x] Date pickers.
    - [x] Device dropdown (auto-populated).
- [x] Implement "Organizer" tab:
    - [x] Path inputs with folder browser modal.
    - [x] Progress bar/Log view.
    - [x] Copy/Move mode selection.
- [x] Implement "Persons" tab (Google Photos style):
    - [x] Grid view of all persons with representative avatars.
    - [x] Person detail modal with all faces.
    - [x] Inline name editing.
    - [x] Face exclusion ("Not this person" button).
    - [x] Merge mode with multi-select.
    - [x] Merge suggestions with quick-merge buttons.
    - [x] Display filters (photo count, alphabetical).
- [x] Implement "Detected Faces" tab:
    - [x] Grid of all indexed photos with faces.
    - [x] Pagination.
    - [x] Click to open photo with face boxes.

### Phase 6: Refinement & Polish
- [x] Add background task support for long-running scans/organizing.
- [x] Optimize face recognition (InsightFace with ArcFace model).
- [x] Auto-detection of system theme preference (dark/light).
- [x] Photo viewer modal with keyboard navigation.
- [x] Toggle for face markers visibility.
- [x] Right-click context menu on scan status (pause/resume/stop).
- [x] Auto-verification of merge suggestions after scan completes.
- [x] Preserve named person's name when merging with "Pessoa X".
- [x] Fix merge bug (faces not being moved properly).

### Phase 7: Person Groups
- [x] Create PersonGroup model with name and color.
- [x] Add group_id to Person model.
- [x] API endpoints for groups (CRUD).
- [x] API endpoint to assign person to group.
- [x] Frontend: Groups section with badges.
- [x] Frontend: Filter persons by group.
- [x] Frontend: Create/edit/delete groups modal.
- [x] Frontend: Select group in person detail modal.
- [x] Re-cluster button to regroup orphan faces.
- [x] Drag and drop merge (arraste uma pessoa sobre outra para mesclar).

## Recent Fixes
- Fixed: Merge not moving faces to target person (SQLAlchemy synchronize_session bug).
- Fixed: Name preservation when merging named person with "Pessoa X".
- Fixed: Suggestions endpoint error handling.
- Fixed: Auto-refresh suggestions after scan completion.
- Fixed: Re-cluster now only affects orphan faces, preserving named persons.
