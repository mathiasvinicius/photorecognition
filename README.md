# PhotoRecognition & Organizer

A Dockerized application to manage, search (by Face or Metadata), and organize your photo collection.

## Features

-   **Face Search:** Upload a photo of a person to find all photos of them in your library.
-   **Metadata Search:** Filter photos by Date Range and Camera Model.
-   **Photo Recovery Organizer:** Automatically organize a messy folder (e.g., recovered from a hard drive) into a clean `YYYY/MM` structure based on the *actual* EXIF date taken.
-   **Web Interface:** Simple, responsive dashboard.

## Setup

1.  **Configure Environment:**
    Copy `.env.example` to `.env` and set your paths.
    ```bash
    cp .env.example .env
    ```
    *   `PHOTOS_DIR`: Your main photo library (Read-Only access by default).
    *   `RECOVERY_DIR`: The "messy" folder you want to organize.
    *   `ORGANIZED_DIR`: Where organized photos will be saved.

2.  **Run with Docker:**
    ```bash
    docker-compose up --build -d
    ```

3.  **Access:**
    Open your browser to `http://localhost:8090`.

## Usage

### 1. Indexing (Scanning)
Go to the **Organize & Manage** tab and click **Start Scan**. This will walk through your `PHOTOS_DIR`, detect faces, and extract metadata. This runs in the background.

### 2. Searching
-   **Face:** Upload a clear photo of a face in the "Face Search" box.
-   **Metadata:** Use the date pickers or camera model text input.

### 3. Organizing Recovered Photos
Go to the **Organize & Manage** tab.
-   Select **Copy** (safer) or **Move**.
-   Click **Start Organize**.
-   Photos from `RECOVERY_DIR` will be sorted into `ORGANIZED_DIR/YYYY/MM/YYYY-MM-DD_Filename.jpg`.

## Tech Stack
-   **Backend:** Python (FastAPI), Face Recognition (dlib), SQLAlchemy.
-   **Database:** PostgreSQL.
-   **Frontend:** HTML5, Bootstrap 5, Vanilla JS.
