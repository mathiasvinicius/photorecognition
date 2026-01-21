# PhotoRecognition & Organizer

## PT-BR

Aplicacao Dockerizada para indexar, buscar (por rosto ou metadados) e organizar colecoes de fotos.

### Principais recursos

- **Busca por rosto:** envie uma foto de referencia para localizar imagens da mesma pessoa.
- **Busca por metadados:** filtre por intervalo de datas e modelo da camera.
- **Indexacao em background:** varre `PHOTOS_DIR`, detecta rostos e extrai EXIF.
- **Organizador de recuperacao:** reorganiza pastas baguncadas em `YYYY/MM` usando a data real da foto.
- **Interface web responsiva:** dashboard simples para gerenciar buscas e organizacao.
- **Deploy via Docker:** sobe API + banco PostgreSQL com um unico compose.

### Estrutura do projeto (ASCII)

```text
PhotoRecognition/
|-- app/                # API FastAPI + frontend estatico
|   |-- main.py
|   `-- static/
|-- data/
|   `-- cache/           # thumbs e cache de processamento
|-- backup/              # backups locais opcionais
|-- docker-compose.yml
|-- Dockerfile
|-- requirements.txt
`-- .env.example
```

### Configuracao rapida

1. Copie `.env.example` para `.env` e ajuste os caminhos.
   ```bash
   cp .env.example .env
   ```
   - `PHOTOS_DIR`: biblioteca principal (leitura por padrao).
   - `RECOVERY_DIR`: pasta baguncada para organizar.
   - `ORGANIZED_DIR`: destino das fotos organizadas.
2. Suba com Docker.
   ```bash
   docker-compose up --build -d
   ```
3. Acesse `http://localhost:8090`.

### Uso basico

1. **Indexacao:** em **Organize & Manage**, clique **Start Scan** para detectar rostos e extrair metadados.
2. **Busca:** envie um rosto em **Face Search** ou use datas/modelo da camera.
3. **Organizacao:** selecione **Copy** ou **Move** e clique **Start Organize** para gerar `ORGANIZED_DIR/YYYY/MM/YYYY-MM-DD_Filename.jpg`.

### Tech stack

- **Backend:** Python (FastAPI), Face Recognition (dlib), SQLAlchemy.
- **Database:** PostgreSQL.
- **Frontend:** HTML5, Bootstrap 5, Vanilla JS.

## English

Dockerized application to index, search (face or metadata), and organize photo collections.

### Key features

- **Face search:** upload a reference face to find matching photos.
- **Metadata search:** filter by date range and camera model.
- **Background indexing:** scans `PHOTOS_DIR`, detects faces, and extracts EXIF.
- **Recovery organizer:** converts messy folders into `YYYY/MM` based on the real capture date.
- **Responsive web UI:** simple dashboard for search and organization.
- **Docker-based deploy:** runs API + PostgreSQL with one compose.

### Project structure (ASCII)

```text
PhotoRecognition/
|-- app/                # FastAPI API + static frontend
|   |-- main.py
|   `-- static/
|-- data/
|   `-- cache/           # thumbs and processing cache
|-- backup/              # optional local backups
|-- docker-compose.yml
|-- Dockerfile
|-- requirements.txt
`-- .env.example
```

### Quick start

1. Copy `.env.example` to `.env` and set your paths.
   ```bash
   cp .env.example .env
   ```
   - `PHOTOS_DIR`: main library (read-only by default).
   - `RECOVERY_DIR`: messy folder to organize.
   - `ORGANIZED_DIR`: destination for organized photos.
2. Run with Docker.
   ```bash
   docker-compose up --build -d
   ```
3. Open `http://localhost:8090`.

### Basic usage

1. **Indexing:** in **Organize & Manage**, click **Start Scan** to detect faces and extract metadata.
2. **Search:** upload a face in **Face Search** or use date/model filters.
3. **Organize:** choose **Copy** or **Move** and click **Start Organize** to create `ORGANIZED_DIR/YYYY/MM/YYYY-MM-DD_Filename.jpg`.

### Tech stack

- **Backend:** Python (FastAPI), Face Recognition (dlib), SQLAlchemy.
- **Database:** PostgreSQL.
- **Frontend:** HTML5, Bootstrap 5, Vanilla JS.
