# Documentação do Projeto: PhotoRecognition

## Visão Geral
O **PhotoRecognition** é uma aplicação web containerizada (Docker) desenvolvida para gerenciar, indexar e organizar grandes bibliotecas de fotos pessoais. Seu principal diferencial é o reconhecimento facial local (sem envio para nuvem) e ferramentas de recuperação de arquivos baseadas em metadados EXIF.

---

## Arquitetura do Sistema

### Componentes
1.  **Backend (App):**
    *   **Linguagem:** Python 3.11
    *   **Framework:** FastAPI
    *   **Bibliotecas Principais:** `face_recognition` (dlib), `numpy` (cálculo vetorial), `Pillow` (Processamento de imagem/EXIF), `SQLAlchemy` (ORM).
    *   **Função:** Processamento de imagens, detecção de rostos, API REST e lógica de organização de arquivos.
2.  **Banco de Dados (DB):**
    *   **Sistema:** PostgreSQL 15 (Alpine)
    *   **Função:** Armazenamento persistente de caminhos de arquivos, metadados (data, câmera) e vetores faciais (encodings).
    *   **Localização dos Dados:** `/DATA/AppData/Photorecognition/postgres` (Mapeado no Host).
3.  **Frontend:**
    *   **Tecnologia:** HTML5, Bootstrap 5, Vanilla JavaScript.
    *   **Design:** Dashboard Responsivo (SPA - Single Page Application logic).

### Infraestrutura (Docker)
*   **Porta:** `8090` (Mapeada para 8000 interna).
*   **Acesso ao Disco:**
    *   Usa `privileged: true` e `propagation: rslave` para garantir acesso a montagens complexas (como NAS via CasaOS).
    *   Mapeia `/mnt/nas/arquivos` (Host) para `/mnt/photos` (Container) como somente leitura.

---

## Funcionalidades Implementadas

### 1. Indexação e Scanner
*   **Seleção de Pastas:** Permite listar e selecionar subpastas específicas dentro do diretório raiz (`/mnt/nas/arquivos`) para escanear.
*   **Processamento:**
    *   Extrai data de captura e modelo da câmera via EXIF.
    *   Detecta rostos na imagem e gera um "encoding" (assinatura matemática do rosto).
    *   Salva tudo no PostgreSQL.
*   **Status em Tempo Real:** Exibe na interface qual arquivo está sendo processado no momento.

### 2. Busca e Reconhecimento Facial
*   **Busca por Upload:** O usuário envia uma foto, o sistema detecta o rosto e compara com o banco de dados.
*   **Busca por Clique (Navegação):**
    *   Aba "Rostos Detectados" lista fotos onde rostos foram encontrados.
    *   Ao clicar numa foto, abre-se um **Modal**.
    *   O Modal desenha **quadrados verdes** sobre os rostos.
    *   Clicar no quadrado busca automaticamente todas as fotos daquela pessoa na biblioteca.
*   **Busca por Metadados:** Filtros por intervalo de datas e modelo da câmera.

### 3. Organizador de Recuperação
*   **Cenário:** Organizar pastas bagunçadas (ex: recuperação de HD formatado).
*   **Funcionamento:** Lê os arquivos da pasta de origem (`RECOVERY_DIR`), extrai a data real da foto (EXIF) e move/copia para uma estrutura limpa: `ANO/MÊS/ANO-MES-DIA_Arquivo.jpg`.

---

## Configuração e Instalação

### Estrutura de Arquivos
```text
PhotoRecognition/
├── app/
│   ├── main.py          # Ponto de entrada da API e rotas
│   ├── models.py        # Modelos do Banco de Dados (SQLAlchemy)
│   ├── database.py      # Conexão com o DB
│   ├── scanner.py       # Lógica de varredura e detecção facial
│   ├── organizer.py     # Lógica de organização de arquivos
│   ├── status.py        # Gerenciamento de estado em memória
│   ├── utils.py         # Utilitários (EXIF)
│   └── static/          # Frontend (index.html)
├── .env                 # Variáveis de ambiente
├── docker-compose.yml   # Orquestração dos containers
├── Dockerfile           # Definição da imagem Python
├── requirements.txt     # Dependências Python
└── tasklist.md          # Histórico de planejamento
```

### Variáveis de Ambiente (.env)
```ini
APP_PORT=8090
DB_USER=user
DB_PASSWORD=password
DB_NAME=photorec_db
PHOTOS_DIR=/mnt/nas/arquivos           # Origem das fotos
DATA_DIR=/DATA/AppData/Photorecognition # Onde o DB é salvo
RECOVERY_DIR=./data/recovery           # Origem para organizar
ORGANIZED_DIR=./data/organized         # Destino da organização
```

### Comandos de Gestão
*   **Iniciar:** `docker compose up -d`
*   **Ver Logs:** `docker logs -f photorec_app`
*   **Resetar Dados:** Parar o docker e limpar a pasta `/DATA/AppData/Photorecognition/postgres`.

---

## Histórico de Correções Importantes
1.  **Persistência:** Mudança do volume do banco de dados para `/DATA/AppData` para garantir backup facilitado no CasaOS.
2.  **Permissões de Disco:** Adição de `privileged: true` e `propagation: rslave` no docker-compose para resolver erro onde o container não enxergava arquivos do NAS.
3.  **Interface de Seleção:** Implementação de listagem de pastas no backend para permitir scans parciais.
4.  **Correção de Busca:** Ajuste na tolerância de "re-detecção" (de 0.1 para 0.4) para garantir que o clique no rosto funcione corretamente.
