## Plan: Multi-Session Support via PostgreSQL (Triple-Emotion Model)

Transition the backend from in-memory dictionaries to a scalable, persistent storage system using asynchronous SQLAlchemy and PostgreSQL. Introduces basic User management (to group sessions) and structured Session/Message tracking with placeholder chat titles (ready for future LLM auto-titling), integrated with a Triple-Emotion Multi-Modal Profile.

**Steps**

### Phase 1: Database Setup & Infrastructure
1. Add necessary dependencies to `requirements.txt`: `sqlalchemy[asyncio]`, `asyncpg`, `passlib`, `bcrypt`, optionally `alembic`.
2. Add `DATABASE_URL` setting in `backend/config.py` and `.env.example` (e.g., `postgresql+asyncpg://user:pass@localhost:5432/mentimdb`).
3. Create `backend/db/database.py` to configure the async SQLAlchemy engine, `sessionmaker`, and the `get_db()` FastAPI dependency.
4. Set up an `init_db` function in `database.py` (to run `Base.metadata.create_all()` during FastAPI's lifespan in `main.py`) to easily bootstrap schemas locally without migration overhead initially.

### Phase 2: Object-Relational Models (SQLAlchemy)
1. Create `backend/db/models.py`.
2. **User Model**: `id` (UUID), `name`, `email` (unique), `password_hash`, `created_at`, `updated_at`. One-to-Many with `Session`.
3. **Session Model**: `id` (UUID), `user_id` (ForeignKey), `title` (default "New Chat"), `created_at`, `updated_at`. One-to-Many with `Message`.
4. **Message Model**: `id` (UUID), `session_id` (ForeignKey), `role`, `content`, `timestamp`, `source`. Introduce **Triple-Emotion Model** fields: `text_emotion`, `voice_emotion`, and `face_emotion` (stored as strings or JSON) to allow for cross-modal comparison.

### Phase 3: Pydantic Schemas & Services Re-wire
1. Update `backend/models/schemas.py` to include:
   - `UserCreate`, `UserResponse`.
   - `SessionListResponse` (id, title, updated_at - for sidebar).
   - Update `SessionHistoryResponse` and chat payloads to optionally include `user_id`.
   - Update `MessageResponse` to export the new `text_emotion`, `voice_emotion`, and `face_emotion`.
2. Create `backend/services/user_service.py` to handle simple user creation with password hashing via `passlib`.
3. Refactor `backend/services/session_manager.py`:
   - Replace the dictionary with async DB CRUD operations.
   - `get_or_create_session(db_session, session_id, user_id)`
   - `add_message(db_session, session_id, role, content, source, text_emotion, voice_emotion, face_emotion)`
4. Refactor `backend/services/langchain_service.py`:
   - Update context processing logic to receive all three emotion labels.
   - Modify the LLM prompt to actively identify **'emotional dissonance'** (e.g., when `text_emotion` is 'happy' but `voice_emotion` is 'anxious') and respond empathetically to the conflict.

### Phase 4: Route Updates & New Endpoints
1. Create `backend/routes/user.py`:
   - `POST /users` (Create user)
   - `GET /users/{user_id}/sessions` (Get all chat sessions for a user, returned in descending `updated_at` order).
2. Create `backend/routes/session.py`:
   - Move `GET /chat/history/{session_id}` and `DELETE /chat/history/{session_id}` here as `GET /sessions/{session_id}` and `DELETE /sessions/{session_id}`.
   - Add `PATCH /sessions/{session_id}/title` (modular endpoint intended for future LLM-generated title updates).
3. Refactor `backend/routes/chat.py`:
   - `POST /chat/text`, `POST /chat/voice` (and plan for a future `POST /chat/video`) to inject `db: AsyncSession = Depends(get_db)`.
   - Process all available modalities (text, audio, visual) concurrently and extract corresponding emotional signals.
   - Save the complete Triple-Emotion profile to Postgres for long-term multi-modal sentiment tracking.
4. Hook up new routers in `backend/main.py`.

**Relevant files**
- `backend/requirements.txt` — Add driver/ORM (+ password hashing)
- `backend/db/models.py` — Schema definition for Triple-Emotion profile
- `backend/routes/chat.py` — Concurrent modality execution & database injection
- `backend/services/session_manager.py` — Switch from Dict to async SQLAlchemy selects/inserts with three emotion states
- `backend/services/langchain_service.py` — Emotional dissonance prompting
- `backend/models/schemas.py` — DTOs for the updated models

**Verification**
1. Bring up a local Postgres instance on 5432.
2. `POST /chat/voice` with sample audio. Ensure Whisper & Wav2Vec2 capture BOTH `text_emotion` and `voice_emotion` and write them accurately to the database.
3. Observe Langchain's outputs when supplying conflicting tone datasets (test for 'emotional dissonance').
4. `GET /sessions/{session_id}` returns identical historical message arrays loaded completely with all present multi-modal emotional data.

**Decisions**
- Incorporating missing structural states (`face_emotion`) with default `None`/`neutral` values today to prepare API models for seamless video expansions tomorrow.
- Enforcing LLM Prompt directives specifically to resolve multi-modal discrepancies effectively bridges conversational gaps for the bot.