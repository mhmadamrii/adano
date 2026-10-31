import os
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Uuid, create_engine, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql+psycopg:///fraudless")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class ModelVersion(Base):
    """Registry of artifacts the service has loaded. Metadata only; the artifact lives on the model volume."""

    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(primary_key=True)
    model_version: Mapped[str] = mapped_column(String(100), unique=True)
    schema_version: Mapped[str] = mapped_column(String(100))
    artifact_sha256: Mapped[str] = mapped_column(String(64))
    trained_at: Mapped[str] = mapped_column(String(40))
    model_metadata: Mapped[dict] = mapped_column(JSONB)
    registered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AuditRun(Base):
    """One row per upload. Aggregates only: no transaction rows, scores, values or filenames are stored."""

    __tablename__ = "audit_runs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    status: Mapped[str] = mapped_column(String(20))  # completed | rejected
    error_code: Mapped[str | None] = mapped_column(String(60))
    model_version: Mapped[str] = mapped_column(String(100))
    schema_version: Mapped[str] = mapped_column(String(100))
    file_sha256: Mapped[str] = mapped_column(String(64))
    file_size_bytes: Mapped[int] = mapped_column(Integer)
    total_rows: Mapped[int | None] = mapped_column(Integer)
    scored_rows: Mapped[int | None] = mapped_column(Integer)
    rejected_rows: Mapped[int | None] = mapped_column(Integer)
    flagged_rows: Mapped[int | None] = mapped_column(Integer)
    warning_codes: Mapped[list] = mapped_column(JSONB, default=list)
    duration_ms: Mapped[int] = mapped_column(Integer)
