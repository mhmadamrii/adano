import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session", autouse=True)
def _test_db():
    # must be set before app.db is imported
    os.environ["DATABASE_URL"] = "postgresql+psycopg:///adano_test"


@pytest.fixture()
def client():
    from sqlalchemy import text

    from app.db import engine
    from app.main import app

    with TestClient(app) as c:
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE notes, categories RESTART IDENTITY CASCADE"))
        yield c


CATS = [
    ("Todo", "Things I need to do, buy, pay, or remember; errands and reminders"),
    ("Important News", "News, market updates, or announcements I want to keep track of"),
    ("Ideas", "Project ideas, things to try, creative thoughts"),
]


def seed(client):
    ids = {}
    for name, desc in CATS:
        r = client.post("/categories", json={"name": name, "description": desc})
        assert r.status_code == 201
        ids[name] = r.json()["id"]
    return ids


def test_cold_start_classification_and_persistence(client):
    ids = seed(client)
    n = client.post("/notes", json={"text": "I need to take money from ATM"}).json()
    assert n["category_id"] == ids["Todo"]
    assert n["is_confirmed"] is False
    # persisted in postgres
    from sqlalchemy import text

    from app.db import engine

    with engine.connect() as conn:
        assert conn.execute(text("select count(*) from notes where text like '%ATM%'")).scalar() == 1
        assert conn.execute(text("select count(*) from categories")).scalar() == 3


def test_no_categories_goes_uncategorized(client):
    n = client.post("/notes", json={"text": "hello"}).json()
    assert n["category_id"] is None


def test_feedback_confirms_and_filters(client):
    ids = seed(client)
    n = client.post("/notes", json={"text": "Bitcoin will go to 100K"}).json()
    r = client.patch(f"/notes/{n['id']}/category", json={"category_id": ids["Important News"]}).json()
    assert r["is_confirmed"] and r["category_id"] == ids["Important News"]
    listed = client.get("/notes", params={"category_id": ids["Important News"]}).json()
    assert [x["id"] for x in listed] == [n["id"]]
    cats = {c["name"]: c for c in client.get("/categories").json()}
    assert cats["Important News"]["count"] == 1


def test_confirmed_examples_steer_future_predictions(client):
    ids = seed(client)
    # teach: crypto talk belongs to Important News
    for t in ["Bitcoin will go to 100K", "Ethereum ETF approved", "Fed raises interest rates"]:
        n = client.post("/notes", json={"text": t}).json()
        client.patch(f"/notes/{n['id']}/category", json={"category_id": ids["Important News"]})
    n = client.post("/notes", json={"text": "Bitcoin hits new all time high"}).json()
    assert n["category_id"] == ids["Important News"]


def test_delete_category_moves_notes_to_uncategorized(client):
    ids = seed(client)
    n = client.post("/notes", json={"text": "buy milk and pay electricity bill"}).json()
    client.patch(f"/notes/{n['id']}/category", json={"category_id": ids["Todo"]})
    assert client.delete(f"/categories/{ids['Todo']}").status_code == 204
    after = client.get("/notes", params={"uncategorized": True}).json()
    assert n["id"] in [x["id"] for x in after]


def test_duplicate_category_409_and_edit(client):
    ids = seed(client)
    assert client.post("/categories", json={"name": "todo"}).status_code == 409
    r = client.patch(f"/categories/{ids['Ideas']}", json={"description": "startup ideas"})
    assert r.json()["description"] == "startup ideas"


def test_delete_note(client):
    n = client.post("/notes", json={"text": "temporary thought"}).json()
    assert client.delete(f"/notes/{n['id']}").status_code == 204
    assert client.get("/notes").json() == []
    assert client.delete(f"/notes/{n['id']}").status_code == 404
