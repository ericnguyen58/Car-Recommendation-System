def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_filters_endpoint(client):
    resp = client.get("/cars/filters")
    assert resp.status_code == 200
    body = resp.json()
    assert "Sedan" in body["bodytypes"]
    assert "Pickup" in body["bodytypes"]
    assert set(body["makes"]) == {"Toyota", "Honda", "Tesla", "Ford"}


def test_car_detail_found(client):
    resp = client.get("/cars/1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["make"] == "Toyota"
    assert body["model"] == "Camry"
    assert body["features"]["safety"]["Traction Control"] == 2


def test_car_detail_not_found(client):
    resp = client.get("/cars/999")
    assert resp.status_code == 404


def test_recommendations_basic(client):
    resp = client.post("/recommendations", json={"bodytype": ["Sedan"], "top_n": 10})
    assert resp.status_code == 200
    body = resp.json()
    makes = {c["make"] for c in body["cars"]}
    assert makes == {"Toyota", "Honda", "Tesla"}
    # ranked by predicted_consumer_rating desc — FakeModel scores by hp, Tesla has the most
    assert body["cars"][0]["make"] == "Tesla"


def test_recommendations_ev_filter(client):
    resp = client.post("/recommendations", json={"is_ev": True})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["cars"]) == 1
    assert body["cars"][0]["make"] == "Tesla"


def test_recommendations_no_match(client):
    resp = client.post("/recommendations", json={"make": ["Nonexistent"]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_matched"] == 0
    assert body["cars"] == []
