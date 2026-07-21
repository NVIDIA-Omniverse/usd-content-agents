# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage for USD search client and tool wrappers."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from world_understanding.functions.knowledge import usd_search
from world_understanding.tools.knowledge import usd_search as usd_search_tool_mod


class FakeApiClient:
    def __init__(self, configuration: Any) -> None:
        self.configuration = configuration

    async def __aenter__(self) -> FakeApiClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None


class FakeRequest:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class FakeToDict:
    def to_dict(self) -> dict[str, Any]:
        return {"kind": "to_dict"}


class FakeObject:
    def __init__(self) -> None:
        self.kind = "object"


class FakeApiException(Exception):
    def __init__(self) -> None:
        super().__init__("api")
        self.status = 500
        self.reason = "bad"
        self.body = "body"


def test_usd_search_async_request_and_result_formatting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        usd_search.usd_search_client,
        "Configuration",
        lambda host: {"host": host},
    )
    monkeypatch.setattr(usd_search.usd_search_client, "ApiClient", FakeApiClient)
    monkeypatch.setattr(usd_search, "BasicSearchRequest", FakeRequest)
    monkeypatch.setattr(
        usd_search,
        "Query",
        lambda actual_instance: {"query": actual_instance},
    )
    monkeypatch.setattr(
        usd_search,
        "VectorQuery",
        lambda **kwargs: {"vector": kwargs},
    )
    monkeypatch.setattr(
        usd_search,
        "VectorQueryType",
        type("VectorQueryType", (), {"TEXT": "TEXT"}),
    )

    async def fake_search_hybrid(request: FakeRequest, *, api_client: FakeApiClient):
        captured["request"] = request.kwargs
        captured["host"] = api_client.configuration["host"]
        return [FakeToDict(), FakeObject(), {"kind": "dict"}, "raw"]

    monkeypatch.setattr(
        usd_search.usd_search_client, "search_hybrid", fake_search_hybrid
    )
    client = usd_search.USDSearchClient(host="http://search")
    results = asyncio.run(
        client.search_async(
            "metal",
            limit=4,
            return_metadata=False,
            return_images=False,
            file_extension_include=["mdl", "usd"],
        )
    )

    assert captured["host"] == "http://search"
    assert captured["request"]["file_extension_include"] == "mdl,usd"
    assert captured["request"]["return_metadata"] is False
    assert results == [
        {"kind": "to_dict"},
        {"kind": "object"},
        {"kind": "dict"},
        {"data": "raw"},
    ]

    asyncio.run(client._search_async("wood", file_extension_include="mdl"))
    assert captured["request"]["file_extension_include"] == "mdl"


def test_usd_search_error_paths_and_extract_results(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(usd_search.usd_search_client, "ApiClient", FakeApiClient)
    monkeypatch.setattr(usd_search, "BasicSearchRequest", FakeRequest)
    monkeypatch.setattr(usd_search, "Query", lambda actual_instance: actual_instance)
    monkeypatch.setattr(usd_search, "VectorQuery", lambda **kwargs: kwargs)
    monkeypatch.setattr(
        usd_search,
        "VectorQueryType",
        type("VectorQueryType", (), {"TEXT": "TEXT"}),
    )
    monkeypatch.setattr(usd_search, "ApiException", FakeApiException)

    async def api_error(*args: Any, **kwargs: Any) -> Any:
        raise FakeApiException()

    monkeypatch.setattr(usd_search.usd_search_client, "search_hybrid", api_error)
    client = usd_search.USDSearchClient()
    assert asyncio.run(client.search_async("bad")) == []
    assert "API Exception occurred" in capsys.readouterr().out

    async def runtime_error(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr(usd_search.usd_search_client, "search_hybrid", runtime_error)
    assert asyncio.run(client.search_async("bad")) == []
    assert "Unexpected error occurred" in capsys.readouterr().out

    assert client._extract_results(["a"]) == ["a"]
    assert client._extract_results(type("Response", (), {"items": [1, 2]})()) == [1, 2]
    assert client._extract_results(FakeToDict()) == []
    assert client._extract_results(FakeObject()) == []
    dict_response = {"payload": [], "matches": [{"id": 1}]}
    assert client._extract_results(dict_response) == [{"id": 1}]
    assert client._extract_results({"first_list": [1]}) == [1]
    assert client._extract_results(object()) == []


def test_usd_search_sync_and_convenience_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = usd_search.USDSearchClient()

    async def fake_search_async(*args: Any, **kwargs: Any) -> list[Any] | None:
        return [FakeToDict(), FakeObject(), {"kind": "dict"}, "raw"]

    monkeypatch.setattr(client, "_search_async", fake_search_async)
    assert client.search("metal") == [
        {"kind": "to_dict"},
        {"kind": "object"},
        {"kind": "dict"},
        {"data": "raw"},
    ]

    async def fake_empty_search_async(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(client, "_search_async", fake_empty_search_async)
    assert client.search("metal") == []

    async def call_sync_inside_loop() -> None:
        with pytest.raises(RuntimeError, match="Cannot use synchronous search"):
            client.search("metal")

    asyncio.run(call_sync_inside_loop())

    class FakeClient:
        def __init__(self, host: str | None = None) -> None:
            self.host = host

        def search(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
            return [{"host": self.host, "args": args, "kwargs": kwargs}]

        async def _search_async(self, *args: Any, **kwargs: Any) -> list[Any]:
            return [FakeToDict(), FakeObject(), {"kind": "dict"}, "raw"]

        def _extract_results(self, response: Any) -> Any:
            return response

    monkeypatch.setattr(usd_search, "USDSearchClient", FakeClient)
    assert usd_search.search_usd_materials("q", host="host")[0]["host"] == "host"
    assert asyncio.run(usd_search.search_usd_materials_async("q")) == [
        {"kind": "to_dict"},
        {"kind": "object"},
        {"kind": "dict"},
        {"data": "raw"},
    ]

    class EmptyClient(FakeClient):
        async def _search_async(self, *args: Any, **kwargs: Any) -> None:
            return None

    monkeypatch.setattr(usd_search, "USDSearchClient", EmptyClient)
    assert asyncio.run(usd_search.search_usd_materials_async("q")) == []


def test_usd_search_tool_success_and_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeToolClient:
        def __init__(self, host: str | None = None) -> None:
            self.host = host

        def search(self, **kwargs: Any) -> list[dict[str, Any]]:
            return [{"query": kwargs["query"], "host": self.host}]

    monkeypatch.setattr(usd_search_tool_mod, "USDSearchClient", FakeToolClient)
    output = usd_search_tool_mod.usd_search_tool(
        usd_search_tool_mod.USDSearchInput(
            query="metal",
            limit=2,
            api_host="host",
            file_extension_include=["mdl", "usd"],
        )
    )
    assert output.success is True
    assert output.num_results == 1
    assert output.file_extensions == ["mdl", "usd"]

    output = usd_search_tool_mod.usd_search_tool(
        usd_search_tool_mod.USDSearchInput(
            query="metal",
            file_extension_include="mdl",
        )
    )
    assert output.file_extensions == ["mdl"]

    def make_raising_client(exc_type: type[Exception]):
        class RaisingClient:
            def __init__(self, host: str | None = None) -> None:
                pass

            def search(self, **kwargs: Any) -> list[dict[str, Any]]:
                raise exc_type("boom")

        return RaisingClient

    for exc_type, prefix in [
        (ImportError, "Import error:"),
        (ConnectionError, "Connection error:"),
        (ValueError, "Invalid input:"),
        (RuntimeError, "Unexpected error:"),
    ]:
        monkeypatch.setattr(
            usd_search_tool_mod, "USDSearchClient", make_raising_client(exc_type)
        )
        output = usd_search_tool_mod.usd_search_tool(
            usd_search_tool_mod.USDSearchInput(query="metal")
        )
        assert output.success is False
        assert output.errors[0].startswith(prefix)
