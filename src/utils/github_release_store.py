from __future__ import annotations

import json
from typing import Any, Dict, Optional
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def _api_request(
    url: str,
    token: str,
    method: str = "GET",
    data: Optional[bytes] = None,
    accept: str = "application/vnd.github+json",
    content_type: str = "application/json",
) -> Any:
    request = Request(
        url,
        data=data,
        method=method,
        headers={
            "Accept": accept,
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "aml25-bundle-release-store",
            "Content-Type": content_type,
        },
    )
    with urlopen(request) as response:
        payload = response.read()
        if not payload:
            return {}
        if accept == "application/octet-stream":
            return payload
        return json.loads(payload.decode("utf-8"))


def get_release_by_tag(repo: str, token: str, tag: str) -> Optional[Dict[str, Any]]:
    url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    try:
        return _api_request(url, token=token)
    except HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def create_release(repo: str, token: str, tag: str, name: str, body: str = "") -> Dict[str, Any]:
    payload = json.dumps(
        {
            "tag_name": tag,
            "name": name,
            "body": body,
            "draft": False,
            "prerelease": False,
        }
    ).encode("utf-8")
    return _api_request(
        f"https://api.github.com/repos/{repo}/releases",
        token=token,
        method="POST",
        data=payload,
    )


def ensure_release(repo: str, token: str, tag: str, name: str, body: str = "") -> Dict[str, Any]:
    existing = get_release_by_tag(repo=repo, token=token, tag=tag)
    if existing is not None:
        return existing
    return create_release(repo=repo, token=token, tag=tag, name=name, body=body)


def delete_release_asset(repo: str, token: str, asset_id: int) -> None:
    _api_request(
        f"https://api.github.com/repos/{repo}/releases/assets/{asset_id}",
        token=token,
        method="DELETE",
    )


def upload_release_asset(
    release: Dict[str, Any],
    repo: str,
    token: str,
    asset_name: str,
    asset_bytes: bytes,
    content_type: str = "application/zip",
) -> Dict[str, Any]:
    for asset in release.get("assets", []):
        if asset.get("name") == asset_name and asset.get("id"):
            delete_release_asset(repo=repo, token=token, asset_id=int(asset["id"]))

    upload_url = str(release["upload_url"]).split("{", 1)[0]
    url = f"{upload_url}?{urlencode({'name': asset_name})}"
    return _api_request(
        url,
        token=token,
        method="POST",
        data=asset_bytes,
        accept="application/vnd.github+json",
        content_type=content_type,
    )


def download_release_asset(asset_url: str, token: str) -> bytes:
    return _api_request(
        asset_url,
        token=token,
        accept="application/octet-stream",
    )
