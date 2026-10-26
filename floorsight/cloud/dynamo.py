"""DynamoDB sink for WorldState snapshots (track history), auto-deleted by TTL.

Table layout (matches infra/terraform/main.tf):
    partition key  session_id  (S)
    sort key       t_ms        (N)  session time in milliseconds
    world          (S)  WorldState JSON
    n_tracks       (N)
    expires_at     (N)  epoch seconds; DynamoDB TTL deletes the item after this
"""

from __future__ import annotations

import json
import time

from floorsight.cloud._aws import make_client
from floorsight.contracts import WorldState


class DynamoTrackSink:
    def __init__(
        self,
        table: str,
        session_id: str = "default",
        ttl_s: float = 24 * 3600,
        min_interval_s: float = 1.0,
        region: str | None = None,
        client=None,
        clock=time.time,
    ):
        self.table = table
        self.session_id = session_id
        self.ttl_s = ttl_s
        self.min_interval_s = min_interval_s
        self._ddb = make_client("dynamodb", region, client)
        self._clock = clock
        self._last_t: float | None = None

    def put_world(self, world: WorldState) -> bool:
        """Store a snapshot at most every `min_interval_s` of session time. Returns True if written."""
        if self._last_t is not None and world.t - self._last_t < self.min_interval_s:
            return False
        body = json.dumps(world.model_dump(mode="json"), separators=(",", ":"))
        if len(body.encode()) > 380_000:  # DynamoDB item limit is 400 KB
            raise ValueError("WorldState snapshot too large for one DynamoDB item")
        self._ddb.put_item(
            TableName=self.table,
            Item={
                "session_id": {"S": self.session_id},
                "t_ms": {"N": str(int(round(world.t * 1000)))},
                "world": {"S": body},
                "n_tracks": {"N": str(len(world.tracks))},
                "expires_at": {"N": str(int(self._clock() + self.ttl_s))},
            },
        )
        self._last_t = world.t
        return True

    def query(self, t0: float, t1: float, limit: int = 500) -> list[WorldState]:
        """Snapshots with t0 <= t <= t1 for this session, oldest first."""
        out: list[WorldState] = []
        kwargs = {
            "TableName": self.table,
            "KeyConditionExpression": "session_id = :s AND t_ms BETWEEN :a AND :b",
            "ExpressionAttributeValues": {
                ":s": {"S": self.session_id},
                ":a": {"N": str(int(t0 * 1000))},
                ":b": {"N": str(int(t1 * 1000))},
            },
            "Limit": limit,
        }
        while True:
            resp = self._ddb.query(**kwargs)
            for item in resp.get("Items", []):
                out.append(WorldState.model_validate_json(item["world"]["S"]))
            if "LastEvaluatedKey" not in resp or len(out) >= limit:
                return out[:limit]
            kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
