"""SNS implementation of Notifier."""

from __future__ import annotations

from floorsight.cloud._aws import make_client
from floorsight.contracts import Alert


class SnsNotifier:
    """Publishes alerts to one SNS topic (email subscribers get the text).

    By default only alerts whose channel is "sns" are published, so the same
    alert stream can be fanned out to the dashboard and SNS without double-sending.
    Pass only_channel=None to publish everything.
    """

    def __init__(self, topic_arn: str, region: str | None = None, client=None, only_channel: str | None = "sns"):
        self.topic_arn = topic_arn
        self.only_channel = only_channel
        self._sns = make_client("sns", region, client)
        self.sent: list[str] = []  # message ids

    def send(self, alert: Alert) -> None:
        if self.only_channel and alert.channel != self.only_channel:
            return
        subject = f"[Floorsight] {alert.audience}: {alert.message}"
        subject = "".join(ch if ch.isprintable() else " " for ch in subject)[:100]  # SNS subject limit
        resp = self._sns.publish(
            TopicArn=self.topic_arn,
            Subject=subject,
            Message=f"{alert.message}\n\nevent: {alert.event_id}\nalert: {alert.alert_id}\nsession t: {alert.t:.1f} s",
            MessageAttributes={
                "audience": {"DataType": "String", "StringValue": alert.audience},
                "event_id": {"DataType": "String", "StringValue": alert.event_id},
            },
        )
        self.sent.append(resp["MessageId"])
