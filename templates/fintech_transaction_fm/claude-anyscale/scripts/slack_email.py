#!/usr/bin/env python3
"""Post stdin to a Slack channel via its "send emails to this channel" address.

Delivery fallback for boxes where an incoming webhook is admin-blocked. Needs an
authenticated SMTP relay (the box can't send mail itself). Env:
  SLACK_CHANNEL_EMAIL  - the idle-sweep-…@<org>.slack.com address
  SMTP_USER            - your @anyscale.com address (the authenticated sender)
  SMTP_APP_PASSWORD    - a Gmail *app password* (Google Account → Security → App passwords)
  SMTP_HOST/SMTP_PORT  - optional; default smtp.gmail.com:587

Usage:  echo "message body" | python3 slack_email.py
"""
import smtplib, os, sys
from email.message import EmailMessage

msg = EmailMessage()
msg["Subject"] = "idle-sweep"                       # -> Slack message title
msg["From"]    = os.environ["SMTP_USER"]
msg["To"]      = os.environ["SLACK_CHANNEL_EMAIL"]  # -> posts into the channel
msg.set_content(sys.stdin.read() or "(empty)")     # -> Slack message body

host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
port = int(os.environ.get("SMTP_PORT", "587"))
with smtplib.SMTP(host, port, timeout=30) as s:
    s.starttls()
    s.login(os.environ["SMTP_USER"], os.environ["SMTP_APP_PASSWORD"])
    s.send_message(msg)
