# Infra Postmortem: API Latency SEV2

**Date:** 2026-06-18 · **Attendees:** Priya, Wei, Infra lead

## Notes

- Root cause: a deploy introduced an N+1 query on the customer lookup path.
- Detected via the api-latency dashboard 8 minutes after deploy, rolled back within 20 minutes.
- Action items: add a query-count regression check to CI (owner: Wei, due 6/30); update oncall runbook with this pattern (owner: Priya, done, see wiki_oncall_runbook.md).
