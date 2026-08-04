# Engineering On-Call Runbook

**Owner:** Infra · **Last updated:** 2026-03-01

## Escalation path

Page the on-call engineer via PagerDuty first. If unacknowledged after 10
minutes, it auto-escalates to the secondary on-call, then to the Infra lead.

## Common incidents

- **API latency spike:** check the api-latency dashboard first; usually
  correlates with a deploy or a downstream provider issue.
- **Elevated error rate:** check recent deploys before anything else; most
  incidents in the last quarter were deploy-caused and resolved by rollback.
- **Database connection exhaustion:** check for a runaway job holding
  connections open; see the DB runbook for the query to identify it.

## Postmortems

Every SEV1/SEV2 gets a postmortem doc within 3 business days, reviewed in
the weekly infra sync.
