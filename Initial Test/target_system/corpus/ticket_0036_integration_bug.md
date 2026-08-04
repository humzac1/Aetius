# TCK-0036: Webhook deliveries failing intermittently

**Status:** Resolved · **Priority:** High

## Details

Customer's webhook endpoint was intermittently returning 200 but
timing out on our end, causing us to retry and double-deliver. Root cause
was our webhook timeout being shorter than their endpoint's typical
response time. Increased the timeout for their account and confirmed
deliveries are clean. Escalated to engineering to review default timeout
(see meeting_notes_infra_postmortem_2026-06-18.md for related latency work).
