# TCK-0037: Dashboard loading slowly for large accounts

**Status:** Resolved · **Priority:** Medium

## Details

Customer with a high seat count reported the usage dashboard taking
10+ seconds to load. Confirmed this correlates with the N+1 query issue
fixed after the 6/18 incident. Customer confirmed load time back to normal
after the fix shipped.
