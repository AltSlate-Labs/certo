# The boundary: real language

The v1 model is trained on structured signatures, so on free-flowing prose it returns a near-uniform distribution and **abstains** rather than guess. Handling real language is the v2 goal (real data + a paraphrase layer). This is the honest edge of what v1 can do.

*Real outputs of `altslate/certo-decision-model`.*

> Customer: I was charged twice for the same order and I'd like a refund.

candidate profiles:
- `billing` — billing and payments: duplicate charges, refunds, invoices
- `tech` — the app or website is broken
- `account` — login and account access problems

certo → billing 0.33 · tech 0.33 · **account 0.34**
→ top: **account**
→ *abstains (top below 0.60 → hand off / ask for more)*
