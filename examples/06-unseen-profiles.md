# Unseen profiles at request time

The candidate profiles are supplied at request time and were never in training — certo binds each description to the evidence and stays calibrated.

*Real outputs of `altslate/certo-decision-model`.*

> We measured cadence as gale, luster as frost.

candidate profiles:
- `Peak` — texture frost, tempo dawn, density dawn, salinity ember, cadence ember, luster ember
- `Ridge` — texture frost, tempo dawn, density ember, salinity gale, cadence gale, luster frost
- `Dune` — texture frost, tempo dawn, density dawn, salinity frost, cadence frost, luster gale
- `Moor` — texture ember, tempo dawn, density ember, salinity dawn, cadence ember, luster frost
- `Glen` — texture frost, tempo ember, density ember, salinity dawn, cadence dawn, luster frost
- `Heath` — texture gale, tempo frost, density dawn, salinity dawn, cadence gale, luster ember

certo → Peak 0.00 · **Ridge 0.90** · Dune 0.00 · Moor 0.04 · Glen 0.03 · Heath 0.03
→ top: **Ridge**
exact  → Peak 0.00 · Ridge 0.88 · Dune 0.00 · Moor 0.04 · Glen 0.04 · Heath 0.04
