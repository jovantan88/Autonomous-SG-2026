# 3-minute pitch: TIP Watcher

TIP = Tier and Impact Probability. Every number below is in `outputs/event_metrics.json` and reproduces with
`python src/event_model.py`.

## 0:00 – 0:30 · The user and the problem

* Our user is the conjunction-assessment analyst on shift. Every close approach brings about a week of warning
  messages, and the manoeuvre decision has to be made about 2 days before closest approach.
* Today the analyst reads the messages one by one and goes by the latest. The question that matters is
  different: how will this event look at closest approach?

## 0:30 – 1:30 · Demo

1. **Events view:** the warnings panel shows how many events reach closest approach within 3 days, how many
   are already within 1 km, and how many we forecast to end Critical.
2. Open the top event: risk, miss distance and relative position (r, t, n) plotted as closest approach gets
   nearer, with the forecast probability of a final Critical tier.
3. Read the reasons out loud, then show the avoidance options: burn size and the resulting collision
   probability, from a physics model that reproduces ESA's own Pc.
4. Press **Escalate**: the analyst decides, and the log records it.
5. **Live sky:** today's real conjunctions from CelesTrak, with both satellites' flight paths and the
   collision point on the globe.

## 1:30 – 2:20 · Evidence

* Fixed 80/20 split of train events; thresholds chosen on cross-validation only; hold-out scored once.
* **Critical events caught: 94 % vs 89 %** for "go with the latest message", so 10 missed instead of 18.
  On events whose outcome is genuinely in the future: 91 % vs 85 %.
* **Where it falls short:** 60 false alarms instead of 25, and at equal false alarms the model does not beat
  the latest message. That matches the ESA competition paper. So we use the model to escalate on top of the
  latest message, not to replace it.

## 2:20 – 2:50 · Trust and impact

* Every flag comes with plain-language reasons; thresholds are published; the analyst makes the call.
* Physics check: our full-covariance Pc matches ESA's to 0.004 decades on average.
* Impact on the hold-out: about 8 fewer Critical events slip through per 2,400 events, for about 35 extra
  alerts to review.

## 2:50 – 3:00 · Close

"TIP Watcher tells the analyst, two days ahead, which close approaches will end up dangerous, and why."
