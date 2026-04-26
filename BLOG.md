# SepsisGuard: Teaching AI Agents to Catch What Humans Miss

*A multi-agent RL environment for sepsis early warning coordination*

---

## The number that should haunt you

**11 million people die from sepsis every year.** ([WHO, 2020](https://www.who.int/news-room/fact-sheets/detail/sepsis))

That is approximately **20% of all deaths on earth** — one in five. Nearly 49 million cases every year. Almost half of those are children under five. More deaths than all cancers combined. More than heart disease in most countries. More than any single natural disaster in living memory. And most people have never heard of it.

We did not build this project because we are doctors. We built it because once you understand what sepsis is and why people keep dying from it, it becomes very hard to look away.

---

## What is sepsis, actually?

Sepsis is not a disease in the way we usually think of disease. It is a response — specifically, it is what happens when your immune system responds to an infection and then cannot stop.

It starts small. A urinary tract infection. A pneumonia. A small cut that gets infected after surgery. Your immune system sends the alarm: fight. White blood cells mobilize. Inflammatory signals flood the bloodstream. Normally this works. The infection is contained, the immune response winds down, you recover.

In sepsis, it does not stop. The immune response escalates beyond the infection it was fighting. Inflammatory chemicals damage blood vessels. Blood pressure drops. Oxygen cannot reach organs. The kidneys, the lungs, the liver — they begin to fail, not because of the original infection, but because of the body's own defense mechanism turning on itself.

The mortality rate increases by approximately **7% for every hour that antibiotics are delayed** after sepsis onset. Not 7% total. 7% per hour. Someone who could have been saved at hour one may not survive to hour six.

The treatment — when caught in time — is straightforward. Fluids to stabilize blood pressure. Antibiotics to address the underlying infection. Every hospital has these. The treatment is not the problem.

---

## Why people keep dying

Here is the part that is genuinely hard to accept: the reason people die from sepsis at this scale is not because medicine does not know how to treat it. It is because of how hospital information flows.

A septic patient does not arrive waving a flag. They arrive with pneumonia, or recovering from surgery, or managing a chronic condition. The signs of early sepsis — a heart rate that climbs, a blood pressure that drifts downward, a lab value that edges outside normal range — are easily attributed to the primary illness.

And these signals are held by different people.

The **nurse** notices the heart rate has been rising across the shift. She has five patients. She documents it.

The **lab analyst** sees that this patient's lactate is elevated and their white cell count is abnormal. She processes dozens of samples. She sends the result.

The **pharmacist** reviews the medication list. This patient is on tacrolimus — an immunosuppressant. That drug suppresses the body's fever response and blunts the white cell count. The usual alarm signals won't appear. He notes this.

The **physician** sees the patient during rounds. He has the complaint that brought the patient in, the notes from the morning, and a hundred other decisions competing for his attention. The lactate result is in the system. The pharmacist's note is in the chart. The nurse's vital trend is in the flowsheet. None of them have been connected.

Hours pass.

This is not negligence. This is how hospitals work. Information is fragmented across roles, systems, and shifts. No one person holds the whole picture, and the coordination overhead to assemble it in real time is enormous.

**An AI agent that monitors all four streams simultaneously, never gets tired, and never misses a shift handover could assemble this picture hours before a human would.**

---

## Why AI hasn't already solved this

Reasonable question. Hospitals are computerized. EHR systems exist. Why isn't there already an alert?

Some hospitals do have early warning scores — numerical thresholds that trigger an alert when vitals breach certain values. They generate an enormous number of alerts. Nurses learn to dismiss most of them. The one real sepsis case is buried under thirty false alarms. This is alarm fatigue, and it is a well-documented clinical problem.

Better systems exist, but AI in clinical settings faces genuine constraints. Liability is a major one. If an AI model recommends an antibiotic and the patient has an adverse reaction, who is responsible? These questions are unresolved and they have significantly slowed deployment.

But **coordination and monitoring** is different from diagnosis and prescription. A system that says "these four signals, appearing in this pattern, have historically preceded sepsis by four to six hours — consider reviewing this patient" is not practicing medicine. It is doing what a very attentive coordinator would do if such a person existed and had access to all the data simultaneously.

This project does not try to make clinical decisions. It tries to train agents to coordinate information so that the right human can make the right decision at the right time.

---

## The environment

SepsisGuard is a multi-agent reinforcement learning environment built on the OpenEnv framework. It simulates a hospital ward running in 30-minute ticks, with five to ten patients, over 24 to 72 simulated hours.

Four agents operate simultaneously. Each sees a different, incomplete slice of the patient information:

**The Nurse** monitors vitals for her five assigned patients: heart rate, blood pressure, respiratory rate, temperature, oxygen saturation. She can escalate a patient to the physician, request lab tests, or flag a concern. She cannot see lab results or the medication list. She must decide, from vitals alone, whether this patient warrants an escalation — knowing that over-escalating erodes the physician's trust in her future warnings.

**The Lab Analyst** sees lab values for all patients: lactate, white cell count, procalcitonin, creatinine. She can flag critical values and recommend follow-up tests. She does not know which patients have abnormal vitals. She cannot see what medications they are on.

**The Pharmacist** reviews medication lists and the hospital's antibiogram (drug resistance rates). He knows which patients are immunosuppressed — a crucial piece of information that changes how you interpret the absence of fever. He can recommend antibiotics and flag concerning drug interactions. He cannot see vitals or lab values directly.

**The Physician** sees only what the other three agents have formally escalated to him. He must synthesize incomplete, secondhand summaries and decide: order antibiotics, admit to ICU, request more labs, or wait. Acting too aggressively on every escalation wastes resources and erodes trust. Missing a real sepsis case costs a life.

No single agent has enough information to diagnose sepsis. They must learn, through training, to communicate the right information at the right time.

We built three task difficulty levels:

| Task | Patients | Sepsis Cases | False Alarms | Duration | Pass Threshold |
|---|---|---|---|---|---|
| Textbook (Easy) | 5 | 1 | 0 | 24 hours | 0.70 |
| Atypical (Medium) | 10 | 3 | 0 | 48 hours | 0.55 |
| High Acuity (Hard) | 10 | 4 | 2 | 72 hours | 0.40 |

The hard task includes false alarm patients — people whose vitals and labs suggest sepsis but who do not have an infection. Agents must learn not just sensitivity (catch the real cases) but specificity (avoid burning physician trust on patients who are fine).

---

## Training

We used **Qwen2.5-3B-Instruct** as the base model, quantized to 4-bit with Unsloth, and trained it with **TRL GRPO** (Grouped Relative Policy Optimization) on Task 1 (the textbook case). A single model serves all four roles, receiving role-conditioned system prompts that specify what it can see, what actions it can take, and what clinical thresholds it should use.

The reward function is not a static dataset. It connects directly to the live environment. The model generates an action (as a JSON object), the environment executes it and advances the patient's physiology, and the resulting clinical outcome — did the patient's condition improve, did the right escalation reach the right person, did sepsis get caught — determines the reward signal.

We trained for approximately 450 steps on a HF A10G GPU.

### What happened

![Training results across all four roles](training_results%20(3).png)

*Training reward curves (left) and per-role score comparison: heuristic baseline, pre-training, and post-training (center and right).*

| Role | Heuristic | Pre-Training | Post-Training | Change |
|---|---|---|---|---|
| Nurse | 0.82 | 0.75 | 0.84 | **+0.09** |
| Lab Analyst | 0.82 | 0.40 | 0.65 | **+0.25** |
| Pharmacist | 0.82 | 0.67 | 0.76 | **+0.09** |
| Physician | 0.82 | 0.76 | 0.45 | −0.31 |

Three of four roles improved. The Lab Analyst — which started weakest — improved the most. Before training, the model was unsure when to flag results, flagging too much or too late. After training, it learned to flag critical lactate and WBC values at the right moment in the episode.

The Physician role degraded. This is expected. The physician's task is structurally the hardest: acting on secondhand, incomplete information with the highest-stakes consequences. Task 1 has one sepsis patient and no false alarms — not enough training variety for the model to learn the full conditional logic the physician role requires. Improving the physician will need a larger task curriculum, more training time, and likely a larger model.

---

## What we learned

Building this environment clarified something that is easy to miss when thinking about AI in healthcare: **the bottleneck is often not the individual decision, it is the information flow that precedes it.**

A model that can perfectly diagnose sepsis given complete patient information is not the hard problem. The hard problem is that complete information never arrives cleanly assembled. It arrives in fragments, across roles, with delays, with noise.

Training agents to handle this — to escalate precisely, to flag at the right threshold, to synthesize incomplete signals — is the actual problem. And it is a problem that reinforcement learning is well-suited to, because the reward signal (did the patient survive? did the sepsis get treated in time?) is natural and unambiguous.

---

## What this is not

To be very direct: this is a research prototype built in 30 hours at a hackathon. It should not be deployed in any clinical setting. It has not been validated against real patient data. The physician role, in particular, is not performing well enough to be trusted with real decisions.

This is not medical advice. It is not a diagnostic tool. It is a demonstration that the coordination problem can be framed as an RL environment, that training on it produces measurable improvement, and that the direction is worth pursuing.

The direction is worth pursuing.

---

## Next steps

### Technical roadmap

To move from MVP to something that could plausibly help real patients:

1. **Train the physician role seriously** — more task variety, longer training, larger base model
2. **Calibrate physiology against real data** — the patient simulation needs validation against actual sepsis progression curves
3. **Expand to Task 2 and Task 3** — the harder tasks test generalization in ways Task 1 does not
4. **Evaluation by clinical experts** — before any deployment conversation, the reward function and agent behavior need review by people who treat sepsis

### The business case

The numbers make this worth pursuing beyond a hackathon.

**The cost of sepsis is staggering.** In wealthy nations, treating a single sepsis patient costs over $32,000 on average (WHO). Across 49 million cases globally per year, the economic burden runs into the trillions. A system that catches sepsis hours earlier — reducing ICU stays, ventilator days, and organ failure complications — does not need to be dramatically better to generate enormous value.

**The market already exists.** Hospitals spend heavily on early warning systems, clinical decision support, and patient monitoring platforms. The existing products (NEWS scores, sepsis alert tools) are rule-based and generate excessive false alarms. A trained multi-agent system that understands context — that a patient on immunosuppressants will not show the usual fever response, that the lab result and the vital trend together mean something the individual values do not — is a meaningfully better product.

**The regulatory path is clearer than people assume.** A system positioned as a coordination and alerting tool — one that surfaces information for a clinician to act on rather than acting autonomously — fits within existing FDA and CE frameworks for clinical decision support software. It does not require the same level of validation as a diagnostic device. This is the right wedge into a heavily regulated market.

**The deployment model is low-friction.** Modern hospitals already have digitized vitals, labs, pharmacy systems, and EHR integration APIs. The environment we built mirrors exactly the data feeds that already exist. Deployment does not require new hardware — it requires connecting to systems that are already running.

The realistic path: validate with a hospital partner on retrospective data, demonstrate earlier detection in controlled trials, deploy as a monitoring overlay with human confirmation required at every step. The goal is not to replace clinical judgment. It is to make sure clinical judgment gets applied before it is too late.

The scale of the problem makes it worth trying to get this right. 11 million deaths per year. A treatment that works when caught in time. A coordination problem that AI is, in principle, capable of helping solve.

---

*SepsisGuard was built at the OpenEnv Hackathon, Meta x Hugging Face, Bangalore, April 2026.*

*Environment: [HuggingFace Space](https://huggingface.co/spaces/Jishnu-Vijayan-03/Sepsis-Guard)*
*Code: [GitHub](https://github.com/JishnuVijayan/Sepsis-Guard)*
*Training notebook: [colab_training_sepsisguard.ipynb](https://huggingface.co/spaces/Jishnu-Vijayan-03/Sepsis-Guard/blob/main/training/colab_training_sepsisguard.ipynb)*
