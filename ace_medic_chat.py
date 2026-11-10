#!/usr/bin/env python3
"""
ACE MEDIC CHAT - a conversational medical AI assistant (the "medical-field counterpart" of a coding assistant)

Put this file next to ace_medic.py (the self-evolving diagnostic core) and run:
    python ace_medic_chat.py        ->  http://localhost:8000

WHAT IT ADDS ON TOP OF THE CORE
  * Real dialogue: remembers the patient's symptoms across messages, asks follow-up questions, understands yes/no.
  * Intent engine: greeting / symptom triage / drug info / drug interactions / lab results / BMI / first aid /
    disease info / confirm-diagnosis / help. It is an online-learning classifier: every thumbs-down + correction
    from a user is stored in SQLite and the classifier retrains itself instantly.
  * Tools: lab-value interpreter, BMI calculator, drug database, interaction checker, first-aid protocols,
    and the self-evolving diagnostic model from ace_medic.py (champion/challenger evolution).
  * Learning from conversation: "my doctor confirmed it was influenza" feeds a real case into the core model.
  * Unanswered questions are logged (and sent to a language model if ANTHROPIC_API_KEY is set, optional).
  * Safety: red-flag rules and emergency keywords always interrupt the conversation.

!! Educational prototype. Not a medical device, not a substitute for a clinician. Gives general information only,
   no personal dosing instructions.
"""
import json, math, os, re, threading, time, uuid, difflib, urllib.request
import ace_medic as core

# --------------------------------------------------------------------------------------
# KNOWLEDGE BASES
# --------------------------------------------------------------------------------------
DRUGS = {
    "paracetamol": {"alias": ["acetaminophen", "tylenol", "panadol"], "class": "Analgesic / antipyretic",
        "uses": "pain and fever", "side": "rare at normal doses; liver damage in overdose",
        "caution": "never exceed the maximum daily dose on the label; check other products for hidden paracetamol; take care with liver disease or heavy alcohol use"},
    "ibuprofen": {"alias": ["advil", "brufen", "nurofen"], "class": "NSAID",
        "uses": "pain, inflammation, fever", "side": "stomach irritation, heartburn, raised blood pressure, kidney strain",
        "caution": "take with food; avoid with stomach ulcers, kidney disease, late pregnancy, or if you take blood thinners unless a clinician agrees"},
    "aspirin": {"alias": ["acetylsalicylic acid"], "class": "NSAID / antiplatelet",
        "uses": "pain, fever, heart-attack prevention in selected patients", "side": "stomach bleeding, bruising",
        "caution": "not for children with viral illness (Reye syndrome); bleeding risk with other blood thinners"},
    "amoxicillin": {"alias": ["amoxil"], "class": "Penicillin antibiotic", "uses": "bacterial infections (ear, chest, throat, urinary)",
        "side": "diarrhea, rash, nausea", "caution": "do not use with penicillin allergy; finish the prescribed course; useless against viruses"},
    "metformin": {"alias": ["glucophage"], "class": "Biguanide antidiabetic", "uses": "type 2 diabetes",
        "side": "nausea, diarrhea, metallic taste, rarely lactic acidosis", "caution": "kidney function must be monitored; hold before contrast scans or when dehydrated"},
    "omeprazole": {"alias": ["prilosec", "losec"], "class": "Proton-pump inhibitor", "uses": "reflux (GERD), stomach ulcers",
        "side": "headache, diarrhea; long-term low magnesium/B12", "caution": "long-term use should be reviewed by a clinician"},
    "salbutamol": {"alias": ["albuterol", "ventolin"], "class": "Short-acting bronchodilator", "uses": "relief of wheeze and asthma attacks",
        "side": "tremor, fast heartbeat", "caution": "needing it more often than usual means asthma control is poor: see a clinician; no relief in an attack is an emergency"},
    "warfarin": {"alias": ["coumadin"], "class": "Anticoagulant", "uses": "prevent blood clots",
        "side": "bleeding", "caution": "needs regular INR tests; many drugs and foods interact"},
    "lisinopril": {"alias": ["zestril"], "class": "ACE inhibitor", "uses": "high blood pressure, heart failure",
        "side": "dry cough, high potassium, dizziness", "caution": "not in pregnancy; kidney function and potassium monitoring"},
    "atorvastatin": {"alias": ["lipitor"], "class": "Statin", "uses": "lowering cholesterol, cardiovascular prevention",
        "side": "muscle aches, raised liver enzymes", "caution": "report unexplained muscle pain or dark urine; avoid large amounts of grapefruit juice"},
}
INTERACTIONS = {
    frozenset(["ibuprofen", "aspirin"]): ("moderate", "Ibuprofen can blunt aspirin's heart-protective effect and both raise stomach-bleeding risk."),
    frozenset(["ibuprofen", "warfarin"]): ("major", "NSAIDs with warfarin significantly raise bleeding risk."),
    frozenset(["aspirin", "warfarin"]): ("major", "Combined anti-clotting effect: high bleeding risk unless prescribed deliberately."),
    frozenset(["ibuprofen", "lisinopril"]): ("moderate", "NSAIDs can weaken blood-pressure control and strain the kidneys with ACE inhibitors."),
    frozenset(["paracetamol", "warfarin"]): ("minor", "Regular high-dose paracetamol can raise INR; occasional normal use is usually fine."),
    frozenset(["aspirin", "lisinopril"]): ("minor", "Low-dose aspirin is usually fine; high doses may reduce the ACE inhibitor effect."),
    frozenset(["metformin", "ibuprofen"]): ("minor", "NSAIDs may harm kidney function, which affects metformin safety."),
    frozenset(["atorvastatin", "warfarin"]): ("minor", "May modestly change INR; monitoring is recommended when starting."),
}
LABS = {  # name: (aliases, low, high, unit, meaning if low, meaning if high)
    "hemoglobin": (["hemoglobin", "haemoglobin", "hb", "hgb"], 12.0, 17.5, "g/dL", "anemia (iron, B12 or folate deficiency, blood loss, chronic disease)", "dehydration, smoking, lung disease or polycythemia"),
    "fasting glucose": (["fasting glucose", "fasting blood sugar", "fbs", "glucose", "blood sugar"], 70, 99, "mg/dL", "hypoglycemia (can be dangerous if symptomatic)", "100-125 = prediabetes range, 126 or more on two tests = diabetes range"),
    "hba1c": (["hba1c", "a1c", "glycated hemoglobin"], 4.0, 5.6, "%", "unusual; consider anemia or lab variation", "5.7-6.4 = prediabetes, 6.5 or more = diabetes range"),
    "ldl cholesterol": (["ldl", "ldl cholesterol"], 0, 100, "mg/dL", "generally favourable", "raises cardiovascular risk; lifestyle and sometimes statins"),
    "potassium": (["potassium"], 3.5, 5.0, "mmol/L", "weakness, cramps, arrhythmia risk (diuretics, vomiting, diarrhea)", "arrhythmia risk (kidney disease, ACE inhibitors) - needs prompt review"),
    "sodium": (["sodium"], 135, 145, "mmol/L", "hyponatremia: confusion, headache; needs medical review", "dehydration or fluid loss"),
    "creatinine": (["creatinine"], 0.6, 1.3, "mg/dL", "low muscle mass", "reduced kidney function"),
    "white blood cells": (["wbc", "white blood cells", "white cell count"], 4.0, 11.0, "x10^9/L", "viral infection, bone marrow or immune problems", "infection, inflammation, stress or steroids"),
    "platelets": (["platelets", "plt"], 150, 450, "x10^9/L", "bleeding/bruising risk", "inflammation, iron deficiency, clotting risk"),
    "tsh": (["tsh"], 0.4, 4.0, "mIU/L", "possible overactive thyroid", "possible underactive thyroid"),
}
FIRST_AID = {
    "burn": ["Cool the burn under cool running water for 20 minutes.", "Remove rings or tight items before swelling starts.", "Cover loosely with clean non-fluffy film or cloth.", "Do not use ice, butter or toothpaste; do not burst blisters.", "Seek emergency care for burns larger than the casualty's palm, on face/hands/genitals, chemical or electrical burns."],
    "choking": ["If the person can cough, encourage coughing.", "If they cannot cough, speak or breathe: call emergency services.", "Give up to 5 sharp back blows between the shoulder blades.", "Then up to 5 abdominal thrusts (Heimlich); alternate until cleared.", "If they become unconscious start CPR."],
    "nosebleed": ["Sit up and lean slightly forward (not back).", "Pinch the soft part of the nose firmly for 10-15 minutes without releasing.", "Breathe through the mouth.", "Seek care if bleeding lasts over 20 minutes or follows a head injury."],
    "bleeding": ["Apply firm direct pressure with a clean cloth.", "Do not remove soaked cloths; add more on top.", "Raise the injured limb if possible.", "Call emergency services for heavy or spurting bleeding."],
    "sprain": ["Rest the joint, apply a cold pack wrapped in cloth for 15-20 minutes at a time.", "Compress with an elastic bandage and elevate above heart level.", "See a clinician if you cannot bear weight, or there is deformity or numbness."],
    "seizure": ["Stay calm, note the time, move hard objects away.", "Do not restrain them or put anything in their mouth.", "Place them on their side after the shaking stops.", "Call emergency services if it lasts more than 5 minutes, repeats, or it is their first seizure."],
    "cpr": ["Check safety and responsiveness; call emergency services and get an AED.", "If not breathing normally, push hard and fast in the centre of the chest, about 5-6 cm deep, 100-120 per minute.", "Use the AED as soon as it arrives and follow its voice prompts."],
    "fainting": ["Lay the person flat and raise their legs.", "Loosen tight clothing, ensure fresh air.", "If not waking within a minute, or injured, or with chest pain: emergency services."],
}
FA_ALIASES = {"burn": ["burn", "burned", "scald"], "choking": ["choking", "choke", "heimlich"], "nosebleed": ["nosebleed", "nose bleed", "bleeding nose"],
              "bleeding": ["bleeding", "cut", "wound", "laceration"], "sprain": ["sprain", "twisted ankle", "sprained"], "seizure": ["seizure", "convulsion", "fit"],
              "cpr": ["cpr", "cardiac arrest", "resuscitation", "not breathing"], "fainting": ["faint", "fainted", "passed out", "fainting"]}

# --------------------------------------------------------------------------------------
# INTENT ENGINE (online-learning Naive Bayes over words; learns from corrections)
# --------------------------------------------------------------------------------------
SEED_INTENTS = {
    "greeting": ["hello", "hi there", "hey", "good morning", "good evening", "hello doctor"],
    "thanks": ["thanks", "thank you", "thanks a lot", "great thank you", "appreciate it"],
    "help": ["what can you do", "help", "how do you work", "what are your features", "who are you"],
    "triage": ["i have a fever and cough", "my head hurts and i feel dizzy", "what could be causing my stomach pain", "i feel sick and tired",
               "i have symptoms", "diagnose me", "what do i have", "i have had chest pain since morning", "my throat is sore"],
    "drug_info": ["what is ibuprofen used for", "tell me about metformin", "side effects of amoxicillin", "is paracetamol safe", "what does omeprazole do", "drug information"],
    "interaction": ["can i take ibuprofen with warfarin", "interaction between aspirin and lisinopril", "is it safe to combine these medicines", "do these drugs interact"],
    "lab": ["my hemoglobin is 9.5", "interpret my lab results", "blood test glucose 130", "my hba1c is 6.8 what does it mean", "what is a normal potassium level"],
    "bmi": ["calculate my bmi", "i weigh 70 kg and i am 175 cm", "what is my body mass index", "am i overweight"],
    "first_aid": ["how do i treat a burn", "first aid for choking", "what to do for a nosebleed", "someone fainted what do i do", "how to do cpr", "i cut my finger"],
    "disease_info": ["tell me about pneumonia", "what is asthma", "explain migraine", "what are the symptoms of influenza", "what is gerd", "information about diabetes"],
    "confirm": ["my doctor confirmed it was influenza", "i was diagnosed with strep throat", "the diagnosis was gastroenteritis", "confirmed diagnosis pneumonia"],
}
TOK = lambda s: re.findall(r"[a-z0-9']+", s.lower())

class IntentModel:
    def __init__(self):
        self.counts, self.totals, self.docs, self.vocab = {}, {}, {}, set()

    def learn(self, text, intent):
        c = self.counts.setdefault(intent, {})
        self.docs[intent] = self.docs.get(intent, 0) + 1
        for w in TOK(text):
            c[w] = c.get(w, 0) + 1
            self.totals[intent] = self.totals.get(intent, 0) + 1
            self.vocab.add(w)

    def predict(self, text):
        words, n_docs = TOK(text), sum(self.docs.values())
        sc = {}
        for it in self.counts:
            lp = math.log(self.docs[it] / n_docs)
            for w in words:
                lp += math.log((self.counts[it].get(w, 0) + 0.3) / (self.totals.get(it, 0) + 0.3 * (len(self.vocab) + 1)))
            sc[it] = lp
        m = max(sc.values())
        e = {k: math.exp(v - m) for k, v in sc.items()}
        z = sum(e.values())
        return {k: v / z for k, v in e.items()}

INTENTS = IntentModel()
for _it, _ex in SEED_INTENTS.items():
    for _t in _ex:
        INTENTS.learn(_t, _it)

# --------------------------------------------------------------------------------------
# TOOLS
# --------------------------------------------------------------------------------------
def find_drugs(text):
    low, found = " " + " ".join(TOK(text)) + " ", []
    names = {n: n for n in DRUGS}
    names.update({a: n for n, d in DRUGS.items() for a in d["alias"]})
    for alias, n in names.items():
        if f" {alias} " in low and n not in found:
            found.append(n)
    for w in TOK(text):  # typo tolerance
        if len(w) > 5 and w not in names:
            hit = difflib.get_close_matches(w, list(names), n=1, cutoff=0.84)
            if hit and names[hit[0]] not in found:
                found.append(names[hit[0]])
    return found

def tool_drug(drugs):
    d = DRUGS[drugs[0]]
    return (f"{drugs[0].title()} ({d['class']}). Used for {d['uses']}. Common/important side effects: {d['side']}. "
            f"Cautions: {d['caution']}. Dosing depends on age, weight, kidney/liver function and the product, so follow the label or ask a pharmacist or doctor.")

def tool_interaction(drugs):
    if len(drugs) < 2:
        return "Tell me the two (or more) medicines you want to check, for example: 'ibuprofen and warfarin'."
    lines = []
    for i in range(len(drugs)):
        for j in range(i + 1, len(drugs)):
            sev, msg = INTERACTIONS.get(frozenset([drugs[i], drugs[j]]), (None, None))
            lines.append(f"{drugs[i].title()} + {drugs[j].title()}: " + (f"{sev.upper()} interaction. {msg}" if sev else "no notable interaction in my database (not a guarantee; ask a pharmacist)."))
    return "\n".join(lines)

def tool_labs(text):
    low, out = text.lower(), []
    for name, (aliases, lo, hi, unit, lowm, himsg) in LABS.items():
        for a in sorted(aliases, key=len, reverse=True):
            m = re.search(r"\b" + re.escape(a) + r"\b\s*(?:is|was|of|=|:|at)?\s*(-?\d+(?:\.\d+)?)", low)
            if m:
                v = float(m.group(1))
                st = "LOW" if v < lo else "HIGH" if v > hi else "within the usual adult range"
                note = lowm if v < lo else himsg if v > hi else "no concern from this value alone"
                out.append(f"{name.title()} {v:g} {unit} (usual adult range {lo:g}-{hi:g}): {st}. Meaning: {note}.")
                break
    return out

def tool_bmi(text):
    low = text.lower()
    w = re.search(r"(\d+(?:\.\d+)?)\s*(kg|kilo|kilos|lb|lbs|pounds)", low)
    h = re.search(r"(\d+(?:\.\d+)?)\s*(cm|m|meters|metres)\b", low)
    if not (w and h):
        return None
    kg = float(w.group(1)) * (0.4536 if w.group(2).startswith(("lb", "pound")) else 1)
    hm = float(h.group(1)) / (100 if h.group(2) == "cm" else 1)
    if not (1.0 < hm < 2.6 and 20 < kg < 400):
        return "Those numbers look unusual. Please give weight in kg and height in cm, e.g. '70 kg and 175 cm'."
    b = kg / hm ** 2
    cat = "underweight" if b < 18.5 else "healthy range" if b < 25 else "overweight" if b < 30 else "obesity range"
    return f"BMI = {b:.1f} ({cat}). BMI is a screening number only; it ignores muscle mass, age and body fat distribution."

def tool_first_aid(text):
    low = " " + " ".join(TOK(text)) + " "
    for k, al in FA_ALIASES.items():
        if any(f" {a} " in low for a in al):
            return k
    return None

def tool_disease(text):
    low = text.lower()
    names = list(core.DISEASES)
    hit = next((d for d in names if d.lower() in low or d.lower().split(" (")[0] in low), None)
    if not hit:
        word_map = {w: d for d in names for w in TOK(d) if len(w) > 4}
        hit = next((word_map[w] for w in TOK(text) if w in word_map), None)
    if not hit:
        return None
    sy = sorted(core.DISEASES[hit].items(), key=lambda kv: -kv[1])[:6]
    urg, adv = core.ADVICE.get(hit, (2, "Consult a clinician."))
    return f"{hit}: typical features are " + ", ".join(s.replace("_", " ") for s, _ in sy) + f". Urgency level {urg}/3. {adv}"

def llm_fallback(question):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        body = json.dumps({"model": os.environ.get("ACE_LLM", "claude-sonnet-5-5"), "max_tokens": 500,
                           "system": "You are the language module of Ace Medic, a medical information assistant. Give accurate, concise, general health information. "
                                     "Never give a definitive diagnosis or personal dosing instructions, always mention when to seek care, and keep answers under 150 words.",
                           "messages": [{"role": "user", "content": question[:1500]}]}).encode()
        req = urllib.request.Request("https://api.anthropic.com/v1/messages", body,
                                     {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read())["content"][0]["text"]
    except Exception:
        return None

# --------------------------------------------------------------------------------------
# DIALOGUE MANAGER
# --------------------------------------------------------------------------------------
YES = {"yes", "yeah", "yep", "yup", "sure", "correct", "i do", "definitely", "true"}
NO = {"no", "nope", "nah", "never", "i dont", "not really", "false"}
UNSURE = {"not sure", "unsure", "skip", "dont know", "i dont know", "maybe"}
SESSIONS, SLOCK = {}, threading.Lock()
DISCLAIMER = "General information only, not a diagnosis. If you feel very unwell, call your local emergency number."

def session(sid):
    with SLOCK:
        s = SESSIONS.setdefault(sid, {"present": set(), "denied": set(), "asked": set(), "q": None, "last": None, "t": time.time()})
        s["t"] = time.time()
        for k in [k for k, v in SESSIONS.items() if time.time() - v["t"] > 6 * 3600]:
            SESSIONS.pop(k, None)
        return s

def log_msg(sid, text, intent, conf, reply):
    db, lock = core.ENGINE.db, core.ENGINE.lock
    with lock:
        cur = db.execute("INSERT INTO chat_log(ts,sid,text,intent,conf,reply) VALUES(?,?,?,?,?,?)", (time.time(), sid, text, intent, conf, reply[:2000]))
        db.commit()
        return cur.lastrowid

def triage_reply(s, text, r):
    s["present"], s["denied"] = set(r["present"]), set(r["denied"])
    out = {"cards": [], "chips": []}
    if r["flags"]:
        out["flags"] = r["flags"]
    if not r["ranking"]:
        return "I could not recognise any symptoms. Describe them in your own words, e.g. 'fever, dry cough and body aches since yesterday'.", out
    top = r["ranking"][0]
    lines = ["Here is what the symptoms you mentioned most resemble (" + ", ".join(x.replace("_", " ") for x in r["present"]) + "):"]
    out["ranking"] = r["ranking"][:4]
    lines.append(f"Most likely: {top['disease']} ({top['prob'] * 100:.0f}%). {top['advice']}")
    q = r["question"]
    s["q"] = q["symptom"] if q else None
    if q:
        lines.append(q["text"] + " (yes / no / not sure)")
        out["chips"] = ["Yes", "No", "Not sure"]
    else:
        lines.append("If a clinician confirms the real diagnosis, tell me (for example 'my doctor confirmed it was influenza') and I will learn from it.")
    s["last"] = top
    return "\n".join(lines), out

def respond(sid, text):
    s = session(sid)
    text = text.strip()[:2000]
    low = " ".join(TOK(text))
    extra = {}
    # 1. emergencies always win
    if core.emergency_text(text) and not re.search(r"suicid|kill myself|end my life|want to die", low):
        reply, intent, conf = "This sounds like a possible EMERGENCY. Call your local emergency number now (999, 911, 112 or your country's number). Do not wait for an online answer.", "emergency", 1.0
        return {"reply": reply, "intent": intent, "flags": ["Emergency keyword detected"], "msg_id": log_msg(sid, text, intent, conf, reply)}
    if re.search(r"suicid|kill myself|end my life|want to die", low):
        reply = ("I am really sorry you are in this much pain, and I am glad you said it. You deserve support right now. Please contact your local emergency number or a crisis line "
                 "(in the US call or text 988; elsewhere see findahelpline.com), or tell someone you trust to stay with you. If you are in immediate danger, call emergency services now.")
        return {"reply": reply, "intent": "emergency", "flags": ["Crisis support"], "msg_id": log_msg(sid, text, "emergency", 1.0, reply)}
    if low in ("new patient", "reset", "start over", "new case", "clear"):
        s.update({"present": set(), "denied": set(), "asked": set(), "q": None, "last": None})
        reply = "Started a new patient. Describe the symptoms."
        return {"reply": reply, "intent": "reset", "msg_id": log_msg(sid, text, "reset", 1.0, reply)}
    # 2. answer to a pending follow-up question
    if s["q"] and (low in YES or low in NO or low in UNSURE):
        sym = s["q"]
        s["asked"].add(sym)
        if low in YES:
            s["present"].add(sym)
        elif low in NO:
            s["denied"].add(sym)
        r = core.ENGINE.diagnose("", s["present"], s["denied"], s["asked"])
        reply, extra = triage_reply(s, text, r)
        return {"reply": reply, "intent": "triage", "msg_id": log_msg(sid, text, "triage", 1.0, reply), **extra}
    # 3. classify
    probs = INTENTS.predict(text)
    present_now, _ = core.extract(text)
    if present_now:
        probs["triage"] = probs.get("triage", 0) + 0.35
    drugs, labs, bmi, fa = find_drugs(text), tool_labs(text), tool_bmi(text), tool_first_aid(text)
    if labs:
        probs["lab"] = probs.get("lab", 0) + 1
    if bmi:
        probs["bmi"] = probs.get("bmi", 0) + 1
    if len(drugs) >= 2:
        probs["interaction"] = probs.get("interaction", 0) + 0.6
    elif drugs:
        probs["drug_info"] = probs.get("drug_info", 0) + 0.4
    if fa and not present_now - {"fever"} :
        probs["first_aid"] = probs.get("first_aid", 0) + 0.4
    intent = max(probs, key=probs.get)
    conf = min(probs[intent], 1.0)
    if conf < 0.4 and not (present_now or drugs or labs or bmi or fa):
        intent = "unknown"
    reply = ""
    if intent == "greeting":
        reply = "Hello, I am Ace Medic, your medical information assistant. Describe your symptoms, ask about a medicine, paste lab values, or ask for first aid steps."
    elif intent == "thanks":
        reply = "You are welcome. Take care, and see a clinician if anything worsens."
    elif intent == "help":
        reply = ("I can: 1) triage symptoms and ask smart follow-up questions, 2) explain medicines and check interactions, 3) interpret lab values (e.g. 'hemoglobin 9.5'), "
                 "4) calculate BMI, 5) give first-aid steps, 6) explain conditions. Say 'new patient' to reset. I learn from your corrections and from clinician-confirmed cases.")
    elif intent == "triage":
        r = core.ENGINE.diagnose(text, s["present"], s["denied"], s["asked"])
        reply, extra = triage_reply(s, text, r)
    elif intent == "drug_info" and drugs:
        reply = tool_drug(drugs)
    elif intent == "interaction":
        reply = tool_interaction(drugs)
    elif intent == "lab" and labs:
        reply = "\n".join(labs) + "\nInterpretation depends on age, sex, lab and symptoms. Discuss results with your clinician."
    elif intent == "bmi" and bmi:
        reply = bmi
    elif intent == "first_aid" and fa:
        reply = f"First aid for {fa}:\n" + "\n".join(f"{i + 1}. {st}" for i, st in enumerate(FIRST_AID[fa])) + "\nWhen in doubt, call emergency services."
    elif intent == "disease_info" and tool_disease(text):
        reply = tool_disease(text)
    elif intent == "confirm":
        m = re.search(r"(?:confirmed|diagnosed with|diagnosis (?:was|is)|it was|said i have|i have)\s*:?\s*(?:an?\s+|the\s+)?([a-z0-9 ()\-]{3,60})$", low)
        dx = m.group(1).strip() if m else ""
        known = next((d for d in core.DISEASES if dx and (d.lower() in dx or dx in d.lower())), dx)
        if known and s["present"]:
            last = s["last"] or {}
            res = core.ENGINE.feedback(s["present"], s["denied"], known, last.get("disease", ""), last.get("prob", 0))
            reply = f"Thank you. I stored this as confirmed case #{res['case_id']} ({res['disease']}) and learned from it. Self-evolution runs in {res['next_evolution_in']} more case(s)."
        else:
            reply = "To learn from a confirmed diagnosis I need your symptoms first, then say for example 'my doctor confirmed it was influenza'."
    if not reply:  # unknown or missing entities
        ans = llm_fallback(text)
        if ans:
            reply, intent = ans, "llm"
        elif intent in ("drug_info", "interaction"):
            reply = "Which medicine do you mean? I know: " + ", ".join(sorted(DRUGS)) + "."
        elif intent in ("lab", "bmi", "first_aid", "disease_info"):
            reply = {"lab": "Give a value like 'hemoglobin 9.5' or 'hba1c 6.8'. I know: " + ", ".join(LABS) + ".",
                     "bmi": "Give weight and height, e.g. '70 kg and 175 cm'.",
                     "first_aid": "I know first aid for: " + ", ".join(FIRST_AID) + ".",
                     "disease_info": "I know: " + ", ".join(core.DISEASES) + "."}[intent]
        else:
            reply = "I am not sure I understood. Try describing symptoms, naming a medicine, or pasting a lab value. Your feedback below teaches me."
            intent = "unknown"
    reply = reply.rstrip()
    return {"reply": reply, "intent": intent, "conf": round(conf, 2), "msg_id": log_msg(sid, text, intent, conf, reply), **extra}

def rate(msg_id, rating, correct_intent=None):
    db, lock = core.ENGINE.db, core.ENGINE.lock
    with lock:
        row = db.execute("SELECT text,intent FROM chat_log WHERE id=?", (msg_id,)).fetchone()
        if not row:
            return {"error": "unknown message"}
        db.execute("UPDATE chat_log SET rating=?, corrected=? WHERE id=?", (rating, correct_intent, msg_id))
        learned = False
        if correct_intent in SEED_INTENTS:
            db.execute("INSERT INTO intent_examples(text,intent) VALUES(?,?)", (row[0], correct_intent))
            INTENTS.learn(row[0], correct_intent)   # instant self-improvement
            learned = True
        elif rating > 0:
            INTENTS.learn(row[0], row[1]) if row[1] in SEED_INTENTS else None
            learned = row[1] in SEED_INTENTS
        db.commit()
    return {"ok": True, "learned": learned}

def init_chat_tables():
    db = core.ENGINE.db
    db.executescript("""CREATE TABLE IF NOT EXISTS chat_log(id INTEGER PRIMARY KEY, ts REAL, sid TEXT, text TEXT, intent TEXT, conf REAL, reply TEXT, rating INTEGER, corrected TEXT);
                        CREATE TABLE IF NOT EXISTS intent_examples(id INTEGER PRIMARY KEY, text TEXT, intent TEXT);""")
    for t, i in db.execute("SELECT text,intent FROM intent_examples").fetchall():
        INTENTS.learn(t, i)   # replay everything ever learned from users

# --------------------------------------------------------------------------------------
# FRONTEND (chat UI)
# --------------------------------------------------------------------------------------
PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ace Medic</title><style>
:root{--bg:#0a111f;--card:#111b2e;--line:#1e2b47;--tx:#e8eef8;--mut:#8a9bb9;--acc:#3ddc97;--warn:#ffb020;--bad:#ff5d5d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);font:15px/1.55 system-ui,Segoe UI,Roboto,sans-serif;height:100vh;display:flex;flex-direction:column}
header{padding:14px 22px;border-bottom:1px solid var(--line);display:flex;gap:12px;align-items:center}header b{font-size:20px}header span{color:var(--acc)}small,.mut{color:var(--mut)}
.wrap{flex:1;display:grid;grid-template-columns:minmax(0,1fr) 300px;min-height:0}@media(max-width:860px){.wrap{grid-template-columns:1fr}aside{display:none}}
#chat{overflow:auto;padding:20px;display:flex;flex-direction:column;gap:12px}.m{max-width:760px;padding:12px 14px;border-radius:14px;white-space:pre-wrap;word-wrap:break-word}
.u{align-self:flex-end;background:#1c3a5e}.a{align-self:flex-start;background:var(--card);border:1px solid var(--line)}
.flag{background:#3a1519;border:1px solid var(--bad);color:#ffd9d9;padding:8px 10px;border-radius:9px;margin:6px 0;font-weight:600}
.bar{height:7px;border-radius:4px;background:#1b2a47;overflow:hidden;margin:3px 0 7px}.bar i{display:block;height:100%;background:var(--acc)}
.u3{color:var(--bad)}.u2{color:var(--warn)}.u1{color:var(--acc)}
.row{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}button{background:#1b2a47;color:var(--tx);border:0;border-radius:9px;padding:6px 12px;cursor:pointer;font:inherit}
button.p{background:var(--acc);color:#04210f;font-weight:700}select{background:#0d1626;color:var(--tx);border:1px solid var(--line);border-radius:8px;padding:5px}
form{display:flex;gap:8px;padding:14px 20px;border-top:1px solid var(--line)}input{flex:1;background:#0d1626;color:var(--tx);border:1px solid var(--line);border-radius:12px;padding:12px;font:inherit}
aside{border-left:1px solid var(--line);padding:16px;overflow:auto}h3{margin:0 0 8px;font-size:12px;text-transform:uppercase;letter-spacing:1px;color:var(--mut)}
.disc{font-size:12px;color:var(--mut);padding:0 20px 10px}
</style></head><body><header><b>ACE <span>MEDIC</span></b><small>your self-improving medical AI assistant</small></header>
<div class="wrap"><div style="display:flex;flex-direction:column;min-height:0"><div id="chat"></div>
<div class="disc">Educational prototype: general information only, not a diagnosis or a substitute for a doctor. Emergency? Call your local emergency number.</div>
<form id="f"><input id="i" autocomplete="off" placeholder="Describe symptoms, ask about a drug, paste a lab value, ask for first aid..." autofocus><button class="p">Send</button></form></div>
<aside><h3>Model brain</h3><div id="st" class="mut">loading...</div><div class="row"><button id="evo">Evolve now</button></div><div id="em" class="mut"></div>
<h3 style="margin-top:20px">Try</h3><div class="row" id="try"></div></aside></div>
<script>
const $=id=>document.getElementById(id),sid=crypto.randomUUID();
const h=(t,c,x)=>{const e=document.createElement(t);if(c)e.className=c;if(x!==undefined)e.textContent=x;return e};
const api=(u,b)=>fetch(u,b?{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(b)}:{}).then(r=>r.json());
const INTENTS=["triage","drug_info","interaction","lab","bmi","first_aid","disease_info","confirm","greeting","help"];
function add(who,text){const m=h("div","m "+who,text);$("chat").append(m);$("chat").scrollTop=1e9;return m}
async function send(text){if(!text)return;add("u",text);const r=await api("/api/chat",{sid,text});const m=add("a",r.reply||r.error||"error");
  (r.flags||[]).forEach(f=>m.prepend(h("div","flag","URGENT: "+f)));
  if(r.ranking){const box=h("div");r.ranking.forEach(o=>{const t=h("div");t.append(h("b","u"+o.urgency,o.disease+" "),h("span","mut",(o.prob*100).toFixed(0)+"%"));const b=h("div","bar"),i=h("i");i.style.width=o.prob*100+"%";b.append(i);box.append(t,b)});m.append(box)}
  const row=h("div","row");(r.chips||[]).forEach(c=>{const b=h("button","p",c);b.onclick=()=>send(c);row.append(b)});
  if(r.msg_id){const up=h("button","","helpful"),dn=h("button","","wrong");
    up.onclick=async()=>{await api("/api/rate",{msg_id:r.msg_id,rating:1});up.textContent="thanks";status()};
    dn.onclick=()=>{const s=h("select");s.append(new Option("what did I mean?",""));INTENTS.forEach(x=>s.append(new Option(x.replace("_"," "),x)));
      s.onchange=async()=>{if(!s.value)return;const a=await api("/api/rate",{msg_id:r.msg_id,rating:-1,correct_intent:s.value});s.replaceWith(h("small","mut",a.learned?"learned, thank you":"noted"));status()};dn.replaceWith(s)};
    row.append(up,dn)}
  m.append(row);$("chat").scrollTop=1e9}
$("f").onsubmit=e=>{e.preventDefault();const v=$("i").value.trim();$("i").value="";send(v)};
$("evo").onclick=async()=>{$("em").textContent="evolving...";const r=await api("/api/evolve",{});$("em").textContent=r.status+(r.cand_loss?" ("+r.champ_loss+" -> "+r.cand_loss+")":"");status()};
["I have a fever, cough and body aches","new patient","can I take ibuprofen with warfarin","my hemoglobin is 9.5 and glucose 130","I weigh 70 kg and I am 175 cm","how do I treat a burn","what is pneumonia"].forEach(t=>{const b=h("button","",t);b.onclick=()=>send(t);$("try").append(b)});
async function status(){const s=await api("/api/status");$("st").replaceChildren();[["Version","v"+s.version],["Diseases",s.diseases],["Symptoms",s.symptoms],["Cases learned",s.cases+" ("+s.pending+" pending)"],["Benchmark acc",s.benchmark_acc],["Live acc",s.live_acc_last50===null?"n/a":s.live_acc_last50]].forEach(([k,v])=>{const d=h("div");d.append(h("b","",k+": "),h("span","",String(v)));$("st").append(d)})}
add("a","Hello, I am Ace Medic. Tell me what you are experiencing, or ask about a medicine, lab result or first aid.");status();setInterval(status,10000);
</script></body></html>"""

class ChatHandler(core.Handler):
    def do_GET(self):
        if self.path == "/":
            return self._send(200, PAGE, "text/html")
        super().do_GET()

    def do_POST(self):
        if self.path in ("/api/chat", "/api/rate"):
            try:
                n = int(self.headers.get("Content-Length", 0))
                if n > 50_000:
                    return self._send(413, {"error": "too large"})
                b = json.loads(self.rfile.read(n) or b"{}")
                if self.path == "/api/chat":
                    r = respond(str(b.get("sid", "anon"))[:64], str(b.get("text", "")))
                else:
                    r = rate(int(b.get("msg_id", 0)), int(b.get("rating", 0)), b.get("correct_intent"))
                return self._send(200, r)
            except Exception as e:
                return self._send(400, {"error": str(e)})
        super().do_POST()

def main():
    core.ENGINE = core.Engine()
    init_chat_tables()
    print(f"Ace Medic Chat ready -> http://localhost:{core.PORT}  (core model v{core.ENGINE.model.version})")
    core.ThreadingHTTPServer(("0.0.0.0", core.PORT), ChatHandler).serve_forever()

if __name__ == "__main__":
    main()
