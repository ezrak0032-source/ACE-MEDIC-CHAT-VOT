#!/usr/bin/env python3
"""
ACE MEDIC - a self-improving medical decision-support AI (backend + frontend in one file)

Pure Python 3.9+, standard library only. Run:  python ace_medic.py   ->  http://localhost:8000

HOW IT BUILDS ITSELF ON EXPERIENCE
  1. Bootstrap      : pre-trains on synthetic patients generated from a seed medical knowledge base.
  2. Experience     : every clinician-confirmed case is stored in SQLite (the "experience memory").
  3. Instant learning: a Naive-Bayes layer (symptom frequencies per disease) updates on every case.
  4. Self-evolution : every N cases a CHALLENGER model is trained (softmax regression) on
                      experience replay + generative replay (synthetic cases sampled from what it has
                      learned so far, which prevents forgetting). It must beat the CHAMPION on a
                      hold-out set to be promoted, otherwise it is rolled back. The ensemble weight is
                      tuned automatically.
  5. Growth         : unknown diagnoses become new disease classes; new symptom phrases become new
                      vocabulary, so the model's knowledge grows with use.
  6. Active asking  : picks the follow-up question with the highest expected information gain.
  7. Safety         : hard-coded red-flag rules are NOT learnable and always override the model.

!! Educational decision-support prototype. NOT a medical device and NOT a substitute for a clinician.
   Seed knowledge is simplified. Validate with real clinical data and oversight before any real use.
"""
import copy, difflib, json, math, os, random, re, sqlite3, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DB_PATH = os.environ.get("ACE_DB", "ace_medic.db")
PORT = int(os.environ.get("PORT", "8000"))
EVOLVE_EVERY = 10          # new confirmed cases that trigger self-evolution
HOLDOUT_MOD = 5            # case ids divisible by this are never trained on (hold-out)
NB_STRENGTH = 15.0         # how strongly seed knowledge resists experience

# --------------------------------------------------------------------------------------
# 1. SEED KNOWLEDGE  (P(symptom | disease), simplified for demonstration)
# --------------------------------------------------------------------------------------
DISEASES = {
    "Common cold":        {"runny_nose": .9, "sneezing": .7, "sore_throat": .6, "cough": .6, "fatigue": .3, "headache": .2, "fever": .1},
    "Influenza":          {"fever": .9, "body_aches": .85, "fatigue": .85, "cough": .8, "headache": .6, "sore_throat": .4, "chills": .6},
    "COVID-19":           {"fever": .7, "cough": .75, "loss_smell": .5, "fatigue": .7, "body_aches": .5, "sore_throat": .4, "shortness_of_breath": .3, "headache": .4},
    "Strep throat":       {"sore_throat": .95, "fever": .7, "swollen_glands": .6, "headache": .3, "nausea": .2, "rash": .1},
    "Pneumonia":          {"cough": .9, "fever": .85, "shortness_of_breath": .75, "chest_pain": .4, "sputum": .7, "fatigue": .7, "chills": .5},
    "Acute bronchitis":   {"cough": .95, "sputum": .7, "fatigue": .5, "sore_throat": .3, "wheezing": .3, "fever": .2},
    "Asthma exacerbation": {"wheezing": .9, "shortness_of_breath": .9, "cough": .7, "chest_pain": .4},
    "Allergic rhinitis":  {"sneezing": .9, "runny_nose": .85, "itchy_eyes": .8, "cough": .2, "headache": .2},
    "Migraine":           {"headache": .98, "nausea": .7, "photophobia": .75, "vomiting": .4, "dizziness": .3},
    "Meningitis":         {"fever": .9, "severe_headache": .85, "stiff_neck": .8, "photophobia": .6, "confusion": .4, "vomiting": .5, "rash": .2},
    "Gastroenteritis":    {"diarrhea": .9, "vomiting": .75, "nausea": .8, "abdominal_pain": .75, "fever": .4, "fatigue": .4, "loss_appetite": .5},
    "Appendicitis":       {"abdominal_pain": .95, "right_lower_pain": .85, "nausea": .7, "fever": .6, "loss_appetite": .8, "vomiting": .5},
    "GERD":               {"heartburn": .95, "chest_pain": .3, "cough": .2, "nausea": .3},
    "Myocardial infarction": {"chest_pain": .95, "arm_pain": .6, "shortness_of_breath": .6, "sweating": .65, "nausea": .4, "dizziness": .3},
    "Urinary tract infection": {"painful_urination": .9, "frequent_urination": .85, "abdominal_pain": .3, "fever": .25, "flank_pain": .2},
    "Kidney infection":   {"flank_pain": .85, "fever": .85, "painful_urination": .6, "frequent_urination": .5, "nausea": .5, "vomiting": .4, "chills": .6},
    "Iron-deficiency anemia": {"fatigue": .9, "pale_skin": .8, "dizziness": .6, "palpitations": .4, "shortness_of_breath": .3, "headache": .3},
    "Type 2 diabetes (hyperglycemia)": {"excessive_thirst": .85, "frequent_urination": .85, "fatigue": .6, "blurred_vision": .5, "weight_loss": .4, "dry_mouth": .5},
    "Dehydration":        {"dry_mouth": .8, "excessive_thirst": .8, "dizziness": .6, "fatigue": .6, "headache": .4, "frequent_urination": .05},
}
ADVICE = {  # urgency 1 = self-care, 2 = see a clinician soon, 3 = urgent / emergency
    "Common cold": (1, "Rest, fluids, symptom relief. See a clinician if it lasts >10 days or worsens."),
    "Influenza": (2, "Rest and fluids; antivirals work best within 48h of onset. Seek care if high-risk or breathing worsens."),
    "COVID-19": (2, "Test if possible, isolate, hydrate. Seek care for breathing difficulty or persistent high fever."),
    "Strep throat": (2, "Needs a throat swab; antibiotics are used if confirmed."),
    "Pneumonia": (3, "Needs medical assessment (exam, chest X-ray) the same day."),
    "Acute bronchitis": (1, "Usually viral; rest and fluids. See a clinician if cough lasts >3 weeks or fever persists."),
    "Asthma exacerbation": (3, "Use prescribed reliever inhaler; seek urgent care if no quick relief."),
    "Allergic rhinitis": (1, "Avoid triggers; antihistamines or nasal sprays may help."),
    "Migraine": (1, "Dark quiet room, hydration, usual migraine plan. Sudden 'worst ever' headache is an emergency."),
    "Meningitis": (3, "EMERGENCY: fever + stiff neck + headache needs immediate hospital evaluation."),
    "Gastroenteritis": (1, "Oral rehydration. Seek care for blood in stool, severe pain or inability to keep fluids down."),
    "Appendicitis": (3, "URGENT: right-lower abdominal pain with fever/nausea needs emergency surgical assessment."),
    "GERD": (1, "Smaller meals, avoid late eating, antacids. Chest pain with exertion needs cardiac evaluation."),
    "Myocardial infarction": (3, "EMERGENCY: call emergency services now. Do not drive yourself."),
    "Urinary tract infection": (2, "Needs urine test and usually antibiotics; see a clinician within 24-48h."),
    "Kidney infection": (3, "Needs prompt medical care (antibiotics, possibly IV)."),
    "Iron-deficiency anemia": (2, "Needs blood tests (CBC, ferritin) and a cause search."),
    "Type 2 diabetes (hyperglycemia)": (2, "Needs blood glucose / HbA1c testing soon."),
    "Dehydration": (1, "Oral rehydration solution; seek care if confused, very dizzy or not urinating."),
}
LEX = {  # symptom -> phrases used to recognise it in free text
    "fever": ["fever", "high temperature", "febrile", "temperature"], "chills": ["chills", "shivering", "shivers"],
    "cough": ["cough", "coughing"], "sputum": ["phlegm", "mucus", "sputum", "productive cough"],
    "sore_throat": ["sore throat", "throat pain", "painful swallowing", "scratchy throat"],
    "runny_nose": ["runny nose", "stuffy nose", "nasal congestion", "blocked nose", "congestion"],
    "sneezing": ["sneezing", "sneeze"], "headache": ["headache", "head pain", "head hurts"],
    "severe_headache": ["severe headache", "worst headache", "thunderclap"], "stiff_neck": ["stiff neck", "neck stiffness"],
    "photophobia": ["light sensitivity", "sensitive to light", "photophobia", "bright light hurts"],
    "nausea": ["nausea", "nauseous", "feel sick", "queasy"], "vomiting": ["vomiting", "throwing up", "vomit"],
    "diarrhea": ["diarrhea", "diarrhoea", "loose stool", "watery stool"],
    "abdominal_pain": ["stomach pain", "abdominal pain", "belly pain", "stomach ache", "tummy ache"],
    "right_lower_pain": ["right lower abdomen", "lower right abdomen", "right lower abdominal", "pain around belly button moving right"],
    "heartburn": ["heartburn", "acid reflux", "acid indigestion", "burning in chest"],
    "chest_pain": ["chest pain", "chest pressure", "chest tightness", "tight chest"],
    "shortness_of_breath": ["shortness of breath", "difficulty breathing", "breathless", "trouble breathing", "short of breath"],
    "wheezing": ["wheezing", "wheeze"], "fatigue": ["fatigue", "tired", "exhausted", "exhaustion", "no energy", "weakness"],
    "body_aches": ["body aches", "muscle pain", "myalgia", "aching muscles", "body ache"],
    "loss_smell": ["loss of smell", "cant smell", "can't smell", "loss of taste", "cant taste", "can't taste"],
    "dizziness": ["dizzy", "dizziness", "lightheaded", "light headed"], "pale_skin": ["pale", "pallor", "pale skin"],
    "palpitations": ["palpitations", "racing heart", "heart racing", "pounding heart"],
    "sweating": ["sweating", "cold sweat", "sweaty"], "arm_pain": ["left arm pain", "jaw pain", "pain radiating", "arm pain"],
    "frequent_urination": ["frequent urination", "urinating often", "peeing a lot", "urinate frequently"],
    "painful_urination": ["burning urination", "painful urination", "burns when i pee", "pain when urinating"],
    "excessive_thirst": ["very thirsty", "excessive thirst", "thirsty", "thirst"], "blurred_vision": ["blurred vision", "blurry vision"],
    "weight_loss": ["weight loss", "losing weight", "lost weight"], "itchy_eyes": ["itchy eyes", "watery eyes", "itchy watery eyes"],
    "dry_mouth": ["dry mouth"], "rash": ["rash", "skin rash"], "confusion": ["confusion", "confused", "disoriented"],
    "swollen_glands": ["swollen glands", "swollen lymph nodes", "swollen neck glands"],
    "loss_appetite": ["loss of appetite", "no appetite", "not hungry", "poor appetite"],
    "flank_pain": ["flank pain", "kidney pain", "side pain", "pain in my back and side"],
}
NEG_WORDS = {"no", "not", "without", "denies", "never", "nor", "negative"}
EMERGENCY_WORDS = ["suicid", "overdose", "unconscious", "face drooping", "can't breathe", "cannot breathe",
                   "seizure", "severe bleeding", "coughing blood", "stroke", "anaphylaxis", "blue lips"]
# Hard safety rules: (required symptoms, any-of symptoms, message). Never learned, never overridden.
RED_FLAGS = [
    ({"chest_pain"}, {"shortness_of_breath", "arm_pain", "sweating", "nausea"}, "Chest pain with breathlessness/sweating/arm-jaw pain may be a heart attack."),
    ({"fever", "stiff_neck"}, {"severe_headache", "headache", "photophobia", "confusion", "rash"}, "Fever + stiff neck + headache may be meningitis."),
    ({"severe_headache"}, {"confusion", "vomiting", "stiff_neck"}, "Severe headache with confusion/vomiting needs urgent assessment."),
    ({"right_lower_pain"}, {"fever", "nausea", "vomiting", "loss_appetite"}, "Right-lower abdominal pain with fever/nausea may be appendicitis."),
    ({"shortness_of_breath"}, {"confusion", "chest_pain"}, "Severe breathing difficulty with confusion/chest pain needs emergency care."),
    ({"confusion"}, {"fever", "excessive_thirst", "dizziness"}, "New confusion is a medical emergency sign."),
]

# --------------------------------------------------------------------------------------
# 2. NLP: symptom extraction with negation handling and typo tolerance
# --------------------------------------------------------------------------------------
LEX_EXTRA = {}  # learned vocabulary (persisted)

def lexicon():
    d = {k: list(v) for k, v in LEX.items()}
    for k, v in LEX_EXTRA.items():
        d.setdefault(k, []).extend(v)
    for k in list(d):
        d[k].append(k.replace("_", " "))
    return d

def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

def extract(text):
    t = " " + re.sub(r"[^a-z0-9' ]", " ", text.lower()) + " "
    t = re.sub(r"\s+", " ", t)
    lx = lexicon()
    present, denied = set(), set()
    for sym, syns in lx.items():
        for syn in syns:
            for m in re.finditer(r"\b" + re.escape(syn) + r"\b", t):
                before = t[:m.start()].split()[-3:]
                (denied if any(w in NEG_WORDS for w in before) else present).add(sym)
    singles = {syn: sym for sym, syns in lx.items() for syn in syns if " " not in syn and len(syn) >= 5}
    words = t.split()
    for i, w in enumerate(words):
        if len(w) >= 5 and w not in singles:
            hit = difflib.get_close_matches(w, list(singles), n=1, cutoff=0.86)
            if hit:
                neg = any(x in NEG_WORDS for x in words[max(0, i - 3):i])
                (denied if neg else present).add(singles[hit[0]])
    return present, denied - present

def emergency_text(text):
    low = text.lower()
    return [w for w in EMERGENCY_WORDS if w in low]

def red_flags(present, text=""):
    out = [m for req, anyof, m in RED_FLAGS if req <= present and (present & anyof)]
    if emergency_text(text):
        out.append("Your description contains an emergency keyword. Contact emergency services now.")
    return out

# --------------------------------------------------------------------------------------
# 3. THE MODEL: Naive-Bayes (instant learning) + softmax regression (evolved) ensemble
# --------------------------------------------------------------------------------------
class AceModel:
    def __init__(self):
        self.classes, self.W, self.b = [], {}, {}
        self.nb, self.prior = {}, {}       # nb[c] = {"n": int, "s": {sym: [present, observed]}}
        self.alpha, self.version = 0.5, 0   # alpha = weight of softmax in the ensemble

    # -- structure -------------------------------------------------------------------
    def add_class(self, name, profile=None):
        if name in self.W:
            return
        self.classes.append(name)
        self.W[name], self.b[name] = {}, 0.0
        self.nb[name] = {"n": 0, "s": {}}
        self.prior[name] = dict(profile or {})

    def vocab(self):
        v = set()
        for c in self.classes:
            v |= set(self.prior[c]) | set(self.nb[c]["s"])
        return sorted(v)

    def p_sym(self, c, s):
        """Learned P(symptom present | disease): seed prior blended with observed experience."""
        pres, obs = self.nb[c]["s"].get(s, [0, 0])
        p0 = self.prior[c].get(s, 0.03)
        return min(max((pres + NB_STRENGTH * p0) / (obs + NB_STRENGTH), 0.01), 0.99)

    # -- inference -------------------------------------------------------------------
    @staticmethod
    def _softmax(sc):
        m = max(sc.values())
        e = {k: math.exp(v - m) for k, v in sc.items()}
        z = sum(e.values())
        return {k: v / z for k, v in e.items()}

    def p_softmax(self, x):
        return self._softmax({c: self.b[c] + sum(self.W[c].get(s, 0.0) * v for s, v in x.items()) for c in self.classes})

    def p_nb(self, x):
        total = sum(self.nb[c]["n"] for c in self.classes)
        sc = {}
        for c in self.classes:
            lp = math.log((self.nb[c]["n"] + 1) / (total + len(self.classes)))
            for s, v in x.items():
                p = self.p_sym(c, s)
                lp += math.log(p if v > 0 else 1 - p)
            sc[c] = lp
        return self._softmax(sc)

    def predict(self, x):
        a, ps, pn = self.alpha, self.p_softmax(x), self.p_nb(x)
        return {c: a * ps[c] + (1 - a) * pn[c] for c in self.classes}

    # -- learning --------------------------------------------------------------------
    def observe(self, x, label):
        """Instant experience update of the Naive-Bayes layer."""
        n = self.nb[label]
        n["n"] += 1
        for s, v in x.items():
            e = n["s"].setdefault(s, [0, 0])
            e[1] += 1
            e[0] += 1 if v > 0 else 0

    def sgd(self, x, label, lr=0.08, l2=1e-4):
        p = self.p_softmax(x)
        for c in self.classes:
            g = p[c] - (1.0 if c == label else 0.0)
            self.b[c] -= lr * g
            for s, v in x.items():
                self.W[c][s] = self.W[c].get(s, 0.0) - lr * (g * v + l2 * self.W[c].get(s, 0.0))

    def sample(self, rng, label=None):
        """Generative replay: synthesise a partially observed patient from what the model has learned."""
        label = label or rng.choice(self.classes)
        reveal_pos, reveal_neg = rng.uniform(.35, .95), rng.uniform(.05, .35)
        x = {}
        for s in self.vocab():
            has = rng.random() < self.p_sym(label, s)
            if rng.random() < (reveal_pos if has else reveal_neg):
                x[s] = 1 if has else -1
        if not any(v > 0 for v in x.values()):
            return self.sample(rng, label)
        return x, label

    def train_softmax(self, replay, epochs=6, lr=0.08, rng=None):
        rng = rng or random.Random()
        for ep in range(epochs):
            rng.shuffle(replay)
            for x, y in replay:
                self.sgd(x, y, lr=lr * (0.85 ** ep))

    # -- persistence -----------------------------------------------------------------
    def dumps(self):
        return json.dumps(self.__dict__)

    @staticmethod
    def loads(s):
        m = AceModel()
        m.__dict__.update(json.loads(s))
        return m

def evaluate(model, data):
    if not data:
        return 9.9, 0.0
    ll = acc = 0.0
    for x, y in data:
        p = model.predict(x)
        ll -= math.log(max(p[y], 1e-9))
        acc += max(p, key=p.get) == y
    return ll / len(data), acc / len(data)

def build_seed_model():
    m = AceModel()
    for name, prof in DISEASES.items():
        m.add_class(name, prof)
    rng = random.Random(42)
    m.train_softmax([m.sample(rng) for _ in range(5000)], epochs=5, rng=rng)
    m.version = 1
    return m

_gold_rng = random.Random(7)
_seed_for_gold = AceModel()
for _n, _p in DISEASES.items():
    _seed_for_gold.add_class(_n, _p)
GOLD = [_seed_for_gold.sample(_gold_rng) for _ in range(400)]   # fixed benchmark of seed knowledge

# --------------------------------------------------------------------------------------
# 4. ENGINE: experience memory, active questioning, safe self-evolution (champion/challenger)
# --------------------------------------------------------------------------------------
class Engine:
    def __init__(self):
        self.lock = threading.RLock()
        self.db = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS cases(id INTEGER PRIMARY KEY, ts REAL, x TEXT, predicted TEXT, conf REAL, confirmed TEXT, used INTEGER DEFAULT 0);
            CREATE TABLE IF NOT EXISTS evo(id INTEGER PRIMARY KEY, ts REAL, version INTEGER, promoted INTEGER, champ_loss REAL, cand_loss REAL, champ_acc REAL, cand_acc REAL, n_cases INTEGER, alpha REAL);
            CREATE TABLE IF NOT EXISTS kv(k TEXT PRIMARY KEY, v TEXT);""")
        row = self.db.execute("SELECT v FROM kv WHERE k='model'").fetchone()
        self.model = AceModel.loads(row[0]) if row else build_seed_model()
        lx = self.db.execute("SELECT v FROM kv WHERE k='lex'").fetchone()
        if lx:
            LEX_EXTRA.update(json.loads(lx[0]))
        self.evolving = False
        self._save()

    def _save(self):
        with self.lock:
            self.db.execute("REPLACE INTO kv VALUES('model',?)", (self.model.dumps(),))
            self.db.execute("REPLACE INTO kv VALUES('lex',?)", (json.dumps(LEX_EXTRA),))
            self.db.commit()

    # -- diagnosis -------------------------------------------------------------------
    def diagnose(self, text, present, denied, asked):
        with self.lock:
            p_new, d_new = extract(text or "")
            present, denied = set(present) | p_new, (set(denied) | d_new)
            denied -= present
            x = {**{s: -1 for s in denied}, **{s: 1 for s in present}}
            flags = red_flags(present, text or "")
            if not present:
                return {"present": [], "denied": sorted(denied), "ranking": [], "question": None, "flags": flags, "version": self.model.version}
            probs = self.model.predict(x)
            ranking = sorted(probs.items(), key=lambda kv: -kv[1])[:6]
            out = [{"disease": c, "prob": round(p, 4), "urgency": ADVICE.get(c, (2, "Consult a clinician."))[0],
                    "advice": ADVICE.get(c, (2, "Consult a clinician."))[1]} for c, p in ranking]
            if flags:  # safety layer escalates regardless of model output
                for o in out:
                    o["urgency"] = 3
            return {"present": sorted(present), "denied": sorted(denied), "ranking": out,
                    "question": self.next_question(probs, x, set(asked) | set(x)), "flags": flags,
                    "version": self.model.version}

    def next_question(self, probs, x, skip):
        """Choose the symptom whose answer is expected to reduce diagnostic uncertainty the most."""
        ent = lambda d: -sum(v * math.log(v) for v in d if v > 0)
        h0 = ent(probs.values())
        if max(probs.values()) > 0.92:
            return None
        best, best_gain = None, 0.02
        for s in self.model.vocab():
            if s in skip:
                continue
            py = sum(probs[c] * self.model.p_sym(c, s) for c in probs)
            if not 0.02 < py < 0.98:
                continue
            yes = [probs[c] * self.model.p_sym(c, s) / py for c in probs]
            no = [probs[c] * (1 - self.model.p_sym(c, s)) / (1 - py) for c in probs]
            gain = h0 - (py * ent(yes) + (1 - py) * ent(no))
            if gain > best_gain:
                best, best_gain = s, gain
        return {"symptom": best, "text": "Do you have " + best.replace("_", " ") + "?", "gain": round(best_gain, 3)} if best else None

    # -- experience ------------------------------------------------------------------
    def feedback(self, present, denied, confirmed, predicted="", conf=0.0, new_symptoms=()):
        with self.lock:
            confirmed = re.sub(r"\s+", " ", confirmed.strip())[:80]
            if not confirmed:
                raise ValueError("confirmed diagnosis required")
            present = set(present)
            for ns in new_symptoms:                       # vocabulary growth
                k = slug(ns)
                if k:
                    LEX_EXTRA.setdefault(k, [ns.lower().strip()])
                    present.add(k)
            match = next((c for c in self.model.classes if c.lower() == confirmed.lower()), None)
            if match is None:                             # disease growth
                self.model.add_class(confirmed)
                match = confirmed
            x = {**{s: -1 for s in denied if s not in present}, **{s: 1 for s in present}}
            cur = self.db.execute("INSERT INTO cases(ts,x,predicted,conf,confirmed) VALUES(?,?,?,?,?)",
                                  (time.time(), json.dumps(x), predicted, conf, match))
            cid = cur.lastrowid
            if cid % HOLDOUT_MOD != 0:                    # instant learning (hold-out cases stay unseen)
                self.model.observe(x, match)
                self.model.sgd(x, match, lr=0.05)
            self.db.commit()
            self._save()
            pending = self.db.execute("SELECT COUNT(*) FROM cases WHERE used=0").fetchone()[0]
            if pending >= EVOLVE_EVERY and not self.evolving:
                threading.Thread(target=self.evolve, daemon=True).start()
            return {"case_id": cid, "disease": match, "pending": pending, "next_evolution_in": max(0, EVOLVE_EVERY - pending)}

    # -- self-evolution --------------------------------------------------------------
    def evolve(self, force=False):
        with self.lock:
            if self.evolving:
                return {"status": "busy"}
            self.evolving = True
        try:
            rows = self.db.execute("SELECT id,x,confirmed FROM cases").fetchall()
            exp = [(i, {k: v for k, v in json.loads(x).items()}, y) for i, x, y in rows]
            train = [(x, y) for i, x, y in exp if i % HOLDOUT_MOD != 0]
            hold = [(x, y) for i, x, y in exp if i % HOLDOUT_MOD == 0]
            champ = self.model
            cand = copy.deepcopy(champ)
            rng = random.Random(int(time.time()))
            # experience replay (x3 weight) + generative replay of everything learned so far
            replay = train * 3 + [champ.sample(rng) for _ in range(3000 + 100 * len(train))]
            cand.train_softmax(replay, epochs=5, lr=0.06, rng=rng)

            def score(m):  # combined objective: real experience matters, old knowledge must not be forgotten
                gl, ga = evaluate(m, GOLD)
                if len(hold) >= 3:
                    el, ea = evaluate(m, hold)
                    return 0.6 * el + 0.4 * gl, 0.6 * ea + 0.4 * ga
                return gl, ga
            best_alpha, best_ll = cand.alpha, 9e9                   # auto-tune ensemble weight
            for a in (0.2, 0.4, 0.5, 0.6, 0.8):
                cand.alpha = a
                ll, _ = score(cand)
                if ll < best_ll:
                    best_alpha, best_ll = a, ll
            cand.alpha = best_alpha
            champ_loss, champ_acc = score(champ)
            cand_loss, cand_acc = score(cand)
            promote = cand_loss <= champ_loss + 0.005               # champion / challenger gate
            with self.lock:
                if promote:
                    cand.version = champ.version + 1
                    self.model = cand
                    self._save()
                self.db.execute("UPDATE cases SET used=1")
                self.db.execute("INSERT INTO evo(ts,version,promoted,champ_loss,cand_loss,champ_acc,cand_acc,n_cases,alpha) VALUES(?,?,?,?,?,?,?,?,?)",
                                (time.time(), self.model.version, int(promote), champ_loss, cand_loss, champ_acc, cand_acc, len(exp), cand.alpha))
                self.db.commit()
            return {"status": "promoted" if promote else "rolled_back", "champ_loss": round(champ_loss, 4), "cand_loss": round(cand_loss, 4)}
        finally:
            self.evolving = False

    def status(self):
        with self.lock:
            n = self.db.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
            pend = self.db.execute("SELECT COUNT(*) FROM cases WHERE used=0").fetchone()[0]
            hist = self.db.execute("SELECT ts,version,promoted,champ_loss,cand_loss,cand_acc FROM evo ORDER BY id DESC LIMIT 8").fetchall()
            gl, ga = evaluate(self.model, GOLD)
            conf = self.db.execute("SELECT predicted,confirmed FROM cases WHERE predicted!='' ORDER BY id DESC LIMIT 50").fetchall()
            live = sum(1 for p, c in conf if p == c) / len(conf) if conf else None
            return {"version": self.model.version, "diseases": len(self.model.classes), "symptoms": len(self.model.vocab()),
                    "cases": n, "pending": pend, "evolving": self.evolving, "alpha": self.model.alpha,
                    "benchmark_acc": round(ga, 3), "live_acc_last50": None if live is None else round(live, 3),
                    "history": [{"time": time.strftime("%m-%d %H:%M", time.localtime(h[0])), "version": h[1], "promoted": bool(h[2]),
                                 "champ_loss": round(h[3], 3), "cand_loss": round(h[4], 3), "acc": round(h[5], 3)} for h in hist],
                    "diagnoses": sorted(self.model.classes)}

ENGINE = None

# --------------------------------------------------------------------------------------
# 5. HTTP BACKEND + FRONTEND
# --------------------------------------------------------------------------------------
PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ace Medic</title><style>
:root{--bg:#0b1220;--card:#121b2e;--line:#1f2c47;--tx:#e6edf7;--mut:#8b9bb8;--acc:#3ddc97;--warn:#ffb020;--bad:#ff5d5d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);font:15px/1.5 system-ui,Segoe UI,Roboto,sans-serif}
header{padding:18px 24px;border-bottom:1px solid var(--line);display:flex;gap:12px;align-items:center}
header b{font-size:20px;letter-spacing:.5px}header span{color:var(--acc)}small,.mut{color:var(--mut)}
main{display:grid;grid-template-columns:minmax(0,2fr) minmax(0,1fr);gap:18px;padding:18px;max-width:1200px;margin:auto}
@media(max-width:850px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px;margin-bottom:16px}
h3{margin:0 0 10px;font-size:13px;text-transform:uppercase;letter-spacing:1px;color:var(--mut)}
textarea,input{width:100%;background:#0d1626;color:var(--tx);border:1px solid var(--line);border-radius:10px;padding:10px;font:inherit}
button{background:var(--acc);color:#04210f;border:0;border-radius:10px;padding:9px 16px;font-weight:700;cursor:pointer}
button.g{background:#1b2a47;color:var(--tx)}button:disabled{opacity:.5}.row{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}
.chip{padding:3px 10px;border-radius:99px;background:#1b2a47;font-size:13px}.chip.n{background:#3a1f28;text-decoration:line-through}
.bar{height:8px;border-radius:5px;background:#1b2a47;overflow:hidden;margin:4px 0 8px}.bar i{display:block;height:100%;background:var(--acc)}
.u3{color:var(--bad)}.u2{color:var(--warn)}.u1{color:var(--acc)}
.flag{background:#3a1519;border:1px solid var(--bad);color:#ffd5d5;padding:12px;border-radius:10px;margin-bottom:12px;font-weight:600}
.disc{font-size:12px;color:var(--mut)}table{width:100%;font-size:12px;border-collapse:collapse}td,th{padding:3px 4px;text-align:left;border-bottom:1px solid var(--line)}
</style></head><body>
<header><b>ACE <span>MEDIC</span></b><small>self-improving clinical decision support (educational prototype)</small></header>
<main><div>
<div class="card"><h3>Describe the symptoms</h3>
<textarea id="t" rows="3" placeholder="e.g. I have had a fever and a bad cough for 3 days, body aches, no sore throat"></textarea>
<div class="row"><button id="go">Analyze</button><button class="g" id="reset">New patient</button></div>
<div class="row" id="chips"></div></div>
<div id="flags"></div>
<div class="card" id="qcard" style="display:none"><h3>Follow-up question</h3><div id="q" style="font-size:17px"></div>
<div class="row"><button id="yes">Yes</button><button class="g" id="no">No</button><button class="g" id="skip">Not sure</button></div></div>
<div class="card" id="res" style="display:none"><h3>Possible conditions</h3><div id="rank"></div></div>
<div class="card" id="fb" style="display:none"><h3>Teach Ace Medic (clinician confirmation)</h3>
<small>When the real diagnosis is known, confirm it. The model learns from it and evolves itself.</small>
<input id="conf" list="dl" placeholder="Confirmed diagnosis (existing or new)" style="margin-top:8px"><datalist id="dl"></datalist>
<input id="newsym" placeholder="Extra symptoms not recognised (comma separated, optional)" style="margin-top:8px">
<div class="row"><button id="send">Submit confirmed case</button></div><div id="fbmsg" class="mut" style="margin-top:8px"></div></div>
<p class="disc">Ace Medic is an educational decision-support prototype, not a medical device. It does not replace a doctor. In an emergency call your local emergency number.</p>
</div><div>
<div class="card"><h3>Model brain</h3><div id="st" class="mut">loading...</div>
<div class="row"><button class="g" id="evo">Run self-evolution now</button></div><div id="evomsg" class="mut" style="margin-top:8px"></div></div>
<div class="card"><h3>Evolution history</h3><table id="hist"></table></div></div></main>
<script>
const $=id=>document.getElementById(id);let S={present:[],denied:[],asked:[]},last=null,curQ=null;
const h=(t,c,x)=>{const e=document.createElement(t);if(c)e.className=c;if(x!==undefined)e.textContent=x;return e};
const api=(u,b)=>fetch(u,b?{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(b)}:{}).then(r=>r.json());
const pretty=s=>s.replace(/_/g," ");
async function run(text){
  const r=await api("/api/diagnose",{text:text||"",present:S.present,denied:S.denied,asked:S.asked});last=r;
  S.present=r.present;S.denied=r.denied;
  const c=$("chips");c.replaceChildren();r.present.forEach(s=>c.append(h("span","chip",pretty(s))));r.denied.forEach(s=>c.append(h("span","chip n",pretty(s))));
  const f=$("flags");f.replaceChildren();r.flags.forEach(m=>f.append(h("div","flag","URGENT: "+m)));
  const rk=$("rank");rk.replaceChildren();$("res").style.display=r.ranking.length?"block":"none";$("fb").style.display=r.ranking.length?"block":"none";
  r.ranking.forEach(o=>{const d=h("div");const t=h("div");t.append(h("b","u"+o.urgency,o.disease+" "),h("span","mut",(o.prob*100).toFixed(1)+"%"));
    const b=h("div","bar");const i=h("i");i.style.width=(o.prob*100)+"%";b.append(i);d.append(t,b,h("small","mut",o.advice));rk.append(d)});
  curQ=r.question;$("qcard").style.display=curQ?"block":"none";if(curQ)$("q").textContent=curQ.text;
}
async function answer(v){if(!curQ)return;S.asked.push(curQ.symptom);
  if(v===1&&!S.present.includes(curQ.symptom))S.present.push(curQ.symptom);
  if(v===-1&&!S.denied.includes(curQ.symptom))S.denied.push(curQ.symptom);await run("")}
$("go").onclick=()=>{const t=$("t").value.trim();if(t){run(t);$("t").value=""}};
$("yes").onclick=()=>answer(1);$("no").onclick=()=>answer(-1);$("skip").onclick=()=>answer(0);
$("reset").onclick=()=>{S={present:[],denied:[],asked:[]};last=null;["chips","flags","rank"].forEach(i=>$(i).replaceChildren());["qcard","res","fb"].forEach(i=>$(i).style.display="none")};
$("send").onclick=async()=>{const c=$("conf").value.trim();if(!c)return;const top=last&&last.ranking[0];
  const r=await api("/api/feedback",{present:S.present,denied:S.denied,confirmed:c,predicted:top?top.disease:"",conf:top?top.prob:0,
    new_symptoms:$("newsym").value.split(",").map(s=>s.trim()).filter(Boolean)});
  $("fbmsg").textContent=r.error?r.error:"Learned case #"+r.case_id+" as "+r.disease+". Next self-evolution in "+r.next_evolution_in+" case(s).";status()};
$("evo").onclick=async()=>{$("evomsg").textContent="evolving...";const r=await api("/api/evolve",{});$("evomsg").textContent=r.status+(r.cand_loss?" (loss "+r.champ_loss+" -> "+r.cand_loss+")":"");status()};
async function status(){const s=await api("/api/status");
  $("st").textContent="";[["Version","v"+s.version],["Diseases known",s.diseases],["Symptoms known",s.symptoms],["Experience cases",s.cases+" ("+s.pending+" pending)"],
  ["Benchmark accuracy",s.benchmark_acc],["Live accuracy (last 50)",s.live_acc_last50===null?"n/a":s.live_acc_last50],["Ensemble weight",s.alpha]].forEach(([k,v])=>{
    const d=h("div");d.append(h("b","",k+": "),h("span","",String(v)));$("st").append(d)});
  const t=$("hist");t.replaceChildren();const hd=h("tr");["time","ver","result","loss","acc"].forEach(x=>hd.append(h("th","",x)));t.append(hd);
  s.history.forEach(e=>{const r=h("tr");[e.time,"v"+e.version,e.promoted?"promoted":"rolled back",e.champ_loss+">"+e.cand_loss,e.acc].forEach(x=>r.append(h("td","",x)));t.append(r)});
  const dl=$("dl");dl.replaceChildren();s.diagnoses.forEach(d=>{const o=h("option");o.value=d;dl.append(o)})}
status();setInterval(status,8000);
</script></body></html>"""

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else (json.dumps(body) if ctype == "application/json" else body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/":
            self._send(200, PAGE, "text/html")
        elif self.path == "/api/status":
            self._send(200, ENGINE.status())
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n > 100_000:
                return self._send(413, {"error": "too large"})
            b = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/diagnose":
                r = ENGINE.diagnose(str(b.get("text", ""))[:2000], b.get("present", []), b.get("denied", []), b.get("asked", []))
            elif self.path == "/api/feedback":
                r = ENGINE.feedback(b.get("present", []), b.get("denied", []), str(b.get("confirmed", "")), str(b.get("predicted", "")),
                                    float(b.get("conf", 0) or 0), [str(s)[:40] for s in b.get("new_symptoms", [])][:5])
            elif self.path == "/api/evolve":
                r = ENGINE.evolve(force=True)
            else:
                return self._send(404, {"error": "not found"})
            self._send(200, r)
        except Exception as e:  # keep the server alive, report the problem
            self._send(400, {"error": str(e)})

def main():
    global ENGINE
    print("Booting Ace Medic (pre-training seed model on first run)...")
    ENGINE = Engine()
    print(f"Ace Medic v{ENGINE.model.version} ready -> http://localhost:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()

if __name__ == "__main__":
    main()
