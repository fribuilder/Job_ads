# Segmentation Issues — spaCy Pipeline (post-refinement)

Recorded after commit `1797b44` (refined boundary rules).
These are **known issues to fix later** — the time-range fix was the priority and is working correctly.

---

## Issue 1: Verb-conjunction over-splitting (caused by `xcomp` removal)

Removing `xcomp` from `_SHARED_CONTEXT_DEPS` in `spacy_utils.py` allows spaCy to split
coordinate verb phrases that share a subject. Each case below is one original sentence
broken at a coordination comma.

### Doc: Practical Nurse — Outpatient

**Segs 3–4**

Original:
> The LPN performs, reports, and records assigned nursing care in the Specialty Clinics including, but not limited to: Medication administration to include: PO, IM, SQ Vital sign data collection, including height and weight Prepare the Patient for the Specialty clinic appointment.

Split into:
- Seg 3: `The LPN performs, reports`
- Seg 4: `records assigned nursing care in the Specialty Clinics including, but not limited to: Medication administration to include: PO, IM, SQ Vital sign data collection, including height and weight Prepare the Patient for the Specialty clinic appointment.`

---

**Segs 40–41**

Original:
> Knowledge and ability to recognize the need for and to institute emergency measures when indicated, promptly seek the assistance of the RN or MD/DO, and assist in resuscitation procedures in cardiac and/or pulmonary arrest.

Split into:
- Seg 40: `Knowledge and ability to recognize the need for and to institute emergency measures when indicated, promptly seek the assistance of the RN or MD/DO`
- Seg 41: `assist in resuscitation procedures in cardiac and/or pulmonary arrest.`

---

**Segs 52–53**

Original:
> Knowledge and ability to recognize urgent or emergent patient care situations, seek assistance of the RN and/or MD/DO, and initiate appropriate emergency interventions as directed.

Split into:
- Seg 52: `Knowledge and ability to recognize urgent or emergent patient care situations, seek assistance of the RN and/or MD/DO`
- Seg 53: `initiate appropriate emergency interventions as directed.`

---

**Segs 58–60**

Original:
> Knowledge and skill in performing support duties for complex diagnostic tests and/or specialized practices or procedures, which include preparing the patient, assisting in the diagnostic examination, preparing and handling specialized instruments or other specialized equipment, and monitoring the patient's condition before, during, and following the procedure.

Split into:
- Seg 58: `Knowledge and skill in performing support duties for complex diagnostic tests and/or specialized practices or procedures, which include preparing the patient`
- Seg 59: `assisting in the diagnostic examination, preparing and handling specialized instruments or other specialized equipment`
- Seg 60: `monitoring the patient's condition before, during, and following the procedure.`

---

### Doc: Supervisory Health System Specialist (Business Operations Director)

**Segs 17–18**

Original:
> The grade may have been in any occupation, but must have been held in the Federal service.

Split into:
- Seg 17: `The grade may have been in any occupation`
- Seg 18: `must have been held in the Federal service.`

---

**Segs 27–30** (also involves `compound` addition to `has_own_structure`)

Original:
> Work must have involved a close working relationship with facility managers and analysis and/or coordination of administrative, clinical, or other service activities, and provided knowledge of the following: Missions, organizations, programs, and requirements of health care delivery systems; Regulations and standards of various regulatory and credentialing groups; and Government-wide, agency, and facility systems and requirements in various administrative areas such as budget, personnel, and procurement.

Split into:
- Seg 27: `Work must have involved a close working relationship with facility managers and analysis and/or coordination of administrative, clinical, or other service activities`
- Seg 28: `provided knowledge of the following: Missions, organizations, programs`
- Seg 29: `requirements of health care delivery systems`
- Seg 30: `Regulations and standards of various regulatory and credentialing groups and Government-wide, agency, and facility systems and requirements in various administrative areas such as budget, personnel, and procurement.`

---

## Planned Fix

1. **Re-add `xcomp` to `_SHARED_CONTEXT_DEPS`** in `segmenters/spacy_utils.py` — the primary cause of Issues 1–5.
2. **Remove `compound` from `has_own_structure`** in `segmenters/spacy_utils.py` — contributing cause of Issue 6.
3. **Apply lowercase-continuation merge** to the output of `split_by_coordination` inside `segmenters/spacy_segmenter.py`, not only inside `_postprocess` in `nltk_splitter.py`. Segs 41, 53, 18, 28–29 all start with lowercase and would be caught by this check.

---

## Separate (higher-priority) issue: bullet-list merging

All list items that were originally newline-separated in the HTML source are merged into
one run-on segment because `_basic_normalize` in `preprocessing/cleaner.py` (line 74)
replaces `\n` with a space before any segmentation logic runs.

Example — original HTML list:
```
The LPN performs... including, but not limited to:
  Medication administration to include: PO, IM, SQ
  Vital sign data collection, including height and weight
  Prepare the Patient for the Specialty clinic appointment.
  Clinical reminders as indicated
  Check expiration dates of medicines and supplies
  ...
```

Arrives at segmenter as one flat string with no recoverable boundaries.

### Proposed fix (Option A — preferred)

In `preprocessing/cleaner.py`, change `_basic_normalize` to convert `\n` to `. ` instead
of ` `, so newline-separated list items become sentence boundaries before NLTK sees them.
Requires care around double-period cleanup and already-terminated lines.
