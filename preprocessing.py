"""Data correction, column selection, scoring, and participant-level assembly.

Ports the data preparation in scripts/00a-pre-processing.R,
scripts/00b-pre-clean.R, and reports/01-preparing-data.qmd.
"""
from __future__ import annotations

import logging
import re
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
TASKS = ("digit_span", "spatial_span", "melody_span", "nback",
         "raven", "proms", "questionnaires")
MATCHED_ID = "subject_ID of the matched participant"
ID_AS_RECORDED = "subject_ID (as written in the weekly log)"


def clean_id(value: Any) -> str | None:
    """Lowercase an ID and remove embedded whitespace."""
    if pd.isna(value):
        return None
    value = re.sub(r"\s+", "", str(value)).lower()
    return value or None


def _read_csv(path: Path) -> pd.DataFrame:
    last_error = None
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            frame = pd.read_csv(path, encoding=encoding, low_memory=False)
            frame.columns = [str(c).replace("\ufeff", "").strip() for c in frame.columns]
            return frame
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            last_error = exc
    raise ValueError(f"Cannot read {path.name}: {last_error}")


def _read_jspsych(path: Path) -> pd.DataFrame:
    last_error = None
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            lines = path.read_text(encoding=encoding).splitlines()
            # Preserve the first header and remove subsequent metadata/header rows.
            lines = [line for i, line in enumerate(lines)
                     if i == 0 or not line.startswith("date")]
            frame = pd.read_csv(StringIO("\n".join(lines)), sep=";", low_memory=False)
            frame.columns = [str(c).replace("\ufeff", "").strip() for c in frame.columns]
            return frame
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            last_error = exc
    raise ValueError(f"Cannot read {path.name}: {last_error}")


def _detect_task(frame: pd.DataFrame, path: Path) -> str | None:
    cols = {str(c).strip().casefold() for c in frame.columns}
    if "letter" in cols:
        return "nback"
    if "melody1" in cols:
        return "melody_span"
    if "stimulus" in cols:
        return "digit_span"
    if "sequence" in cols:
        return "spatial_span"
    if "s2m36correct" in cols:
        return "raven"
    if "promsmelodystti" in cols:
        return "proms"
    if "ebmrqtime" in cols or len(frame.columns) == 282:
        return "questionnaires"
    if any(c.startswith(("bfi[", "ses", "ebmrq[")) for c in cols):
        return "questionnaires"
    stem = path.stem.casefold().replace("_", "").replace("-", "")
    for task in TASKS:
        if task.replace("_", "") in stem:
            return task
    if "questionnaire" in stem:
        return "questionnaires"
    return None


def load_id_master(path: Path) -> pd.DataFrame:
    """Combine non-criteria sheets in the participant ID workbook."""
    excel = pd.ExcelFile(path)
    parts = []
    for sheet in excel.sheet_names:
        if "criteria" in sheet.casefold():
            continue
        frame = pd.read_excel(excel, sheet_name=sheet)
        if frame.empty:
            continue
        frame.columns = [str(c).replace("\ufeff", "").strip() for c in frame.columns]
        frame["sheet_name"] = sheet
        parts.append(frame)
    if not parts:
        raise ValueError(f"No participant sheets found in {path}")
    master = pd.concat(parts, ignore_index=True, sort=False)
    if ID_AS_RECORDED not in master:
        raise KeyError(f"Missing ID workbook column: {ID_AS_RECORDED}")
    master = master.rename(columns={ID_AS_RECORDED: "subject_ID"})
    rename = {
        col: re.sub(r"\s+", "_", col.strip()).lower()
        for col in master.columns
        if col != "subject_ID" and not col.startswith("subject_ID")
    }
    master = master.rename(columns=rename)
    if "true_subject_id" not in master:
        raise KeyError("ID workbook has no true subject_ID column")
    for col in ("subject_ID", "true_subject_id", MATCHED_ID):
        if col in master:
            master[col] = master[col].map(clean_id)
    return master.dropna(subset=["subject_ID", "true_subject_id"]).copy()


def _id_lookup(master: pd.DataFrame) -> pd.DataFrame:
    fields = [c for c in ("sex", "age", "group", "country", "unit",
                          "years_of_education", MATCHED_ID) if c in master]
    records = []
    for row in master.to_dict("records"):
        canonical = row["true_subject_id"]
        metadata = {c: row.get(c, np.nan) for c in fields}
        for alias in (row.get("subject_ID"), canonical):
            if alias:
                records.append({"alias": alias, "true_subject_id": canonical, **metadata})
    lookup = pd.DataFrame(records).drop_duplicates()
    if lookup.empty:
        raise ValueError("ID workbook contains no usable ID mappings")
    conflicts = lookup.groupby("alias")["true_subject_id"].nunique()
    conflicts = conflicts[conflicts > 1]
    if not conflicts.empty:
        raise ValueError(f"{len(conflicts)} IDs map to multiple canonical participant IDs")
    return lookup.drop_duplicates("alias").set_index("alias")


def _standardise_task_columns(frame: pd.DataFrame, task: str) -> pd.DataFrame:
    frame = frame.copy()
    by_lower = {str(c).casefold(): c for c in frame.columns}
    renames = {}
    for source, target in (("university", "unit"),
                           ("participantid", "subject_ID"),
                           ("edu[years]", "years_of_education")):
        if source in by_lower and target not in frame.columns:
            renames[by_lower[source]] = target
    if renames:
        frame = frame.rename(columns=renames)
    if "subject_ID" not in frame:
        candidate = next((c for c in frame.columns
                          if str(c).casefold() in ("subject_id", "participantid")), None)
        if candidate is not None:
            frame = frame.rename(columns={candidate: "subject_ID"})
    if "subject_ID" not in frame:
        raise KeyError(f"{task} export is missing subject_ID")
    return frame


def fix_raw_sources(project_root: Path, output_root: Path) -> dict[str, pd.DataFrame]:
    """Merge raw task exports, fix IDs/demographics, and save intermediate CSVs."""
    raw_dir = project_root / "data" / "raw-before-fixing"
    if not raw_dir.exists():
        raise FileNotFoundError(raw_dir)
    pieces: dict[str, list[pd.DataFrame]] = {task: [] for task in TASKS}

    for path in sorted(raw_dir.glob("*.txt")):
        frame = _read_jspsych(path)
        task = _detect_task(frame, path)
        if task == "melody_span" and len(frame) == 7459:
            task = "melody_span_old"
        if task == "melody_span_old":
            task = "melody_span"
        if task in pieces:
            pieces[task].append(frame)
        else:
            LOGGER.warning("Skipping unrecognised text export %s", path.name)

    for path in sorted(raw_dir.glob("*.csv")):
        frame = _read_csv(path)
        task = _detect_task(frame, path)
        if task in pieces:
            pieces[task].append(frame)
        else:
            LOGGER.warning("Skipping unrecognised CSV export %s", path.name)

    merged = {}
    for task, frames in pieces.items():
        if not frames:
            continue
        if task == "questionnaires":
            frames = [
                frame.loc[:, [c for c in frame.columns
                              if not str(c).startswith("groupTime")]]
                for frame in frames
            ]
        merged[task] = pd.concat(frames, ignore_index=True, sort=False)

    missing = set(TASKS).difference(merged)
    if missing:
        raise ValueError("Could not identify raw task(s): " + ", ".join(sorted(missing)))

    master = load_id_master(project_root / "data" / "id" / "id.xlsx")
    lookup = _id_lookup(master)
    corrected = {}
    for task, original in merged.items():
        frame = _standardise_task_columns(original, task)
        ids = frame["subject_ID"].map(clean_id)
        valid = ids.isin(lookup.index)
        frame = frame.loc[valid].copy()
        ids = ids.loc[valid]
        frame["subject_ID"] = ids.map(lookup["true_subject_id"]).to_numpy()
        for col in lookup.columns:
            if col != "true_subject_id":
                frame[col] = ids.map(lookup[col]).to_numpy()
        corrected[task] = frame.reset_index(drop=True)
        LOGGER.info("%s: retained %s ID-verified rows", task, len(frame))

    out_dir = output_root / "data" / "raw-before-selection"
    out_dir.mkdir(parents=True, exist_ok=True)
    for task, frame in corrected.items():
        frame.to_csv(out_dir / f"{task}.csv", index=False)
    return corrected


def load_dictionary(path: Path) -> dict[str, pd.DataFrame]:
    excel = pd.ExcelFile(path)
    return {sheet: pd.read_excel(excel, sheet_name=sheet)
            for sheet in excel.sheet_names}


def select_raw_columns(
    corrected: dict[str, pd.DataFrame], dictionary_path: Path, output_root: Path
) -> dict[str, pd.DataFrame]:
    """Apply the keep=1 entries in each task's data-dictionary sheet."""
    dictionaries = load_dictionary(dictionary_path)
    out_dir = output_root / "data" / "raw-after-selection"
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = {}
    for task, frame in corrected.items():
        if task not in dictionaries:
            raise KeyError(f"No dictionary sheet for {task}")
        sheet = dictionaries[task]
        keep = pd.to_numeric(sheet["keep"], errors="coerce").eq(1)
        cols = sheet.loc[keep, "column"].astype(str).tolist()
        absent = [c for c in cols if c not in frame]
        if absent:
            raise KeyError(f"{task} is missing dictionary field(s): {', '.join(absent[:10])}")
        selected[task] = frame.loc[:, cols].copy()
        selected[task].to_csv(out_dir / f"{task}.csv", index=False)
    return selected


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _strict_sum(series: pd.Series) -> float:
    values = _numeric(series)
    if values.isna().any():
        return np.nan
    return float(values.sum())


def _score_block(frame: pd.DataFrame, block_col: str, correct_col: str, score_name: str) -> pd.DataFrame:
    work = frame.loc[_numeric(frame[block_col]).ne(0),
                     ["subject_ID", block_col, correct_col]]
    block = (work.groupby(["subject_ID", block_col], dropna=False)[correct_col]
             .apply(_strict_sum).rename("score").reset_index())
    result = block.groupby("subject_ID", as_index=False)["score"].mean()
    return result.rename(columns={"score": score_name})


def _map_responses(series: pd.Series, mapping: dict[str, int]) -> pd.Series:
    values = (series.astype("string").str.strip().str.casefold()
              .str.replace(r"\s+", "", regex=True))
    values = values.replace({"<na>": pd.NA, "nan": pd.NA, "none": pd.NA})
    canonical = {re.sub(r"\s+", "", str(k)).casefold(): v for k, v in mapping.items()}
    return values.map(canonical).astype(float)


def _score_questionnaires(q: pd.DataFrame, fixing_path: Path) -> dict[str, pd.DataFrame]:
    q = q.copy()
    if "subject_ID" not in q:
        raise KeyError("Questionnaire data is missing subject_ID")

    if fixing_path.exists():
        fixes = pd.read_excel(fixing_path).iloc[:, :3].copy()
        fixes.columns = [str(c).strip() for c in fixes.columns]
        if "subject_ID" in fixes:
            fixes["subject_ID"] = fixes["subject_ID"].map(clean_id)
            q["subject_ID"] = q["subject_ID"].map(clean_id)
            for item in ("BFI[i01]", "BFI[i33]"):
                if item in q and item in fixes:
                    replacement = (fixes.dropna(subset=[item]).drop_duplicates("subject_ID")
                                   .set_index("subject_ID")[item])
                    q[item] = q[item].where(q[item].notna(), q["subject_ID"].map(replacement))

    # BFI-2 reverse scoring and five domain sums.
    bfi_items = [f"BFI[i{i:02d}]" for i in range(1, 61)]
    bfi = pd.DataFrame({"subject_ID": q["subject_ID"]})
    if all(c in q for c in bfi_items):
        scores = q[bfi_items].copy()
        levels = {"Disagree strongly": 1, "Disagree a little": 2, "Neutral": 3,
                  "Agree a little": 4, "Agree strongly": 5}
        for col in bfi_items:
            scores[col] = _map_responses(scores[col], levels)
        reverse = (11, 16, 26, 31, 36, 51, 12, 17, 22, 37, 42, 47,
                   3, 8, 23, 28, 48, 58, 4, 9, 24, 29, 44, 49,
                   5, 25, 30, 45, 50, 55)
        for n in reverse:
            col = f"BFI[i{n:02d}]"
            scores[col] = 6 - scores[col]
        domains = {
            "bfi_extraversion": (1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56),
            "bfi_agreeableness": (2, 7, 12, 17, 22, 27, 32, 37, 42, 47, 52, 57),
            "bfi_conscientiousness": (3, 8, 13, 18, 23, 28, 33, 38, 43, 48, 53, 58),
            "bfi_negative_emotionality": (4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59),
            "bfi_open_mindedness": (5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60),
        }
        for name, items in domains.items():
            cols = [f"BFI[i{i:02d}]" for i in items]
            bfi[name] = scores[cols].sum(axis=1, min_count=len(cols))

    # eBMRQ item recoding, reverse keys, subscales, and totals.
    ebmrq = pd.DataFrame({"subject_ID": q["subject_ID"]})
    e_items = [f"eBMRQ[i{i:02d}]" for i in range(1, 25)]
    if all(c in q for c in e_items):
        values = q[e_items].copy()
        levels = {"Completelydisagree": 1, "Disagree": 2,
                  "Neither agree nor disagree": 3, "Agree": 4,
                  "Completely agree": 5}
        for col in e_items:
            values[col] = _map_responses(values[col], levels)
        values["eBMRQ[i02]"] = 6 - values["eBMRQ[i02]"]
        values["eBMRQ[i05]"] = 6 - values["eBMRQ[i05]"]
        domains = {
            "eBMRQ_MusicSeeking": (8, 13, 20, 2),
            "eBMRQ_EmotionEvocation": (3, 9, 14, 21),
            "eBMRQ_MoodRegulation": (4, 10, 16, 22),
            "eBMRQ_SensoryMotor": (11, 17, 23, 5),
            "eBMRQ_SocialReward": (1, 7, 15, 19),
            "eBMRQ_MusicalAbsorption": (6, 12, 18, 24),
        }
        for name, items in domains.items():
            cols = [f"eBMRQ[i{i:02d}]" for i in items]
            ebmrq[name] = values[cols].sum(axis=1, min_count=len(cols))
        subscales = list(domains)
        ebmrq["eBMRQ_TOT"] = ebmrq[subscales].sum(axis=1, min_count=6)
        ebmrq["BMRQ_TOT"] = ebmrq[[c for c in subscales
                                     if c != "eBMRQ_MusicalAbsorption"]].sum(axis=1, min_count=5)

    # Gold-MSI has 38 contiguous columns in the dictionary's questionnaire order.
    gold = pd.DataFrame({"subject_ID": q["subject_ID"]})
    if "GMSI[AE01]" in q and "AE08" in q:
        left, right = q.columns.get_loc("GMSI[AE01]"), q.columns.get_loc("AE08")
        cols = list(q.columns[left:right + 1])
        if len(cols) != 38:
            raise ValueError(f"Expected 38 Gold-MSI fields; found {len(cols)}")
        values = q[cols].copy()
        likert = {
            "Completely disagree": 1, "Strongly disagree": 2, "Disagree": 3,
            "Neither agree or disagree": 4, "Agree": 5, "Strongly agree": 6,
            "Completely agree": 7,
        }
        for col in cols[:31]:
            values[col] = _map_responses(values[col], likert)
        for n in (9, 11, 13, 14, 17, 21, 23, 25, 27):
            values[cols[n - 1]] = 8 - values[cols[n - 1]]
        maps = (
            {"0": 1, "1": 2, "2": 3, "3": 4, "4-5": 5, "6-9": 6, "10 or more": 7},
            {"0": 1, "0.5": 2, "1": 3, "1.5": 4, "2": 5, "3-4": 6, "5 or more": 7},
            {"0": 1, "1": 2, "2": 3, "3": 4, "4-6": 5, "7-10": 6, "11 or more": 7},
            {"0": 1, "0.5": 2, "1": 3, "2": 4, "3": 5, "4-6": 6, "7 or more": 7},
            {"0": 1, "0.5": 2, "1": 3, "2": 4, "3-5": 5, "6-9": 6, "10 or more": 7},
            {"0": 1, "1": 2, "2": 3, "3": 4, "4": 5, "5": 6, "6 or more": 7},
            {"0-15 min": 1, "15-30 min": 2, "30-60 min": 3, "60-90 min": 4,
             "2 hrs": 5, "2-3 hrs": 6, "4 hrs or more": 7},
        )
        for offset, mapping in enumerate(maps, start=31):
            values[cols[offset]] = _map_responses(values[cols[offset]], mapping)
        domains = {
            "gold_msi_active_engagement": (1, 3, 8, 15, 21, 24, 28, 34, 38),
            "gold_msi_perceptual_abilities": (5, 6, 11, 12, 13, 18, 22, 23, 26),
            "gold_msi_musical_training": (14, 27, 32, 33, 35, 36, 37),
            "gold_msi_emotions": (2, 9, 16, 19, 20, 31),
            "gold_msi_singing_abilities": (4, 7, 10, 17, 25, 29, 30),
            "gold_msi_general_sophistication": (1, 3, 4, 7, 10, 12, 14, 15,
                                                 17, 19, 23, 24, 25, 27, 29, 32, 33, 37),
        }
        for name, items in domains.items():
            gold[name] = values[[cols[i - 1] for i in items]].sum(axis=1, skipna=True)

    # Selected questionnaire fields plus instrument count/list.
    custom_re = re.compile(
        r"student|studyField|musicSchool|musicDegree|expertiseMusic|mainInstr|"
        r"otherInstr|AddMusicTraining|lastMusicActivity\[years\]|training|"
        r"practice|relativePitch|AP01|expertise|BI01|ST01|MT05",
        flags=re.IGNORECASE,
    )
    custom_cols = [c for c in q.columns
                   if custom_re.search(str(c)) and not str(c).endswith("Time")]
    custom = q[custom_cols].copy()
    custom["subject_ID"] = q["subject_ID"]
    instruments = [c for c in custom_cols
                   if str(c).startswith("otherInstr[") and c != "otherInstr[NO]"]
    count_items = [c for c in instruments if c != "otherInstr[other]"]
    instrument = pd.DataFrame({"subject_ID": q["subject_ID"]})
    instrument["n_instrument"] = q[count_items].eq("Yes").sum(axis=1) if count_items else 0
    def instrument_list(row: pd.Series) -> str:
        names = []
        for col in instruments:
            if pd.notna(row[col]) and str(row[col]).strip().casefold() != "no":
                match = re.search(r"\[(.*?)\]", str(col))
                if match:
                    names.append(match.group(1))
        return ",".join(names)
    instrument["played_instrument"] = q.apply(instrument_list, axis=1) if instruments else ""
    custom = custom.drop(columns=[c for c in custom if str(c).startswith("otherInstr[")],
                                 errors="ignore")
    custom = custom.merge(instrument, on="subject_ID", how="left")

    # Hollingshead socioeconomic status.
    ses = pd.DataFrame({"subject_ID": q["subject_ID"]})
    ses_fields = ("SESedu", "SESjob", "SESeduma", "SESjobma", "SESeudpa", "SESjobpa")
    if all(c in q for c in ses_fields):
        education = {
            "not applicable or unknown": 1, "less than 7th grade": 1,
            "junior high school, including 9th grade": 2,
            "partial high school, 10th or 11th grade": 3,
            "high school graduate": 4,
            "partial college, at least one year of specialized training": 5,
            "standard college or university graduation": 6,
            "graduate/professional training": 7,
        }
        def education_score(column: pd.Series) -> pd.Series:
            return column.astype("string").str.strip().str.casefold().map(education).astype(float)
        def job_score(column: pd.Series) -> pd.Series:
            def one(value: Any) -> float:
                if pd.isna(value):
                    return np.nan
                text = str(value).strip()
                match = re.match(r"([A-J])\s*[-–]", text, flags=re.IGNORECASE)
                if match:
                    return float(ord(match.group(1).upper()) - ord("A"))
                if "not applicable" in text.casefold() or "unknown" in text.casefold():
                    return 0.0
                return np.nan
            return column.map(one)
        for role, edu_col, job_col in (
            ("participant", "SESedu", "SESjob"),
            ("mother", "SESeduma", "SESjobma"),
            ("father", "SESeudpa", "SESjobpa"),
        ):
            edu = education_score(q[edu_col])
            job = job_score(q[job_col])
            ses[f"ses_{role}_edu"] = edu
            ses[f"ses_{role}_job"] = job
            ses[f"ses_{role}_final"] = (edu * 3 + job * 5).where(job.ne(0))
        relation = q.get("SESrelparent", pd.Series(index=q.index, dtype="object"))
        single = q.get("SESsingleParent", pd.Series(index=q.index, dtype="object"))
        separated = q.get("SESseparatedParent", pd.Series(index=q.index, dtype="object"))
        in_charge = single.where(single.notna(), separated)
        ses["SESrelparent"] = relation
        ses["parent_in_charge_of_the_economy"] = in_charge
        nuclear = []
        for i, value in enumerate(relation):
            rel = "" if pd.isna(value) else str(value).casefold()
            person = "" if pd.isna(in_charge.iloc[i]) else str(in_charge.iloc[i]).casefold()
            if "other" in rel:
                nuclear.append(np.nan)
            elif "divorc" in rel or "widow" in rel:
                key = "mother" if "mother" in person else "father" if "father" in person else None
                nuclear.append(ses[f"ses_{key}_final"].iloc[i] if key else np.nan)
            else:
                parents = [ses["ses_mother_final"].iloc[i], ses["ses_father_final"].iloc[i]]
                parents = [x for x in parents if pd.notna(x)]
                nuclear.append(float(np.mean(parents)) if parents else np.nan)
        ses["ses_nuclear_family"] = nuclear
    else:
        ses["ses_nuclear_family"] = np.nan

    return {"bfi": bfi, "ebmrq": ebmrq, "gold_msi": gold,
            "custom": custom, "ses": ses}


def _score_raven(frame: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in frame if str(c).endswith("correct") and c != "S1M02correct"]
    vals = frame[cols].apply(_numeric)
    return pd.DataFrame({
        "subject_ID": frame["subject_ID"],
        "raven": vals.sum(axis=1, min_count=len(cols)) if cols else np.nan,
    })


def score_tasks(
    selected: dict[str, pd.DataFrame], master: pd.DataFrame, fixing_path: Path,
    dictionary_path: Path,
) -> pd.DataFrame:
    """Calculate task/questionnaire scores and join the participant master."""
    missing = set(TASKS).difference(selected)
    if missing:
        raise ValueError("Missing selected task(s): " + ", ".join(sorted(missing)))

    digit = _score_block(selected["digit_span"], "block", "was_correct", "digit_span")
    spatial = _score_block(selected["spatial_span"], "NBlock", "correct", "spatial_span")

    melody_raw = selected["melody_span"].copy()
    dates = pd.to_datetime(melody_raw["date"], errors="coerce", format="mixed")
    if dates.isna().all():
        dates = pd.to_datetime(melody_raw["date"], errors="coerce", format="%a %b %d %Y")
    melody_raw["_date"] = dates
    cutoff = pd.Timestamp("2023-09-18")
    before = melody_raw.loc[
        (melody_raw["_date"] < cutoff) & _numeric(melody_raw["block_number"]).eq(1)
    ]
    before_score = before.groupby("subject_ID")["correct"].apply(_strict_sum).rename("melody_span").reset_index()
    before_score["melody_span_score"] = "block1"
    after = melody_raw.loc[
        (melody_raw["_date"] > cutoff) & _numeric(melody_raw["block_number"]).ne(0)
    ]
    after_blocks = (after.groupby(["subject_ID", "block_number"], dropna=False)["correct"]
                    .apply(_strict_sum).rename("score").reset_index())
    after_score = after_blocks.groupby("subject_ID", as_index=False)["score"].mean()
    after_score = after_score.rename(columns={"score": "melody_span"})
    after_score["melody_span_score"] = "block12"
    melody = pd.concat([before_score, after_score], ignore_index=True, sort=False)

    nb = selected["nback"].copy()
    eligible = nb.loc[_numeric(nb["block_number"]).between(1, 5)].copy()
    correct = eligible["correct"].astype("string").str.strip().str.casefold()
    eligible["_hit"] = correct.eq("hit").fillna(False).astype(int)
    eligible["_fa"] = correct.eq("false alarm").fillna(False).astype(int)
    nback = eligible.groupby("subject_ID", as_index=False).agg(
        hit=("_hit", "sum"), fa=("_fa", "sum")
    )
    nback["nback"] = nback["hit"] - nback["fa"]
    nback = nback[["subject_ID", "nback"]]

    raven = _score_raven(selected["raven"])
    p = selected["proms"]
    proms_cols = [c for c in p if str(c).endswith("SCORE")]
    proms = p[["subject_ID", *proms_cols]].copy()
    if {"PROMSTOTALSCORE", "PROMSMELODYSCORE"}.issubset(proms.columns):
        proms["PROMSTOTALSCORENOMELODY"] = (
            _numeric(proms["PROMSTOTALSCORE"]) - _numeric(proms["PROMSMELODYSCORE"])
        )
    q_scores = _score_questionnaires(selected["questionnaires"], fixing_path)
    score_tables = [raven, digit, spatial, melody, nback, proms, *q_scores.values()]

    # ID workbook supplies demographics, lab, matched ID, and WAIS vocabulary.
    anag = master.drop(columns=["subject_ID"], errors="ignore").drop_duplicates().copy()
    anag = anag.rename(columns={"true_subject_id": "subject_ID"})
    if "wais_vocabulary_score" in anag:
        score = pd.to_numeric(anag["wais_vocabulary_score"], errors="coerce")
        is_wais3 = anag["sheet_name"].isin({"BR_SaoPaulo", "PT_Lisbon"})
        anag["wais3"] = score.where(is_wais3)
        anag["wais4"] = score.where(~is_wais3)
        anag["waisvocabulary"] = np.where(is_wais3, score / 66, score / 57)

    final = anag
    for table in score_tables:
        final = final.merge(table, on="subject_ID", how="left", sort=False)
    final = final.rename(columns={"sheet_name": "unit_short"})

    dictionaries = load_dictionary(dictionary_path)
    clean_dict = dictionaries["clean"]
    keep = pd.to_numeric(clean_dict["keep"], errors="coerce").eq(1)
    columns = clean_dict.loc[keep, "column"].astype(str).tolist()
    columns = ["unit_short" if c == "sheet_name" else c for c in columns if c != "to_keep"]
    for col in columns:
        if col not in final:
            final[col] = np.nan
    final = final.loc[:, columns].copy()
    final["ses_nuclear_family"] = pd.to_numeric(
        final["ses_nuclear_family"], errors="coerce"
    ).fillna(8)

    # Group-specific preregistered eligibility rules.
    group = final["group"].astype("string").str.casefold()
    add_training = pd.to_numeric(final["AddMusicTraining[years]"], errors="coerce")
    last_activity = pd.to_numeric(final["lastMusicActivity[years]"], errors="coerce")
    training_years = pd.to_numeric(final["training[years]"], errors="coerce")
    mt_levels = {"0": 0, "0.5": 1, "1": 2, "2": 3, "3-5": 4, "6-9": 5, "10 or more": 6}
    mt = final["MT05"].astype("string").str.strip().map(mt_levels)
    nonmusician = group.eq("nonmusician")
    musician = group.eq("musician")
    fail = pd.Series(False, index=final.index)
    fail |= (nonmusician & add_training.gt(2)).fillna(False)
    fail |= (nonmusician & last_activity.lt(5)).fillna(False)
    # In R, missing musician training falls through to FALSE; missing MT05 is NA
    # and is ignored by the any(..., na.rm=TRUE) exclusion check.
    fail |= (musician & ~training_years.ge(10).fillna(False)).fillna(False)
    fail |= (musician & mt.notna() & mt.ne(6)).fillna(False)
    fail |= (nonmusician & mt.ge(4)).fillna(False)
    final["to_keep"] = ~fail
    if MATCHED_ID in final:
        excluded_partner_ids = set(final.loc[~final["to_keep"], MATCHED_ID].dropna().map(clean_id))
        final.loc[final["subject_ID"].map(clean_id).isin(excluded_partner_ids), "to_keep"] = False
    return final


def run_preprocessing(project_root: Path, output_root: Path) -> tuple[Path, Path]:
    """Run raw correction, dictionary selection, scoring, and clean CSV export."""
    corrected = fix_raw_sources(project_root, output_root)
    dictionary_path = project_root / "files" / "dictionary.xlsx"
    selected = select_raw_columns(corrected, dictionary_path, output_root)
    master = load_id_master(project_root / "data" / "id" / "id.xlsx")
    full = score_tasks(
        selected, master, project_root / "data" / "fixing" / "missing_items_BFI2.xlsx",
        dictionary_path,
    )
    main = full.copy()
    if MATCHED_ID in main:
        matched = main[MATCHED_ID]
        has_match = matched.notna() & matched.astype("string").str.casefold().ne("na")
        main = main.loc[has_match]
    if "unit_short" in main:
        main = main.loc[~main["unit_short"].isin({"AR_BuenosAires", "US_NewYork"})]

    clean_dir = output_root / "data" / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    full_path = clean_dir / "dat-clean-full.csv"
    main_path = clean_dir / "dat-clean.csv"
    full.to_csv(full_path, index=False)
    main.to_csv(main_path, index=False)
    LOGGER.info("Saved %s full rows and %s AMPPS rows", len(full), len(main))
    return main_path, full_path




