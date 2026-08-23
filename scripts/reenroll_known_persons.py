#!/usr/bin/env python3
"""
scripts/reenroll_known_persons.py — re-enrollment helper for known persons
whose stored face template may have been captured through the pre-fix,
buggy `detect_face()` (see TECH_DEBT.md: "Resolved: `detect_face()`'s
Haar cascade sometimes returned the WRONG PERSON'S face"). Turns "someone
should re-enroll these people" into "run this one command."

WHY THIS EXISTS: confirmed (same TECH_DEBT.md entry) that no raw
images/crops are ever retained anywhere, so there is no way to
retroactively AUDIT whether an existing `faceencoding` row is corrupted —
the only remediation is capturing a fresh photo and re-registering through
the now-fixed detection path.

This script does NOT reimplement registration. It imports and calls the
exact same functions the live `/register` endpoint uses —
`app.services.face_recognition.main._sync_decode_and_embed` (detection +
embedding, via the fixed `detect_face()`/`crop_face()`) and
`_sync_register_existing_person` (the DB insert + face-cache invalidation)
— just driven from a local image file path instead of an HTTP multipart
upload, via a tiny shim that satisfies the same `file.file.read()`
interface `_sync_decode_and_embed` expects.

USAGE
    # Safe, read-only — lists who needs re-enrollment, no DB writes
    python scripts/reenroll_known_persons.py --list

    # Interactive — walks through each enrolled person, asks for a photo
    # file path, shows what was detected, confirms before writing
    python scripts/reenroll_known_persons.py

    # Re-enroll one specific person non-interactively
    python scripts/reenroll_known_persons.py --personid 3 --photo photo.jpg

    # Also remove the old (possibly corrupted) encoding(s) for a person
    # after the new one is confirmed written — off by default, since it's
    # a destructive DB delete; requires an explicit extra confirmation.
    python scripts/reenroll_known_persons.py --personid 3 --photo photo.jpg --delete-old-encodings

A successful re-enrollment ADDS a new `faceencoding` row through the fixed
path. It does NOT delete the old (possibly corrupted) one automatically —
use --delete-old-encodings if you want the superseded encoding removed
rather than kept alongside the new one.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.db.session import create_session  # noqa: E402
from app.models.person import KnownPerson  # noqa: E402
from app.models.face_encoding import FaceEncoding  # noqa: E402
from sqlalchemy import select  # noqa: E402

import app.services.face_recognition.face_service as fs  # noqa: E402
import app.services.face_recognition.main as m  # noqa: E402


class _LocalFileUpload:
    """Minimal shim satisfying the `file.file.read()` interface
    _sync_decode_and_embed() expects from a FastAPI UploadFile, so it can
    be called with a local file path instead of an HTTP multipart upload
    — no reimplementation of its detection/embedding logic."""

    def __init__(self, path: str):
        self.file = open(path, "rb")

    def close(self):
        self.file.close()


def p(msg=""):
    print(msg, flush=True)


def list_enrolled_persons(db):
    persons = db.execute(select(KnownPerson)).scalars().all()
    rows = []
    for person in persons:
        encodings = db.execute(
            select(FaceEncoding).where(FaceEncoding.personid == person.personid)
        ).scalars().all()
        rows.append((person, encodings))
    return rows


def print_roster(rows):
    p(f"{'personid':>8}  {'name':<20} {'notes':<28} encodings")
    p("-" * 80)
    for person, encodings in rows:
        ids = ", ".join(str(e.faceencodingid) for e in encodings) or "(none)"
        p(f"{person.personid:>8}  {person.name or '':<20} {(person.notes or ''):<28} "
          f"{len(encodings)} [{ids}]")


def reenroll_one(db, personid: int, photo_path: str, delete_old: bool) -> bool:
    """Returns True on success. Reuses the real registration functions —
    see module docstring."""
    person = db.get(KnownPerson, personid)
    if person is None:
        p(f"  ERROR: no knownperson with personid={personid}")
        return False

    before = db.execute(
        select(FaceEncoding).where(FaceEncoding.personid == personid)
    ).scalars().all()
    before_ids = [e.faceencodingid for e in before]
    p(f"  BEFORE: {person.name!r} (personid={personid}) has {len(before)} "
      f"encoding(s): {before_ids}")

    if not os.path.exists(photo_path):
        p(f"  ERROR: photo file not found: {photo_path}")
        return False

    # Sanity check the fix is actually in place before writing anything —
    # abort loudly rather than silently re-enrolling through a reverted
    # detector. This is the "confirm the fixed path was used" guarantee.
    assert fs.FACE_LOCALIZATION_DETECTOR == "fastmtcnn", (
        f"face_service.FACE_LOCALIZATION_DETECTOR is {fs.FACE_LOCALIZATION_DETECTOR!r}, "
        "not 'fastmtcnn' — refusing to re-enroll through an unexpected detector. "
        "Check TECH_DEBT.md before proceeding."
    )

    upload = _LocalFileUpload(photo_path)
    try:
        embedding, _ = m._sync_decode_and_embed(upload)
    finally:
        upload.close()

    if embedding is None:
        p(f"  ERROR: no face detected in {photo_path} (via {fs.FACE_LOCALIZATION_DETECTOR}) "
          "— try a clearer photo with the person's face visible.")
        return False

    p(f"  Face detected via {fs.FACE_LOCALIZATION_DETECTOR} — generated a "
      f"{len(embedding)}-dim embedding.")

    try:
        faceencoding_id = m._sync_register_existing_person(personid, embedding)
    except Exception as e:
        p(f"  ERROR: registration failed: {e}")
        return False

    p(f"  AFTER:  new faceencoding id={faceencoding_id} written via the fixed "
      f"{fs.FACE_LOCALIZATION_DETECTOR} path (confirmed above, not silently "
      "falling back to the old Haar-cascade code — that code path no longer exists).")

    if delete_old and before_ids:
        p(f"  Deleting {len(before_ids)} superseded encoding(s): {before_ids}")
        for enc in before:
            db.delete(enc)
        db.commit()
        p("  Old encoding(s) deleted.")
    elif before_ids:
        p(f"  Old encoding(s) {before_ids} were KEPT (pass --delete-old-encodings "
          "to remove them after confirming the new one works).")

    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="List enrolled persons and exit (read-only, no writes).")
    ap.add_argument("--personid", type=int, help="Re-enroll only this person (non-interactive if --photo is also given).")
    ap.add_argument("--photo", help="Path to the new photo for --personid (non-interactive mode).")
    ap.add_argument("--delete-old-encodings", action="store_true",
                     help="After a successful re-enrollment, delete that person's prior encoding(s). "
                          "Off by default — this is a destructive DB delete.")
    args = ap.parse_args()

    db = create_session()
    try:
        rows = list_enrolled_persons(db)
        p(f"Found {len(rows)} enrolled known person(s) in the database right now "
          "(queried fresh, not assumed from a prior round):")
        print_roster(rows)
        p()

        if args.list:
            return

        if args.personid is not None and args.photo:
            p(f"Re-enrolling personid={args.personid} from {args.photo}...")
            ok = reenroll_one(db, args.personid, args.photo, args.delete_old_encodings)
            sys.exit(0 if ok else 1)

        if args.personid is not None and not args.photo:
            p("--personid given without --photo — nothing to do. Provide --photo, "
              "or omit --personid to run the interactive walkthrough for everyone.")
            sys.exit(1)

        # Interactive walkthrough — every enrolled person, one at a time.
        p("Interactive re-enrollment. For each person below, provide a path to a")
        p("fresh photo of their face, or press Enter to skip them for now.\n")
        results = []
        for person, encodings in rows:
            p(f"--- {person.name} (personid={person.personid}), "
              f"currently {len(encodings)} encoding(s) ---")
            photo_path = input(f"  Photo file path for {person.name} (Enter to skip): ").strip()
            if not photo_path:
                p("  Skipped.\n")
                continue
            delete_old = args.delete_old_encodings
            if not delete_old and encodings:
                answer = input(f"  Delete the {len(encodings)} old encoding(s) after "
                                f"the new one is confirmed? [y/N]: ").strip().lower()
                delete_old = answer == "y"
            ok = reenroll_one(db, person.personid, photo_path, delete_old)
            results.append((person.name, ok))
            p()

        if results:
            p("=" * 60)
            p("SUMMARY")
            for name, ok in results:
                p(f"  {name}: {'re-enrolled OK' if ok else 'FAILED'}")
        else:
            p("No one was re-enrolled this run.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
